#!/bin/bash
# strad32 공용 VLM 서빙 기동/정지 스크립트
#
#   ./serve.sh up         레플리카 N개 + 앞단 LB 기동
#   ./serve.sh down       전부 정리
#   ./serve.sh status     상태 확인
#   ./serve.sh logs [이름]
#   ./serve.sh preflight  기동하지 않고 몫 규칙(GPU 배정·RAM 총합·LB upstream 정합)만 검사
#   ./serve.sh add <GPU>  운영 중인 레플리카를 건드리지 않고 한 대 추가 (무중단 확장)
#
# 절차 원본 = docs/04 §1, 플래그 원본 = docs/08 §8.
# 아래 값들은 2026-07-28 strad32 실측으로 확정된 것이며, 바꾸기 전에 주석의 근거를 읽을 것.
set -euo pipefail

# ── 구성 ────────────────────────────────────────────────────────────────────
IMAGE="${IMAGE:-local-vlm/vllm:v0.24.0-dmt}"     # docker/Dockerfile 로 빌드
MODEL="${MODEL:-unsloth/Qwen3.6-35B-A3B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-qwen36-35b-a3b}"     # 클라이언트가 model 필드에 쓰는 이름
GPUS="${GPUS:-0 1}"                              # 사용할 GPU 인덱스. docs/05 상 dst+dmt 몫은 0-3
GPU_UUIDS="${GPU_UUIDS:-}"                       # 비우면 GPUS 인덱스로 조회. 카드 탈락 등 인덱스를 못 믿을 때 UUID 직접 나열
ALLOWED_GPUS="${ALLOWED_GPUS:-0 1 2 3}"          # docs/05 dst+dmt 몫. 이 밖의 인덱스는 기동 거부한다
BASE_PORT="${BASE_PORT:-8001}"                   # 레플리카는 8001, 8002, ... 순서
LB_PORT="${LB_PORT:-8000}"                       # docs/01 이 정한 공개 주소
HF_CACHE="${HF_CACHE:-/data01/dmt/hf-cache}"
CPUSET="${CPUSET:-0-15,32-47}"                   # docs/05 dst+dmt 몫 (NUMA node 0)
MEMSET="${MEMSET:-0}"
MEM_LIMIT="${MEM_LIMIT:-64g}"                    # 레플리카 1개당. DP=4 로 늘리면 64×4+2=258g > 몫 225g → 48g 이하로 내릴 것
LB_MEM="${LB_MEM:-2g}"                            # nginx 는 가볍다. 상한만 걸어 둔다
SHARE_MEM_GIB="${SHARE_MEM_GIB:-225}"            # docs/05 dst+dmt RAM 몫 (slice MemoryMax=241591910400)
ADD_MEM_LIMIT="${ADD_MEM_LIMIT:-32g}"            # add 로 붙이는 레플리카의 RAM 상한.
                                                 # 실사용은 레플리카당 약 10g 이므로 32g 로 충분하고,
                                                 # DP=4 에서 몫(225g)을 넘지 않는다 (64+64+32+32+2=194g)
RESTART="${RESTART:-unless-stopped}"             # 첫 기동 검증 때는 RESTART=no 권장. 실패 시 무한 재시작으로 GPU 를 붙잡는다
# nginx 는 이 디렉토리의 *.conf 만 include 한다. 여기에 다른 서비스의 .conf 를 두면 LB 에 함께 로드된다.
LB_CONF_DIR="${LB_CONF_DIR:-$(cd "$(dirname "$0")/../docker" && pwd)}"

# ── 확정된 vLLM 플래그와 근거 ───────────────────────────────────────────────
#   --gpu-memory-utilization 0.95
#       0.90 으로는 기동 실패한다. 가중치가 23.25 GiB 를 쓰는데(체크포인트가 순수 NVFP4 가
#       아니라 MIXED_PRECISION 이라 docs/02 의 "약 20GB" 추정보다 크다) 0.90 예산에서는
#       "Available KV cache memory: -1.09 GiB" 로 ValueError 가 난다.
#       0.95 의 결과(실측): KV 3.14 GiB = 270,767 tokens, 32K 요청 기준 최대 동시성 8.26x.
#       vLLM 0.21+ 는 CUDA graph 메모리도 프로파일에 포함하므로 0.95 의 실효는 0.9347 이고,
#       엔진이 로그로 "동일 KV 를 원하면 0.9653" 을 권고한다. VRAM 여유는 카드당 약 2.7 GiB.
#       KV 를 늘리려면 0.96-0.965 로 A/B 할 것 (부하 중 활성화 피크와 상충하므로 실측 필수).
#   --max-num-batched-tokens 10240
#       한 스텝 활성화 피크를 줄인다. 16384 이던 2026-08-03, 멀티이미지 긴 prefill 이
#       16K 배치를 채운 순간 flashinfer MoE workspace 일시 할당 1.04 GiB > VRAM 여유
#       1.02 GiB 로 레플리카 4대가 순차 전멸했다 (docs/08 §8). 10240 은 피크를 약
#       0.65 GiB 로 낮춰 마진을 확보한다.
#       주의: --disable-chunked-mm-input 이라 mm 아이템은 한 청크에 통째로 들어가야
#       한다. native 이미지 1장 = 약 9.4K 토큰이므로 9.5K 미만으로 내리면 native
#       해상도 요청이 스케줄 불가가 된다.
#   --max-num-seqs 16
#       hybrid 는 Mamba cache 블록이 한정적이라 캡이 필수 (docs/02 Hybrid 주의 1번).
#   --kv-cache-dtype fp8 은 넣지 않는다
#       가동 검증 게이트 1(NVFP4 정확도)과 교락된다. 게이트 통과 후 별도 A/B.
#   MoE FP4 플래그(VLLM_USE_FLASHINFER_MOE_FP4 등)도 넣지 않는다
#       vLLM 0.24.0 이 FLASHINFER_CUTLASS NvFp4 MoE 백엔드를 자동 선택한다(실측).
#   참고: block-wise FP8 체크포인트를 쓸 때는 VLLM_USE_DEEP_GEMM=0 이 필요하다.
#       DeepGEMM 이 SM120 의 scale factor 레이아웃을 몰라 "Unknown SF transformation" 으로
#       엔진이 죽는다. 현재 모델은 NVFP4 라 해당 없음.
VLLM_ARGS=(
  --served-model-name "$SERVED_NAME"
  --gpu-memory-utilization 0.95
  --max-model-len 32768
  --max-num-seqs 16
  --max-num-batched-tokens 10240
  --disable-chunked-mm-input
  --mm-processor-cache-gb 0
  --limit-mm-per-prompt '{"image": 4}'
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --enable-auto-tool-choice
)

gpu_uuid() {  # 인덱스 -> UUID. 인덱스는 카드가 bus 에서 빠지면 밀리므로 핀은 UUID 로 한다(docs/04 §2)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$1" '$1==i{print $2}'
}

replicas() {  # 이름을 앵커로 묶는다. docker 의 name 필터는 부분 문자열 매칭이라
              # 앵커가 없으면 남의 컨테이너(my-vlm-rag 등)까지 rm -f 대상이 된다.
  docker ps -a --filter "name=^vlm-r[0-9]+$" --format '{{.Names}}'
}

resolve_uuids() {  # 기동할 GPU 의 UUID 목록
  local g uuid out=""
  if [ -n "$GPU_UUIDS" ]; then echo "$GPU_UUIDS"; return 0; fi
  for g in $GPUS; do
    uuid="$(gpu_uuid "$g")"
    [ -n "$uuid" ] || { echo "GPU 인덱스 $g 를 찾을 수 없다" >&2; return 1; }
    out="$out $uuid"
  done
  echo "${out# }"
}

preflight() {  # docs/05 의 몫 규칙을 코드로 강제한다. 위반이면 기동하지 않는다.
  local n="$1" g cnt ups per lb total

  # 1) 배정 밖 GPU 차단 ("타 팀 GPU 사용 금지"). GPU_UUIDS 를 직접 준 경우는 인덱스 검사 대상이 아니다.
  if [ -z "$GPU_UUIDS" ]; then
    for g in $GPUS; do
      case " $ALLOWED_GPUS " in
        *" $g "*) ;;
        *) echo "거부: GPU $g 는 배정($ALLOWED_GPUS) 밖이다"; exit 1 ;;
      esac
    done
    # 카드가 하나라도 빠지면 인덱스가 밀려 배정 밖 카드를 잡을 수 있다(docs/04 §2).
    cnt="$(nvidia-smi -L | wc -l | tr -d ' ')"
    [ "$cnt" = "8" ] || { echo "거부: GPU ${cnt}장만 보인다(정상 8장). 인덱스 밀림 위험 → GPU_UUIDS 로 직접 지정할 것"; exit 1; }
  fi

  # 2) RAM 총합. 컨테이너는 계정 slice 밖이라 cgroup 이 몫을 막아주지 않는다(docs/05 §도커 사용 시).
  per="${MEM_LIMIT%g}"; lb="${LB_MEM%g}"; total=$((per * n + lb))
  [ "$total" -le "$SHARE_MEM_GIB" ] \
    || { echo "거부: RAM 총합 ${total}g > 몫 ${SHARE_MEM_GIB}g. MEM_LIMIT 를 내릴 것"; exit 1; }

  # 3) 레플리카 수와 LB upstream 수 정합. 어긋나면 죽은 포트로 보내거나 레플리카가 유휴로 남는다.
  ups="$(grep -cE '^[[:space:]]*server[[:space:]]+host\.docker\.internal:' "$LB_CONF_DIR/nginx-lb.conf")"
  [ "$ups" = "$n" ] \
    || { echo "거부: 레플리카 ${n}개인데 nginx upstream 은 ${ups}개다 ($LB_CONF_DIR/nginx-lb.conf)"; exit 1; }

  echo ">> preflight ok: 레플리카 ${n}개, RAM ${total}g/${SHARE_MEM_GIB}g, upstream ${ups}개, restart=$RESTART"
}

up() {
  local uuids n=0
  uuids="$(resolve_uuids)"
  preflight "$(echo "$uuids" | wc -w | tr -d ' ')"

  for uuid in $uuids; do
    local name="vlm-r${n}" port=$((BASE_PORT + n))
    echo ">> $name  $uuid  ->  :$port"
    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run -d --name "$name" --restart "$RESTART" \
      --gpus "\"device=$uuid\"" \
      --cpuset-cpus="$CPUSET" --cpuset-mems="$MEMSET" \
      --memory="$MEM_LIMIT" --memory-swap="$MEM_LIMIT" --shm-size=16g \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -v "$HF_CACHE":/cache \
      -p "$port":8000 \
      "$IMAGE" "$MODEL" "${VLLM_ARGS[@]}" >/dev/null
    n=$((n + 1))
  done

  echo ">> vlm-lb  ->  :$LB_PORT"
  docker rm -f vlm-lb >/dev/null 2>&1 || true
  # 설정은 파일이 아니라 디렉토리로 마운트한다.
  # 파일 마운트는 inode 를 묶기 때문에 호스트에서 편집(sed -i 등)해도 컨테이너에 반영되지 않는다.
  # LB 도 자원 제한을 명시한다. 컨테이너는 계정 cgroup slice 밖이므로(docs/05 §도커 사용 시)
  # 가벼운 프로세스라도 배정 밖 CPU·NUMA 에서 스케줄될 수 있다.
  docker run -d --name vlm-lb --restart "$RESTART" \
    --add-host=host.docker.internal:host-gateway \
    --cpuset-cpus="$CPUSET" --cpuset-mems="$MEMSET" \
    --memory="$LB_MEM" --memory-swap="$LB_MEM" \
    -v "$LB_CONF_DIR":/etc/nginx/conf.d:ro \
    -p "$LB_PORT":8000 nginx:alpine >/dev/null

  echo ">> 기동 대기 (모델 로드 + CUDA graph 캡처로 수 분 걸린다)"
  # starting 이 아닌 상태(exited, unhealthy)는 즉시 실패로 본다.
  # 그냥 healthy 만 기다리면 죽은 컨테이너를 15분(60×15s) 동안 기다린 뒤 성공처럼 끝난다.
  for _ in $(seq 1 60); do
    local all=1 st
    for c in $(replicas); do
      st="$(docker inspect "$c" --format '{{.State.Status}}/{{.State.Health.Status}}')"
      case "$st" in
        running/healthy)  ;;
        running/starting) all=0 ;;
        *) echo ">> $c 기동 실패 ($st). 마지막 로그 30줄:"; docker logs --tail 30 "$c" 2>&1; exit 1 ;;
      esac
    done
    [ "$all" = "1" ] && { echo ">> 전 레플리카 healthy"; break; }
    sleep 15
  done
  status
}

down() {
  docker rm -f vlm-lb >/dev/null 2>&1 || true
  for c in $(replicas); do
    docker rm -f "$c" >/dev/null 2>&1 || true
  done
  echo "정리 완료"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}

status() {
  docker ps --filter "name=^vlm-" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  echo
  printf "LB  :%s  " "$LB_PORT"; curl -s -m 5 "localhost:$LB_PORT/lb-health" || echo "(응답 없음)"
  printf "모델: "; curl -s -m 10 "localhost:$LB_PORT/v1/models" \
    | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])' 2>/dev/null || echo "(조회 실패)"
}

add() {  # 운영 레플리카를 건드리지 않고 한 대 추가한다. 무중단 확장 경로.
  local g="$1" n port name uuid used total ups
  case " $ALLOWED_GPUS " in
    *" $g "*) ;;
    *) echo "거부: GPU $g 는 배정($ALLOWED_GPUS) 밖이다"; exit 1 ;;
  esac
  local cnt; cnt="$(nvidia-smi -L | wc -l | tr -d ' ')"
  [ "$cnt" = "8" ] || { echo "거부: GPU ${cnt}장만 보인다(정상 8장). 인덱스 밀림 위험"; exit 1; }

  # 대상 GPU 가 비어 있어야 한다. 남의 실험이나 우리 레플리카가 이미 올라와 있으면 거부.
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")"
  [ "$used" -lt 1000 ] || { echo "거부: GPU $g 가 이미 ${used}MB 사용 중이다"; exit 1; }

  n="$(replicas | wc -l | tr -d ' ')"
  name="vlm-r${n}"; port=$((BASE_PORT + n))
  docker inspect "$name" >/dev/null 2>&1 && { echo "거부: $name 이 이미 있다"; exit 1; }

  # RAM 몫. 기존 컨테이너 상한 합에 신규를 더해 확인한다.
  local cur=0 m
  for c in $(docker ps -a --format '{{.Names}}' | grep -E '^vlm-'); do
    m="$(docker inspect -f '{{.HostConfig.Memory}}' "$c" 2>/dev/null || echo 0)"
    cur=$((cur + m / 1073741824))
  done
  total=$((cur + ${ADD_MEM_LIMIT%g}))
  [ "$total" -le "$SHARE_MEM_GIB" ] \
    || { echo "거부: RAM 총합 ${total}g > 몫 ${SHARE_MEM_GIB}g (ADD_MEM_LIMIT 를 내릴 것)"; exit 1; }

  uuid="$(gpu_uuid "$g")"
  [ -n "$uuid" ] || { echo "GPU 인덱스 $g 를 찾을 수 없다"; exit 1; }

  echo ">> $name  GPU$g ($uuid)  ->  :$port  RAM ${cur}g+${ADD_MEM_LIMIT}=${total}g/${SHARE_MEM_GIB}g"
  docker run -d --name "$name" --restart "$RESTART" \
    --gpus "\"device=$uuid\"" \
    --cpuset-cpus="$CPUSET" --cpuset-mems="$MEMSET" \
    --memory="$ADD_MEM_LIMIT" --memory-swap="$ADD_MEM_LIMIT" --shm-size=16g \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$HF_CACHE":/cache \
    -p "$port":8000 \
    "$IMAGE" "$MODEL" "${VLLM_ARGS[@]}" >/dev/null

  echo ">> 기동 대기 (모델 로드 + CUDA graph 캡처)"
  for _ in $(seq 1 60); do
    local st; st="$(docker inspect "$name" --format '{{.State.Status}}/{{.State.Health.Status}}')"
    case "$st" in
      running/healthy) echo ">> $name healthy"; break ;;
      running/starting) ;;
      *) echo ">> 기동 실패 ($st). 마지막 로그 30줄:"; docker logs --tail 30 "$name" 2>&1; exit 1 ;;
    esac
    sleep 15
  done

  # LB 는 이 레플리카를 아직 모른다. upstream 을 맞춘 뒤 reload 해야 트래픽이 간다.
  ups="$(grep -cE '^[[:space:]]*server[[:space:]]+host\.docker\.internal:' "$LB_CONF_DIR/nginx-lb.conf")"
  n="$(replicas | wc -l | tr -d ' ')"
  if [ "$ups" != "$n" ]; then
    echo ">> 주의: nginx upstream ${ups}개 != 레플리카 ${n}개."
    echo "   $LB_CONF_DIR/nginx-lb.conf 에 'server host.docker.internal:$port ...' 를 추가하고"
    echo "   docker exec vlm-lb nginx -t && docker exec vlm-lb nginx -s reload 를 실행할 것"
  fi
  status
}

case "${1:-status}" in
  up)     up ;;
  add)    add "${2:?GPU 인덱스}" ;;
  down)   down ;;
  status) status ;;
  logs)   docker logs -f "${2:-vlm-r0}" ;;
  # 기동하지 않고 몫 규칙만 검사한다. 설정을 바꿨을 때 먼저 이걸로 확인할 것
  preflight) preflight "$(resolve_uuids | wc -w | tr -d ' ')" ;;
  *)      echo "usage: $0 {up|add <GPU>|down|status|logs [name]|preflight}"; exit 1 ;;
esac
