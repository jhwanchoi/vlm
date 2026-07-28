#!/bin/bash
# strad32 공용 VLM 서빙 기동/정지 스크립트
#
#   ./serve.sh up       레플리카 N개 + 앞단 LB 기동
#   ./serve.sh down     전부 정리
#   ./serve.sh status    상태 확인
#   ./serve.sh logs [이름]
#
# 절차 원본 = docs/04 §1, 플래그 원본 = docs/08 §8.
# 아래 값들은 2026-07-28 strad32 실측으로 확정된 것이며, 바꾸기 전에 주석의 근거를 읽을 것.
set -euo pipefail

# ── 구성 ────────────────────────────────────────────────────────────────────
IMAGE="${IMAGE:-local-vlm/vllm:v0.24.0-dmt}"     # docker/Dockerfile 로 빌드
MODEL="${MODEL:-unsloth/Qwen3.6-35B-A3B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-qwen36-35b-a3b}"     # 클라이언트가 model 필드에 쓰는 이름
GPUS="${GPUS:-0 1}"                              # 사용할 GPU 인덱스. docs/05 상 dst+dmt 몫은 0-3
BASE_PORT="${BASE_PORT:-8001}"                   # 레플리카는 8001, 8002, ... 순서
LB_PORT="${LB_PORT:-8000}"                       # docs/01 이 정한 공개 주소
HF_CACHE="${HF_CACHE:-/data01/dmt/hf-cache}"
CPUSET="${CPUSET:-0-15,32-47}"                   # docs/05 dst+dmt 몫 (NUMA node 0)
MEMSET="${MEMSET:-0}"
MEM_LIMIT="${MEM_LIMIT:-64g}"
LB_CONF_DIR="${LB_CONF_DIR:-$(cd "$(dirname "$0")/../docker" && pwd)}"

# ── 확정된 vLLM 플래그와 근거 ───────────────────────────────────────────────
#   --gpu-memory-utilization 0.95
#       0.90 으로는 기동 실패한다. 가중치가 23.25 GiB 를 쓰는데(체크포인트가 순수 NVFP4 가
#       아니라 MIXED_PRECISION 이라 docs/02 의 "약 20GB" 추정보다 크다) 0.90 예산에서는
#       "Available KV cache memory: -1.09 GiB" 로 ValueError 가 난다.
#   --max-num-batched-tokens 16384
#       한 스텝 활성화 피크를 줄여 KV 여유를 만든다. 이미지 1장(약 9.4K 토큰)은
#       여전히 한 청크에 들어간다.
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
  --max-num-batched-tokens 16384
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

up() {
  local n=0
  for g in $GPUS; do
    local name="vlm-r${n}" port=$((BASE_PORT + n)) uuid
    uuid="$(gpu_uuid "$g")"
    [ -n "$uuid" ] || { echo "GPU 인덱스 $g 를 찾을 수 없다"; exit 1; }
    echo ">> $name  GPU$g ($uuid)  ->  :$port"
    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run -d --name "$name" --restart unless-stopped \
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
  docker run -d --name vlm-lb --restart unless-stopped \
    --add-host=host.docker.internal:host-gateway \
    -v "$LB_CONF_DIR":/etc/nginx/conf.d:ro \
    -p "$LB_PORT":8000 nginx:alpine >/dev/null

  echo ">> 기동 대기 (모델 로드 + CUDA graph 캡처로 수 분 걸린다)"
  for _ in $(seq 1 60); do
    local all=1
    for c in $(docker ps -a --filter "name=vlm-r" --format '{{.Names}}'); do
      [ "$(docker inspect "$c" --format '{{.State.Health.Status}}')" = "healthy" ] || all=0
    done
    [ "$all" = "1" ] && { echo ">> 전 레플리카 healthy"; break; }
    sleep 15
  done
  status
}

down() {
  docker rm -f vlm-lb >/dev/null 2>&1 || true
  for c in $(docker ps -a --filter "name=vlm-r" --format '{{.Names}}'); do
    docker rm -f "$c" >/dev/null 2>&1 || true
  done
  echo "정리 완료"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}

status() {
  docker ps --filter "name=vlm-" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  echo
  printf "LB  :%s  " "$LB_PORT"; curl -s -m 5 "localhost:$LB_PORT/lb-health" || echo "(응답 없음)"
  printf "모델: "; curl -s -m 10 "localhost:$LB_PORT/v1/models" \
    | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])' 2>/dev/null || echo "(조회 실패)"
}

case "${1:-status}" in
  up)     up ;;
  down)   down ;;
  status) status ;;
  logs)   docker logs -f "${2:-vlm-r0}" ;;
  *)      echo "usage: $0 {up|down|status|logs [name]}"; exit 1 ;;
esac
