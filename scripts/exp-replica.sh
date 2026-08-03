#!/bin/bash
# 실험용 레플리카를 유휴 배정 GPU에 띄운다. 운영 레플리카(vlm-r0, vlm-r1)는 건드리지 않는다.
#
#   ./exp-replica.sh up   <GPU인덱스> <포트> [vLLM 플래그 덮어쓰기...]
#   ./exp-replica.sh down <포트>
#   ./exp-replica.sh status
#
#   예) 플래그 A/B. 운영 서빙 재기동 없이 비교군을 만든다
#       ./exp-replica.sh up 2 8003 --max-num-seqs 32
#       ./exp-replica.sh up 3 8004 --gpu-memory-utilization 0.965
#
# 플래그 하나를 바꿔 보려고 운영 레플리카를 재기동하면 dst 에 다운타임이 생긴다.
# 남는 배정 GPU(docs/05 상 dst+dmt 몫의 실험 여유분)에 비교군을 따로 띄우면
# 다운타임 없이 A/B 가 된다. 측정은 stress/run.sh 로 이 포트를 직접 때린다.
#
# 기준 플래그는 scripts/serve.sh 의 VLLM_ARGS 와 같아야 한다. 바꾼 것만 인자로 넘긴다.
set -euo pipefail

IMAGE="${IMAGE:-local-vlm/vllm:v0.24.0-dmt}"
MODEL="${MODEL:-unsloth/Qwen3.6-35B-A3B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-qwen36-35b-a3b}"
ALLOWED_GPUS="${ALLOWED_GPUS:-0 1 2 3}"
HF_CACHE="${HF_CACHE:-/data01/dmt/hf-cache}"
CPUSET="${CPUSET:-0-15,32-47}"
MEMSET="${MEMSET:-0}"
MEM_LIMIT="${MEM_LIMIT:-32g}"          # 실험 레플리카는 몫 총합을 고려해 운영(64g)보다 작게
SHARE_MEM_GIB="${SHARE_MEM_GIB:-225}"
IDLE_VRAM_MB="${IDLE_VRAM_MB:-1000}"   # 이 값 이상 쓰는 GPU 는 유휴로 보지 않는다

# 운영과 동일한 기준 플래그. 실험 인자가 뒤에 붙어 같은 플래그를 덮어쓴다.
BASE_ARGS=(
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

our_mem_gib() {
  local total=0 m
  for c in $(docker ps -a --format '{{.Names}}' | grep -E '^vlm-'); do
    m="$(docker inspect -f '{{.HostConfig.Memory}}' "$c" 2>/dev/null || echo 0)"
    total=$((total + m / 1073741824))
  done
  echo "$total"
}

up() {
  local gpu="$1" port="$2"; shift 2
  local name="vlm-exp-$port"

  case " $ALLOWED_GPUS " in
    *" $gpu "*) ;;
    *) echo "거부: GPU $gpu 는 배정($ALLOWED_GPUS) 밖이다"; exit 1 ;;
  esac
  case "$port" in
    8[0-9][0-9][0-9]) ;;
    *) echo "거부: 포트 $port 는 dst+dmt 몫(8xxx) 밖이다"; exit 1 ;;
  esac
  case "$port" in
    8000|8001|8002) echo "거부: 포트 $port 는 운영 서빙이 쓴다"; exit 1 ;;
  esac
  ss -ltn 2>/dev/null | grep -q ":$port " && { echo "거부: 포트 $port 이미 사용 중"; exit 1; }

  local cnt; cnt="$(nvidia-smi -L | wc -l | tr -d ' ')"
  [ "$cnt" = "8" ] || { echo "거부: GPU ${cnt}장만 보인다(정상 8장). 인덱스 밀림 위험"; exit 1; }

  # 유휴 확인. 남의 실험이 올라와 있으면 붙지 않는다.
  local used uuid
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"
  [ "$used" -lt "$IDLE_VRAM_MB" ] || { echo "거부: GPU $gpu 가 이미 ${used}MB 사용 중이다"; exit 1; }
  uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$gpu" '$1==i{print $2}')"
  [ -n "$uuid" ] || { echo "거부: GPU $gpu UUID 조회 실패"; exit 1; }

  local used_g total_g
  used_g="$(our_mem_gib)"; total_g=$((used_g + ${MEM_LIMIT%g}))
  [ "$total_g" -le "$SHARE_MEM_GIB" ] \
    || { echo "거부: RAM 총합 ${total_g}g > 몫 ${SHARE_MEM_GIB}g. MEM_LIMIT 를 내리거나 유휴 컨테이너를 정리할 것"; exit 1; }

  echo ">> $name  GPU$gpu ($uuid)  ->  :$port  RAM ${used_g}g+${MEM_LIMIT}=${total_g}g/${SHARE_MEM_GIB}g"
  echo ">> 덮어쓴 플래그: $*"
  docker run -d --name "$name" --restart no \
    --gpus "\"device=$uuid\"" \
    --cpuset-cpus="$CPUSET" --cpuset-mems="$MEMSET" \
    --memory="$MEM_LIMIT" --memory-swap="$MEM_LIMIT" --shm-size=16g \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$HF_CACHE":/cache \
    -p "$port":8000 \
    "$IMAGE" "$MODEL" "${BASE_ARGS[@]}" "$@" >/dev/null

  echo ">> 기동 대기 (모델 로드 + CUDA graph 캡처)"
  for _ in $(seq 1 60); do
    local st; st="$(docker inspect "$name" --format '{{.State.Status}}/{{.State.Health.Status}}')"
    case "$st" in
      running/healthy) echo ">> healthy  http://localhost:$port/v1"; return 0 ;;
      running/starting) ;;
      *) echo ">> 기동 실패 ($st). 마지막 로그 30줄:"; docker logs --tail 30 "$name"; exit 1 ;;
    esac
    sleep 15
  done
  echo ">> 시간 초과. docker logs $name 확인"; exit 1
}

down() {
  local name="vlm-exp-${1:?포트}"
  docker rm -f "$name" >/dev/null 2>&1 || true
  echo "정리: $name"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}

case "${1:-status}" in
  up)     shift; up "$@" ;;
  down)   down "${2:?포트}" ;;
  status) docker ps --filter "name=^vlm-exp-" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' ;;
  *)      echo "usage: $0 {up <GPU> <포트> [플래그...]|down <포트>|status}"; exit 1 ;;
esac
