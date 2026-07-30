#!/bin/bash
# 스트레스 테스트 오케스트레이션. 몫 preflight, 컨테이너 기동, 감시 연결, 결과 수집.
#
#   ./run.sh preflight <이름> <포트> <대상URL>
#   ./run.sh run       <이름> <포트> <대상URL>
#   ./run.sh status
#
#   예) ./run.sh run run1 8089 http://host.docker.internal:8000
#       STAGES="8,16" STAGE_SEC=90 SCENARIOS=S3 ./run.sh run run3c 8093 http://host.docker.internal:8000
#
# 이 스크립트의 1순위 요건은 성능 수치가 아니라 타 팀 자원 무침범이다.
# preflight 를 통과하지 못하면 절대 기동하지 않는다.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

IMAGE="${IMAGE:-locustio/locust:2.32.4}"
MODEL="${MODEL:-qwen36-35b-a3b}"
CPUSET="${CPUSET:-12-15,44-47}"          # docs/05 dst+dmt 몫(0-15,32-47)의 부분집합
MEMSET="${MEMSET:-0}"                     # NUMA 구역 0
MEM_LIMIT="${MEM_LIMIT:-8g}"
ALLOWED_CPUS="${ALLOWED_CPUS:-0-15,32-47}"
SHARE_MEM_GIB="${SHARE_MEM_GIB:-225}"
RESULTS_ROOT="${RESULTS_ROOT:-/data01/dmt/stress-results}"
LB_URL="${LB_URL:-http://localhost:8000}"

STAGES="${STAGES:-1,2,4,8,12,16,24,32,48,64}"
STAGE_SEC="${STAGE_SEC:-90}"
WARMUP_SEC="${WARMUP_SEC:-30}"
SCENARIOS="${SCENARIOS:-S1,S2,S3}"
IGNORE_EOS="${IGNORE_EOS:-1}"
USE_SCHEMA="${USE_SCHEMA:-0}"
MAX_TOKENS="${MAX_TOKENS:-256}"

cpuset_subset() {  # $1 ⊆ $2 인지. 배정 밖 코어를 쓰지 않는다는 것을 코드로 확인한다
  python3 - "$1" "$2" <<'PY'
import sys
def expand(spec):
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out
sys.exit(0 if expand(sys.argv[1]) <= expand(sys.argv[2]) else 1)
PY
}

our_mem_gib() {  # 이미 떠 있는 우리 컨테이너들의 RAM 상한 합 (GiB)
  local total=0 m
  for c in $(docker ps -a --format '{{.Names}}' | grep -E '^vlm-'); do
    m="$(docker inspect -f '{{.HostConfig.Memory}}' "$c" 2>/dev/null || echo 0)"
    total=$((total + m / 1073741824))
  done
  echo "$total"
}

preflight() {
  local name="$1" port="$2" target="$3" fail=0

  echo "== preflight: $name port=$port target=$target"

  docker image inspect "$IMAGE" >/dev/null 2>&1 \
    && echo "  ok  이미지 $IMAGE" \
    || { echo "  NG  이미지 없음: docker pull $IMAGE"; fail=1; }

  if cpuset_subset "$CPUSET" "$ALLOWED_CPUS"; then
    echo "  ok  cpuset $CPUSET ⊆ 배정 $ALLOWED_CPUS"
  else
    echo "  NG  cpuset $CPUSET 가 배정 $ALLOWED_CPUS 밖이다 (타 팀 코어 침범)"; fail=1
  fi

  local used new total
  used="$(our_mem_gib)"; new="${MEM_LIMIT%g}"; total=$((used + new))
  if [ "$total" -le "$SHARE_MEM_GIB" ]; then
    echo "  ok  RAM ${used}g + ${new}g = ${total}g ≤ 몫 ${SHARE_MEM_GIB}g"
  else
    echo "  NG  RAM 총합 ${total}g > 몫 ${SHARE_MEM_GIB}g"; fail=1
  fi

  case "$port" in
    8[0-9][0-9][0-9]) : ;;
    *) echo "  NG  포트 $port 는 dst+dmt 몫(8xxx) 밖이다"; fail=1 ;;
  esac
  case "$port" in
    8000|8001|8002|8003) echo "  NG  포트 $port 는 서빙이 쓰는 포트다"; fail=1 ;;
  esac
  if ss -ltn 2>/dev/null | grep -q ":$port "; then
    echo "  NG  포트 $port 이미 사용 중"; fail=1
  else
    echo "  ok  포트 $port 사용 가능"
  fi

  case "$target" in
    *:8000|*:8001|*:8002|*:8003) echo "  ok  대상 $target (우리 엔드포인트)" ;;
    *) echo "  NG  대상 $target 은 허용 목록(:8000-:8003) 밖이다"; fail=1 ;;
  esac

  mkdir -p "$RESULTS_ROOT" 2>/dev/null || true
  if [ -w "$RESULTS_ROOT" ]; then
    echo "  ok  결과 디렉토리 $RESULTS_ROOT 쓰기 가능"
  else
    echo "  NG  결과 디렉토리 $RESULTS_ROOT 쓰기 불가"; fail=1
  fi

  local missing=0
  for f in img_1mp.jpg img_native.jpg; do
    [ -f "$HERE/assets/$f" ] || { echo "  NG  이미지 없음: assets/$f (make_assets.py 로 생성해 복사)"; missing=1; }
  done
  [ "$missing" = 0 ] && echo "  ok  부하용 이미지 2종 존재"
  fail=$((fail + missing))

  if curl -s -m 8 "$LB_URL/v1/models" | grep -q '"id"'; then
    echo "  ok  서빙 응답 ($LB_URL/v1/models)"
  else
    echo "  NG  서빙이 응답하지 않는다 ($LB_URL)"; fail=1
  fi

  echo "  참고 GPU 는 요청하지 않는다 (--gpus 미지정). 부하기는 GPU 를 인식하지 못한다"

  [ "$fail" -eq 0 ] || { echo "== preflight 실패. 기동하지 않는다"; return 1; }
  echo "== preflight 통과"
}

run() {
  local name="$1" port="$2" target="$3"
  local container="vlm-stress-$name"

  # 테스트가 끝나도 컨테이너를 내리지 않는 정책이라, 같은 이름으로 다시 돌릴 때는
  # 포트가 점유된 상태다. 같은 이름은 명시적 교체로 본다 (다른 런의 컨테이너는 건드리지 않는다).
  if docker inspect "$container" >/dev/null 2>&1; then
    echo ">> 같은 이름의 부하기 컨테이너를 교체한다: $container"
    docker rm -f "$container" >/dev/null
    sleep 2
  fi

  preflight "$name" "$port" "$target"

  local rdir="$RESULTS_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-$name"
  mkdir -p "$rdir"
  # locust 이미지는 uid 1000 으로 돌고 결과 디렉토리는 dmt(1003) 소유이므로 열어준다.
  chmod 777 "$rdir"

  local stage_count total_sec
  stage_count="$(echo "$STAGES" | tr ',' '\n' | grep -c .)"
  total_sec=$((stage_count * STAGE_SEC + 120))

  # 실행 조건을 그대로 남긴다. 나중에 수치를 다시 볼 때 구성이 없으면 해석이 불가능하다.
  {
    echo "{"
    echo "  \"name\": \"$name\", \"target\": \"$target\", \"port\": $port,"
    echo "  \"started_utc\": \"$(date -u +%FT%TZ)\","
    echo "  \"stages\": \"$STAGES\", \"stage_sec\": $STAGE_SEC, \"warmup_sec\": $WARMUP_SEC,"
    echo "  \"scenarios\": \"$SCENARIOS\", \"ignore_eos\": $IGNORE_EOS, \"use_schema\": $USE_SCHEMA,"
    echo "  \"max_tokens\": $MAX_TOKENS, \"locust_image\": \"$IMAGE\","
    echo "  \"cpuset\": \"$CPUSET\", \"memset\": \"$MEMSET\", \"mem_limit\": \"$MEM_LIMIT\","
    echo "  \"serving_image\": \"$(docker inspect -f '{{.Config.Image}}' vlm-r0 2>/dev/null)\","
    echo "  \"vllm_args\": $(docker inspect -f '{{json .Args}}' vlm-r0 2>/dev/null || echo null),"
    echo "  \"containers_at_start\": ["
    docker ps --format '    {"name":"{{.Names}}","image":"{{.Image}}","status":"{{.Status}}"},' | sed '$ s/,$//'
    echo "  ],"
    echo "  \"gpu_at_start\": ["
    nvidia-smi --query-gpu=index,uuid,memory.used,temperature.gpu,clocks.sm --format=csv,noheader,nounits \
      | awk -F', *' '{printf "    {\"gpu\":%s,\"uuid\":\"%s\",\"mem_mb\":%s,\"temp\":%s,\"sm\":%s},\n",$1,$2,$3,$4,$5}' | sed '$ s/,$//'
    echo "  ]"
    echo "}"
  } > "$rdir/meta.json"

  # 감시할 레플리카는 부하 대상에서 정한다. LB 면 뒤의 레플리카 둘, 직결이면 그 포트.
  # 이걸 넘기지 않으면 monitor 가 기본값(8001 8002)만 보고, 실험 레플리카(:8003)로 부하를
  # 걸었을 때 엔진 지표가 통째로 비어 버린다(실측).
  # LB 대상이면 그 뒤에 실제로 떠 있는 레플리카를 전부 찾는다. 개수를 하드코딩하면
  # DP=4 로 확장한 뒤에도 2대만 보고 나머지 엔진 지표가 통째로 빠진다(실측).
  local watch
  case "$target" in
    *:8000) watch="$(for c in $(docker ps --filter "name=^vlm-r[0-9]+$" --format '{{.Names}}'); do
                       docker port "$c" 2>/dev/null | head -1 | sed 's/.*://'
                     done | sort -n | tr '\n' ' ')" ;;
    *)      watch="${target##*:}" ;;
  esac
  watch="${REPLICAS:-$watch}"   # 명시적으로 넘기면 그것을 쓴다
  echo ">> 감시 시작 (부하기와 분리된 프로세스). 감시 대상 레플리카: $watch"
  REPLICAS="$watch" nohup bash "$HERE/monitor.sh" "$rdir" "$port" "$container" >> "$rdir/monitor.out" 2>&1 &
  local mon_pid=$!
  echo "$mon_pid" > "$rdir/monitor.pid"
  sleep 4
  kill -0 "$mon_pid" 2>/dev/null || { echo "감시 프로세스가 즉시 죽었다. 기동 중단"; tail -20 "$rdir/monitor.out"; return 1; }

  echo ">> 부하기 기동 $container -> :$port (대상 $target, 단계 $STAGES × ${STAGE_SEC}s)"
  docker run -d --name "$container" \
    --add-host=host.docker.internal:host-gateway \
    --cpuset-cpus="$CPUSET" --cpuset-mems="$MEMSET" \
    --memory="$MEM_LIMIT" --memory-swap="$MEM_LIMIT" \
    -e TARGET_HOST="$target" -e MODEL="$MODEL" -e SCENARIOS="$SCENARIOS" \
    -e IGNORE_EOS="$IGNORE_EOS" -e USE_SCHEMA="$USE_SCHEMA" -e MAX_TOKENS="$MAX_TOKENS" \
    -e STAGES="$STAGES" -e STAGE_SEC="$STAGE_SEC" -e WARMUP_SEC="$WARMUP_SEC" \
    -e ASSET_DIR=/mnt/locust/assets -e PYTHONPATH=/mnt/locust \
    -v "$HERE":/mnt/locust:ro -v "$rdir":/results \
    -p "$port":"$port" \
    "$IMAGE" \
    -f /mnt/locust/locustfile.py --web-port "$port" --autostart \
    --run-time "${total_sec}s" \
    --csv /results/locust --csv-full-history --html /results/report.html >/dev/null

  echo ">> Web UI: http://localhost:$port (SSH 터널: ssh -L $port:localhost:$port dmt@<서버>)"
  echo ">> 결과: $rdir"

  # watchdog. 감시가 죽으면 부하를 즉시 멈춘다. 감시 없는 부하는 허용하지 않는다.
  local waited=0
  while [ "$waited" -lt "$total_sec" ]; do
    sleep 10; waited=$((waited + 10))
    if ! kill -0 "$mon_pid" 2>/dev/null; then
      echo ">> 감시 프로세스 사망. 부하 중단"
      curl -s -m 5 "http://localhost:$port/stop" >/dev/null 2>&1 || true
      docker stop "$container" >/dev/null 2>&1 || true
      echo "monitor died" > "$rdir/guard_triggered.txt"
      break
    fi
    if ! docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -q true; then
      echo ">> 부하기 컨테이너가 종료됨"; break
    fi
    local st
    st="$(curl -s -m 3 "http://localhost:$port/stats/requests" | grep -o '"state"[[:space:]]*:[[:space:]]*"[a-z]*"' | sed 's/.*"\([a-z]*\)"$/\1/' || true)"
    if [ "$st" = "stopped" ] && [ "$waited" -gt 60 ]; then
      echo ">> 테스트 종료 (state=stopped, ${waited}s 경과)"; break
    fi
    if [ -f "$rdir/guard_triggered.txt" ]; then
      echo ">> 가드 발동: $(cat "$rdir/guard_triggered.txt")"; break
    fi
    [ $((waited % 60)) -eq 0 ] && echo "   ... ${waited}s / ${total_sec}s  state=$st"
  done

  # 무한 부하 차단 3차 장치. shape 종료(1차)와 --run-time(2차)이 모두 안 들었으면 여기서 멈춘다.
  # /stop 은 GET 전용이다 (POST 는 405).
  local st_end
  st_end="$(curl -s -m 3 "http://localhost:$port/stats/requests" | grep -o '"state"[[:space:]]*:[[:space:]]*"[a-z]*"' | sed 's/.*"\([a-z]*\)"$/\1/' || true)"
  if [ "$st_end" = "running" ] || [ "$st_end" = "spawning" ]; then
    echo ">> 예정 시간 경과했는데 아직 실행 중이다. 강제 중단"
    curl -s -m 5 "http://localhost:$port/stop" >/dev/null 2>&1 || true
    sleep 5
  fi

  echo ">> 집계"
  python3 "$HERE/analyze.py" "$rdir" > "$rdir/summary.md" 2>"$rdir/analyze.err" \
    && { echo "== summary.md"; cat "$rdir/summary.md"; } \
    || { echo "집계 실패:"; cat "$rdir/analyze.err"; }

  echo ">> 부하기 컨테이너는 내리지 않는다 ($container, :$port 에서 결과 확인 가능)"
}

status() {
  docker ps --filter "name=^vlm-" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  echo
  ls -1t "$RESULTS_ROOT" 2>/dev/null | head -10
}

case "${1:-status}" in
  preflight) preflight "${2:-preflight}" "${3:-8089}" "${4:-http://host.docker.internal:8000}" ;;
  run)       run "${2:?이름}" "${3:?포트}" "${4:?대상URL}" ;;
  status)    status ;;
  *)         echo "usage: $0 {preflight|run <이름> <포트> <대상URL>|status}"; exit 1 ;;
esac
