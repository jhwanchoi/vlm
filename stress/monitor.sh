#!/bin/bash
# 부하 측정용 관측 + 자동 중단. 관측과 중단만 담당하며 부하기 내부는 모른다
# (Locust REST API 로만 통신한다).
#
#   ./monitor.sh <결과디렉토리> <locust포트> <locust컨테이너명>
#
# 부하기와 프로세스를 분리하는 이유: 부하기가 죽어도 감시와 중단 권한이 남아야 한다.
# 컨테이너가 살아 있는 동안 계속 돌면서, 테스트가 활성일 때만 샘플을 남긴다.
# Web UI 에서 사람이 수동으로 재실행해도 가드가 그대로 걸린다.
#
# 최우선 목적은 성능 수치가 아니라 타 팀(vpt GPU 4-5, dpt GPU 6-7) 무침범이다.
set -uo pipefail

RESULT_DIR="${1:?결과 디렉토리}"
PORT="${2:?locust 포트}"
CONTAINER="${3:?locust 컨테이너명}"

REPLICAS="${REPLICAS:-8001 8002}"
OTHER_GPUS="${OTHER_GPUS:-4 5 6 7}"          # 타 팀 몫. 감시 대상이며 우리는 쓰지 않는다
LB_CONTAINER="${LB_CONTAINER:-vlm-lb}"
OUR_CONTAINERS="${OUR_CONTAINERS:-vlm-r0 vlm-r1}"

# 가드 임계. 하나라도 걸리면 부하를 멈춘다.
GUARD_TEMP="${GUARD_TEMP:-83}"                # 타 팀 GPU 온도 (도)
GUARD_CLK_DROP_PCT="${GUARD_CLK_DROP_PCT:-15}"  # 타 팀 GPU SM 클럭 하락률
GUARD_CLK_SUSTAIN="${GUARD_CLK_SUSTAIN:-30}"  # 클럭 하락 지속 초
# 메모리 가드는 MemAvailable 로 본다. numactl 의 node free 는 page cache 를 제외한 값이라
# 정상 운영 중에도 수 GB 로 낮게 나오며 압박 신호가 아니다 (실측: node0 free 1.9GB,
# 서버 available 450GB). node free 는 참고용으로 CSV 에만 남긴다.
GUARD_MEM_AVAIL_MB="${GUARD_MEM_AVAIL_MB:-20480}"
GUARD_OUR_MEM_GIB="${GUARD_OUR_MEM_GIB:-200}"   # 우리 컨테이너 실사용 합. 몫 225g 방어
GUARD_LOADAVG="${GUARD_LOADAVG:-96}"          # 코어 64의 1.5배
GUARD_5XX_PCT="${GUARD_5XX_PCT:-20}"          # LB 5xx 비율. dst 영향 감지
GUARD_5XX_SUSTAIN="${GUARD_5XX_SUSTAIN:-30}"
SLOW_EVERY="${SLOW_EVERY:-10}"                # 무거운 조회(5xx, docker stats) 주기 초

ENGINE_CSV="$RESULT_DIR/engine.csv"
GPU_CSV="$RESULT_DIR/gpu.csv"
HOST_CSV="$RESULT_DIR/host.csv"
GUARD_LOG="$RESULT_DIR/guard.log"

mkdir -p "$RESULT_DIR"
[ -s "$ENGINE_CSV" ] || echo "ts,port,running,waiting,waiting_capacity,waiting_deferred,kv_usage,preemptions,success,gen_tokens,queue_sum,queue_count,ttft_sum,ttft_count" > "$ENGINE_CSV"
[ -s "$GPU_CSV" ] || echo "ts,gpu,util,mem_used_mb,power_w,temp_c,sm_mhz" > "$GPU_CSV"
[ -s "$HOST_CSV" ] || echo "ts,state,load1,mem_avail_mb,our_mem_gib,node0_free_mb,node1_free_mb,lb_5xx,lb_total,stress_cpu_pct,r0_cpu_pct,r1_cpu_pct" > "$HOST_CSV"

log() { echo "$(date -u +%FT%TZ) $*" >> "$GUARD_LOG"; }

# 실수 비교. load average 와 메모리 사용량은 소수라 bash 의 [ -gt ] 로는 비교할 수 없다
# (임계를 0.1 로 주면 "integer expression expected" 로 조용히 실패해 가드가 안 걸린다).
fgt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a + 0 > b + 0)}'; }

log "monitor 시작 port=$PORT container=$CONTAINER 임계: temp>$GUARD_TEMP clk-$GUARD_CLK_DROP_PCT% mem_avail<${GUARD_MEM_AVAIL_MB}MB our_mem>${GUARD_OUR_MEM_GIB}g load>$GUARD_LOADAVG 5xx>$GUARD_5XX_PCT%"

locust_state() {  # spawning | running | stopped | cleanup | (조회 실패 시 unknown)
  curl -s -m 3 "http://localhost:$PORT/stats/requests" \
    | grep -o '"state"[[:space:]]*:[[:space:]]*"[a-z]*"' | head -1 \
    | sed 's/.*"\([a-z]*\)"$/\1/' || echo unknown
}

stop_load() {  # $1 = 이유
  # Locust 2.32 의 /stop 은 GET 전용이다. POST 는 405 를 돌려주며 curl 은 성공으로
  # 종료하므로, 코드를 확인하지 않으면 "중단했다"고 착각한 채 부하가 계속 돈다(실측).
  local code
  log "가드 발동: $1 -> Locust 중단 요청"
  code="$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://localhost:$PORT/stop")"
  if [ "$code" != "200" ]; then
    sleep 2
    code="$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://localhost:$PORT/stop")"
  fi
  if [ "$code" = "200" ]; then
    log "중단 완료 (HTTP 200)"
  else
    log "중단 실패 (HTTP $code). 컨테이너를 직접 정지한다"
    docker stop "$CONTAINER" >/dev/null 2>&1 \
      && log "컨테이너 $CONTAINER 정지" || log "컨테이너 정지도 실패. 수동 확인 필요"
  fi
  echo "$1" > "$RESULT_DIR/guard_triggered.txt"
}

engine_sample() {  # $1 = 포트. CSV 한 줄 반환
  local m
  m="$(curl -s -m 3 "http://localhost:$1/metrics")" || return 1
  printf '%s\n' "$m" | awk -v p="$1" -v ts="$2" '
    /^vllm:num_requests_running/                       {r=$2}
    /^vllm:num_requests_waiting\{/                     {w=$2}
    /^vllm:num_requests_waiting_by_reason\{.*capacity/  {wc=$2}
    /^vllm:num_requests_waiting_by_reason\{.*deferred/  {wd=$2}
    /^vllm:kv_cache_usage_perc/                        {kv=$2}
    /^vllm:num_preemptions_total/                      {pr=$2}
    /^vllm:request_success_total/                      {s+=$2}
    /^vllm:generation_tokens_total/                    {gt=$2}
    /^vllm:request_queue_time_seconds_sum/             {qs=$2}
    /^vllm:request_queue_time_seconds_count/           {qc=$2}
    /^vllm:time_to_first_token_seconds_sum/            {tts=$2}
    /^vllm:time_to_first_token_seconds_count/          {ttc=$2}
    END {printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
         ts,p,r+0,w+0,wc+0,wd+0,kv+0,pr+0,s+0,gt+0,qs+0,qc+0,tts+0,ttc+0}'
}

# 타 팀 GPU 클럭 기준선.
# 유휴 GPU 의 클럭(수백 MHz)을 기준선으로 잡으면, 그 카드가 더 깊은 절전으로 들어갈 때
# "하락"으로 오탐한다. 그래서 util 이 BUSY_UTIL 이상인 표본만 기준선으로 쓰고,
# 클럭 하락 판정도 busy 상태에서만 한다. throttle 은 부하 중에만 의미가 있다.
BUSY_UTIL="${BUSY_UTIL:-20}"
declare -A BASE_CLK
capture_baseline() {
  local idx util clk
  while IFS=', ' read -r idx util _ _ _ clk; do
    case " $OTHER_GPUS " in *" $idx "*)
      [ "$util" -ge "$BUSY_UTIL" ] || continue
      if [ -z "${BASE_CLK[$idx]:-}" ] || [ "$clk" -gt "${BASE_CLK[$idx]}" ]; then
        BASE_CLK[$idx]="$clk"
      fi ;;
    esac
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm --format=csv,noheader,nounits)
}
for _ in 1 2 3; do capture_baseline; sleep 1; done
log "타 팀 GPU 클럭 기준선(busy 표본만): $(for g in $OTHER_GPUS; do printf '%s=%s ' "$g" "${BASE_CLK[$g]:-미관측}"; done)"

# 부하기 컨테이너가 아직 없을 수 있다. run.sh 는 감시를 먼저 띄우고 컨테이너를 만든다.
WAIT_CONTAINER_SEC="${WAIT_CONTAINER_SEC:-180}"
waited=0
until docker inspect -f '{{.State.Running}}' "$CONTAINER" >/dev/null 2>&1; do
  capture_baseline
  sleep 2; waited=$((waited + 2))
  if [ "$waited" -ge "$WAIT_CONTAINER_SEC" ]; then
    log "컨테이너 $CONTAINER 가 ${WAIT_CONTAINER_SEC}s 안에 나타나지 않았다. 종료"
    exit 1
  fi
done
log "컨테이너 $CONTAINER 확인. 감시 시작"

clk_bad=0
x5_bad=0
tick=0
lb_5xx=0
lb_total=0
stress_cpu=""
r0_cpu=""
r1_cpu=""

while docker inspect -f '{{.State.Running}}' "$CONTAINER" >/dev/null 2>&1; do
  state="$(locust_state)"
  ts="$(date -u +%FT%TZ)"

  capture_baseline

  if [ "$state" != "running" ] && [ "$state" != "spawning" ]; then
    # 유휴. 디스크를 불리지 않도록 샘플을 남기지 않는다 (기준선은 위에서 이미 갱신했다).
    clk_bad=0; x5_bad=0
    sleep 5
    continue
  fi

  tick=$((tick + 1))

  for p in $REPLICAS; do
    engine_sample "$p" "$ts" >> "$ENGINE_CSV" || true
  done

  gpu_out="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm --format=csv,noheader,nounits)"
  printf '%s\n' "$gpu_out" | awk -v ts="$ts" -F', *' '{printf "%s,%s,%s,%s,%s,%s,%s\n", ts,$1,$2,$3,$4,$5,$6}' >> "$GPU_CSV"

  load1="$(cut -d' ' -f1 /proc/loadavg)"
  mem_avail="$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)"
  node0_free="$(numactl --hardware 2>/dev/null | awk '/^node 0 free/{print $4}')"
  node1_free="$(numactl --hardware 2>/dev/null | awk '/^node 1 free/{print $4}')"

  if [ $((tick % SLOW_EVERY)) -eq 1 ]; then
    read -r lb_5xx lb_total < <(docker logs "$LB_CONTAINER" --since "${SLOW_EVERY}s" --tail 5000 2>/dev/null \
      | awk '{s=$9} s ~ /^[0-9][0-9][0-9]$/ {t++; if (s ~ /^5/) f++} END{print (f+0)" "(t+0)}')
    stats="$(docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}' "$CONTAINER" $OUR_CONTAINERS 2>/dev/null)"
    stress_cpu="$(printf '%s\n' "$stats" | awk -v n="$CONTAINER" '$1==n{gsub("%","",$2); print $2}')"
    r0_cpu="$(printf '%s\n' "$stats" | awk '$1=="vlm-r0"{gsub("%","",$2); print $2}')"
    r1_cpu="$(printf '%s\n' "$stats" | awk '$1=="vlm-r1"{gsub("%","",$2); print $2}')"
    # MemUsage 는 "12.3GiB / 64GiB" 형태. 앞쪽 실사용만 GiB 로 합산한다.
    our_mem="$(printf '%s\n' "$stats" | awk '{u=$3; if (u ~ /GiB$/) {sub("GiB","",u); t+=u} else if (u ~ /MiB$/) {sub("MiB","",u); t+=u/1024}} END{printf "%.1f", t+0}')"
  fi

  echo "$ts,$state,$load1,${mem_avail:-},${our_mem:-},${node0_free:-},${node1_free:-},$lb_5xx,$lb_total,${stress_cpu:-},${r0_cpu:-},${r1_cpu:-}" >> "$HOST_CSV"

  # ── 가드 판정 ─────────────────────────────────────────────────────────────
  # 1) 타 팀 GPU 온도. 우리 카드 2장의 지속 부하가 섀시 흡기 온도를 올릴 수 있다.
  hot="$(printf '%s\n' "$gpu_out" | awk -F', *' -v gs=" $OTHER_GPUS " -v lim="$GUARD_TEMP" \
    'index(gs, " "$1" ") && $5+0 > lim {print $1":"$5}' | tr '\n' ' ')"
  if [ -n "$hot" ]; then
    stop_load "타 팀 GPU 온도 ${GUARD_TEMP}도 초과 ($hot)"
    sleep 10; continue
  fi

  # 2) 타 팀 GPU SM 클럭 하락 (throttle 실발생 신호)
  drop=""
  while IFS=', ' read -r idx util _ _ _ clk; do
    case " $OTHER_GPUS " in *" $idx "*)
      base="${BASE_CLK[$idx]:-0}"
      # busy 인 카드만 본다. 유휴 카드의 저클럭은 절전이지 throttle 이 아니다.
      [ "$util" -ge "$BUSY_UTIL" ] || continue
      if [ "$base" -gt 0 ] && [ "$clk" -lt $((base * (100 - GUARD_CLK_DROP_PCT) / 100)) ]; then
        drop="$drop $idx:$clk<$base"
      fi ;;
    esac
  done < <(printf '%s\n' "$gpu_out")
  if [ -n "$drop" ]; then clk_bad=$((clk_bad + 1)); else clk_bad=0; fi
  if [ "$clk_bad" -ge "$GUARD_CLK_SUSTAIN" ]; then
    stop_load "타 팀 GPU SM 클럭 ${GUARD_CLK_DROP_PCT}% 하락이 ${GUARD_CLK_SUSTAIN}초 지속 ($drop)"
    clk_bad=0; sleep 10; continue
  fi

  # 3) 서버 회수 가능 메모리와 우리 몫 사용량
  if [ -n "${mem_avail:-}" ] && [ "$mem_avail" -lt "$GUARD_MEM_AVAIL_MB" ]; then
    stop_load "MemAvailable ${mem_avail}MB < ${GUARD_MEM_AVAIL_MB}MB"
    sleep 10; continue
  fi
  if [ -n "${our_mem:-}" ] && fgt "$our_mem" "$GUARD_OUR_MEM_GIB"; then
    stop_load "우리 컨테이너 메모리 ${our_mem}GiB > ${GUARD_OUR_MEM_GIB}GiB (몫 225g 방어)"
    sleep 10; continue
  fi

  # 4) 서버 전반 부하
  if fgt "$load1" "$GUARD_LOADAVG"; then
    stop_load "load average $load1 > $GUARD_LOADAVG"
    sleep 10; continue
  fi

  # 5) 공용 서비스 영향. 클라이언트 타임아웃은 5xx 를 만들지 않으므로
  #    파괴점 측정과 충돌하지 않는다.
  if [ "${lb_total:-0}" -gt 20 ]; then
    pct=$((lb_5xx * 100 / lb_total))
    if [ "$pct" -gt "$GUARD_5XX_PCT" ]; then x5_bad=$((x5_bad + SLOW_EVERY)); else x5_bad=0; fi
    if [ "$x5_bad" -ge "$GUARD_5XX_SUSTAIN" ]; then
      stop_load "LB 5xx ${pct}% 가 ${GUARD_5XX_SUSTAIN}초 지속 (${lb_5xx}/${lb_total})"
      x5_bad=0; sleep 10; continue
    fi
  fi

  sleep 1
done

log "monitor 종료 (컨테이너 $CONTAINER 없음)"
