#!/bin/bash
# Locust Web UI 에 남아 있는 데이터를 결과 디렉토리로 내보낸다.
#
#   ./export.sh            떠 있는 모든 vlm-stress-* 컨테이너
#   ./export.sh run1       특정 런만
#
# 테스트가 끝난 뒤 부하를 다시 걸지 않았다면 UI 의 통계는 종료 시점 스냅샷이다.
# 컨테이너를 내리면 이 데이터는 사라지므로(메모리에만 있다) 내보내 둔다.
# 내보내는 항목은 UI 의 탭 구성과 같다: Statistics, Charts, Failures, Exceptions,
# Current ratio, Logs.
set -uo pipefail

RESULTS_ROOT="${RESULTS_ROOT:-/data01/dmt/stress-results}"
ONLY="${1:-}"

port_of() {  # 컨테이너의 첫 공개 포트
  docker port "$1" 2>/dev/null | head -1 | sed 's/.*://'
}

export_one() {
  local name="$1" port="$2" dir out code total=0
  dir="$(ls -1td "$RESULTS_ROOT"/*-"$name" 2>/dev/null | head -1)"
  if [ -z "$dir" ]; then
    echo "  건너뜀: $name 의 결과 디렉토리가 없다"
    return 1
  fi
  out="$dir/export"
  mkdir -p "$out"

  fetch() {  # <경로> <저장이름> <설명>
    code="$(curl -s -m 120 -o "$out/$2" -w '%{http_code}' "http://localhost:$port/$1")"
    local sz=0
    [ -f "$out/$2" ] && sz="$(stat -c %s "$out/$2")"
    total=$((total + sz))
    printf '    %-22s %-24s HTTP %s  %s bytes\n' "$3" "$2" "$code" "$sz"
    [ "$code" = "200" ] || return 1
  }

  echo "  $name (:$port) -> $out"
  fetch "stats/requests/csv"      "statistics.csv"     "Statistics"
  fetch "stats/failures/csv"      "failures.csv"       "Failures"
  fetch "exceptions/csv"          "exceptions.csv"     "Exceptions"
  fetch "stats/report?download=1" "charts_report.html" "Charts (HTML)"
  fetch "tasks"                   "ratio.json"         "Current ratio"
  fetch "logs"                    "locust_logs.json"   "Logs"
  fetch "stats/requests"          "stats_snapshot.json" "상태 스냅샷"

  # 컨테이너 표준 출력. warmup 결과와 shape 전환 이력이 여기 남는다.
  docker logs "vlm-stress-$name" > "$out/container.log" 2>&1
  printf '    %-22s %-24s %s bytes\n' "컨테이너 로그" "container.log" "$(stat -c %s "$out/container.log")"

  {
    echo "{"
    echo "  \"exported_utc\": \"$(date -u +%FT%TZ)\","
    echo "  \"run\": \"$name\", \"port\": $port,"
    echo "  \"locust_state\": \"$(grep -o '"state": *"[a-z]*"' "$out/stats_snapshot.json" 2>/dev/null | head -1 | sed 's/.*"\([a-z]*\)"$/\1/')\","
    echo "  \"export_bytes\": $total"
    echo "}"
  } > "$out/export_meta.json"
  echo "    합계 $((total / 1024)) KB"
}

echo "== Locust 데이터 내보내기 (RESULTS_ROOT=$RESULTS_ROOT)"
found=0
for c in $(docker ps --filter "name=^vlm-stress-" --format '{{.Names}}'); do
  name="${c#vlm-stress-}"
  [ -z "$ONLY" ] || [ "$ONLY" = "$name" ] || continue
  port="$(port_of "$c")"
  if [ -z "$port" ]; then
    echo "  건너뜀: $c 의 공개 포트를 찾을 수 없다"
    continue
  fi
  export_one "$name" "$port" && found=$((found + 1))
done
echo "== 완료: ${found}개 런"
