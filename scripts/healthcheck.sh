#!/bin/bash
# strad32 부팅 후 점검 스크립트. 서버에서 실행: bash healthcheck.sh
# 모든 항목 PASS면 vLLM 스모크 테스트(serve-smoke.sh) 진행 가능.

# 드라이버 버전 기준 (docs/03 소프트웨어 스택 표)
#   최소 575  - open kernel module + SM120 경로. 575.57+ 에서 vLLM 성능이 확연히 향상
#   권장 595+ - 안정 브랜치. 필수 아니므로 INFO 로만 안내
MIN_DRIVER=575
REC_DRIVER=595

pass=0; fail=0

ver_ge() { # ver_ge <버전> <기준> : 버전 >= 기준 이면 참
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]
}

check() { # check <이름> <명령> <기대 패턴>
  local name="$1" cmd="$2" expect="$3"
  local out
  out=$(eval "$cmd" 2>&1)
  if echo "$out" | grep -qE "$expect"; then
    echo "PASS  $name"
    pass=$((pass+1))
  else
    echo "FAIL  $name"
    echo "      => $out" | head -3
    fail=$((fail+1))
  fi
}

echo "=== 1. NVIDIA 드라이버 ==="
check "nvidia-smi 동작 (mismatch 해소)" "nvidia-smi --query-gpu=count --format=csv,noheader | head -1" "^[0-9]"
check "GPU 8장 인식" "nvidia-smi -L | wc -l" "^8$"
# 드라이버 버전을 세 곳에서 읽는다. 특정 버전과의 일치가 아니라
#   (a) 최소 버전 충족  (b) 세 값의 정합  (c) open module 여부  를 본다.
# (b)가 깨진 상태 = 드라이버 갱신 후 재부팅 전 -> nvidia-smi 가 죽는 전형적 사고.
drv_loaded=$(head -1 /proc/driver/nvidia/version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)  # 로드된 커널 모듈
drv_ondisk=$(modinfo -F version nvidia 2>/dev/null)                                                          # 디스크의 모듈
drv_nvml=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)                 # 사용자공간 (NVML)

if [ -n "$drv_loaded" ] && ver_ge "$drv_loaded" "$MIN_DRIVER"; then r=ok; else r=ng; fi
check "드라이버 >= ${MIN_DRIVER} (docs/03)" "echo \"$r (실측 ${drv_loaded:-?})\"" "^ok"

if [ -n "$drv_loaded" ] && [ "$drv_loaded" = "$drv_ondisk" ] && [ "$drv_loaded" = "$drv_nvml" ]; then r=ok; else r=ng; fi
check "드라이버 정합 (로드=디스크=NVML)" \
  "echo \"$r (로드 ${drv_loaded:-?} / 디스크 ${drv_ondisk:-?} / NVML ${drv_nvml:-?})\"" "^ok"

check "open kernel module (docs/03 필수)" "head -1 /proc/driver/nvidia/version" "Open Kernel Module"

if [ -n "$drv_loaded" ] && ! ver_ge "$drv_loaded" "$REC_DRIVER"; then
  echo "INFO  드라이버 $drv_loaded - docs/03 권장 안정 브랜치는 ${REC_DRIVER}+ (필수 아님)"
fi

echo "=== 2. GPU 상세 (기록용) ==="
nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.max,pcie.link.width.max,driver_version --format=csv,noheader

echo "=== 3. 자원 제한 생존 확인 ==="
for u in dst dmt vpt dpt; do
  uid=$(id -u "$u" 2>/dev/null)
  if [ -z "$uid" ]; then echo "FAIL  계정 $u 없음"; fail=$((fail+1)); continue; fi
  props=$(systemctl show "user-$uid.slice" -p AllowedCPUs -p AllowedMemoryNodes -p MemoryMax -p MemorySwapMax | tr '\n' ' ')
  case "$u" in
    dst|dmt) expect="AllowedCPUs=0-15 32-47.*AllowedMemoryNodes=0.*MemoryMax=241591910400" ;;
    vpt)     expect="AllowedCPUs=16-23 48-55.*AllowedMemoryNodes=1.*MemoryMax=120259084288" ;;
    dpt)     expect="AllowedCPUs=24-31 56-63.*AllowedMemoryNodes=1.*MemoryMax=120259084288" ;;
  esac
  if echo "$props" | grep -qE "$expect"; then echo "PASS  $u slice"; pass=$((pass+1));
  else echo "FAIL  $u slice"; echo "      => $props"; fail=$((fail+1)); fi
done

echo "=== 4. rke2 잔재 ==="
check "rke2-agent 비활성" "systemctl is-enabled rke2-agent 2>&1" "disabled|not-found|No such file"
check "kubelet 프로세스 없음" "pgrep -c kubelet || echo 0" "^0$"

echo "=== 5. 시스템 ==="
check "swap 꺼짐" "free | awk '/Swap/{print \$2}'" "^0$"
check "NUMA 2노드" "numactl --hardware | grep -c '^node [01] cpus'" "^2$"
check "docker 동작" "docker info --format ok 2>&1" "^ok$"

echo
echo "결과: PASS $pass / FAIL $fail"
[ "$fail" -eq 0 ] && echo "==> 전 항목 통과. 다음 단계: 모델 다운로드(download-models.sh) 또는 스모크 테스트" || echo "==> FAIL 항목 출력 확인 필요"
