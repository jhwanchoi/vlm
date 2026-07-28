#!/bin/bash
# strad32 부팅 후 점검 스크립트. 서버에서 실행: bash healthcheck.sh
# 모든 항목 PASS면 vLLM 스모크 테스트(serve-smoke.sh) 진행 가능.

pass=0; fail=0
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
check "커널 모듈 버전 = 580.159" "cat /proc/driver/nvidia/version | head -1" "580\.159"

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
# docker info --format ok 는 데몬에 접근하지 못해도 템플릿 결과 ok 를 출력하므로
# 권한이 없는 계정에서도 통과한다. 데몬 왕복이 필요한 명령으로 확인한다.
check "docker 데몬 접근" "docker ps -q >/dev/null 2>&1 && echo ok || echo ng" "^ok$"
# 다음 단계인 vLLM 스모크 테스트는 --gpus 로 GPU 를 넘긴다.
# nvidia-container-toolkit 이 런타임으로 등록되어 있어야 동작한다.
check "docker GPU 런타임 (nvidia)" "docker info --format '{{json .Runtimes}}' 2>/dev/null" "\"nvidia\""

echo
echo "결과: PASS $pass / FAIL $fail"
[ "$fail" -eq 0 ] && echo "==> 전 항목 통과. 다음 단계: 모델 다운로드(download-models.sh) 또는 스모크 테스트" || echo "==> FAIL 항목 출력 확인 필요"
