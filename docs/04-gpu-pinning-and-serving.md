# 04. GPU 핀닝 & 서빙 토폴로지

- 작성: 2026-07-22, 갱신: 2026-07-30 (DP=2 가동 완료, 실측 반영)
- 원본 리서치: [research/2026-07-22-brief-deployment.md](research/2026-07-22-brief-deployment.md)

## 0. 현재 가동 상태 (2026-07-30, DP=4)

| | |
|---|---|
| 공개 주소 | `http://${STRAD32_IP}:8000/v1` (nginx LB) · model `qwen36-35b-a3b` · 인증 없음 |
| 레플리카 | `vlm-r0`~`vlm-r3` (GPU 0-3, :8001-:8004). 배정 4장 전부 서빙 (2026-07-30 dst 협의) |
| 이미지 | `local-vlm/vllm:v0.24.0-dmt` (`docker/Dockerfile`) |
| 기동 | `scripts/serve.sh up` — 몫 규칙은 `preflight` 로 사전 검사. 무중단 증설은 `add <GPU>` |
| 용량 | 레플리카당 KV 3.14 GiB = 270,767 tokens. 부하 실측 상한 **1,508 출력 tok/s, 포화점 동시 64, 권고 동시성 44** ([09](09-stress-test-results.md)) |
| RAM | 상한 합 194g / 몫 225g (운영 2대 64g + 증설 2대 32g + LB 2g) |

`:8001`/`:8002` 직결은 지표 조회·디버깅 전용. 클라이언트는 항상 `:8000`.
레플리카에 직접 붙으면 분배도 폴백도 없다.

## 1. 공용 모델 첫 가동 절차

확정 모델 Qwen3.6-35B-A3B NVFP4 ([02](02-model-candidates.md))를 dmt 계정으로 가동한다.
dmt가 서버를 소유·운영하고 dst는 API 클라이언트로 접속한다 ([01](01-project-overview.md) 공용 몫 관례).

| 단계 | 내용 | 도구 | 상태 (2026-07-30) |
|---|---|---|---|
| 1 | 모델 다운로드. dmt 계정, 캐시는 `/data01/dmt/hf-cache` (로그인 시 자동 설정) | `scripts/download-models.sh` | **완료** (35G) |
| 2 | 단일 레플리카 기동 (GPU 0, 포트 8001). 플래그는 [08 8번 플래그 스택](08-optimization-catalog.md#8-플래그-스택-실측-확정) + tool calling 3종(`--reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice`) | docker + vLLM **0.24.0** (cu130) | **완료** |
| 3 | 스모크: 텍스트 요청, 이미지 요청, tool call 왕복 각 1회 | `examples/quickstart.py`, `examples/curl.md` | **완료** (tool call 왕복은 미실행) |
| 4 | 가동 검증 게이트 5항목 실측 (NVFP4 OCR, prefix cache, bbox 좌표, DST 골든셋, vision+tool 동시) | [02 게이트 표](02-model-candidates.md#가동-검증-게이트-서빙-직후-실측-5항목) | 게이트 2만 완료 |
| 5 | DP 확장: 우선 레플리카 2개 (GPU 0-1) + 앞단 LB, 포트 8000. 남는 2장(GPU 2-3)은 실험 여유분 | 아래 3번 토폴로지 (C)안 | **완료** (nginx `least_conn` + `zone`) |
| 6 | 두 팀 공개: dst는 `http://${STRAD32_IP}:8000/v1` 접속 확인 | OpenAI 호환 클라이언트 | **완료** |

- 게이트 4(DST 골든셋)는 DST의 라벨 데이터가 필요하므로 나머지 게이트와 병렬로 준비한다
- 포트는 dst+dmt 몫 규칙(8xxx)을 따른다 ([05 공유 자원 규칙](05-strad32-team-resource-split.md#공유-자원-규칙))
- 서빙 규모는 DP=2로 시작했고 (미팅 합의: 실험 병목 최소화),
  2026-07-30 dst 협의로 **DP=4까지 확장 완료**했다.
  **증설은 `serve.sh add <GPU>` 로 한다.** 운영 레플리카를 건드리지 않아 다운타임이 없다.
  전체 재기동(`up`)으로 DP=4를 만들려면 `MEM_LIMIT` 을 48g 이하로 내려야 한다 (64g×4 + LB 2g = 258g > 몫 225g).
  `docker/nginx-lb.conf` 의 upstream 도 함께 늘려야 하며, 개수가 어긋나면 `serve.sh` 가 기동을 거부한다.
  **순서가 중요하다: 레플리카가 healthy 가 된 뒤에 upstream 을 늘리고 reload 한다.**
  뒤집으면 LB 가 죽은 포트로 보내 dst 가 502 를 받는다
- **LB 주의**: nginx upstream 에 `zone` 이 없으면 워커마다 연결 카운터가 따로여서
  동시 요청이 전부 첫 레플리카로 간다 (실측: 동시 6요청 6:0 → `zone` 추가 후 3:3)

## 2. 특정 GPU만 쓰게 하기

### 주의 두 가지

1. CUDA 기본 열거 순서는 `FASTEST_FIRST`이며 nvidia-smi 인덱스와 불일치 가능.
   GPU 하나가 bus에서 떨어지면 뒤 인덱스가 전부 밀림 → **UUID가 안전**
2. vLLM은 UUID 형식 `CUDA_VISIBLE_DEVICES` 파싱 버그 이력 (#32569, 2026-01 수정)

→ **권장 패턴: 컨테이너 레벨에서 UUID로 핀. vLLM은 내부에서 깨끗한 0-3 인덱스만 본다.**

### 방법별 정리

```bash
nvidia-smi -L                              # UUID 확보

# 베어메탈 - 항상 페어로
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3        # 또는 GPU-uuid,... 형식

# Docker (따옴표 주의 - 쉼표가 docker까지 도달해야 함)
docker run --gpus '"device=GPU-uuid1,GPU-uuid2,GPU-uuid3,GPU-uuid4"' ...

# CDI - 2026 표준 (Docker ≥28.3 기본 활성). 드라이버 업데이트 후 재생성
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
docker run --device nvidia.com/gpu=0 --device nvidia.com/gpu=1 ...
```

docker compose: `device_ids: ["GPU-uuid1", ...]` + `capabilities: [gpu]` (`count`와 상호배타).

### 강제 격리 옵션 (환경변수는 권고 수준)

```bash
sudo nvidia-smi -i 0,1,2,3 -c EXCLUSIVE_PROCESS   # 타 프로세스 컨텍스트 차단
sudo systemctl enable --now nvidia-persistenced

# EXCLUSIVE_PROCESS는 재부팅 시 리셋 → systemd oneshot으로 고정할 것
```

- strad32 운영에서는 사고 반복 시에만 도입한다 ([05 Level 2](05-strad32-team-resource-split.md#사고-발생-시-단계적-추가-조치))
- systemd 유닛이면 `DeviceAllow=/dev/nvidia0 rw` + `DevicePolicy=closed`로 cgroup 강제 격리 가능
- vLLM TP와 EXCLUSIVE_PROCESS는 호환 (GPU당 워커 1프로세스)

### Kubernetes

- 기존 device plugin(`nvidia.com/gpu: 4`)은 **특정 GPU 지정 불가** (플러그인이 임의 선택)
- 2026 답: **DRA** (K8s 1.34 GA). ResourceClaim의 CEL 셀렉터로 UUID/인덱스 지정 가능.
  이 박스를 KServe/Ray 클러스터에 편입하면 DRA로 갈 것
  (편입해도 GPU 노드는 strad32 1대, 단일 노드 전제, 스케일아웃 없음)

## 3. 서빙 토폴로지: TP vs PP vs DP

P2P 없는 PCIe 박스의 원칙: **모델이 들어가는 최소 병렬을 쓴다.**

| | TP=4 | PP | DP (레플리카) |
|---|---|---|---|
| 통신 비용 | 레이어당 all-reduce 2회 × 매 토큰 (host RAM 경유, 최악) | 스테이지 경계 activation만 | **0** |
| 처리량 | sub-linear (~1.4×/카드) | 고동시성에서 양호 | 거의 선형 4× |
| 장애 반경 | GPU 1개 에러 = 엔진 전체 다운 | 동일 | 레플리카 1개만 |
| 용도 | 진짜 4장 필요한 모델만 | 용량 + 처리량 | **32GB에 들어가면 무조건** |

```bash
# (A) 큰 모델 단일 엔진 (122B AWQ 등)
vllm serve <model> --tensor-parallel-size 4 \
     --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 4}'
# 반드시 TP2×PP2도 A/B: --tensor-parallel-size 2 --pipeline-parallel-size 2

# (B) 작은 모델 DP=4 - vLLM 내장 LB, 단일 엔드포인트
vllm serve <model> --data-parallel-size 4 --api-server-count 4

# (C) 완전 독립 레플리카 4개 + 외부 LB - 운영 최단순, 처리량 상한
CUDA_VISIBLE_DEVICES=0 vllm serve $M --port 8001 &   # GPU 1,2,3 반복
```

- dense 모델이면 (C)가 (B)와 동등하며 단순.
  `--data-parallel-size` 조율이 진짜 필요한 것은 **MoE** (expert 동기화, `--enable-expert-parallel`).
  단 MoE+EP는 P2P 부재에 가장 불리한 트래픽 패턴
- 외부 LB: nginx `least_conn` / **LiteLLM proxy** (least-busy + 키/쿼터/로깅) / vllm-router
- SGLang 대응: `--tp-size`, `sglang_router.launch_server --dp-size 4`
  (cache-aware 라우팅, shared-prefix 워크로드에 강점)

## 4. 서빙 스택 비교

| | vLLM | SGLang | TensorRT-LLM | Ollama/llama.cpp |
|---|---|---|---|---|
| VLM 커버리지 | 최광 | 우수 (Qwen day-0) | 좁음 | 제한적 |
| Structured output | xgrammar 기본 | xgrammar 기본 | 있음(덜 성숙) | JSON schema |
| SM120 성숙도 | **최고** | 이슈 잔존 | 최악 | 해당 없음 |
| 결론 | **기본 선택** | shared-prefix 헤비면 고려 | 보류 | 개인 실험용 |

Vision 요청: OpenAI 호환 `/v1/chat/completions` + `image_url` (URL 또는 base64 data URI).
대형 이미지는 토큰 폭발 주의. `--mm-processor-kwargs '{"max_pixels": ...}'` 캡 ([06](06-inspection-agent-design.md) 해상도 사다리).

## 5. 모니터링

- **dcgm-exporter** (`:9400/metrics`) + Prometheus + Grafana 대시보드 **12239**.
  GPU별 UUID 라벨 → 팀별 대시보드 분리 용이.
  GeForce라 `DCGM_FI_PROF_*` 제한. 문제 시 폴백 `utkuozdemir/nvidia_gpu_exporter`
- **엔진 자체 `/metrics` 필수**: `num_requests_running/waiting`, TTFT/TPOT 히스토그램, KV-cache 사용률.
  "박스가 바쁜 이유"(큐잉 vs KV 포화)는 엔진 지표로만 구분 가능
- 서빙 GPU UUID에 타 프로세스 VRAM 할당 시 알림 (EXCLUSIVE_PROCESS와 이중 방어)
