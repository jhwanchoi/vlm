# 03. RTX 5090 (SM120) 하드웨어 노트

- 작성: 2026-07-22
- 원본 리서치: [research/2026-07-22-brief-hardware.md](research/2026-07-22-brief-hardware.md) (이슈 번호/출처 포함)

## 카드 기본

- 32GB GDDR7, PCIe Gen5 x16, TDP 575W, Blackwell **SM120** (컨슈머)
- GeForce 세그먼트 제약: **NVLink 없음 · P2P 드라이버 차단 · MIG 없음 · ECC 없음.** MPS는 가능
- 주의: "Blackwell 지원" 커널이 데이터센터용 SM100/SM110만 커버하는 경우 많음. SM120은 별도 확인

## 인터커넥트: 이 박스 설계의 지배 변수

P2P 차단(NVIDIA 공식 확인, 2025-03)의 결과:

- NCCL은 shared-memory 폴백 → **모든 all-reduce가 GPU → host RAM → GPU** 경유
- vLLM 기동 시 "custom allreduce is disabled … lacks GPU P2P capability" 경고 = 정상, 영구
- 실측 (CloudRift 2025): 4×5090 TP=4가 **RTX PRO 6000 1장에 패배**.
  최고 처리량은 레플리카/PP 구성 (30B-AWQ 4레플리카 = 12,744 tok/s)
- MoE + expert parallel이 P2P 부재에 가장 불리한 트래픽 패턴

비공식 P2P 패치 드라이버 존재 (aikitoria/open-gpu-kernel-modules, 5090 지원, 실측 10-30% 향상).
단 ReBAR + `iommu=pt` + ACS off 필요 = DMA 격리 포기 + vLLM 코드 패치.
**팀 공용 박스에는 비권장.**

## 소프트웨어 스택: 2026-07 기준 권장 경로

| 컴포넌트 | 권장 | 비고 |
|---|---|---|
| 드라이버 | open kernel modules 필수, 575+ (595.71 안정) | 575.57에서 vLLM 성능 확연히 향상 |
| PyTorch | ≥2.9 (cu128/cu130) | sm_120은 2.7부터 공식 지원 |
| 추론 엔진 | **vLLM ≥0.17** | SM120 전용 FP8 GEMM 경로 |
| Attention | **FlashInfer** (또는 FA2/Triton) | 필요 시 `FLASHINFER_CUDA_ARCH_LIST=12.0f` |
| SGLang | 2순위. 모델별 검증 후 | SM120 blockwise FP8 미지원 등 이슈 잔존 |
| TensorRT-LLM | 계획에서 제외 | GeForce Blackwell multi-GPU 경로 미성숙 |

피할 것:

- 공식 flash-attn: SM120 미지원 (WGMMA 없음, shared mem 99KB라 Hopper 커널 이식 불가)
- xformers: 소스빌드 시 torch 다운그레이드 유발 사례

NCCL 환경변수:

```bash
NCCL_P2P_DISABLE=1            # P2P 탐색 실패 방지 (어차피 없음)
CUDA_DEVICE_ORDER=PCI_BUS_ID
# 행 걸리면: --disable-custom-all-reduce, NCCL_DEBUG=TRACE
```

## 정밀도

| 정밀도 | 판정 |
|---|---|
| **FP8 (W8A8)** | Blackwell 네이티브, vLLM ≥0.17에 SM120 전용 커널. **프로덕션 기본값** |
| FP8 KV cache | `--kv-cache-dtype fp8`. 32GB 카드 표준 |
| NVFP4 | 하드웨어 네이티브. 소프트웨어는 2026 들어 빠르게 성숙 중. [08 1번](08-optimization-catalog.md#1-양자화-사다리-sm120-판정표) 판정표 참조 |

FP4의 "3배" 홍보 수치는 이미지 생성/TensorRT 기준.
LLM decode는 bandwidth-bound라 이득 완만. 실이득은 **용량**(32GB에 큰 모델)과 prefill.

## 전력 / 쿨링 / 안정성 (8장 박스)

- 575W × 8 = GPU만 4.6kW, 실측 벽전력 4kW+. 카드당 <1ms 과도전류 **901W**. PSU/회로 여유 필수
- 12V-2x6 커넥터: 지속 575W에서 PSU측 배선 150°C 실측 사례. 커넥터 모니터링
- **전력 캡: strad32에 450W 적용 완료** (`gpu-power-cap.service`, [05 공유 자원 규칙](05-strad32-team-resource-split.md#공유-자원-규칙)).
  decode는 memory-bound라 손실 한 자릿수~15%, 전력 22% 절감
- 공랭 8장: 73-75°C + 스로틀링 사례 / 수랭: 58-60°C 무스로틀. 라이저/OCuLink는 Xid 79와 상관. 지양
- 알려진 이슈: Xid 79 / GSP heartbeat timeout (특정 AMD AGESA BIOS 트리거 사례, 롤백으로 해결), Xid 13/109/119
- **풀로드 번인 테스트 후 운영 투입**
- BIOS: Above-4G Decoding + Resizable BAR 활성

## 대안 참고: RTX PRO 6000 Blackwell 96GB

> 참고 기록용. **현 전제(strad32 1대 고정)에선 해당 없음.** 장비 계획이 바뀔 때만 유효.

같은 GB202 계열. 96GB, P2P/ECC/MIG 지원, ~$8-9.4K (2026-06).
32GB 초과 단일 VLM이 목적이면 1장이 4×5090 TP보다 빠르고 전력 1/4 (이 니치의 정석).
박스 내 실현은 5090 일부를 PRO 6000으로 교체하는 구성인데, 팀 공용 서버라 비현실적.
