# 08. 추론 최적화 전체 카탈로그

- 작성: 2026-07-22
- [07 로드맵](07-inference-optimization-roadmap.md)이 "실행 순서"라면,
  이 문서는 "전체 기법 목록 + 우리 스택(vLLM ≥0.17, SM120, PCIe-only) 기준 판정"
- 원본 리서치:
  [VLM 특화](research/2026-07-22-brief-opt-vlm-specific.md) ·
  [일반 서빙](research/2026-07-22-brief-opt-general-serving.md) (이슈/PR 번호, 출처 포함)

워크로드 재확인: **prefill-dominated**
(이미지 1K-9.4K 토큰 + few-shot prefix vs 출력 ~200-500 토큰).
prefill 축소/공유 기법이 지배적, decode 기법은 2차.

## 0. 한눈 요약

우선순위 (효과 × 확실성):

1. `max_pixels` 사다리: prefill 5-7× 스윙, 어떤 pruning 논문보다 큼
2. FP8/W4 양자화 + prefix 캐시 위생 (few-shot byte-identical, 배치 정렬)
3. `--mm-encoder-tp-mode data` + ViT CUDA graphs
4. `max_num_batched_tokens` 16-32K + full CUDA graphs + `-O3` compile
5. FP8 KV cache (정확도 스팟체크 후)
6. suffix/ngram spec decode (QA 엔드포인트)
7. 전력 스윕 450/500/575W

**Skip 확정**: INT8 W8A8 · W4A8 · 2:4 sparsity · GGUF · MPS ·
visual token pruning 논문 직접 통합 · Medusa ·
**P/D disaggregation · EPD** (서버 1대 고정 전제라 사실상 영구 skip. 워크로드가 요청당 크롭 4장+ 상시로 바뀌면 EPD만 재검토)

## 1. 양자화 사다리: SM120 판정표

| 기법 | SM120 상태 (2026-07) | 판정 |
|---|---|---|
| **FP8 W8A8** | 네이티브. v0.17에 SM120 전용 GEMM. `FP8_DYNAMIC` 레시피는 캘리브레이션 불필요 | **기본값. 성숙** |
| INT8 / SmoothQuant | vLLM 문서가 Blackwell 명시 미지원 ("FP8 쓰라") | skip |
| **AWQ/GPTQ W4A16** | Marlin 커널로 동작. Machete는 SM120 미지원. 5090 실측: GPTQ-Marlin이 AWQ보다 8-19% 빠름 | 성숙. VRAM-bound일 때 |
| W4A8 | CUTLASS 커널 CC90 전용 | skip |
| **NVFP4** | 동작: dense 직로드, MoE는 `VLLM_USE_FLASHINFER_MOE_FP4=1` + `--moe-backend=flashinfer_b12x` (2026-05 머지) | 빠르게 성숙 중. **1장/레플리카 구성의 열쇠. 확정 모델의 배포 정밀도** |
| MXFP4 | Triton 백엔드로 동작 | 체크포인트가 MXFP4일 때만 |
| GGUF | out-of-tree 플러그인, "highly experimental" | skip |
| **FP8 KV cache** | `--kv-cache-dtype fp8` (e4m3). KV 비용 ~54%, break-even 4-7K ctx. 무보정 손실 통상 1-2pt 이내 | 정확도 체크 후 활성. 주의: head_dim=256 모델은 prefill ~1.6× 퇴행 |
| llm-compressor + AutoRound | 표준 워크플로. AutoRound 통합(2025-12) | 자체 양자화 시 이 경로 |
| 2:4 sparsity | SM120 지원 없음 + pruned 체크포인트 없음 | skip |

> 비전 타워 주의: 공식 FP8 VLM 체크포인트는 **ViT+projector를 BF16으로 남김**
> (llm-compressor도 의도적 skip). ViT 직접 양자화 금물. 이득 미미, LM이 승부처.

## 2. VLM 특화 기법

### Visual token pruning/merging: 논문 직접 통합 비권장

- FastV/SparseVLM/VisionZip/PyramidDrop/ToMe/PruMerge 전부 vLLM/SGLang **네이티브 미지원** (2026-07)
- vLLM RFC #45098 (2026-06, 네이티브 image pruning 제안)이 첫 경로. **Q4 2026 재확인**
- 커스텀 통합은 Qwen DeepStack 구조와 궁합 나쁨 + 작은 텍스트 drop 위험이 큼.
  오버레이 숫자 판독 태스크에 치명적
- **같은 FLOPs 절감을 `max_pixels`로 정확·안전·캐시친화적으로 달성한다**
- EVS (video 전용, vLLM 머지됨): 연속 프레임 pseudo-video 묶음 시 사용 가능하나
  프레임별 JSON 판정과 상충. 보류
- 모델측 압축: InternVL3.5-Flash (ViR, visual token 50% 감축)는 모델 교체 옵션으로만

### Vision encoder

| 기법 | 판정 |
|---|---|
| `--mm-encoder-tp-mode data` | **필수 플래그.** ViT를 all-reduce (58-126회/forward) 대신 rank별 분담. P2P 없는 박스에서 이득 최대 (10-45%+) |
| ViT CUDA graphs (`cudagraph_mm_encoder: true`) | **활성.** Blackwell 실측 encoder 지연 11-44%↓, 수치 무변 |
| encoder torch.compile | ~3%. CUDA graphs와 동시 사용 "unspecified". graphs 우선 |
| EPD (encoder 분리) | **skip.** 요청당 이미지 4-8장 + 여유 GPU 플릿용. 단일 노드 + 1이미지/요청이면 colocate + 위 두 플래그가 우위 |

### 멀티모달 캐싱 3계층

1. **Processor cache** (`--mm-processor-cache-gb`)
   - 9.4K-토큰 프레임 1장 = pixel_values ~80-90MB → 기본 4GiB에 ~45장
   - 수천 유니크 프레임 배치에선 스래싱만 발생 → **배치 엔드포인트는 0으로 끔**
   - 인터랙티브(같은 프레임 재질의)는 `shm` 유지
2. **Encoder/embedding cache**: few-shot 예시 이미지가 매 요청 히트.
   예시를 프롬프트 **앞쪽에**, byte-identical로
3. **KV prefix cache**: 이미지 content hash(mm_hash, sha256)가 블록 해시에 포함.
   - per-request `max_pixels` 다르면 해시 다름 (오히려 안전)
   - 주의: Qwen3.5/3.6 hybrid는 GDN 때문에 prefix 캐시 granularity 528토큰 + "align mode experimental".
     긴 few-shot prefix라 구조상 문제없으나 **실동작 검증 필수** (가동 검증 게이트 2번)

보너스:

- **`image_embeds` 사전계산 입력**: 같은 프레임 코퍼스를 프롬프트만 바꿔 재실험할 때
  (encode 1회 오프라인, LM-only 서빙). shape 이슈 잔존, 중간 성숙도
- KV offload: native `--kv-offloading-backend native` (225GB NUMA-local RAM 활용) 먼저,
  재시작 간 영속 필요하면 LMCache

## 3. Attention / 커널 / 컴파일

- **Attention 백엔드**: SM120은 trtllm-gen FMHA cubin 미컴파일 → FlashInfer도 FA2급 폴백.
  `FLASH_ATTN` vs `FLASHINFER` vs `TRITON_ATTN` 실벤치 필수 (5-15% 차이 통상).
  v0.17의 FA4 통합이 SM120 커버하는지 빌드에서 확인
- **CUDA graphs**: sm_120 full graphs 동작 확인 (enforce-eager 대비 8× 사례).
  `--enforce-eager` 프로덕션 금지.
  FULL_AND_PIECEWISE 캡처 실패 시 `FULL_DECODE_ONLY` (균일한 짧은 decode에 적합)
- **torch.compile**: `-O3`. 나이틀리 배치는 스타트업이 수천 프레임에 상각.
  `VLLM_CACHE_ROOT` 영속화로 재기동 가속
- **Chunked prefill** (V1 기본): `--max-num-batched-tokens` 16-32K
  (이미지 1장이 ~1청크에 들어가게) + **`--disable-chunked-mm-input`**

## 4. 스케줄링 / 배칭

- `--max-num-batched-tokens`: 배치 16-32K / 인터랙티브 2-8K (ITL 보호)
- `--max-num-seqs`: KV-bound까지 상향. 주의: Qwen3.6 hybrid는 Mamba cache 한정, 캡 필요 (16-32 시작)
- **Priority scheduling** (`--scheduling-policy priority`):
  배치+인터랙티브 한 엔진 동거 (interactive=0, batch=100). GPU 분할 대신 검토 가능
- `--async-scheduling` 확인: 짧은 출력 = CPU 오버헤드 비중 큼
- **오프라인 배치는 `LLM.generate`/`LLM.chat`** (HTTP/SSE/토크나이저 오버헤드 제거).
  서버 필수면 `--api-server-count N` + `VLLM_MEDIA_LOADING_THREAD_COUNT` 상향
- **Parallel sampling (n>1)**: 공유 프롬프트 KV 1회 계산 + ref-count 공유.
  **self-consistency 투표 비용이 낮다** (에스컬레이션 패스 설계에 반영)

## 5. Speculative decoding 메뉴

전제: E2E에서 decode는 ~20-30%, 상한은 한 자릿수~20%. 0번 요약의 상위 항목 먼저.

| 방법 | 상태 | 판정 |
|---|---|---|
| **Suffix decoding** | `{"method":"suffix"}` + arctic-inference. 이전 생성물까지 매칭. 거의 동일한 JSON 수천 개에 최적 (ngram 대비 1.4-3.9×) | **decode측 1순위 실험** |
| ngram/prompt-lookup | 무료. 주의: `prompt_lookup_min≥8` (기본 2는 Qwen3계+structured output 오염 버그 #40875) | 2순위 |
| MTP (Qwen3.5/3.6 내장) | accept ~3.19 (n=3). **고동시성 처리량 퇴행. 배치 OFF**, 인터랙티브 전용 | 조건부 |
| DFlash (Qwen3.6) | 블록 draft. 동시성 유지하며 2.3-2.5× | Qwen3.6 채택으로 MTP보다 우선 검토 |
| EAGLE-3 | AngelSlim draft는 Qwen3-VL 2B/4B/8B/30B-A3B만. VLM 이득 작고 역효과 사례 (0.96×) | Qwen3.6은 내장 MTP/DFlash 우위라 해당 없음 |
| Medusa / draft-model | EAGLE-3에 대체 / 유지보수 경로 아님 | skip |

structured output과 spec decode는 V1에서 정합.
문법이 draft를 검증하므로 정확성 리스크 없음 (기각 draft는 낭비 컴퓨트일 뿐).

## 6. 출력측 다이어트

- 스키마 = 성능 튜닝 대상: 짧은 키, enum, `additionalProperties: false`,
  긴 문자열 `pattern` 금지, float 대신 int
- **출력 200→120 토큰 = decode 1.7× 직접 절감**
- 스키마 객체 1개를 런 전체 재사용 (xgrammar 컴파일 20-50ms 1회 상각)
- 배치는 non-streaming, tight `max_tokens`

## 7. 시스템 레벨

- **NUMA**: vLLM `--numa-bind-nodes`/`--numa-bind-cpus` 지원.
  [05 런북](05-strad32-team-resource-split.md)의 user-slice 분할과 정확히 정합 (dst+dmt = 구역 0).
  이미지 디코드/전처리도 GPU-local NUMA 노드에. 5-15% 통상
- **클라이언트 사전 리사이즈**: max_pixels에 맞춰 JPEG를 미리 줄여 전송.
  media decode + H2D 자체가 축소
- 이미지 사이즈 **버킷팅** (max_pixels 고정 = vision 토큰 균일) → CUDA graph shape 안정
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: SM120 단편화 OOM 방어 정석
- **전력 스윕**: prefill-heavy는 캡 손실이 decode보다 큼. 450/500/575W 실배치 측정
  (현재 서버 정책은 450W 캡, [05](05-strad32-team-resource-split.md#공유-자원-규칙). 스윕 실험은 팀 합의 후)
- MoE 후보는 `--enable-expert-parallel` A/B
- **Sleep mode** (v0.17 level-0): 배치 모델을 런 사이 host RAM으로 스왑.
  야간 배치/주간 인터랙티브 교대 운영
- 워밍업 위생: (스키마, 이미지 버킷)별 1요청 워밍 후 측정

## 8. 플래그 스택 (실측 확정)

**단일 원본은 `scripts/serve.sh` 의 `VLLM_ARGS` 다.** 아래는 그 값과 근거이며,
바꾸려면 serve.sh 주석의 근거를 먼저 읽을 것. 2026-07-28 strad32 실측으로 확정.

```bash
# 배치 엔드포인트 - DP 레플리카 (Qwen3.6-35B-A3B NVFP4, GPU당 1개 × N)
# 컨테이너 레벨에서 UUID 로 핀하므로 CUDA_VISIBLE_DEVICES 는 쓰지 않는다 (docs/04 §2)
vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name qwen36-35b-a3b \
  --gpu-memory-utilization 0.93 \            # 0.90 은 기동 실패, 0.95 는 OOM 전멸 이력 (아래)
  --max-model-len 32768 \
  --max-num-seqs 16 \                        # hybrid Mamba cache 캡
  --max-num-batched-tokens 16384 \           # 하한. disable-chunked-mm-input 이 16384 미만을 거부 (아래)
  --disable-chunked-mm-input \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image": 4}' \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice
# 나머지 GPU 동일 반복, 앞단 LB = nginx least_conn + zone (docker/nginx-lb.conf)
```

초안 대비 변경점과 이유:

| 항목 | 초안 | 확정 | 이유 |
|---|---|---|---|
| `--gpu-memory-utilization` | 미지정 | **0.93** | 0.90 은 기동 실패(가중치 23.25 GiB 실측, `Available KV cache memory: -1.09 GiB` ValueError). 처음 0.95 로 운영하다 2026-08-03 장애로 하향: 부하 시 VRAM 여유가 약 1.0 GiB 뿐인데 멀티이미지 긴 prefill 이 16K 배치를 채우자 flashinfer cutlass MoE workspace 일시 할당(1.04 GiB)이 이를 넘어 `MemoryError: CUDA out of memory` 로 레플리카 4대 순차 전멸. 0.93 은 여유 약 1.6 GiB, 대가는 KV 3.14 -> 약 2.5 GiB (실측 피크 2.0 GiB 수용) |
| `--max-num-batched-tokens` | 32768 | **16384** | 스텝 활성화 피크 축소 목적으로 32768 에서 하향. **더 내릴 수 없다**: `--disable-chunked-mm-input` 하에서 vLLM 이 `max_tokens_per_mm_item`(이 모델 16384) 미만을 거부한다. 2026-08-03 OOM 조치로 10240 을 시도했으나 `Chunked MM input disabled but max_tokens_per_mm_item (16384) is larger than max_num_batched_tokens` 로 기동 실패, 조치는 util 0.93 하향으로 전환했다. 활성화 피크를 더 줄이려면 chunked mm input 허용을 exp-replica 로 A/B |
| `--max-num-seqs` | 32 | **16** | Mamba cache 블록 한정. KV 3.14 GiB 로는 32 를 지탱하지 못한다 |
| `--kv-cache-dtype fp8` | 있음 | **제외** | 가동 검증 게이트 1(NVFP4 정확도)과 교락. 게이트 통과 후 별도 A/B |
| `--limit-mm-per-prompt` | image: 5 | **image: 4** | 단, 4464x2160 프레임은 9.4K 토큰이므로 32K 컨텍스트에는 **실제 3장**까지. few-shot 이미지는 크롭/다운스케일 전제 |
| `-O3`, `--mm-encoder-tp-mode data`, `cudagraph_mm_encoder` | 있음 | **미적용** | 아직 미검증. 기본 설정으로 먼저 가동선을 확보했다. 최적화 Phase 에서 A/B 대상 |
| MoE FP4 env (`VLLM_USE_FLASHINFER_MOE_FP4` 등) | — | **불필요** | vLLM 0.24.0 이 FLASHINFER_CUTLASS NvFp4 MoE 백엔드를 자동 선택 (실측) |

가동 후 용량 실측 (레플리카 1개 = GPU 1장):

| 항목 | 값 |
|---|---|
| KV 캐시 | 3.14 GiB = **270,767 tokens** |
| 최대 동시성 | 32K 요청 기준 **8.26x** (`--max-num-seqs 16` 보다 작다 → 긴 요청 몰리면 preemption) |
| VRAM | 29.85 / 32.6 GiB (여유 약 2.7 GiB) |
| prefix cache | **자동 비활성** (`prefix_cache_queries_total` = 0 고정) |

- 위 용량 표는 util 0.95 시절 실측이다. 0.93 에서는 KV 약 2.5 GiB 로 준다
- **util 상향으로 KV 를 키우는 최적화는 2026-08-03 장애 이후 봉인.** 부하 중 MoE workspace 일시 할당(16K 배치에서 1.04 GiB)과 정면 상충한다. 되돌리려면 멀티이미지 16K prefill 부하로 workspace 피크를 먼저 실측할 것
- 참고: block-wise FP8 체크포인트를 쓸 때는 `VLLM_USE_DEEP_GEMM=0` 이 필요하다.
  DeepGEMM 이 SM120 의 scale factor 레이아웃을 몰라 `Unknown SF transformation` 으로 엔진이 죽는다. 현재 NVFP4 라 해당 없음

# 큰 모델 단일 엔진 대안 (122B AWQ 등)
vllm serve <model> --tensor-parallel-size 4 ...   # TP2×PP2도 A/B

# 인터랙티브 QA 엔드포인트 (작은 모델 레플리카)
vllm serve <small-model> \
  --mm-processor-cache-type shm \
  --kv-offloading-backend native --kv-offloading-size 64 \
  --speculative-config '{"method":"suffix","num_speculative_tokens":16}' \
  --max-num-batched-tokens 4096
```

## 9. Watch 목록 (분기 재확인)

- vLLM RFC #45098: 네이티브 image token pruning
- VLCache (arXiv 2512.12977), SpecVLM, ElasticMM: 연구 단계
- FlashInfer SM120 trtllm-gen 커널 wiring (#2555): attention 백엔드 판도 변경 가능
- llm-compressor 2026 로드맵 (W4A8 등)
- Qwen3.6 hybrid의 Mamba prefix caching 성숙도 (experimental → stable 전환 여부)
