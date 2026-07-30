# 07. 추론 최적화 실험 로드맵

- 작성: 2026-07-22, 갱신: 2026-07-27 (베이스라인을 확정 모델 기준으로)
- 원본 리서치: [research/2026-07-22-brief-task-design.md](research/2026-07-22-brief-task-design.md) 5번 절

> 이 문서는 "실행 순서".
> 전체 기법 목록 + SM120 채택/skip 판정은 **[08 카탈로그](08-optimization-catalog.md)**.
> 로드맵에 없는 기법(FP8 KV 세부, suffix decoding, mm-encoder-tp-mode, NUMA bind, 스키마 다이어트 등)은 전부 거기 있음.

## 추적 지표 (전 단계 공통)

- 성능: TTFT, TPOT/ITL, E2E latency, **images/hour** (비즈니스 지표), tokens/s/GPU
- 자원: KV-cache 사용률, prefix-cache 히트율, VRAM 헤드룸
- **품질 게이트: 매 단계 frozen 평가셋의 클래스별 F1/κ가 합의된 델타(예: ≤1pt) 이내.**
  깨지면 그 최적화는 불채택
- 도구: `vllm bench serve`, 엔진 `/metrics`, promptfoo 회귀

## Phase 0: 베이스라인

확정 모델 Qwen3.6-35B-A3B NVFP4, GPU당 1레플리카
(첫 가동 구성 = [04 1번](04-gpu-pinning-and-serving.md#1-공용-모델-첫-가동-절차) + [08 8번 플래그 스택](08-optimization-catalog.md#8-플래그-스택-실측-확정)).
frozen 평가셋 + 벤치 1회 → 이후 모든 단계는 이 대비 diff.
BF16 대비 품질 확인은 가동 검증 게이트 1번(단발 TP=4 실행)에서 수행 ([02](02-model-candidates.md)).

## Phase 1: 무비용/저비용 (week 1)

| # | 실험 | 기대 효과 |
|---|---|---|
| 1 | **해상도 사다리** (`max_pixels`) | 최대 단일 레버. native 9.4K → 2.5K tokens = prefill ~4× 절감 |
| 2 | **Prefix/encoder 캐시 확인** | few-shot 이미지 byte-identical 유지 시 해시 기반 캐시 히트. 로그로 히트율 검증 |
| 3 | **오프라인 배치 + 스케줄러 튜닝** | `LLM.generate`/`LLM.chat`, `max_num_batched_tokens` 8K-64K 스윕, `max_num_seqs` |
| 4 | **DP vs TP A/B** | 레플리카가 P2P 없는 박스에서 실측 ~2.8× (4×5090) |
| 4b | **Encoder 플래그 2종** | `--mm-encoder-tp-mode data` (10-45%) + ViT CUDA graphs. 무위험. [08 2번](08-optimization-catalog.md#2-vlm-특화-기법) |
| 5 | **`--gpu-memory-utilization` 0.95 → 0.96-0.965** | KV 가 3.14 GiB(270,767 tokens)뿐. 엔진이 로그로 0.9653 을 권고. 단 부하 실측에서 KV 는 64퍼센트만 찼으므로 6번보다 뒤다 |
| 6 | **`--max-num-seqs` 16 → 24 또는 32** | 부하 실측 1순위. `running` 이 32(16×2) 상한에 닿고 KV 는 남았다. preemption 0 ([09 6번](09-stress-test-results.md#6-병목은-kv가-아니라-max-num-seqs)) |

Phase 1 을 시작하기 전에 알아야 할 가동 실측 ([08 8번](08-optimization-catalog.md#8-플래그-스택-실측-확정)):

- **prefix caching 은 자동 비활성**(게이트 2 확정). 위 2번은 "히트율 검증"이 아니라
  "캐시를 켤 수 있는가"부터 확인해야 하며, 현재 전제로는 few-shot prefix 절감 효과가 0이다
- **32K 컨텍스트에 4464x2160 프레임은 3장까지.** few-shot 이미지를 쓰려면 1번(해상도 사다리)이 선행 조건이다
- **해상도가 처리량을 지배한다(실측).** 동시 16에서 텍스트 1885 tok/s, 1MP 1614 tok/s, native 206 tok/s.
  1MP 까지는 사실상 공짜이고 native 는 처리량을 1/9 로 떨어뜨린다 ([09 5번](09-stress-test-results.md#5-해상도가-처리량을-지배한다))
- 측정 시 타 팀의 `--cpuset-cpus` 없는 컨테이너가 우리 코어에 끼어들 수 있다
  ([05 도커 사용 시](05-strad32-team-resource-split.md#도커-사용-시)). load average 를 함께 기록할 것

주의: 과거 고동시성 MM prefix 캐시 정합성 버그(#20261) 이력. 타깃 동시성에서 출력 스팟체크.
chunked prefill은 V1 기본이라 대개 건드릴 필요 없음.

## Phase 2: 양자화 사다리 (weeks 2-3)

매 단계 품질 게이트 통과 필수. 판정표는 [08 1번](08-optimization-catalog.md#1-양자화-사다리-sm120-판정표).

| # | 실험 | 비고 |
|---|---|---|
| 5 | **FP8 (W8A8)** | Blackwell 네이티브, VLM 거의 무손실, prefill ~1.5-2×. NVFP4 품질 이슈 시 착지점 |
| 6 | **FP8 KV cache** | KV 비용 ~54%. prefill-heavy라 지연보다 **동시성(images/hour)** 이득 |
| 7 | **W4 (NVFP4 우선, AWQ/GPTQ 폴백)** | 1장/레플리카 구성의 열쇠. VLM은 모델별 품질 저하 사례 있음. 자체 평가셋 클래스별 검증 |
| 8 | **NVFP4 심화** | Qwen3.6 계열은 NVFP4 체크포인트 기성품 존재. 타 모델은 llm-compressor + 타임박스 |

## Phase 3: Decode 가속 (weeks 3-4)

| # | 실험 | 비고 |
|---|---|---|
| 9 | **Speculative decoding** | 1순위 suffix decoding (반복적 JSON에 최적), 2순위 ngram (`prompt_lookup_min≥8`, 버그 #40875), Qwen3.6은 내장 MTP/DFlash. 짧은 JSON + 고동시성 배치에선 이득 축소. **인터랙티브 QA용**. [08 5번](08-optimization-catalog.md#5-speculative-decoding-메뉴) |
| 10 | **Vision encoder 분리 (EPD)** | **skip 판정** ([08 2번](08-optimization-catalog.md#2-vlm-특화-기법)). 서버 1대 고정이라 재검토 조건은 "요청당 크롭 4장+ 상시" 워크로드로 바뀔 때뿐 |

## Phase 4: 조합 + 동결

유력 최종 조합:

> NVFP4/FP8 가중치 + FP8 KV + 튜닝된 max_pixels 패널 크롭 + 캐시된 few-shot prefix
> + DP 레플리카 + 튜닝된 max_num_seqs. 인터랙티브 엔드포인트만 spec decode.

전체 frozen 평가셋 + 사람 대비 κ 재실행 →
`{model, quant, prompt_sha, schema, engine version}`을 릴리스 아티팩트로 동결.

## 실험 순서 원칙

1. 비용 낮은 것부터: 해상도/캐시가 양자화보다 먼저
2. 한 번에 한 변수
3. 품질 게이트 통과 못 하면 불채택. 속도는 품질 안에서만 의미
4. 모든 런에 설정 태깅 (재현성)
