# 리서치 브리프: Modular LLM Inference Handbook 전수 조사

- 작성: 2026-08-04
- 출처: https://handbook.modular.com/ (Modular사, 52페이지. 전체 색인 https://handbook.modular.com/llms.txt,
  소스 https://github.com/modular/llm-inference-handbook. 페이지 URL 뒤에 `.md`를 붙이면 마크다운 원문)
- 조사 방식: 6개 섹션(planning/metrics, model-preparation, model-interaction,
  inference-optimization, kernel-optimization, infra/ops)을 병렬 정독 후
  이 저장소 문서(07/08/09)와 대조해 "이미 함 / 갭 / 적용 가치" 판정
- 반영: [10-operations-metrics-and-slo.md](../10-operations-metrics-and-slo.md) 신설,
  [07](../07-inference-optimization-roadmap.md) [08](../08-optimization-catalog.md)
  [06](../06-inspection-agent-design.md) 갱신

## 총평

새로 배울 개별 기법은 적다. 08 카탈로그가 대부분 항목에서 핸드북보다 깊고,
09 실측 방법론(포화점 탐색, 단계 180초, A/B 레플리카)은 핸드북에 아예 없는 수준이다.
실제 가치는 운영 체계 프레임(goodput/SLO, 유스케이스별 지표, 롤백 기준, TCO 항목)과
로드맵 우선순위 조정 근거다.

핸드북의 한계 3가지:

1. VLM/vision token 내용 사실상 0 (52페이지 중 prefix caching 페이지의 한 문장뿐).
   고해상도 이미지 토큰 비용, encoder 캐시, 멀티모달 스케줄링 전무
2. NVFP4/Blackwell/SM120/Mamba(hybrid attention) 언급 0. 하드웨어 기준선이 Hopper 세대
3. 비용 모델 없음(infra 섹션은 사실상 managed 서비스 마케팅), roofline/정량 병목 판정 절차 없음,
   프로파일링 도구 가이드 없음

우리 검증 게이트 5개(NVFP4 OCR, Mamba prefix caching, bbox 규약, DST 골든셋,
vision+tool calling)에 대한 답은 핸드북에 없다. 실측 노선 유지가 맞다.

## 채택 항목 (반영처 포함)

### 지표/운영 체계 → 10번 문서 신설

- **goodput**: SLO를 만족한 요청만 세는 처리량. 핸드북이 주 지표로 지목.
  기존 실측(09)의 재해석만으로 도입 가능
- **유스케이스별 우선 지표 매핑**: 인터랙티브 챗 TTFT+ITL/TPOT, 롱폼 스트리밍 ITL+E2EL,
  에이전틱 워크플로 E2EL(첫 토큰이 아니라 전체 완료 시간), 오프라인 배치 TPS+토큰당 비용
- **평균/P50/P99 3관점**: 평균은 추세용(아웃라이어 오염), P50은 전형 경험, P99는 SLA 판정.
  Bedrock 비교 자료에 평균만 쓰면 반박당함
- **스케일 판단 신호**: GPU utilization은 커널이 잠깐만 돌아도 사용 중으로 잡히는 허수.
  CPU util은 GIL, QPS는 길이 변동성 문제. concurrency(+큐 대기)가 배치 크기와 직접 상관되어 최선
- **롤백 트리거 2축 명문화**: 지연 급증 + 품질 저하. 플래그 변경마다 적용
- **TCO에 엔지니어 시간 포함**: LLM 인프라 인력은 일반 DevOps 대비 30-50% 프리미엄(핸드북 인용).
  자체 서빙 유리 결론에서 인건비를 빼면 반박당함
- **콜드스타트(레플리카 기동 시간) 실측 항목화**
- **멀티모델 파이프라인 과설계 경고**: 스테이지당 지연이 생성 시작 전에 누적.
  단일 모델로 요구를 충족하면 붙이지 말 것 (적정 규모 원칙과 일치)

### 로드맵/카탈로그 보강 → 07, 08 갱신

- **speculative decoding 배치 적용 부정 근거(핸드북 실측)**: 단일 GPU에서 동시 20-30부터
  throughput 이득 소멸(TPOT만 개선), draft가 KV 풀 공간 경쟁, 부하 오를수록 조율 오버헤드 증가.
  MoE/vision 검증 데이터 전무. 기존 "인터랙티브 QA용" 판정을 외부 실측으로 확정
- **vLLM 버전업 = 성능 변수**: "커널 층이 천장을 정하고 시스템 튜닝은 커널 비효율을 보상 못 한다".
  Qwen3.6 MoE + NVFP4는 신생 경로라 현재 수치는 하드웨어 상한이 아니라 현 버전 커널의 상한일 수 있음.
  업그레이드마다 재측정
- **FA-3/FA-4 하드웨어 대응 명확화**: FA-3는 Hopper 전용, FA-4는 H100/B200(SM100) 타겟.
  consumer Blackwell(SM120)은 별개 아키텍처라 FA-4 벤치 수치(B200 1613 TFLOPS)를 기대치로 삼지 말 것
- **P/D disaggregation skip 재확인**: 핸드북 스스로 "작거나 미튜닝 시스템에서 20-30% 성능 하락" 경고
- **텐서 코어 매핑 주의**: 양자화를 걸어도 타일 정렬/레이아웃이 어긋나면 저정밀 경로를 못 탐.
  NVFP4 효과가 기대보다 작으면 커널 경로부터 의심

### agent 설계 보강 → 06 갱신

- **샘플링 extraction 프리셋**: 판정/추출/OCR은 temperature 0-0.2, top_p 1.0이 표준.
  1차 판정 호출에 적용 (2차 self-consistency 투표의 0.6-0.8과 구분)
- **logprobs 신뢰도 게이트**: 판정 토큰의 logprobs를 confidence 점수로. 모델 자기보고
  confidence 필드보다 캘리브레이션 신뢰 가능. 에스컬레이션(2차 트리아지) 라우팅 기준 후보
- **seed는 재현성 보장이 아님**: 모델 버전/하드웨어/커널이 바뀌면 같은 seed도 출력이 다름.
  A/B 판정은 반복 샘플 통계로
- **stop sequence는 스키마 강제의 대체재가 아님** (우리는 xgrammar라 이미 해당 없음)
- **prefix caching 프롬프트 위생**: 불변 지시문 맨 앞, 타임스탬프/요청 ID/파일명을 앞쪽에 금지,
  JSON 직렬화 키 순서 고정. 08 캐싱 3계층에 이미 있으나 "앞쪽 가변 요소 금지"를 명시 추가

## 기각/보류 항목

| 항목 | 판정 | 이유 |
|---|---|---|
| assistant prefilling | 기각 | xgrammar constrained decoding이 이미 스키마를 보장. 중복 |
| Instructor류 재프롬프팅 | 폴백만 | constrained decoding 하위 호환. bbox 스키마가 문법 컴파일 실패할 때만 |
| cache-aware 라우팅 (SGLang router, llm-d) | 보류 | 우리 캐시 대상은 전 요청 공통 prefix라 4레플리카 모두 곧 보유. nginx least_conn으로 충분 |
| KV cache offloading (LMCache) | 보류 | 판단 기준 "전송이 재계산보다 싼가" 이전에 "재사용 prefix가 실제로 축출되는가"부터. 축출 실측 없이 도입 금지. 08 보너스 항목 유지 |
| MCP | 기각 | 단일 목적 로컬 agent에 과함 |
| Mojo/MAX, 커스텀 CUDA 커널 | 기각 | "대부분의 추론 팀은 커스텀 커널을 쓰지 말아야 한다"(핸드북 FAQ). 소규모 팀 vLLM 유지 결정의 외부 인용 근거로만 사용 |
| Anthropic 호환 API 경로 | 기각 | 자체 호스팅에서 이미지 모달리티 미동작 가능 경고. OpenAI 호환 유지 |
| GPU/모델 선택, BYOC, 멀티클라우드, 오토스케일링 | 해당 없음 | HW/모델 확정, 온프렘 단일 서버 고정 |

## 유용한 인터랙티브 도구 (팀 설명/교육용)

| 도구 | 위치 | 주의 |
|---|---|---|
| Latency Metrics Playground | /llm-inference-basics/llm-inference-metrics/ | TTFT/TPOT/goodput 관계 시뮬레이션 |
| KV Cache Memory Calculator | /inference-optimization/kv-cache-offloading/ | 순수 MHA 기준. GQA/Mamba 미반영이라 우리 모델은 과대 추정 |
| Chunked Prefill Scheduler | /inference-optimization/static-dynamic-continuous-batching/ | max-num-batched-tokens 트레이드오프 직관 |
| Batching Strategy Simulator | 상동 | static/dynamic/continuous 비교 |
| GPU Memory Calculator | /getting-started/calculating-gpu-memory-for-llms/ | dense 전제. MoE는 총 파라미터(35B) 기준으로 입력 |
| GPU Execution and Memory Map | /kernel-optimization/gpu-architecture-fundamentals/ | 메모리 계층 교육용 |

## 참고한 핸드북 페이지 (주요)

- 지표: /llm-inference-basics/llm-inference-metrics/ (goodput, TPOT 정의, 유스케이스 매핑)
- 배칭/chunked prefill: /inference-optimization/static-dynamic-continuous-batching/
- speculative decoding 실측: /inference-optimization/speculative-decoding/
- P/D 분리 경고: /inference-optimization/prefill-decode-disaggregation/
- prefix caching 위생/hybrid 경고: /inference-optimization/prefix-caching/
- 병렬화(TP vs DP): /inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism/
- 스케일 신호 비교: /infrastructure-and-operations/fast-scaling/
- 운영 절차(롤백/카나리): /infrastructure-and-operations/inferenceops-and-management/
- 인건비 프리미엄: /infrastructure-and-operations/build-and-maintenance-cost/
- structured output 3방식: /model-interaction/structured-outputs/
- 샘플링 프리셋/logprobs/seed: /model-interaction/inference-parameters/
- 커널 천장/도구 사다리: /kernel-optimization/kernel-optimization-for-llm-inference/, /kernel-optimization/kernel-optimization-tools/
- FlashAttention 버전표: /kernel-optimization/flashattention/
