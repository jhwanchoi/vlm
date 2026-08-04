# 10. 운영 지표와 SLO

- 작성: 2026-08-04
- 원본 리서치: [research/2026-08-04-brief-modular-handbook.md](research/2026-08-04-brief-modular-handbook.md)
- 배경: [09 실측](09-stress-test-results.md)이 "지연 요건이 상한을 정한다"고 결론냈지만
  그 요건(SLO)이 미정의 상태였다. 이 문서가 그 빈칸을 채운다.
  모니터링 스택 구성은 [04 5번](04-gpu-pinning-and-serving.md#5-모니터링), 측정 하네스는 [stress/](../stress/)

## 1. 원칙: 처리량이 아니라 goodput

**goodput = SLO를 만족한 요청만 센 초당 처리량.** 지금까지의 실측 수치(예: DP=4 최대 1,508 tok/s)는
TTFT 12초짜리 응답도 포함한 값이다. SLO를 정하면 같은 데이터에서 "실사용 가능한 처리량"이 따로 나온다.
새 측정 없이 [09 2번](09-stress-test-results.md#2-동시성별-실측-혼합-워크로드-lb-경유) 표의 재해석으로 시작한다.

보고 통계는 3관점을 병기한다: 평균(추세), P50(전형 경험), P99 또는 p95(SLA 판정).
평균 단독 보고 금지. Bedrock 비교 자료도 동일.

## 2. 워크로드별 우선 지표와 SLO 제안

용도별로 지표가 다르다. 단일 평균으로는 배치와 인터랙티브 둘 다 놓친다.

| 워크로드 | 우선 지표 | SLO 제안 (실측 근거) | 상태 |
|---|---|---|---|
| 오프라인 배치 (inspection, DST 마이닝) | **images/hour, TPS, 토큰당 비용** | 지연 무관, 처리량 극대화. 동시 44-64 (DP=4 포화점의 70-100%) | 제안 |
| 인터랙티브 QA (dst 수동 질의) | **TTFT p95, TPOT** | TTFT p95 ≤ 7초 → 동시 8 이하로 운용 ([09 3번](09-stress-test-results.md#3-파괴점-동시-64에서도-실패-0퍼센트) 표) | 제안 |
| tool calling agent (최종 목표) | **E2EL** (요청→마지막 토큰) | 미정. 첫 토큰이 아니라 전체 완료 시간이 체감을 결정. TTFT 최적화 우선순위는 여기서 낮다 | 측정 전 |

- SLO 값은 제안이며 dst와 사용 패턴 확인 후 확정한다 ([09 9번](09-stress-test-results.md#9-다음-할-일) 4순위 항목의 답)
- 배치와 인터랙티브가 같은 엔진을 쓰는 동안은 인터랙티브 SLO가 배치 동시성의 상한을 정한다.
  충돌이 실제로 발생하면 priority scheduling ([08 4번](08-optimization-catalog.md#4-스케줄링--배칭)) 검토

## 3. 상시 관측 최소 세트

**2026-08-04 가동됨**: Prometheus + Grafana + vLLM 공식 대시보드 3종, `http://${STRAD32_IP}:3000`.
구성과 기동 절차는 [monitoring/](../monitoring/README.md). dcgm-exporter(GPU 지표)는 2차 보류.
부하 측정 시의 고빈도(1초) 수집은 종전대로 [stress/monitor.sh](../stress/monitor.sh).

| 지표 | 용도 |
|---|---|
| `vllm:request_queue_time_seconds` (sum/count) | **1순위 신호.** 큐 대기 증가는 레플리카 열화/용량 부족의 최선행 지표 |
| `vllm:num_requests_running` / `waiting` | 슬롯 포화(waiting 적체 = max-num-seqs 병목) vs 여유 구분 |
| `vllm:kv_cache_usage_perc` | KV 병목 여부. [09 6번](09-stress-test-results.md#6-병목은-kv가-아니라-max-num-seqs)의 판정 재사용 |
| `vllm:num_preemptions_total` | KV 압박 경보. 0이 정상 기대값 |
| TTFT/TPOT 히스토그램 | SLO 판정 원자료 |

판단 규칙:

- **DP 레플리카 증설/축소 판단은 동시성과 큐 대기 시간으로.** GPU utilization은 커널이 잠깐만 돌아도
  100%로 잡히는 허수라 보조 지표로만 쓴다 (nvidia-smi 눈대중으로 증설 판단 금지)
- "박스가 바쁜 이유"는 엔진 지표로만 구분 가능: waiting 적체 = 슬롯, KV 고사용 + preemption = 메모리,
  둘 다 아니고 TTFT만 상승 = prefill 혼잡

## 4. 플래그 변경과 롤백 기준

최근 OOM 장애 이력(0.95 → 0.93, [08 8번](08-optimization-catalog.md#8-플래그-스택-실측-확정))의 교훈을 절차로 고정한다.

1. **변경은 실험 레플리카 A/B를 먼저 통과** ([scripts/exp-replica.sh](../scripts/exp-replica.sh),
   [09 측정 메모](09-stress-test-results.md#9-다음-할-일): 비교 측정은 단계 180초 이상)
2. **롤백 트리거 2축**: 운영 반영 후 (a) 지연 급증 (TTFT/큐 대기가 A/B 대비 유의미 초과),
   (b) 품질 저하 (frozen 평가셋 게이트 위반, 평가셋 구축 전에는 출력 스팟체크) 중 하나면 직전 플래그로 복귀.
   serve.sh의 VLLM_ARGS가 단일 원본이므로 롤백 = git revert + 레플리카 순차 재기동
3. **vLLM 버전업은 기능 변경이 아니라 성능 변수.** 추론 속도의 천장은 커널 층이 정하고
   Qwen3.6 MoE + NVFP4는 신생 경로라, 버전업만으로 수치가 오르내릴 수 있다.
   업그레이드마다 Phase 0 베이스라인 재측정 ([07](07-inference-optimization-roadmap.md))

## 5. POC 비용 비교 체크리스트 (Bedrock 대비)

[01 POC 계획](01-project-overview.md#poc-계획-1개월-2026-07-27-미팅-합의)의 비용 비교에 포함할 항목.

- 로컬 측: GPU 서버 상각 + 전력(450W 캡 × 4장 실효) + **엔지니어 시간**.
  LLM 인프라 운영 인력은 일반 DevOps 대비 30-50% 프리미엄이 업계 통설이다.
  인건비를 빼고 "로컬이 싸다"는 결론을 내면 반박당한다
- 로컬 측 운영 항목: 레플리카 콜드스타트(기동→healthy 시간) 실측치, 장애 대응 이력(OOM 1건 등)을
  가용성 비용으로 명시
- Bedrock 측: 토큰 단가 × 실측 토큰 분포 (평균이 아니라 이미지 해상도별 분포로),
  리전 간 단가 차이 존재(클라우드 GPU 리전차 최대 60% 사례) 각주
- 양측 공통: 같은 평가셋, 같은 지연 통계(P50/P99)로 비교. goodput 기준 통일
