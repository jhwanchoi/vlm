# VLM Visual Inspection Agent 프로젝트 문서

8× RTX 5090 서버(strad32, 4팀 공유: dst+dmt 공용 4장 / vpt 2장 / dpt 2장)에서 로컬 VLM을 서빙하고,
svnet3 추론 결과 시각화 이미지의 자동 visual inspection agent와
DST 라벨 누락 객체 탐지(Bedrock 마이그레이션)를 만드는 프로젝트.

- 상태 (2026-07-30): **공용 모델 서빙 가동 완료.** Qwen3.6-35B-A3B NVFP4 × DP=2 (GPU 0-1),
  앞단 nginx LB, 공개 주소 `http://${STRAD32_IP}:8000/v1`, model `qwen36-35b-a3b`
  ([04 0번 현재 가동 상태](04-gpu-pinning-and-serving.md#0-현재-가동-상태-2026-07-30)).
  다음 단계는 가동 검증 게이트 잔여 4항목 + DST 골든셋 ([02 실측 현황](02-model-candidates.md#가동-검증-게이트-서빙-직후-실측-5항목))
- 고정 제약: **서버 strad32 1대 (스케일아웃 없음)** · GeForce P2P 차단 · svnet3 소스 보안 정책상 접근 불가 ·
  모델 학습(fine-tuning) 범위 외
- 용어: 문서 내 "NUMA 노드"는 서버 내부의 CPU 소켓별 메모리 구역 (1대 안에 2개). 서버 대수와 무관.
  그림 설명은 [05 1번 서버 구조](05-strad32-team-resource-split.md#서버-구조)
- 모델/소프트웨어 정보는 빠르게 낡음. 분기마다 재검증 권장

## 핵심 결론

1. **모델 (확정)**: 공용 모델 **Qwen3.6-35B-A3B NVFP4 × DP=N** (레플리카당 GPU 1장. N = 몫: 2장 몫은 2, dst+dmt 풀은 4).
   DMT inspection과 DST 라벨 누락 탐지가 하나의 서빙을 공유.
   품질 폴백 Qwen3.6-27B NVFP4, 에스컬레이션 전용 Qwen3.5-122B-A10B AWQ TP=4 (**4장 풀에서만**).
   DST 마이닝은 VLM 단독 시작, 골든셋 recall 미달 시에만 detector 사이드카(LLMDet 등) 도입 (02 문서).
2. **GPU 핀닝**: 컨테이너 레벨에서 UUID로 고정하고 vLLM은 내부에서 0-3 인덱스만 보게.
   `EXCLUSIVE_PROCESS` 등 강제 수단은 사고 반복 시에만 (05 Level 2).
3. **분산 토폴로지**: 5090은 NVLink 없음 + GeForce P2P 차단.
   → "32GB에 들어가면 레플리카(DP), 안 들어가면 최소 병렬". TP=4는 최후 수단.
4. **자원 분할**: 3개 몫 확정: dst+dmt 공용(GPU 0-3, 225G) / vpt(4-5, 112G) / dpt(6-7, 112G), NUMA 정렬.
   적용은 user-slice 명령 4줄이 전부, GPU는 배정표 관례 (05 런북).

## 문서 구성

| 문서 | 내용 |
|---|---|
| [01-project-overview.md](01-project-overview.md) | 목표, 데이터 특성, 제약(보안), 마일스톤, 미결정 사항 |
| [02-model-candidates.md](02-model-candidates.md) | **[확정]** 모델 선정: 확정 구성, 가동 검증 게이트, DST 용도 검토, 후보 비교, 검토 이력 |
| [03-rtx5090-hardware-notes.md](03-rtx5090-hardware-notes.md) | RTX 5090(SM120) 하드웨어 현실: P2P, 스택, 정밀도, 전력/운영 |
| [04-gpu-pinning-and-serving.md](04-gpu-pinning-and-serving.md) | **첫 가동 절차**, GPU 핀닝, TP/PP/DP 선택, 서빙 스택 비교, 모니터링 |
| [05-strad32-team-resource-split.md](05-strad32-team-resource-split.md) | **[운영 안내]** 자원 분할: 분할 전략(3개 몫), 검증 결과, 사용 방법, 변경·사고 시 절차 |
| [06-inspection-agent-design.md](06-inspection-agent-design.md) | 프롬프트 구조, 이미지 전처리, structured output, 평가 |
| [07-inference-optimization-roadmap.md](07-inference-optimization-roadmap.md) | 최적화 실험 로드맵 (Phase 0-4, 실행 순서) |
| [08-optimization-catalog.md](08-optimization-catalog.md) | 최적화 기법 전체 카탈로그 (SM120 판정표, skip/watch 목록) |
| [research/](research/) | 원본 리서치 브리프 (출처 포함, 수정 금지) |
| [policy/](policy/) | 정책 문서 보관소: 도메인별 하위 폴더 (현재 [inspection/](policy/inspection/) 수신 대기) |
| [../samples/](../samples/) | 입력 샘플 이미지 (svnet3 추론 결과 스크린샷 등) |
| [../scripts/](../scripts/) | `healthcheck.sh` 부팅 후 점검 · `download-models.sh` · **`serve.sh` 기동/정지/preflight** · `serve-smoke-test.sh` |
| [../docker/](../docker/) | `Dockerfile` (dmt 사용자 내장 vLLM 파생 이미지) · `nginx-lb.conf` (LB 설정) |
| [../examples/](../examples/) | 클라이언트 예제: `quickstart.py` · `curl.md` · `prompts.md` · 합성 샘플 생성 |

## 읽는 순서

- 처음 온 사람: README → 01 → 02
- 서빙 가동 담당: 04 → 02 → 08 → `scripts/serve.sh`
- 서빙 쓰는 사람(dst 등): [../examples/](../examples/) 만 보면 된다
- 서버 셋업 담당: 05 → 04 → 03
- agent 구현 담당: 06 → 02 → 07 → 08
