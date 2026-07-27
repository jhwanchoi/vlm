# 01. 프로젝트 개요

- 작성: 2026-07-22, 갱신: 2026-07-27 (모델 확정, 마일스톤 재편)

## 목표

1. 사내 8× RTX 5090 서버(strad32)에서 로컬 VLM을 서빙한다.
   확정 분배: dst+dmt 공용 4장 / vpt 2장 / dpt 2장 ([05](05-strad32-team-resource-split.md)).
2. 첫 유스케이스: **svnet3 추론 결과 시각화 이미지의 자동 visual inspection agent**.
3. (2026-07-27 합의) 서빙 모델은 DST의 **라벨 누락 객체 탐지**(Bedrock 마이그레이션) 용도와 공용.
   최종 목표는 image + text 입력 agent. 공용 모델은 **Qwen3.6-35B-A3B NVFP4로 확정** ([02](02-model-candidates.md)).
4. 이 서버를 **추론 최적화 실험 플레이그라운드**로도 활용한다
   (양자화, 캐싱, speculative decoding 등: [07](07-inference-optimization-roadmap.md)/[08](08-optimization-catalog.md)).
   모델 학습(fine-tuning)은 범위에서 제외한다.

## 입력 데이터 특성

svnet3 추론 결과 시각화 스크린샷, 약 4464×2160.
샘플: `samples/svnet3_inference_result_sample.jpg` (git 미포함, 보안 정책 범위 확인 전까지 사내 별도 공유)

- 전방 카메라 패널: 차량/VRU/신호등 바운딩 박스, 차선 라인, 거리·속도 수치 오버레이
- 우측 BEV(bird's-eye-view) 패널
- HUD 스타일 디버그 텍스트 (영문)
- 배경에 한국어 간판/건물 텍스트
- 패널 레이아웃 고정 → 전처리에서 결정적 크롭 가능

핵심 수치: Qwen 계열 native resolution 기준 이미지 1장 ≈ **9,400 vision tokens** (`tokens ≈ H×W/1024`).
해상도·크롭 전략이 비용과 품질을 동시에 지배한다.

## 검사 대상 defect (초안, [policy/inspection/](policy/inspection/) 정책 수신 시 확정본으로 교체)

| 클래스 | 설명 |
|---|---|
| Missed detection | 있어야 할 객체 미검출 |
| False positive | 없는 객체 오검출 |
| Box misalignment | 박스 정렬/크기 불량 |
| Lane fit error | 차선 피팅 오류 |
| Implausible value | 오버레이 거리/속도 값의 물리적 비정합 |
| Panel inconsistency | 카메라 패널 ↔ BEV 패널 불일치 |

## 제약 사항

### 데이터 보안 제약

- svnet3 **소스코드**는 해외 리전 LLM(Claude 등)에 입력 불가 (사내 판정).
  요약·자연어 설명도 불가.
- 이 프로젝트가 **로컬 VLM**인 이유 중 하나: 추론 결과 이미지가 온프렘을 벗어나지 않아 정책 방향과 정합.
- 미확인: **추론 결과 오버레이 이미지 자체**가 보안 정책 범위인지 확인 필요.
  (외부 LLM 세션 첨부 가능 여부의 문제. 로컬 VLM 파이프라인에는 영향 없음.)

### 하드웨어

- **서버는 strad32 1대 고정, 스케일아웃 없음.** 처리량 확장은 박스 내부(몫 협의, 최적화)로만 해결.
  K8s 편입하더라도 GPU 노드 1대짜리 단일 노드 클러스터
- RTX 5090 = GeForce: NVLink 없음, **P2P 드라이버 차단**, MIG 없음, ECC 없음.
  상세: [03](03-rtx5090-hardware-notes.md)
- 서버는 4팀이 3개 몫으로 분할 사용: dst+dmt 공용 4장 / vpt 2장 / dpt 2장 (NUMA 정렬 CPU/RAM).
  상세: [05 운영 안내](05-strad32-team-resource-split.md)
- 공용 몫 내 관례 (dst·dmt 간 합의): vLLM 서버는 **dmt가 소유·운영** (GPU 0-3, 포트 8000번대),
  **dst는 API 클라이언트** (`http://${STRAD32_IP}:8000/v1`). dst의 자체 GPU 실험은 dmt와 협의 후

## POC 계획 (1개월, 2026-07-27 미팅 합의)

- 목표: 로컬 LLM의 **Bedrock 대체 가능성 판단** (성능·비용 비교).
  결과물보다 실행 증명과 추가 기간 확보 근거 마련에 집중
- 종료 시: DST 담당 태스크의 마이그레이션 결과 + 이미지 인풋 테스트 결과 공유 → 연장/확대 결정
- 비용 비교 준비: Bedrock 사용 비용 vs 로컬 GPU 운영비 분석.
  **현재 AWS 권한 부재로 비용/트래픽 관찰 불가 → 옵저버빌리티 권한 확보 요청 필요 (담당: 서버 운영자)**
- 자동화 범위는 작게 시작: 전방 카메라 등 특정 상황 한정, TSTLD 리뷰 자동화가 초기 후보

## 마일스톤

1. **공용 모델 서빙 가동 (현재 단계, 최우선).**
   다운로드 → 단일 레플리카 기동 → 가동 검증 게이트 5항목 → DP=2 확장 → 두 팀 공개.
   절차: [04 1번](04-gpu-pinning-and-serving.md#1-공용-모델-첫-가동-절차)
2. **DST 골든셋 실측**: 누락 라벨을 아는 프레임 50-100장에서 VLM 단독 마이닝 recall과
   crop 판정 정밀도 측정 (게이트 4번). 결과에 따라 사이드카 detector 도입 여부 결정
3. **inspection 개발 (정책 문서 수신 후)**: 크롭 스크립트, JSON 스키마, 프롬프트,
   미니 평가셋(~50장) 반복 ([06](06-inspection-agent-design.md))
4. POC 결과 공유 (1개월 시점): Bedrock 대비 성능·비용, 연장/확대 판단
5. 평가셋 300-1,000장 확장, 최적화 실험 착수 ([07](07-inference-optimization-roadmap.md))

## 미결정 사항

| 항목 | 영향 |
|---|---|
| 주 모드: 오프라인 배치 vs 인터랙티브 QA 비중 | 토폴로지, spec decoding 적용 여부 |
| Defect taxonomy 확정: **inspection 정책 문서 수신 대기 ([policy/inspection/](policy/inspection/))** | 라벨셋 설계, JSON 스키마의 뿌리. 위 초안은 정책 수신 시 교체 |
| 한국어 간판 텍스트가 판정에 중요한가 | OCR 사이드카 필요 여부 |
| 이 박스의 K8s 편입 계획 | 편입 시 DRA 기반 GPU 핀닝으로 전환 |
