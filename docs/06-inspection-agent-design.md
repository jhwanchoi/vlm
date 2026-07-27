# 06. Inspection Agent 설계

- 작성: 2026-07-22
- 원본 리서치: [research/2026-07-22-brief-task-design.md](research/2026-07-22-brief-task-design.md) (논문/출처 포함)

> defect taxonomy·심각도·판정 기준은 외부 inspection 정책 수신 대기 중 ([policy/inspection/](policy/inspection/)).
> 수신 시 이 문서의 JSON 스키마 enum과 라벨링 기준을 정책 버전에 맞춰 갱신하고,
> 배치 출력 태깅에 `policy_version`을 포함할 것.
> 모델 학습(fine-tuning)은 프로젝트 범위 외 ([01](01-project-overview.md)).

## 1. 프롬프트 구조

### 기본 패턴: 체크리스트형 루브릭 + 2단 트리아지

연구 합의: 체크리스트 분해가 단일 메가 프롬프트보다 신뢰도·디버깅성 우위.
단 배치 비용 때문에 다음처럼 절충한다.

1. **1차 (전 프레임)**: defect 클래스별 필드를 가진 단일 JSON을 뽑는 섹션형 루브릭 프롬프트 1회 호출
2. **2차 (애매 판정 프레임만)**: 클래스별 분해 질문 + self-consistency 투표
   (N=3-5, temp 0.6-0.8, 다수결)

원칙:

- 전 프레임 투표는 3-5배 비용이므로 에스컬레이션에만
- 배치 모드는 single-shot (multi-turn은 prefix 캐시 히트율 파괴)
- 회귀 비교를 위해 temperature/seed 고정 (judge의 run 간 자기 불일치는 문서화된 현상)

### 선행연구 경고

BetterCheck (ITSC 2025, Waymo 데이터): VLM은 교통 장면 fine detail은 잘 보지만
**agent 환각(FP)과 VRU 누락(FN)** 경향.
→ 자유 서술 금지. 클래스별 강제 판정 + 근거 필드가 필수인 이유.

## 2. 이미지 전처리 (4464×2160 대응)

### 패널 크롭: 전처리에서 결정적으로

레이아웃이 고정이므로 VLM에게 패널 찾기를 시키지 않는다.

- ① 전방 카메라 패널 ② BEV 패널 ③ HUD/수치 오버레이 정밀 크롭 2-3장
- **카메라↔BEV 일치 검사**: 각 패널에서 JSON(객체 수/위치/신호등 상태)을 추출한 뒤
  **텍스트 대 텍스트 비교** (이미지 2장 비교 프롬프트보다 신뢰도 높음)

### 해상도 사다리

Qwen 계열 `tokens ≈ H×W/1024`, `max_pixels`로 제어:

| max_pixels | 토큰/장 | 용도 |
|---|---|---|
| ~1 MP | ~1K | 객체 단위 누락 여부 수준의 gross check |
| ~2.6 MP | ~2.5K | 중간, knee 후보 |
| ~5 MP | ~5K | 정밀 |
| native (~9.6 MP) | ~9.4K | 박스 정렬/작은 숫자 판독 |

- 운영 패턴: 1 MP 풀프레임 저비용 패스 + 고해상 크롭 정밀 패스
- 과해상도가 grounding을 해치는 사례 보고 있음. 평가셋에서 knee 찾기
- 타일링이 불가피하면 5-10% 오버랩 + 경계 중복 플래그 dedupe.
  패널 정렬 크롭이면 이 문제 자체가 없음

### Few-shot

- 예시 이미지(정상 1 + defect 유형별 1씩, gold JSON 포함) 2-4장 인터리브
- `--limit-mm-per-prompt` 상향
- **예시 블록은 byte-identical 유지**: vLLM이 이미지 해시로 prefix/encoder 캐시 히트
  ([08 2번 캐싱 3계층](08-optimization-catalog.md#멀티모달-캐싱-3계층))

## 3. Structured Output (vLLM 2026)

- 구 `guided_json`은 v0.12에서 제거됨.
  현행: 서버 `response_format`/`structured_outputs` extra-body, 오프라인 `StructuredOutputsParams`
- 백엔드 xgrammar(기본). 단일 스키마 공유 시 컴파일 비용 상각, 오버헤드 near-zero

스키마 설계 원칙:

- 체크 항목당 플랫 객체:
  `{check: enum, verdict: enum[pass|fail|uncertain], severity: enum, evidence: str, region: str|null, confidence: number}`
- **enum 최대한**: constrained decoding = 강제 선택 = 최고 신뢰도
- **`evidence`를 `verdict`보다 앞에**: autoregressive라 추론 후 판정하게
- **`uncertain` 선택지 필수**: constrained decoding은 거부 불가. 없으면 근거 없는 확정 출력이 나온다
- 깊은 중첩/`oneOf`/무한 배열 회피, 배열 길이 캡
- Pydantic → `model_json_schema()` → 응답 재검증 → 실패는 재시도 큐
- Thinking 변형: 구조화는 reasoning 이후 적용이 기본 (그대로 둘 것)

## 4. 평가: 모든 최적화의 게이트

### 라벨셋

- 300-1,000 프레임, defect 클래스 = JSON 스키마와 동일 taxonomy
- 층화: 정상(다수) + 클래스별 + hard negative + 주야/우천/밀집
- 라벨러 2명 이상 겹치기 → **사람 간 κ 먼저** (judge가 라벨 노이즈를 이길 수 없음)

### 지표

- 클래스별 P/R/F1. missed-detection recall = 안전 핵심 수치, precision = 리뷰어 신뢰
- 사람 대비 Cohen's κ. 2026 기준 **0.6 합격 / 0.8 우수** (GPT-4o급 judge 참고치 κ 0.72-0.83)
- flag-rate 드리프트를 카나리로

### 회귀 하네스

- promptfoo (이미지 입력 + JSON assertion + CI 게이트, 완전 자체호스팅이라 에어갭 적합)
- 프롬프트/루브릭/스키마는 git 관리
- 배치 출력에 `{prompt_sha, model, quant, max_pixels, sampling}` 태깅
- 집계 벤치는 이미지 단위 판정 뒤집힘을 숨긴다.
  **양자화 등 모든 변경은 자체 평가셋의 클래스별 지표로 게이트**
