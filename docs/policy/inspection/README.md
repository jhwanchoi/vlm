# policy/inspection/ — Visual Inspection 판정 정책

svnet3 결과 검사의 **판정 정책**(defect 정의, 심각도, 판정 규칙)을 외부에서 전달받으면 여기에 정리한다.
2026-07-22 기준 미수신 — 수신 대기 중.

파일 규칙은 [상위 README](../README.md) 공통 규칙을 따름
(정리본: `YYYY-MM-DD-inspection-policy-v<N>.md`).

## 수신 시 체크리스트

- [ ] Defect 클래스 정의 — [01](../../01-project-overview.md)의 초안 taxonomy와 대조, 차이 기록
- [ ] 클래스별 심각도(severity) 기준
- [ ] 판정 기준선 — 무엇이 pass/fail인지, 경계 사례 처리
- [ ] 예외/무시 조건 (예: 특정 거리 이상 객체, 특정 날씨)
- [ ] 요구 지표/합격선 (있다면 — recall 하한 등)

## 수신 후 전파 절차

정책은 아래 3곳의 뿌리 — 수신 즉시 반영:

1. [01](../../01-project-overview.md) — defect taxonomy 초안 → 확정본으로 교체
2. [06](../../06-inspection-agent-design.md) — JSON 스키마(enum/severity)와 라벨링 가이드 반영
3. 평가셋 라벨 기준 — 기존 라벨이 있으면 정책 버전과의 정합 재검토

배치 출력 태깅(`{prompt_sha, model, quant, ...}`)에 **`policy_version` 필드 추가** —
어떤 정책 기준 판정인지 추적 가능해야 함.
