# 프롬프트 예시

용도별 출발점. 확정본이 아니라 **실험의 시작점**이며, inspection 정책 문서 수신 후
[06 inspection agent 설계](../docs/06-inspection-agent-design.md)에서 다듬는다.

공통 규칙 세 가지:

1. **출력 언어를 명시한다.** 한국어로 물어도 다른 언어로 답할 때가 있다 (실측).
2. **배치·자동화는 thinking 을 끈다.** `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
   기본값이 ON 이라 켜 두면 느리다 ([02 서빙 노트](../docs/02-model-candidates.md)).
3. **자유 텍스트 대신 스키마를 준다.** 파싱이 안정되고 출력 토큰이 줄어 decode 비용이 직접 내려간다
   ([08 6번 출력측 다이어트](../docs/08-optimization-catalog.md)).

---

## 1. 판독 확인 (bring-up 스모크)

`sample_overlay.jpg` 로 모델이 오버레이를 읽는지만 본다. 정답지는 `make_sample.py` 의 `GROUND_TRUTH`.

```
이 이미지의 바운딩 박스와 오버레이 텍스트를 한국어로 그대로 나열해라.
각 박스마다 라벨, 거리, 속도(있으면), 박스 색을 적어라. 추측하지 말고 보이는 것만 적어라.
```

기대: `CAR 12.5m 38km/h` (녹색), `VRU 7.2m` (적색), `TL:GREEN 24.1m` (하늘색).

---

## 2. inspection — 검출 정합성 판정 (DMT)

[01 검사 대상 defect](../docs/01-project-overview.md) 초안의 6개 클래스를 그대로 스키마로 옮긴 형태.

```
당신은 자율주행 인식 결과를 검수하는 검사자다.
이미지는 전방 카메라에 검출 결과가 오버레이된 화면이다.

아래 JSON 스키마로만 답하라. 설명 문장을 붙이지 마라.

{
  "defects": [
    {
      "type": "missed_detection | false_positive | box_misalignment | lane_fit_error | implausible_value | panel_inconsistency",
      "evidence": "화면에서 보이는 근거를 한국어 한 문장으로",
      "confidence": 0.0
    }
  ],
  "overall": "pass | fail | uncertain"
}

판정이 애매하면 defects 를 비우고 overall 을 uncertain 으로 두어라.
없는 것을 지어내지 마라.
```

> `type` 을 enum 으로, `confidence` 를 float 하나로 좁힌 것이 의도적이다.
> 긴 문자열 필드와 자유 서술을 줄이면 출력 토큰이 줄고 xgrammar 컴파일도 가벼워진다.

---

## 3. 라벨 누락 후보 판정 (DST)

[02 DST 용도 검토](../docs/02-model-candidates.md)의 2단 구조에서 **VLM 이 맡는 판정 단계**.
detector 가 뽑은 후보 crop 이 진짜 객체인지만 본다.

```
아래 이미지는 어떤 장면에서 잘라낸 후보 영역이다.
이 영역에 "{class_name}" 에 해당하는 객체가 실제로 존재하는가?

한국어로, 아래 JSON 으로만 답하라.
{"exists": true|false, "reason": "한 문장", "confidence": 0.0}

부분적으로 가려졌거나 잘렸어도 식별 가능하면 true 로 본다.
판단 근거가 화면에 없으면 false 로 하고 reason 에 그 이유를 적어라.
```

---

## 4. 좌표 규약 확인 (가동 검증 게이트 3)

bbox 좌표가 절대 픽셀인지 0-1000 정규화인지 자료마다 다르다
([02 아키텍처](../docs/02-model-candidates.md)). 실측으로 확정하는 프롬프트.

```
이 이미지에서 가장 큰 초록색 바운딩 박스의 좌표를 알려줘.
아래 JSON 으로만 답하라. 좌표계가 무엇인지도 함께 적어라.

{"bbox": [x0, y0, x1, y1], "coord_system": "absolute_pixel | normalized_0_1 | normalized_0_1000", "image_size": [width, height]}
```

받은 좌표를 원본에 되그려 IoU 를 재면 규약이 확정된다.

---

## 5. tool calling + vision 동시 (가동 검증 게이트 5)

서버가 `--tool-call-parser qwen3_coder --enable-auto-tool-choice` 로 떠 있어야 한다.
이미지 입력과 tool call 을 **동시에** 쓸 때 파서가 깨지는지가 확인 대상이다.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "report_defect",
        "description": "검출 결함을 보고한다",
        "parameters": {
            "type": "object",
            "properties": {
                "defect_type": {"type": "string", "enum": ["missed_detection", "false_positive"]},
                "note": {"type": "string"},
            },
            "required": ["defect_type"],
        },
    },
}]
# messages 에 image_url 을 포함한 채로 tools 를 함께 넘긴다
```

`response.choices[0].message.tool_calls` 가 채워지면 통과. 파서 오류가 나면
`--tool-call-parser qwen3_xml` 로 교체 후 재시도 ([02 가동 검증 게이트](../docs/02-model-candidates.md)).

---

## 비용 감각

이미지 토큰 수는 대략 `H x W / 1024 + 2` 다.

| 해상도 | 대략 토큰 |
|---|---|
| 1280 x 720 (`sample_overlay.jpg`) | 약 900 |
| 1920 x 1080 | 약 2,000 |
| 4464 x 2160 (실제 프레임) | 약 9,400 |

출력은 보통 200~500 토큰이므로 **prefill 이 지배적**이다. 해상도를 낮추는 것이 가장 큰
비용 레버이며, 패널 크롭으로 필요한 영역만 보내는 쪽이 전체 축소보다 판독에 유리하다
([08 0번 요약](../docs/08-optimization-catalog.md)).

현재 서버는 요청당 이미지 **최대 4장**, 컨텍스트 **32,768 토큰**이다.
