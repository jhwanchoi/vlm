# examples — 바로 써 보는 클라이언트 예제

서빙이 떠 있으면(→ [scripts/serve.sh](../scripts/serve.sh)) 아래로 곧장 확인할 수 있다.

```bash
pip install openai pillow

python examples/quickstart.py --base-url http://<STRAD32_IP>:8000/v1
```

`VLM_BASE_URL` / `VLM_MODEL` 환경변수로도 지정할 수 있다.

| 파일 | 용도 |
|---|---|
| [quickstart.py](quickstart.py) | 접속·텍스트·이미지·스트리밍 4항목을 한 번에 확인 |
| [make_sample.py](make_sample.py) | 합성 오버레이 이미지 생성 (해상도 지정 가능) |
| `sample_overlay.jpg` | 미리 만들어 둔 1280x720 샘플. 정답지는 `make_sample.py` 의 `GROUND_TRUTH` |
| [prompts.md](prompts.md) | 용도별 프롬프트 예시 (inspection · 라벨 판정 · 좌표 규약 · tool calling) |

## 접속 정보

| | |
|---|---|
| base_url | `http://<STRAD32_IP>:8000/v1` (`.env` 의 `STRAD32_IP`) |
| model | `qwen36-35b-a3b` |
| api_key | **인증 없음.** 아무 값이나 (`"EMPTY"` 등) |

`:8000` 은 앞단 LB 이고 뒤에 레플리카가 여러 개 있다.
개별 레플리카(`:8001`, `:8002`)에 직접 붙는 것은 디버깅 용도로만.

## 알아둘 것

- **첫 요청은 느리다.** 커널 JIT 컴파일 때문이며 수십 초 걸릴 수 있다. 두 번째부터 정상.
- **출력 언어를 프롬프트에 명시한다.** 한국어로 물어도 다른 언어로 답할 때가 있다.
- **배치·자동화는 thinking 을 끈다.** `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
  기본값이 ON 이다.
- **이미지 비용**: 대략 `H x W / 1024` 토큰. 4464x2160 프레임 한 장이 약 9,400 토큰이다.
  요청당 최대 4장, 컨텍스트 32,768 토큰.
- POC 단계라 예고 없이 재기동될 수 있다.

## 실제 프레임은 레포에 넣지 않는다

svnet3 추론 결과 이미지가 보안 정책 범위인지 미확인이다 ([01 제약 사항](../docs/01-project-overview.md)).
`samples/` 는 `.gitignore` 대상이며, 여기 `examples/` 에는 **합성 이미지만** 둔다.
실제 프레임으로 시험할 때는 `--image` 로 로컬 경로를 넘긴다.
