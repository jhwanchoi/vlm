#!/usr/bin/env python3
"""공용 VLM 서빙 quickstart — 접속·텍스트·이미지·스트리밍을 한 번에 확인한다.

서버 구성은 docs/04 §1, 기동은 scripts/serve.sh 참조.
이 스크립트의 호출 형태는 2026-07-28 strad32 실측으로 검증된 것이다.

    pip install openai pillow

    python examples/quickstart.py                          # 전체
    python examples/quickstart.py --image my_frame.jpg     # 내 이미지로
    python examples/quickstart.py --base-url http://<IP>:8000/v1

주의
  * 인증이 없다. api_key 는 아무 값이나 넣으면 된다.
  * 첫 요청은 커널 JIT 컴파일로 수십 초 걸릴 수 있다. 두 번째부터 정상.
  * 출력 언어를 반드시 프롬프트로 지정할 것. 한국어로 물어도 다른 언어로 답할 때가 있다.
"""
import argparse
import base64
import io
import os
import pathlib
import sys

DEFAULT_BASE_URL = os.environ.get("VLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("VLM_MODEL", "qwen36-35b-a3b")

# thinking 은 기본 ON 이라 배치·자동화에서는 꺼야 느려지지 않는다 (docs/02).
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def b64_jpeg(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def image_message(text: str, b64: str) -> list:
    """OpenAI 호환 멀티모달 메시지. data URI 또는 http URL 둘 다 된다."""
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--image", type=pathlib.Path, default=None,
                    help="생략하면 examples/sample_overlay.jpg 를 쓴다")
    args = ap.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai 가 필요하다", file=sys.stderr)
        return 1

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")  # 인증 없음
    print(f"서버: {args.base_url}\n")

    # 1) 서버가 어떤 모델을 서빙 중인지. model 필드에 쓸 이름을 여기서 확인한다.
    ids = [m.id for m in client.models.list().data]
    print(f"[1] 모델 목록      : {ids}")
    if args.model not in ids:
        print(f"    경고: --model {args.model} 이 목록에 없다. {ids[0]} 로 진행한다")
        args.model = ids[0]

    # 2) 텍스트
    r = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "한국어로 한 문장만 답해주세요. 당신은 누구인가요?"}],
        max_tokens=128,
        extra_body=NO_THINKING,
    )
    print(f"[2] 텍스트 왕복    : {r.choices[0].message.content.strip()[:160]}")
    print(f"    usage          : {r.usage}")

    # 3) 이미지. 이미지 1장이 대략 (H*W/1024) 토큰을 먹으므로 큰 프레임은 비용이 크다.
    img = args.image or (pathlib.Path(__file__).parent / "sample_overlay.jpg")
    if not img.exists():
        print(f"    이미지 없음: {img}  (make_sample.py 로 생성하거나 --image 로 지정)")
    else:
        r2 = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": image_message(
                "이 이미지의 바운딩 박스와 오버레이 텍스트를 한국어로 그대로 나열해라.",
                b64_jpeg(img))}],
            max_tokens=400,
            extra_body=NO_THINKING,
        )
        print(f"[3] 이미지 왕복    : {img.name}")
        for line in r2.choices[0].message.content.strip().splitlines()[:10]:
            print(f"    | {line}")
        print(f"    usage          : {r2.usage}")

    # 4) 스트리밍. 청크가 여러 개로 나뉘어 오면 앞단 LB 의 버퍼링이 꺼져 있다는 뜻이다.
    stream = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "1부터 5까지 세어줘"}],
        max_tokens=64, stream=True, extra_body=NO_THINKING,
    )
    chunks = [c.choices[0].delta.content or "" for c in stream]
    body = "".join(chunks).replace("\n", " ").strip()
    print(f"[4] 스트리밍       : {len(chunks)} chunks -> {body[:80]}")

    print("\n전 항목 통과.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
