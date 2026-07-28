#!/usr/bin/env python3
"""테스트용 합성 오버레이 이미지 생성.

svnet3 추론 결과 시각화(docs/01 입력 데이터 특성)를 흉내 낸다.
전방 카메라 패널의 바운딩 박스 + 거리·속도 오버레이, 차선, HUD 텍스트.

실제 프레임은 보안 정책 확인 전까지 레포에 넣지 않는다(.gitignore 의 samples/).
이 스크립트가 만드는 것은 **합성 이미지**이며 판독 정확도를 눈으로 보는 용도다.

    pip install pillow
    python examples/make_sample.py                    # examples/sample_overlay.jpg
    python examples/make_sample.py -o /tmp/x.jpg --width 4464 --height 2160
"""
import argparse
import pathlib

from PIL import Image, ImageDraw

# 정답지. 모델이 이걸 그대로 읽어내는지가 판독 확인의 기준이 된다.
GROUND_TRUTH = [
    ("CAR", "12.5m", "38km/h", (0, 255, 90), (0.09, 0.25, 0.37, 0.72)),
    ("VRU", "7.2m", None, (255, 90, 90), (0.59, 0.33, 0.77, 0.78)),
    ("TL:GREEN", "24.1m", None, (90, 200, 255), (0.44, 0.10, 0.52, 0.24)),
]


def build(width: int, height: int) -> Image.Image:
    im = Image.new("RGB", (width, height), (18, 22, 30))
    d = ImageDraw.Draw(im)
    s = max(1, width // 320)  # 선 굵기를 해상도에 비례시킨다

    d.text((int(width * 0.015), int(height * 0.03)),
           "FRONT CAM  |  frame 004217  |  ego 41km/h", fill=(200, 200, 200))

    for label, dist, speed, color, (x0, y0, x1, y1) in GROUND_TRUTH:
        box = [int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)]
        d.rectangle(box, outline=color, width=s)
        cap = f"{label}  {dist}" + (f"  {speed}" if speed else "")
        d.text((box[0] + 2, box[1] - int(height * 0.035)), cap, fill=color)

    # 차선 (좌우 2개)
    d.line([(0, int(height * 0.95)), (int(width * 0.42), int(height * 0.62))],
           fill=(90, 160, 255), width=s)
    d.line([(width, int(height * 0.92)), (int(width * 0.60), int(height * 0.62))],
           fill=(90, 160, 255), width=s)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", type=pathlib.Path,
                    default=pathlib.Path(__file__).parent / "sample_overlay.jpg")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    build(args.width, args.height).save(args.out, quality=92)
    tokens = args.width * args.height // 1024 + 2  # docs/02 의 이미지 토큰 수식
    print(f"저장: {args.out}  ({args.width}x{args.height}, 약 {tokens:,} vision tokens)")
    print("정답지:")
    for label, dist, speed, *_ in GROUND_TRUTH:
        print(f"  {label}  {dist}" + (f"  {speed}" if speed else ""))


if __name__ == "__main__":
    main()
