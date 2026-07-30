#!/usr/bin/env python3
"""부하용 합성 이미지 생성. Pillow 가 필요하므로 로컬에서 실행한다.

서버에는 pip 이 없어 Pillow 를 설치할 수 없다. 여기서 만들어 run.sh 가
서버로 복사한다. 실 svnet3 프레임은 반입하지 않는다. 부하 특성은 픽셀 수가
결정하므로 합성으로 충분하고, 판독 정확도는 이 테스트의 관심사가 아니다.

    python3 stress/make_assets.py

  img_1mp.jpg     1024x1024   약 1.0K vision tokens
  img_native.jpg  4464x2160   약 9.4K vision tokens (docs/01 실 프레임과 동일 해상도)
"""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).resolve().parent / "assets"

SIZES = {
    "img_1mp.jpg": (1024, 1024),
    "img_native.jpg": (4464, 2160),
}


def _load_builder():
    """examples/make_sample.py 의 build() 를 재사용한다 (오버레이 생성 로직 중복 방지)."""
    path = ROOT / "examples" / "make_sample.py"
    spec = importlib.util.spec_from_file_location("_make_sample", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build


def main():
    build = _load_builder()
    OUT.mkdir(exist_ok=True)
    for name, (w, h) in SIZES.items():
        dst = OUT / name
        build(w, h).save(dst, quality=92)
        tokens = w * h // 1024 + 2
        print(f"{dst}  {w}x{h}  {dst.stat().st_size / 1e6:.2f} MB  약 {tokens:,} vision tokens")


if __name__ == "__main__":
    main()
