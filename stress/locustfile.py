"""VLM 서빙 부하 시나리오.

이 파일의 책임은 요청 1건을 어떻게 보내고 어떻게 재느냐 뿐이다.
부하의 형태(동시성 계단)는 shapes.py, 감시와 중단은 monitor.sh 가 담당한다.

계측 방식
  Locust 기본 client 를 쓰지 않고 requests 세션을 직접 쓴다. SSE 첫 청크 시각을
  재려면 응답 스트림을 직접 돌아야 하고, Locust client 를 쓰면 헤더 도착 시각이
  자동 이벤트로 한 번 더 집계되어 통계가 이중으로 남는다.

  요청당 이벤트를 2개 발생시킨다.
    TTFT  <시나리오>   response_time = 첫 토큰까지 ms,  response_length = prompt_tokens
    total <시나리오>   response_time = 마지막 청크까지 ms, response_length = completion_tokens

  response_length 에 토큰 수를 실어두면 Locust 의 Average Content Size 집계가
  그대로 "요청당 토큰 수"가 되고, analyze.py 가 처리량(토큰/s)을 RPS 와 곱해 낸다.

환경변수 (run.sh 가 주입)
  TARGET_HOST   http://host.docker.internal:8000 등. 우리 엔드포인트만 허용된다
  MODEL         served-model-name
  SCENARIOS     쉼표 구분. 기본 "S1,S2,S3"
  IGNORE_EOS    1이면 max_tokens 까지 무조건 생성 (동시성별 비교용 고정 길이)
  USE_SCHEMA    1이면 response_format json_schema 사용 (xgrammar 비용 포함)
  MAX_TOKENS    기본 256
"""
import base64
import json
import os
import pathlib
import sys
import time

import requests
from locust import User, constant, events, task  # noqa: F401  task는 API 호환용

# locust 가 locustfile 디렉토리를 sys.path 에 넣어주지만, 컨테이너/버전에 따라
# 보장되지 않으므로 직접 넣는다.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from shapes import StepLoad  # noqa: E402,F401  Locust가 이 이름으로 shape를 찾는다

TARGET_HOST = os.environ.get("TARGET_HOST", "http://host.docker.internal:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "qwen36-35b-a3b")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
IGNORE_EOS = os.environ.get("IGNORE_EOS", "1") == "1"
USE_SCHEMA = os.environ.get("USE_SCHEMA", "0") == "1"
SCENARIOS = [s.strip() for s in os.environ.get("SCENARIOS", "S1,S2,S3").split(",") if s.strip()]
ASSET_DIR = pathlib.Path(os.environ.get("ASSET_DIR", "/mnt/locust/assets"))
TTFT_TIMEOUT = float(os.environ.get("TTFT_TIMEOUT", "60"))
TOTAL_TIMEOUT = float(os.environ.get("TOTAL_TIMEOUT", "300"))

# 부하 대상 화이트리스트. 타 팀 엔드포인트로 실수로 쏘는 것을 코드에서 막는다.
# 8000 LB, 8001-8002 운영 레플리카, 8003 실험용 레플리카(GPU 2, 우리 몫 여유분).
_ALLOWED_PORTS = {"8000", "8001", "8002", "8003"}

# inspection 판정 프롬프트. examples/prompts.md 2번을 부하용으로 줄인 형태.
# 출력 형식을 좁게 고정해야 completion_tokens 가 흔들리지 않는다.
_SCHEMA_TEXT = """아래 JSON 스키마로만 답하라. 설명 문장을 붙이지 마라.
{"defects": [{"type": "missed_detection | false_positive | box_misalignment | lane_fit_error | implausible_value | panel_inconsistency", "evidence": "한국어 한 문장", "confidence": 0.0}], "overall": "pass | fail | uncertain"}"""

PROMPT_IMAGE = (
    "당신은 자율주행 인식 결과를 검수하는 검사자다. "
    "이미지는 전방 카메라에 검출 결과가 오버레이된 화면이다.\n" + _SCHEMA_TEXT
)
PROMPT_TEXT = (
    "당신은 자율주행 인식 결과를 검수하는 검사자다. "
    "다음 검출 결과를 검수하라: CAR 12.5m 38km/h, VRU 7.2m, TL:GREEN 24.1m, ego 41km/h.\n"
    + _SCHEMA_TEXT
)

JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "inspection",
        "schema": {
            "type": "object",
            "properties": {
                "defects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "evidence": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["type", "evidence", "confidence"],
                    },
                },
                "overall": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
            },
            "required": ["defects", "overall"],
        },
    },
}

# 시나리오 정의. weight 는 혼합 실행 시의 비중이다.
SCENARIO_SPECS = {
    "S1": {"weight": 2, "image": None, "prompt": PROMPT_TEXT},
    "S2": {"weight": 5, "image": "img_1mp.jpg", "prompt": PROMPT_IMAGE},
    "S3": {"weight": 3, "image": "img_native.jpg", "prompt": PROMPT_IMAGE},
}


def _load_b64(name):
    """이미지를 프로세스당 1회만 base64 로 만들어 재사용한다.

    요청마다 인코딩하면 부하기 CPU 가 측정 대상이 되어버린다. 서버 쪽에서는
    prefix cache 와 mm-processor cache 가 모두 꺼져 있어 동일 이미지라도
    매번 재인코딩되므로, 이미지를 재사용해도 캐시 특혜는 생기지 않는다.
    """
    path = ASSET_DIR / name
    if not path.exists():
        raise SystemExit(f"이미지가 없다: {path} (make_assets.py 로 생성해 서버에 복사할 것)")
    return base64.b64encode(path.read_bytes()).decode()


_B64_CACHE = {}


def _payload(scn):
    spec = SCENARIO_SPECS[scn]
    prompt = spec["prompt"]
    if spec["image"]:
        b64 = _B64_CACHE.setdefault(spec["image"], _load_b64(spec["image"]))
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
    else:
        content = prompt

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if IGNORE_EOS:
        body["ignore_eos"] = True
    if USE_SCHEMA:
        body["response_format"] = JSON_SCHEMA
    return body


def _fire(request_type, scn, ms, length, exc=None):
    events.request.fire(
        request_type=request_type,
        name=scn,
        response_time=ms,
        response_length=length,
        exception=exc,
        context={},
    )


def _one_request(session, scn):
    """요청 1건 전송 + 계측. 예외는 호출자에서 이벤트로 변환한다."""
    body = _payload(scn)
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    usage = None

    with session.post(
        f"{TARGET_HOST}/v1/chat/completions",
        json=body,
        stream=True,
        timeout=(10, TTFT_TIMEOUT),
    ) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        for line in r.iter_lines(decode_unicode=True):
            if time.perf_counter() - t0 > TOTAL_TIMEOUT:
                raise TimeoutError(f"전체 {TOTAL_TIMEOUT}s 초과")
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    chunks += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0

    total = time.perf_counter() - t0
    if ttft is None:
        raise RuntimeError("본문 토큰이 오지 않았다")
    if ttft > TTFT_TIMEOUT:
        raise TimeoutError(f"TTFT {ttft:.1f}s 초과")

    # usage 청크가 없으면 청크 수로 대체한다. 서버 설정에 따라 빠질 수 있다.
    completion = int(usage["completion_tokens"]) if usage else chunks
    prompt_tokens = int(usage["prompt_tokens"]) if usage else 0
    return ttft * 1000, total * 1000, prompt_tokens, completion


def _make_task(scn):
    def run(self):
        try:
            ttft_ms, total_ms, ptok, ctok = _one_request(self.session, scn)
        except Exception as exc:  # noqa: BLE001  실패도 통계에 남겨야 한다
            _fire("total", scn, 0, 0, exc=exc)
            return
        _fire("TTFT", scn, ttft_ms, ptok)
        _fire("total", scn, total_ms, ctok)

    run.__name__ = f"task_{scn}"
    return run


def _enabled_tasks():
    unknown = [s for s in SCENARIOS if s not in SCENARIO_SPECS]
    if unknown:
        raise SystemExit(f"알 수 없는 시나리오: {unknown}")
    return {_make_task(s): SCENARIO_SPECS[s]["weight"] for s in SCENARIOS}


class VLMUser(User):
    # 폐루프 부하. 사용자 수가 그대로 동시 요청 수가 된다.
    wait_time = constant(0)
    tasks = _enabled_tasks()

    def on_start(self):
        self.session = requests.Session()

    def on_stop(self):
        self.session.close()


@events.init.add_listener
def _check_target(environment, **_):
    port = TARGET_HOST.rsplit(":", 1)[-1].split("/")[0]
    if port not in _ALLOWED_PORTS:
        raise SystemExit(
            f"대상 포트 {port} 는 허용 목록 {sorted(_ALLOWED_PORTS)} 밖이다. "
            "타 팀 엔드포인트에 부하를 걸 수 없다"
        )


@events.test_start.add_listener
def _warmup(environment, **_):
    """첫 요청의 커널 JIT 컴파일을 측정 구간에서 배제한다."""
    session = requests.Session()
    for scn in SCENARIOS:
        try:
            _one_request(session, scn)
        except Exception as exc:  # noqa: BLE001
            print(f"[warmup] {scn} 실패: {exc}")
        else:
            print(f"[warmup] {scn} ok")
    session.close()
