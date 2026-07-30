# curl 로 확인하기

SDK 없이 바로 두드려 보는 용도. Python 예제는 [quickstart.py](quickstart.py).

```bash
export HOST=<STRAD32_IP>                   # .env 의 STRAD32_IP. 서버 안에서는 localhost
export VLM=http://$HOST:8000               # 8000 = LB. 클라이언트는 항상 이 주소를 쓴다
export MODEL=qwen36-35b-a3b
```

`:8001`, `:8002` 는 개별 레플리카 직결이며 **디버깅·지표 조회 전용**이다.
레플리카에 직접 붙으면 분배도 폴백도 없다.

응답 파싱에 `jq` 가 없으면 `python3 -m json.tool` 이나 `python3 -c` 로 대체하면 된다.

---

## 0. 살아 있나

```bash
curl -s $VLM/lb-health                      # LB 자체 (업스트림을 거치지 않는다)
curl -s $VLM/v1/models | python3 -m json.tool
```

`lb ok` 와 모델 목록이 나오면 LB·레플리카 둘 다 정상이다.

## 1. 텍스트

```bash
curl -s $VLM/v1/chat/completions -H "Content-Type: application/json" -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"한국어로 한 문장만 답해주세요. 당신은 누구인가요?\"}],
  \"max_tokens\": 128,
  \"chat_template_kwargs\": {\"enable_thinking\": false}
}" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"]); print(d["usage"])'
```

- `enable_thinking: false` 를 빼면 thinking 이 켜져 훨씬 느리다. 기본값이 ON 이다.
- 출력 언어를 프롬프트에 명시할 것. 한국어로 물어도 다른 언어로 답할 때가 있다.

## 2. 이미지

```bash
IMG=examples/sample_overlay.jpg              # 또는 본인 프레임
B64=$(base64 -w0 "$IMG")                     # macOS 는 base64 -i "$IMG" | tr -d '\n'

curl -s $VLM/v1/chat/completions -H "Content-Type: application/json" -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": [
    {\"type\": \"text\", \"text\": \"이 이미지의 바운딩 박스와 오버레이 텍스트를 한국어로 그대로 나열해라.\"},
    {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$B64\"}}
  ]}],
  \"max_tokens\": 400,
  \"chat_template_kwargs\": {\"enable_thinking\": false}
}" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"]); print(d["usage"])'
```

`usage.prompt_tokens` 를 보면 이미지 비용이 그대로 드러난다. 1280x720 이 약 915 토큰,
4464x2160 실제 프레임은 약 9,400 토큰이다 (`H x W / 1024 + 2`).

## 3. 스트리밍

```bash
curl -N -s $VLM/v1/chat/completions -H "Content-Type: application/json" -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"1부터 10까지 세어줘\"}],
  \"max_tokens\": 128,
  \"stream\": true,
  \"chat_template_kwargs\": {\"enable_thinking\": false}
}"
```

`-N`(버퍼링 해제)이 없으면 청크가 한꺼번에 몰려 보인다. `data: {...}` 가 여러 줄로
흘러나오면 정상이다. 한 덩어리로 오면 앞단 LB 의 `proxy_buffering` 을 의심할 것.

## 4. structured output (JSON 스키마 강제)

문법 위반 토큰을 생성 단계에서 막으므로, 프롬프트로 "JSON 으로 답해줘" 라고 부탁하는 것과 다르다.

```bash
curl -s $VLM/v1/chat/completions -H "Content-Type: application/json" -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"서울과 부산의 인구를 알려줘\"}],
  \"max_tokens\": 200,
  \"chat_template_kwargs\": {\"enable_thinking\": false},
  \"response_format\": {
    \"type\": \"json_schema\",
    \"json_schema\": {
      \"name\": \"cities\",
      \"schema\": {
        \"type\": \"object\",
        \"properties\": {
          \"cities\": {\"type\": \"array\", \"items\": {
            \"type\": \"object\",
            \"properties\": {\"name\": {\"type\": \"string\"}, \"population\": {\"type\": \"integer\"}},
            \"required\": [\"name\", \"population\"]
          }}
        },
        \"required\": [\"cities\"]
      }
    }
  }
}" | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

## 5. tool calling + 이미지 동시 (가동 검증 게이트 5)

이미지 입력과 tool call 을 **동시에** 쓸 때 파서가 깨지는지가 확인 대상이다
(GLM-4.6V 가 이 이슈로 후보에서 탈락했다 — `docs/02`).

```bash
B64=$(base64 -w0 examples/sample_overlay.jpg)

curl -s $VLM/v1/chat/completions -H "Content-Type: application/json" -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": [
    {\"type\": \"text\", \"text\": \"이미지에서 결함으로 의심되는 것을 report_defect 로 보고해라.\"},
    {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$B64\"}}
  ]}],
  \"tools\": [{\"type\": \"function\", \"function\": {
    \"name\": \"report_defect\",
    \"description\": \"검출 결함을 보고한다\",
    \"parameters\": {\"type\": \"object\", \"properties\": {
      \"defect_type\": {\"type\": \"string\", \"enum\": [\"missed_detection\", \"false_positive\", \"box_misalignment\"]},
      \"note\": {\"type\": \"string\"}
    }, \"required\": [\"defect_type\"]}
  }}],
  \"tool_choice\": \"auto\",
  \"max_tokens\": 300,
  \"chat_template_kwargs\": {\"enable_thinking\": false}
}" | python3 -c 'import sys,json; m=json.load(sys.stdin)["choices"][0]["message"]; print(json.dumps(m.get("tool_calls"), ensure_ascii=False, indent=2))'
```

`tool_calls` 가 채워지면 통과. `null` 이거나 파서 오류가 나면 서버를
`--tool-call-parser qwen3_xml` 로 재기동해 재시도한다.

## 6. 엔진 지표

LB 를 거치지 않고 **개별 레플리카**에 직접 물어야 한다. `:8000` 은 요청을 분배하므로
지표가 한쪽 것만 보인다.

```bash
curl -s http://$HOST:8001/metrics | grep -E '^vllm:(num_requests_running|num_requests_waiting\{|gpu_cache_usage_perc)'
curl -s http://$HOST:8001/metrics | grep -E '^vllm:prefix_cache_(queries|hits)_total'
```

| 지표 | 읽는 법 |
|---|---|
| `num_requests_running` / `waiting` | 박스가 바쁜 이유가 처리 중인지 큐잉인지 가른다 |
| `gpu_cache_usage_perc` | KV 캐시 포화도. 100% 에 붙으면 동시성이 KV 로 막힌 것 |
| `prefix_cache_*` | 이 모델은 prefix caching 이 자동 비활성이라 0 으로 고정이다 |

## 부하 한 번 줘 보기

동시 요청을 넣어야 `num_requests_running` 이 올라가고 LB 분배도 관찰된다.
분배가 한쪽으로 쏠리면 `docker/nginx-lb.conf` 의 `zone` 을 확인할 것
(zone 이 없으면 nginx 워커마다 연결 카운터가 따로여서 동시 요청이 전부 첫 레플리카로 간다).

```bash
for i in $(seq 1 8); do
  curl -s -o /dev/null $VLM/v1/chat/completions -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"MLOps 를 길게 설명해줘 ($i)\"}],
    \"max_tokens\": 500,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }" &
done
sleep 5
for p in 8001 8002; do
  printf ":%s running=%s\n" "$p" \
    "$(curl -s http://$HOST:$p/metrics | awk '/^vllm:num_requests_running/{print $2}')"
done
wait
```

## 흔한 증상

| 증상 | 원인 |
|---|---|
| 첫 요청이 수십 초 | 커널 JIT 컴파일. 두 번째부터 정상 |
| 응답이 느리고 `reasoning` 이 채워짐 | `enable_thinking` 을 안 껐다 |
| 스트리밍이 한 덩어리로 옴 | `curl -N` 누락, 또는 LB `proxy_buffering` |
| 413 Request Entity Too Large | 이미지가 LB 의 `client_max_body_size` 초과 |
| 응답 언어가 다름 | 프롬프트에 출력 언어를 명시할 것 |
