#!/bin/bash
# vLLM 스모크 테스트: Qwen3-VL-8B-FP8을 GPU 1장에 올리고 vision 요청 1건 확인.
# 인프라 검증 전용 (inspection 로직 아님 — 그건 정책 수신 후).
# dst 또는 dmt 계정에서 실행 권장 (GPU 0-3 몫).
set -e

MODEL="Qwen/Qwen3-VL-8B-Instruct-FP8"
PORT=8001
GPU=0

docker run -d --name vlm-smoke \
  --gpus "\"device=$GPU\"" \
  --cpuset-cpus="0-15,32-47" --cpuset-mems="0" \
  --memory=64g --memory-swap=64g --shm-size=16g \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -p $PORT:8000 \
  vllm/vllm-openai:latest \
  --model "$MODEL" \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 2}' \
  --max-model-len 16384

echo "기동 대기 중 (모델 로드 1-3분)..."
for i in $(seq 1 60); do
  curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1 && break
  sleep 5
done
curl -sf "http://localhost:$PORT/v1/models" || { echo "기동 실패. 로그:"; docker logs --tail 30 vlm-smoke; exit 1; }

echo
echo "=== 텍스트 요청 스모크 ==="
curl -s "http://localhost:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"1+1=?\"}], \"max_tokens\": 10}" | head -c 500

echo
echo "=== 이미지 요청 스모크 (인자로 이미지 경로 주면 실행) ==="
if [ -n "$1" ] && [ -f "$1" ]; then
  B64=$(base64 -w0 "$1")
  curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": [
          {\"type\": \"text\", \"text\": \"이 이미지에 보이는 것을 두 문장으로 설명해라.\"},
          {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$B64\"}}
        ]}], \"max_tokens\": 150}" | head -c 1000
else
  echo "(이미지 미지정 — bash serve-smoke-test.sh /path/to/sample.jpg 로 재실행)"
fi

echo
echo "정리: docker rm -f vlm-smoke"
