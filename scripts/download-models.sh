#!/bin/bash
# 모델 다운로드 스크립트. 자기 계정으로 로그인한 상태에서 실행 (HF_HOME이 .bashrc에 설정되어 있음).
# 사내망 프록시가 필요하면 HTTPS_PROXY 설정 후 실행.
set -e

command -v hf >/dev/null || pip install -U "huggingface_hub[cli]"

echo "HF_HOME=${HF_HOME:-기본값(~/.cache/huggingface)}"
df -h "${HF_HOME:-$HOME}" | tail -1

# 1) 공용 모델 (확정, 2026-07-27): NVFP4, 1장/레플리카 (약 20GB)
#    unsloth 판이 기본. NVIDIA 공식판 확인되면 교체.
hf download unsloth/Qwen3.6-35B-A3B-NVFP4

# 2) 선택: 파이프라인 bring-up용 소형 (약 10GB, GPU 1장)
hf download Qwen/Qwen3-VL-8B-Instruct-FP8

# 3) 품질 폴백 (약 16GB) - 가동 검증 게이트 실패 시에만
#    27B NVFP4는 커뮤니티판 리포지토리명 재확인 후 주석 해제할 것.
# hf download <27B-NVFP4-repo>

# 4) 후순위 (필요 시):
# hf download Qwen/Qwen3.5-122B-A10B-Instruct-AWQ   # 2차 검증기 (4장 풀에서만, 리포명 확인)
# hf download zai-org/GLM-4.6V-AWQ                  # bbox 교차검증 비교군 (리포명 확인)
# hf download PaddlePaddle/PaddleOCR-VL-1.5         # 한국어 OCR 사이드카 (리포명 확인)
# 사이드카 detector (게이트 4a 미달 시에만): LLMDet Swin-L, SAM 3 (transformers 경로, hf 다운로드 불필요)

echo "다운로드 완료. 용량 확인:"
du -sh "${HF_HOME:-$HOME/.cache/huggingface}"
