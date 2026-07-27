# Research Brief: 생성형 VLM의 exhaustive detection 성능과 agentic 신뢰성 (2026-07-27)

배경: DMT(svnet3 inspection)와 DST(라벨 누락 객체 탐지) 공용 모델 선정.
DST 용도가 요구하는 "카테고리 X 전부 찾기"(exhaustive open-vocabulary detection)에 대해
shortlist(Qwen3.6-35B-A3B, Qwen3.6-27B, Qwen3.5-122B-A10B, GLM-4.6V)를 검증한 원본 브리프.

---

## Q1. Qwen 계열의 exhaustive detection ("카테고리 X 전부 찾기") 성능

### 벤치마크 근거

**ODinW-13 (open-vocab detection, 증거 강도: 강함)**
- Qwen3-VL-235B-A22B가 ODinW-13에서 48.6 mAP. 공식 기술 리포트 수치이며, VLM 중에서는 최상위지만 이것은 가장 큰 235B 모델 기준. confidence score를 1.0으로 고정해 평가하므로 실제 precision-recall 트레이드오프 조절이 불가능한 구조.
- 출처: [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631), [ODinW-13 평가 코드](https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/ODinW-13)

**RF100-VL / 도메인 특화 zero-shot (강함)**
- Roboflow100-VL에서 zero-shot 최고 성능은 GroundingDINO 15.7 mAP 수준이고, Qwen2.5-VL 등 생성형 VLM은 특수 도메인(의료 등)에서 2% 미만. 항공/산업 이미지 포함 100개 데이터셋 기준으로, 학습 분포 밖 카테고리의 exhaustive detection은 VLM 단독으로는 크게 부족.
- Qwen3-VL-30B-A3B + DetPO(프롬프트 엔지니어링 + in-context) 조합이 RF20-VL에서 21.6 mAP로 특화 detector를 넘긴 사례가 있음. 즉 프롬프트/후처리 파이프라인으로 상당히 끌어올릴 여지는 있음.
- 출처: [RF100-VL 논문](https://arxiv.org/abs/2505.20612), [rf100-vl.org](https://rf100-vl.org/), [DetPO](https://arxiv.org/pdf/2603.23455)

**밀집 장면 counting = 최대 약점 (강함)**
- HoloCount(2026-07) 벤치마크: 고밀도 장면에서 Qwen3-VL-8B는 exact match 0% (MAE 40.11), Qwen3-VL-32B 3% (MAE 32.94), Qwen2.5-VL-72B 1%. 최신 Qwen3.5-27B조차 고밀도에서 20% (다른 subset에서는 80% 이상). 오픈소스 모델 전반이 실제 개수의 1/3~1/2만큼 벗어나는 systematic undercounting.
- 일반 밀도 counting은 반대로 우수: Qwen3.5-27B가 CountBench 리더보드 1위(0.978).
- 시사점: 사진 한 장에 대상 객체가 수십 개 이상이면 "빠뜨린 라벨 찾기"의 recall이 급락함. 주행 이미지에서 차량/보행자 밀집 구간이 정확히 이 케이스.
- 출처: [HoloCount](https://arxiv.org/html/2607.06420v1), [CountBench 리더보드](https://llm-stats.com/benchmarks/countbench)

**고해상도 좌표 정확도 (중간~강함)**
- Qwen3-VL은 0-1000 정규화 좌표를 출력. 4464x2160 입력이면 가로 1 단위가 약 4.5px라 bbox 검증(IoU 확인)에는 충분하지만 아주 작은 객체 localization에는 한계.
- 비정사각형/리사이즈 조건에서 bbox aspect ratio가 틀어지는 실측 이슈 보고(llama.cpp에서 1000x1000이 아닌 이미지에서 좌표 오류, 1.25x 리사이즈 필요 사례). vLLM에서는 전처리 파이프라인이 다르므로 동일 재현은 아니지만, "리사이즈 정책과 좌표 역변환을 반드시 검증해야 한다"는 교훈.
- GroundingME(2025-12): 최강 Qwen3-VL-235B도 Acc@0.5 45.1%에 그침. 고해상(8K) 이미지의 소형 객체(면적비 중앙값 1%)와 occlusion에서 실패, IoU 0.75/0.9 기준에서는 성능 급락(박스가 느슨함). 그리고 "존재하지 않는 대상" rejection에서 대부분 모델이 0%, 즉 없는 객체도 박스를 그려내는 hallucination 경향.
- 출처: [llama.cpp issue #16880](https://github.com/ggml-org/llama.cpp/issues/16880), [QwenLM/Qwen3-VL issue #1486](https://github.com/QwenLM/Qwen3-VL/issues/1486), [GroundingME](https://arxiv.org/html/2512.17495)

**커뮤니티/프로덕션 리포트 (anecdotal)**
- HF 블로그(bbox RL 학습기): 소형 Qwen VL 모델은 out-of-box로 개수 오판(과소/과대)과 대상 영역 놓침이 잦아 RL 튜닝이 필요했다고 보고. 정량 before/after는 없음.
- Qwen2.5-VL로 주행 데이터 auto-annotation QA pair 775K 생성 사례(ReCogDrive)는 있으나, 이는 캡션/QA 생성이지 exhaustive bbox mining이 아님.
- 출처: [bbox RL env 블로그](https://huggingface.co/blog/UlrickBL/bbox-rl-env), [ReCogDrive](https://arxiv.org/pdf/2506.08052), [Roboflow open-vocab detection 노트북](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/open-vocabulary-object-detection-with-qwen3-vl.ipynb)

### Q1 결론

Qwen 계열은 "이미지에 있는 X를 전부 박스로" 태스크에서 VLM 중 최상급이지만, (1) 밀집 장면 recall 붕괴, (2) 느슨한 박스(IoU 0.75+에서 급락), (3) 없는 객체 hallucination, (4) 고해상도 소형 객체 미검출이 문서화된 약점. 미싱 라벨 마이닝을 VLM 단독으로 돌리면 안 되고, 타일링(crop 단위 질의) + 기존 라벨 제외 프롬프트 + open-vocab detector(GroundingDINO/OWLv2류) 후보와 교차 검증하는 파이프라인이 필요.

## Q2. GLM-4.6V의 동일 태스크 성능

- 공식 문서상 GLM-4.6V는 "지정 카테고리의 모든 해당 박스"를 `[[xmin,ymin,xmax,ymax]]` 형식으로 출력하는 multi-object detection을 명시 지원. RefCOCO 포함 20여 개 벤치마크에서 동급 오픈소스 SOTA 주장 (강도: 중간, 벤더 자료).
- GLM-4.5V 계보는 원래 grounding 정밀도가 강점: Human-MME에서 bounding box 관련 문항 1위, 정밀 localization 우위 보고 (중간).
- 제3자 벤치마크(수중 도메인 UWBench 계열)에서 "GLM-4.6V가 Qwen3-VL보다 F1/IoU는 높고 Hit@1은 낮다"는 결과, 즉 박스는 더 타이트하지만 첫 후보 적중률은 낮은 경향 (약함, 단일 도메인).
- 밀집 counting 데이터 없음: HoloCount에 GLM 계열 미평가. exhaustive recall에 대한 직접 증거는 Qwen보다 더 부족함 (증거 공백).
- 출처: [Z.ai GLM-4.6V 문서](https://docs.z.ai/guides/vlm/glm-4.6v), [GLM-V GitHub](https://github.com/zai-org/GLM-V), [GLM-4.5V 논문](https://arxiv.org/pdf/2507.01006), [Human-MME](https://arxiv.org/pdf/2509.26165), [VentureBeat](https://venturebeat.com/ai/z-ai-debuts-open-source-glm-4-6v-a-native-tool-calling-vision-model-for)

## Q3. 2026-07-22 ~ 07-27 신규 릴리스 체크

**결론: shortlist를 바꿀 릴리스 없음 (강도: 중간, 복수 트래커 교차 확인)**
- 해당 주간 확인된 릴리스: Claude Opus 5 (7/24, API), Gemini 3.6 Flash / 3.5 Flash-Lite (7/21, API), Grok STT 1.0 및 Ant Ling-3.0-flash (7/23), Qwen-Audio-3.0-TTS (7/20, 오디오). 오픈웨이트 VLM 신규 없음. Kimi K3는 7/16.
- 참고할 최근 주변 동향:
  - Unsloth가 Qwen3 계열 NVFP4 W4A4 quant 공개(7월 초순), NVIDIA 공식 W4A16 quant보다 빠름. 5090 서빙에 직접 유효.
  - GLM-5V-Turbo(4/1)는 API 상품으로 보이며 zai-org HF에 오픈웨이트 GLM-5V는 없음. 오픈웨이트 최신은 GLM-5.2(744B, MIT)로 8x5090에 탑재 불가. 즉 Zhipu 오픈 VLM 최신은 여전히 GLM-4.6V.
- 출처: [aireleasetracker](https://aireleasetracker.com/latest), [ThursdAI 7월 릴리스](https://thursdai.news/releases/2026-07), [LLM Daily 7/11](https://buttondown.com/agent-k/archive/llm-daily-july-11-2026/), [zai-org HF](https://huggingface.co/zai-org/GLM-5.2), [GLM-5V-Turbo 해설](https://wavespeed.ai/blog/posts/glm-5v-turbo-developers-2026/)

## Q4. Agentic 축: tool calling / structured output 신뢰성

**Qwen3.6 계열 (강도: 강함, 공식 recipe + 실측 다수)**
- vLLM 공식 recipe 존재: `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`(또는 qwen3_xml) + `--enable-auto-tool-choice`. OpenAI-compatible tool calling이 1st-class로 지원되고 5090 NVFP4 실측 사례 다수(단일 5090에서 105-160 tok/s).
- 에이전트 성능: Qwen3.6-35B-A3B는 SWE-bench Verified 73.4, Terminal-Bench 2.0 51.5. Qwen3.6-27B dense는 SWE-bench 77.2, Terminal-Bench 59.3에 AndroidWorld 70.3(스크린샷 기반 UI 에이전트 검증). Qwen3.5-122B-A10B는 SWE-bench 72.4, function calling 공식 지원.
- 출처: [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B), [Qwen3.6-35B-A3B 카드](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [5090 실측](https://patrickgawron.com/articles/qwen36-35b-nvfp4-vllm-on-rtx5090/), [Qwen3.6-27B](https://www.marktechpost.com/2026/04/22/alibaba-qwen-team-releases-qwen3-6-27b-a-dense-open-weight-model-outperforming-397b-moe-on-agentic-coding-benchmarks/), [Qwen3.5-122B NIM 카드](https://build.nvidia.com/qwen/qwen3-5-122b-a10b/modelcard)

**GLM-4.6V (강도: 중간, 설계 강점 + 서빙 마찰 실증)**
- 모델 자체는 "native multimodal tool use"가 차별점: 이미지/스크린샷을 tool 인자와 tool 결과로 직접 주고받는 설계이며 MCP 확장, RL로 tool 행동 통합. image+text agent라는 최종 목표에 개념적으로 가장 부합.
- 그러나 vLLM 서빙 신뢰성 이슈가 반복 보고됨: transformers 5.x + glm45 tool parser 비호환으로 "vision과 tool calling 동시 사용 불가" 이슈(#31485, 12월 말 제기, PR #31622 연계, 완전 해소 여부 불명확), reasoning parser가 reasoning_content/content를 잘못 분리하는 문제, think 태그 누락 문제. vLLM이 Unified Parser(RFC #32713)로 정리 중인 과도기.
- BFCL 계열 직접 수치는 GLM-4.6V로는 미공개. 텍스트 계열 GLM-4.5가 BFCL v3 76.7-77.8%로 Qwen3-32B(75.7%)와 백중. 모델 능력 차이보다 서빙 스택 성숙도 차이가 실질 변수.
- 출처: [vLLM issue #31485](https://github.com/vllm-project/vllm/issues/31485), [vLLM issue #30865](https://github.com/vllm-project/vllm/issues/30865), [RFC #32713](https://github.com/vllm-project/vllm/issues/32713), [SiliconFlow 해설](https://medium.com/@SiliconFlowAI/glm-4-6v-now-on-siliconflow-native-multimodal-tool-use-meets-sota-visual-intelligence-b638150246fc), [BFCL 비교](https://llm-stats.com/benchmarks/bfcl-v3)
- 하드웨어 참고: GLM-4.6V는 106B-A12B라 4bit(AWQ 존재)로도 무게만 약 55-60GB, P2P 없는 5090에서 TP 2-4장 필수. 커뮤니티 AWQ INT4: [cyankiwi/GLM-4.6V-AWQ-4bit](https://huggingface.co/cyankiwi/GLM-4.6V-AWQ-4bit)

## Q5. 3D 렌더/LiDAR BEV 이미지를 VLM에 넣는 것

- **"2D-Cheating" 논문 (강도: 강함, 목적에 유리한 방향)**: 점군을 렌더링한 이미지만 줘도 VLM이 전문 3D LLM을 오브젝트 수준 태스크에서 능가(3D MM-Vet 58.1 vs 3D SOTA 43.2). 단 장면(scene) 수준 공간 추론은 단일뷰/멀티뷰 렌더 모두 3D 모델 대비 열세. 결론: "이 렌더에 라벨 안 붙은 객체가 보이는가" 같은 표면 정보 태스크는 렌더 이미지로 충분히 작동하는 영역.
- **BEV 관행 (중간)**: 점군을 지면 투영한 BEV + 동서남북 멀티뷰 렌더를 병용하는 것이 문헌상 표준 전처리.
- **점군 차량 분류 (약함)**: CVPR DriveX 워크숍 논문에서 registration으로 dense하게 만든 렌더 + morphology 전처리 + few-shot으로 "encouraging" 성능. 정량 수치 빈약.
- **함정 정리**: (1) 희소 점군 원본 렌더는 인식률 낮음, 누적/densification 필요, (2) intensity/높이 컬러맵 선택이 결과에 영향, (3) 거리/크기 등 기하 추정은 신뢰 불가, 존재 여부 판단까지만, (4) 3D bbox 좌표를 VLM이 직접 출력하는 것은 근거 없음. 렌더 위에 기존 3D 라벨을 2D로 투영해 그려 넣고 "박스 없는 객체 클러스터 존재 여부"를 묻는 verifier 구도가 증거에 부합.
- 출처: [Revisiting 3D LLM Benchmarks](https://arxiv.org/abs/2502.08503), [VLM point cloud 차량 분류](https://arxiv.org/abs/2504.08154), [VLM 3D detection 리뷰](https://arxiv.org/html/2504.18738v1)

## 실행 요약

1. 어떤 후보도 밀집 장면 exhaustive detection을 단독 수행 못함(고밀도 counting 0-20%, rejection 0%). 미싱 라벨 마이닝은 "타일링 + open-vocab detector 후보 + VLM verifier" 파이프라인 전제로 모델을 골라야 함.
2. 그 전제라면 결정 축은 grounding 원시 성능보다 (a) 서빙/tool calling 신뢰성, (b) 5090 8장(P2P 없음) 적합성이며, 둘 다 Qwen3.6 계열이 우위(공식 vLLM recipe, NVFP4 5090 실측, qwen3_coder parser 안정).
3. GLM-4.6V는 native multimodal tool use와 타이트한 박스가 매력이지만 vLLM에서 vision+tool calling 동시 사용 파서 이슈 이력, 106B로 GPU 2-4장 점유, 밀집 recall 증거 부재로 공유 단일 모델로는 리스크가 더 큼.
4. 단일 공유 모델로는 Qwen3.6-35B-A3B(NVFP4)가 기본 선택: GPU 1-2장으로 서빙 여유, 두 유즈케이스 모두 verifier 역할 수행 가능. 정밀도가 부족하면 Qwen3.5-122B-A10B(counting 계열 최강 가문, function calling 지원)로 승격 검토.
5. 7/22-27 사이 shortlist를 바꿀 신규 릴리스 없음. 단 4464x2160 입력의 리사이즈 정책과 0-1000 좌표 역변환 검증, 고밀도 recall 실측(자체 골든셋)은 도입 전 필수 PoC 항목.
