# Research Brief: 라벨 누락 마이닝용 open-vocabulary detector 지형도 (2026-07-27)

배경: DST 용도(라벨링 누락 객체 탐지, 2D + 3D 렌더, Bedrock 마이그레이션)에 대해
"VLM 단독 vs VLM + 전용 detector 사이드카"를 판정하기 위한 원본 브리프.
요건: 로컬 배포(RTX 5090, SM120), 상용 라이선스.

---

## Q1. 2026년 Open-Vocabulary 검출기 지형도

### API-only / 상용 불가 그룹 (로컬 배포 탈락)

- **Grounding DINO 1.5/1.6 Pro·Edge (IDEA)**: 1.6 Pro가 zero-shot COCO 55.4 AP, LVIS-minival 57.7 AP로 강력하지만, **가중치 미공개, DeepDataSpace API 전용**. GitHub의 Apache 2.0은 API 래퍼 코드에만 적용. "Grounding DINO 2.0"은 2026년 7월 현재 존재하지 않음. [GitHub](https://github.com/IDEA-Research/Grounding-DINO-1.5-API), [arXiv](https://arxiv.org/abs/2405.10300)
- **DINO-X (IDEA)**: LVIS-minival 59.8 AP, rare class에서 GD 1.6 Pro 대비 +5.8 AP로 현존 최고 수준이지만 **API 전용**. DINO-X Edge 오픈소스 요청 이슈는 2025년 1월 이후 무응답. [DINO-X-API](https://github.com/IDEA-Research/DINO-X-API), [Issue #23](https://github.com/IDEA-Research/DINO-X-API/issues/23), [arXiv](https://arxiv.org/html/2411.14347v3)
- **T-Rex2**: 가중치가 HuggingFace에 있으나 **IDEA License 1.0, 비상업 연구 전용** (상용은 IDEA에 별도 라이선스 요청, 중국법 준거). T-Rex3는 없고, 후속은 Rex-Omni. [LICENSE](https://github.com/IDEA-Research/T-Rex/blob/trex2/LICENSE)
- **Rex-Omni (CVPR 2026, IDEA)**: 3B MLLM으로 검출을 next-point prediction으로 재정의, zero-shot SOTA 주장. 그러나 **IDEA License 1.0 + Qwen Research License로 이중 비상업 제약**. [GitHub](https://github.com/IDEA-Research/Rex-Omni)

결론: IDEA 계열 최상위 모델은 전부 로컬 상용 배포 불가. 이 요건에서는 후보 제외.

### 로컬 배포 + 상용 가능 그룹

- **LLMDet (CVPR 2025 Highlight)**: MM-Grounding-DINO를 LLM caption supervision으로 재학습한 모델. **Apache 2.0, 로컬 배포 가능한 open-vocab 검출기 중 최고 정확도**. LVIS-minival zero-shot: Swin-T 44.7 / Swin-B 48.3 / **Swin-L 51.1 AP (rare 45.1)**. API-only인 GD 1.5 Pro와 대등한 수준. transformers 4.55+에 공식 통합. [GitHub](https://github.com/iSEE-Laboratory/LLMDet), [arXiv](https://arxiv.org/abs/2501.18954)
- **SAM 3 / SAM 3.1 (Meta, 2025-11 / 2026-03)**: "concept prompt"(명사구 텍스트 + 이미지 exemplar)로 **해당 개념의 모든 인스턴스를 한 번에 검출·분할**. Zero-shot LVIS 48.8 mask AP (이전 최고 38.5 대비 +27%). 848M 파라미터, H200에서 100+ 객체 이미지 1장 30ms, 24GB 소비자 GPU에서 구동 확인. 체크포인트 공개, **커스텀 SAM License: 상용 허용 (군사/ITAR 등 용도 제한만 존재)**. SAM 3.1은 object multiplexing으로 비디오 처리량 2배. [Meta](https://ai.meta.com/blog/segment-anything-model-3/), [GitHub](https://github.com/facebookresearch/sam3), [arXiv](https://arxiv.org/html/2511.16719v1), [라이선스 분석](https://ai0w.com/en/sam3/)
- **MM-Grounding-DINO (OpenMMLab)**: Apache 2.0, LVIS-minival 41.4 AP 수준. transformers에 포팅되어 mmcv 없이 사용 가능. LLMDet의 베이스라인이므로 지금은 LLMDet이 상위 호환. [HF Docs](https://huggingface.co/docs/transformers/main/model_doc/mm-grounding-dino)
- **Grounding DINO 원본 (ECCV 2024)**: Apache 2.0, zero-shot COCO 52.5 AP. LVIS에서는 MM-GDINO/LLMDet에 밀림. [GitHub](https://github.com/IDEA-Research/GroundingDINO)
- **OWLv2 (Google)**: Apache 2.0, HF transformers. LVIS **rare class 44.6 APr**로 희귀 클래스 recall이 강점, 대신 ViT-L/14라 느림. 고recall 후보 마이닝 보조용으로 여전히 유효. [HF Docs](https://huggingface.co/docs/transformers/en/model_doc/owlv2), [arXiv](https://arxiv.org/html/2306.09683v3)
- **YOLOE / YOLOE26 (ICCV 2025, THU-MIG)**: 텍스트/비주얼/프롬프트-프리 3모드. YOLOE26-L이 **LVIS 36.8 AP, T4에서 161 FPS**. **AGPL-3.0** (내부 QA 도구로 자사 사용이면 통상 문제 없으나, 사내 법무 확인 권장. Ultralytics 상용 라이선스 구매 옵션 존재). [GitHub](https://github.com/THU-MIG/yoloe), [Ultralytics Docs](https://docs.ultralytics.com/models/yoloe)
- **YOLO-World (CVPR 2024)**: LVIS 35.4 AP @ 52 FPS(V100), GPL-3.0. YOLOE에 사실상 대체됨. [GitHub](https://github.com/ailab-cvc/yolo-world)
- **Florence-2 (Microsoft)**: MIT, 230M/770M로 가볍지만 검출 정확도는 위 모델들보다 낮음. 보조 캡셔닝/그라운딩용. Florence-3 공개 소식 없음. [Roboflow](https://blog.roboflow.com/florence-2/)
- **APE (CVPR 2024)**: 공개 모델이나 2026년 기준 성능·생태계 모두 뒤처짐. [GitHub](https://github.com/shenyunhang/APE)
- 신규 동향: OV-DEIM 등 실시간 DETR형 open-vocab 연구가 2026년에 나오고 있으나 아직 초기 단계. [arXiv](https://arxiv.org/pdf/2603.07022)

드라이빙 도메인 참고: 차량/보행자/신호등/표지판은 COCO·Objects365·GoldG의 head class라 위 모델 전부 사전학습 어휘에 충분히 포함됨. BDD100K/nuImages 전용 zero-shot 공식 벤치마크는 발견하지 못함 (항공영상 전이 연구에서 도메인 갭이 확인되므로 소규모 자체 검증셋 권장). [항공 전이 평가](https://arxiv.org/pdf/2601.22164)

## Q2. Missing-Annotation Mining의 업계 표준 (2026)

핵심 결론: **"강한 검출기(또는 앙상블)의 고신뢰 예측 중 GT와 매칭 안 되는 것 = 누락 후보" 방식이 표준**이고, 최근에는 여기에 **VLM을 후보 crop 검증(judge)으로 얹는 2단 구조**가 주류.

- **FiftyOne (Voxel51) `compute_mistakenness()`**: 고신뢰 예측이 GT에 매칭되지 않으면 `possible_missing=True` 플래그. 오픈소스 누락 라벨 마이닝의 사실상 표준 워크플로. [튜토리얼](https://docs.voxel51.com/tutorials/detection_mistakes.html), [가이드](https://docs.voxel51.com/getting_started/object_detection/03_finding_mistakes.html)
- **Cleanlab ObjectLab**: 검출기 예측 vs GT 비교로 라벨 품질 점수화. [Lightly 정리](https://www.lightly.ai/blog/best-data-curation-tools)
- **학술 최신 (2025-2026)**: loss 관찰 기반 라벨 오류 검출([arXiv 2303.06999](https://arxiv.org/pdf/2303.06999)), KITTI 보행자에서 원본 GT의 18%가 누락/부정확 라벨임을 밝힌 검출-교정 벤치마크 "Rechecked" (**최고 기법도 라벨 오류의 최대 66%를 놓침, 즉 인간 검수 루프 필수**) ([arXiv 2508.06556](https://arxiv.org/pdf/2508.06556)), 2016-2025 문헌을 정리한 어노테이션 오류 서베이([Springer 2026](https://link.springer.com/article/10.1007/s10462-026-11502-z)), teacher 모델 + 인간 검증으로 MUSES 라벨을 업그레이드해 mAP 0.13에서 0.56-0.62로 올린 사례([Springer](https://link.springer.com/article/10.1007/s44291-026-00244-5))
- **VLM 활용 동향**: Roboflow는 SAM 3 + Gemini + GPT를 묶어 2-of-3 투표로 자동 라벨링하는 멀티모델 워크플로를 공개([Roboflow](https://blog.roboflow.com/multi-model-auto-labeling-for-segmentation/)). 미지 객체 auto-labeling 실험에서는 **텍스트 프롬프트 기반 open-vocab 검출만이 unknown 객체에 유효**했다는 보고([Medium 2026-04](https://medium.com/@kynkynkyn/auto-labeling-unknown-objects-a-vision-model-pipeline-for-what-models-cant-see-390b2cc1ea39)). 반면 MLLM 단독 검출은 tiny/소형 객체에서 정확도가 급락한다는 것이 반복 확인됨(Qwen3-VL 계열 포함) ([Qwen3-VL 리포트](https://arxiv.org/pdf/2511.21631), [DetPO](https://arxiv.org/pdf/2603.23455))
- 상용 플랫폼(Encord, SuperAnnotate, Labelbox)은 자동 QA + 컨센서스 검사를 내장하나 SaaS 종속. [Encord](https://encord.com/blog/best-image-annotation-tools/)

## Q3. RTX 5090 (SM120) 호환성, 상위 후보 기준

- **공통**: sm_120은 CUDA 12.8+ 필수, PyTorch는 2.7.0부터 cu128 휠로 네이티브 지원(현재 2.11 권장). [PyTorch Issue](https://github.com/pytorch/pytorch/issues/159207), [SaladCloud 가이드](https://docs.salad.com/container-engine/tutorials/machine-learning/pytorch-rtx5090)
- **LLMDet / MM-GDINO / OWLv2 / SAM 3 (HF transformers 경로)**: 순수 PyTorch 구현이라 5090에서 그대로 동작. Deformable attention 커스텀 커널은 옵션이며 미설치 시 pure PyTorch로 폴백. **단, mmdetection/mmcv 네이티브 경로는 sm_120 빌드 실패 사례가 다수 보고되어 있으므로 반드시 HF 포팅 버전 사용 권장.** [HF mm-grounding-dino](https://huggingface.co/docs/transformers/main/model_doc/mm-grounding-dino), [RTX5090 MMDetection 실패기](https://note.com/unco3/n/n6b27d3fcc7bf), [PTX JIT 이슈](https://discuss.huggingface.co/t/ptx-jit-broken-on-rtx-5080-blackwell-sm-120-missing-libnvptxcompiler-so-in-cuda-12-8-12-9/161827)
- **YOLOE (Ultralytics)**: torch 2.7+cu128로 문제 없음, TensorRT 10.8+가 Blackwell 지원이라 TRT export 가능.
- **SAM 3**: 848M 파라미터 (fp16 약 2GB), RTX 4090 24GB에서도 이미지 추론 무난 보고. 5090 32GB에서는 배치 추론 여유. VLM과 GPU를 나눠 쓰더라도 사이드카 검출기(SAM 3 + LLMDet 합산 약 4-6GB)는 5090 1장에 동거 가능. [Spheron 배포 가이드](https://www.spheron.network/blog/deploy-sam-3-gpu-cloud/)

## Q4. 3D (포인트클라우드) 누락 라벨 마이닝

- **실무 표준은 여전히 "2D 검출 + 프러스텀 lifting"**: 카메라 이미지(또는 렌더 뷰)에서 open-vocab 2D 검출 실행, 기존 3D 박스를 이미지에 투영, 매칭 안 되는 2D 검출을 누락 후보로 플래그, 캘리브레이션으로 프러스텀을 잘라 포인트 존재 여부로 확증하는 파이프라인. [2DDATA](https://arxiv.org/pdf/2309.11755), [frustum 기반 자동 라벨](https://arxiv.org/pdf/2303.14893)
- **직접적인 open-vocab 3D 검출기는 아직 연구 단계**: OpenSight, Find n' Propagate(ECCV 2024, urban), ImOV3D, Zoo3D(zero-shot, 2025), HQ-OV3D 등이 있으나 실내 중심이거나 AP가 낮아 프로덕션 QA 도구로는 미성숙. [Zoo3D](https://arxiv.org/pdf/2511.20253), [Find n' Propagate](https://link.springer.com/chapter/10.1007/978-3-031-73661-2_8), [VLM 3D 검출 리뷰](https://arxiv.org/pdf/2504.18738)
- 3D 박스 자체의 계통 오차 교정 연구도 등장([arXiv 2601.14038](https://arxiv.org/pdf/2601.14038)). "3D 렌더 이미지" 입력이라면 2D 마이너를 그대로 적용하되, 렌더 이미지는 자연영상과 도메인이 달라 텍스트 프롬프트 성능 저하 가능성이 있으므로 **SAM 3의 이미지 exemplar 프롬프트나 YOLOE visual prompt로 보완 검증** 권장.

## 로컬 배포 가능 검출기 비교표 (상용 관점)

| 모델 | LVIS minival zero-shot | 속도 | VRAM(추론) | 라이선스 | 비고 |
|---|---|---|---|---|---|
| **LLMDet Swin-L** | **51.1 AP (rare 45.1)** | 수 FPS 급 (DETR형) | 약 3-5GB | **Apache 2.0** | 로컬 최고 정확도, transformers 통합 |
| **SAM 3 / 3.1** | 48.8 mask AP | H200 30ms/장, 5090 추정 10FPS+ | 약 3-6GB | SAM License (상용 OK, 군사 등 제한) | concept prompt로 해당 클래스 전부 탐색, exemplar 프롬프트 |
| MM-GDINO-L | 약 41.4 AP | 수 FPS 급 | 약 3-5GB | Apache 2.0 | LLMDet이 상위 호환 |
| OWLv2-L/14 | rare 44.6 APr | 느림 (약 1-2 img/s) | 약 4-6GB | Apache 2.0 | 희귀 클래스 recall 보조용 |
| YOLOE26-L | 36.8 AP | 161 FPS (T4) | 약 2-4GB | AGPL-3.0 | 대량 스캔용 초고속, 라이선스 법무 확인 |
| Florence-2-L | (LVIS 비SOTA) | 빠름 | 약 2GB | MIT | 보조 캡셔닝용 |
| 참고: DINO-X / GD1.6 Pro / T-Rex2 / Rex-Omni | 57-60 AP | - | - | API-only 또는 비상업 | **로컬 상용 배포 불가로 탈락** |

## Executive Summary

1. **VLM 단독은 부적합**: Qwen3.6급 MLLM도 소형·밀집 객체(원거리 보행자, 신호등)에서 recall이 급락하며, 누락 라벨 마이닝은 recall이 생명이므로 전용 검출기 사이드카가 필요하다.
2. **권장 구성은 VLM + 사이드카 2종**: LLMDet Swin-L(Apache 2.0, 로컬 최고 51.1 LVIS AP)을 주 마이너로, SAM 3(concept prompt, 상용 허용)를 exemplar 기반 보완 마이너로 사용.
3. **VLM은 검증 judge로 활용**: 검출기 고신뢰 예측 중 GT 미매칭 후보(FiftyOne `possible_missing` 패턴)를 crop해서 이미 서빙 중인 VLM에 참/거짓 판정시키면 precision을 확보. 2026년 업계 표준 2단 구조.
4. **5090 호환은 문제 없음**: 전 후보가 HF transformers 순수 PyTorch 경로로 PyTorch 2.7+ cu128에서 동작(mmcv 네이티브 빌드만 회피), 사이드카 합산 VRAM 약 6GB로 VLM과 동거 가능.
5. **탈락 사유 명확**: 절대 성능 1위인 DINO-X/GD 1.6 Pro/T-Rex2/Rex-Omni는 API 전용 또는 비상업 라이선스라 로컬 상용 요건에서 제외. 3D는 open-vocab 3D 검출기가 미성숙하므로 2D 마이닝 + 프러스텀 lifting이 표준.
