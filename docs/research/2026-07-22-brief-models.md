# Open-Weight VLMs for Local Deployment on 4x RTX 5090 — ADAS Inspection Use Case (Research Brief, July 2026)

## 0. Hardware Ground Rules (4x RTX 5090, 128GB total, no NVLink)

- RTX 5090 = 32GB GDDR7, Blackwell SM120. Realistic usable VRAM ≈ 30–30.5GB/GPU (~122GB total) after CUDA/driver overhead. vLLM runs on 5090 with CUDA 12.8 + PyTorch ≥2.6/2.9 builds; FP8 tensor cores are native, and FP4/NVFP4 is Blackwell-native (exposed via TensorRT-LLM and increasingly vLLM `nvfp4` checkpoints) (vLLM issue #13306; vLLM forums "vLLM on RTX5090", 2025-26; RunPod RTX 5090 guide, 2026).
- No NVLink → tensor-parallel over PCIe. This penalizes dense TP=4 throughput somewhat but is a *non-issue for low-active-parameter MoE* models (A3B–A12B), which are the sweet spot here.
- Your input: ~4464×2160 screenshots. At Qwen3-VL-style native-resolution encoding (16px patch, 2×2 merge → 32px/token), one full-res image ≈ **~9,500 vision tokens**. Budget KV cache accordingly (plan 16–32K context per request).

---

## 1. Landscape Snapshot (what changed by mid-2026)

- **Qwen3-VL** (Sep–Oct 2025): dense 2B/4B/8B/32B + MoE 30B-A3B / 235B-A22B, Instruct + Thinking, official FP8 releases, Apache 2.0, 256K context, OCR in 32–39 languages, 2D/3D grounding. vLLM ≥0.11, SGLang supported (QwenLM/Qwen3-VL GitHub; Qwen3-VL Technical Report arXiv 2511.21631).
- **Qwen3.5** (Feb 16, 2026): **natively multimodal (early-fusion) across the whole family** — no separate "-VL" line. Dense 0.8B/2B/4B/9B/27B + MoE 35B-A3B / **122B-A10B** / 397B-A17B. All Apache 2.0. Outperforms Qwen3-VL on vision benchmarks (flagship: MMMU 85.0 vs 80.6; OmniDocBench v1.5 90.8). vLLM/SGLang/KTransformers supported (Codersera Qwen 3.5 guide, 2026; HF Qwen/Qwen3.5-27B & Qwen3.5-122B-A10B cards; morphllm.com Qwen 3.5 page, 2026). **Qwen3.6** (Apr 2026, 27B dense + 35B-A3B, also multimodal, Apache 2.0); **Qwen3.7 Max** (May 2026) is API-only/closed.
- **GLM-4.6V** (Z.ai, Dec 8-9, 2025): 106–108B total / ~12B active MoE + **GLM-4.6V-Flash 9B**. MIT license, 128K context, native multimodal function calling (images as tool parameters), precise grounding with normalized bbox output. Official FP8 checkpoint; vLLM ≥0.12, SGLang ≥0.5.6 (MarkTechPost 2025-12-09; VentureBeat Dec 2025; HF zai-org/GLM-4.6V, GLM-4.6V-FP8). **GLM-5V-Turbo** (Apr 2026) is 744B — API-tier, not practical locally (WinBuzzer 2026-04-02).
- **InternVL3.5** (Aug 2025, still OpenGVLab's flagship VLM line as of mid-2026): dense 1–38B + MoE 20B-A4B/30B-A3B/241B-A28B; Qwen3/GPT-OSS backbones; 448px tiles, up to 36 (train)/128 (test) tiles; Flash variants with ~50% visual-token compression. InternVL-U (Mar 2026) is a 4B unified gen+understanding model, not an inspection workhorse (internvl.github.io blog 2025-08-26; OpenGVLab GitHub).
- **Gemma 4** (Google, Apr 2, 2026): E2B/E4B/26B-MoE(A3.8B)/31B-dense, all multimodal, **Apache 2.0 (first for Gemma)**, aspect-ratio-preserving vision encoder, bbox output as JSON `[y1,x1,y2,x2]`, but **image token budget capped at 1120 tokens/image** — a real limitation for fine 4.4K-wide overlay text unless you tile manually (Datature blog 2026; Gemma 4 Technical Report arXiv 2607.02770; explainx.ai July 2026 update).
- **Llama 4 Scout/Maverick** (Apr 2025): 109B-A17B / 400B-A17B early-fusion multimodal. OCR/doc understanding widely judged mediocre vs Chinese OSS models; Llama-license (EU vision carve-out, branding requirements). Behemoth never shipped. Not competitive for this task (Codersera Llama 4 guide 2026; InsiderLLM 2026).
- **MiniCPM-V**: 4.5 (8B, Sep 2025, OCRBench 89.0, DocVQA 94.7) still the "big" one; **MiniCPM-V 4.6 (May 2026) is a 1.3B phone-class model** (SigLIP2-400M + Qwen3.5-0.8B) — edge only (HF openbmb/MiniCPM-V-4.6; NYU Shanghai RITS 2026).
- **Molmo 2** (Ai2, Dec 17, 2025): 4B / 8B (Qwen3-based) / 7B-O. Best-in-class **pointing, tracking, pixel-level grounding**; 8B beats old Molmo-72B and beats Gemini 3 Pro on some grounding benchmarks. Apache 2.0 weights but **training data includes non-commercial-research datasets** — legal review advised; vLLM support still landing via issue #31331 (allenai.org/blog/molmo2; HPCwire Dec 2025; vllm-project issue #31331).
- **Kimi**: Kimi-VL (16B-A3B, 2025) obsolete; K2.5 (Jan 2026) / K2.6 (Apr 2026) are natively multimodal but **1T-A32B** — ≥500GB at 4-bit, impossible on 128GB (DeepInfra K2.6 overview; kimi.com).
- **DeepSeek**: no new general VLM; **DeepSeek-OCR-2** (Jan 27, 2026, 3B, DeepEncoder V2 "causal flow", OmniDocBench v1.5 91.09) is a document-OCR specialist, useful as a sidecar not a judge (ComfyUI Wiki 2026-01-27; TechNode 2026-01-28).
- **Ovis2.6-30B-A3B** (AIDC-AI, ~May 2026): MoE upgrade of Ovis2.5; native-resolution NaViT (no tiling loss), grounding via normalized bbox/points, thinking budget, mid-CoT crop/rotate tool use; strong self-reported OCR/chart numbers near larger Qwen3-VL variants; vLLM examples provided (HF AIDC-AI/Ovis2.6-30B-A3B; aiHola 2026).
- **Pixtral: dead line.** Mistral folded vision into **Mistral Small 4** (Mar 16, 2026, 119B-A6B MoE, open weights); Pixtral 12B/Large deprecated (Mistral changelog; Serenities AI Mistral 2026 guide).
- **OCR specialists (sidecar candidates)**: PaddleOCR-VL-1.5/1.6 (0.9B, OmniDocBench ~96 self-reported; **best Korean MDPBench score 86.0**), GLM-OCR (0.9B, OmniDocBench 94.62 but Korean MDPBench only 61.2), Surya 2, Nemotron-OCR-v1 (Korean blog lh99tw.github.io 2026-04-17; codesota.com OCR 2026; NVIDIA dev blog).
- **Korean-specialist VLM**: **VARCO-VISION-2.0-14B** (NCSOFT, open-weight, KO/EN bilingual, **layout-aware Korean OCR with text bounding boxes**, +1.7B on-device version) — the only serious open Korean-centric VLM with grounding (HF NCSOFT/VARCO-VISION-2.0-14B; arXiv 2509.10105). Korean VQA benchmarks to test against: KRETA (arXiv 2508.19944), KOCRBench.

---

## 2. Candidate Table (serious contenders only)

| Model | Params (act.) | License | BF16 / FP8 / 4-bit weights | Fits 4x5090? | Native res / tiling | OCR (Korean) | Grounding | vLLM/SGLang | Quant ckpts |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-122B-A10B** | 122B (10B) | Apache 2.0 | 244 / ~122 / **~65GB** | FP8 ✗ (no KV room), **AWQ/INT4 ✓✓** | native-res early fusion, 262K ctx | OCRBench **92.1**, OmniDocBench 89.8; KO among supported langs (untested KRETA) | yes (inherits Qwen grounding, rel. coords) | vLLM/SGLang (latest main) | community GGUF/AWQ growing; no official FP8 yet |
| **GLM-4.6V** | 106–108B (~12B) | **MIT** | ~212 / **~106 (very tight)** / **~55GB ✓✓** | AWQ ✓✓, FP8 marginal | up to ~4K imgs, arbitrary AR, 128K ctx | strong zh/en doc OCR; **Korean not a focus** | **normalized bbox output, native function calling** | vLLM ≥0.12 / SGLang ≥0.5.6 | official FP8; cyankiwi AWQ-4bit |
| **Qwen3-VL-32B-Instruct/Thinking** | 33B dense | Apache 2.0 | **66 ✓ / 33 ✓ / ~18GB** | ✓✓ (BF16 TP=4 easy) | native res, patch16, 256K ctx | OCR 32–39 langs incl. **Korean** (>70% acc on 32/39 langs); OCR Arena top-15 | **2D/3D grounding cookbooks, box+point coords** | vLLM ≥0.11 / SGLang, mature | **official FP8**; QuantTrio+cyankiwi AWQ; RedHatAI **NVFP4** |
| **Qwen3.5-27B** | 27B dense | Apache 2.0 | 54 ✓ / 27 ✓ / ~14GB | ✓✓ | native multimodal, 262K ctx | OCRBench **89.4**, OmniDocBench 88.9, MMMU-Pro 75.0 | yes | vLLM/SGLang (recent builds) | community quants (213+ listed) |
| **InternVL3.5-38B** | 38.4B dense | Apache 2.0 | 77 ✓ / ~39 ✓ / ~20GB | ✓✓ | 448px tiles ×36–128 (4464×2160 ≈ 50 tiles) | good zh/en; Korean weaker than Qwen | decent, bbox capable | vLLM/LMDeploy/SGLang | community AWQ/GPTQ |
| **Ovis2.6-30B-A3B** | 30B (3B) | Apache 2.0 | 60 ✓ / 30 ✓ / ~16GB | ✓✓ (even 1–2 GPUs) | **NaViT native res, no tiling loss** | strong charts/docs (self-reported) | normalized bbox + points | vLLM examples official | community |
| **Gemma 4 31B** | 30.7B dense | **Apache 2.0** | 62 ✓ / 31 ✓ / ~16GB | ✓✓ | AR-preserving but **≤1120 tokens/img** | good DocVQA/ChartQA; multilingual OK | JSON bbox `[y1,x1,y2,x2]` | supported (FA4 update Jul 2026) | Ollama/community |
| **InternVL3.5-241B-A28B** | 241B (28B) | Apache 2.0 | 482 / 241 / **~121GB** | ✗ (no KV headroom) | tiles | top-tier | yes | yes | — |
| **Qwen3-VL-235B-A22B** | 235B (22B) | Apache 2.0 | 470 / 235 / **~117–132GB** | **✗** (INT4 weights alone ≈ usable VRAM; no room for 9.5K-token images + KV) | native res | best-in-class (CC-OCR SOTA) | best-in-class | yes | official FP8, community AWQ |
| **Qwen3.5-397B-A17B** | 397B (17B) | Apache 2.0 | — / — / ~200GB | ✗ | native | flagship scores above | yes | yes | — |
| Llama 4 Scout | 109B (17B) | Llama license (EU vision carve-out) | 218 / 109 / ~55GB ✓ | AWQ ✓ | early fusion | weak vs Qwen/GLM | mediocre | yes | yes | 
| Molmo 2-8B | 8B | Apache 2.0 (data caveat) | 16 ✓ | single GPU | good | fair (not KO-focused) | **best pointing/tracking** | vLLM WIP (#31331) | community NVFP4 (4B) |
| VARCO-VISION-2.0-14B | 14B | open-weight (check CC-BY-NC clauses on HF card) | 28 ✓ | single GPU | LLaVA-style | **best open Korean OCR+layout** | **text bbox output** | transformers; vLLM via llava arch | — |
| MiniCPM-V 4.5 (8B) | 8B | OpenBMB license (commercial OK w/ registration) | 16 ✓ | single GPU | LLaVA-UHD tiles | OCRBench 89.0, DocVQA 94.7 | limited | vLLM ✓ | GGUF/int4 |
| DeepSeek-OCR-2 (3B) / PaddleOCR-VL-1.5 (0.9B) | 3B / 0.9B | open | <8GB | trivially | doc-optimized | **PaddleOCR-VL: Korean MDPBench 86.0 (best)** | text-line boxes | vLLM ✓ (PaddleOCR-VL) | — |

VRAM notes: "4-bit" column = weights only; add ~2–6GB vision encoder + activations + KV (a 9.5K-token image at 32K ctx costs single-digit GB on GQA/MoE models, per-GPU when TP=4).

---

## 3. Fit-for-Purpose Analysis (your 4 requirements)

**(a) Fine overlay text/numbers at 4464×2160**: Requires true native/dynamic resolution with high token budgets → Qwen3-VL / Qwen3.5 (native-res, ~9.5K tokens/img), GLM-4.6V (≤4K images, arbitrary AR — 4464px slightly over; downscale to 3840 or split BEV panel), Ovis NaViT. Gemma 4's 1120-token/img cap is disqualifying at full res; InternVL's 50-tile requirement exceeds its 36-tile training regime (works but off-distribution).

**(b) Detection-correctness reasoning (missed/false/misaligned boxes, lane fits)**: Favors Thinking variants + grounding. Qwen3-VL/3.5 Thinking and GLM-4.6V (RL-trained multimodal reasoner, outputs its *own* bboxes in normalized coords — lets you cross-check the ADAS model's boxes against the VLM's independent localization). Molmo 2 is the best pure pointer/verifier ("point at every pedestrian") and is a strong *second-opinion* module.

**(c) Structured JSON verdicts**: All top candidates work with vLLM guided decoding (xgrammar) for schema-enforced JSON. GLM-4.6V adds native multimodal function calling (`--tool-call-parser glm45`); Qwen3-VL/3.5 have mature JSON-mode behavior and cookbooks.

**(d) Korean OCR**: Qwen3-VL/3.5 officially cover Korean in the 32/39-language OCR set (>70% acc tier); community reports (Korean dev forums, OCR Arena) say Qwen3-VL is the best *general* VLM for Korean but still trails dedicated OCR. GLM-4.6V and InternVL are zh/en-centric. If Korean sign text is verdict-critical, add **PaddleOCR-VL-1.5 (0.9B, Korean MDPBench 86.0)** as a sidecar OCR pass, or test **VARCO-VISION-2.0-14B** (lh99tw blog 2026-04-17; KRETA arXiv 2508.19944).

---

## 4. Top 3 Recommendations

### 1) Qwen3.5-122B-A10B (Instruct/Thinking), AWQ-4bit, vLLM TP=4 — primary judge
- Math: 122B × 0.5B/param ≈ **61–65GB weights** → ~16GB/GPU, leaving **~55–60GB pooled KV/activations** → comfortable for 9.5K-token images + 16–32K contexts at batch 4–8. Only 10B active params → PCIe TP=4 penalty is small; expect roughly 40–70 tok/s/stream class decode.
- Why: best open vision-reasoning per GB in this class (OCRBench 92.1 / OmniDocBench 89.8 / MMMU 83.9 — above Qwen3-VL-235B on several axes), Apache 2.0, 262K context, native-res vision, Korean in OCR language set, first-class vLLM/SGLang. FP8 (122GB) does NOT fit with realistic KV — use AWQ/INT4 (or NVFP4 when a checkpoint lands; Blackwell runs it natively).
- Risk: Feb 2026 release — quant ecosystem younger than Qwen3-VL's; validate AWQ vision quality on your screenshots vs the BF16 API before committing.

### 2) GLM-4.6V (106B-A12B), AWQ-4bit (cyankiwi) or FP8-with-care, vLLM ≥0.12 TP=4 — best grounding/agentic judge
- Math: AWQ-4bit ≈ **55–58GB** → ~65GB headroom (very comfortable). Official FP8 (~106GB) technically loads but leaves <4GB/GPU for KV + a 9.5K-token image — only viable single-request; use AWQ.
- Why: MIT license, native bbox grounding in normalized coordinates (directly comparable to your detector's boxes), native multimodal function calling → cleanest structured-verdict pipeline, 128K ctx, thinking mode. 
- Risk: Korean OCR is its weak axis → pair with PaddleOCR-VL sidecar; 4464px slightly exceeds its ~4K comfort zone (downscale or crop BEV panel separately).

### 3) Qwen3-VL-32B (Instruct + Thinking), BF16 TP=4 or FP8 TP=2 — the proven, boring choice
- Math: BF16 66GB → ~17GB/GPU + huge KV headroom; or run **two independent FP8 replicas (33GB each on 2 GPUs)** to double throughput for batch screenshot triage.
- Why: most battle-tested vLLM/SGLang path of the three; official FP8, RedHatAI NVFP4, QuantTrio/cyankiwi AWQ all exist; 2D grounding cookbooks; 32-language OCR incl. Korean; Apache 2.0. Slightly weaker reasoning than #1/#2 but far more operational maturity and the best ecosystem for structured output + fine-tuning (Unsloth etc.) if you later want to specialize it on ADAS overlay screenshots.

**Explicitly ruled out on this box**: Qwen3-VL-235B-A22B and InternVL3.5-241B (4-bit weights ≈ all 122GB usable VRAM, zero KV room for 9.5K-token images), Qwen3.5-397B, Kimi K2.5/K2.6 (1T), GLM-5V-Turbo (744B), Llama 4 Maverick. Llama 4 Scout fits but underperforms on OCR/doc tasks; Gemma 4 31B fits but the 1120-token/image cap breaks fine-text reading at your resolution.

## 5. Single-GPU Fallbacks (fast experimentation, one 5090 = 32GB)

- **Qwen3-VL-30B-A3B-Instruct-FP8 (~31GB)** — tight on one card (reduce `max_pixels`/ctx) — safer on 2 GPUs; extremely fast (A3B). Best quality/speed fallback.
- **Qwen3-VL-8B-Instruct-FP8 (~10GB)** / **Qwen3.5-9B (~18GB BF16)** — pipeline bring-up, prompt/schema iteration; same tokenizer+behavior family as the big Qwen judges → prompts transfer.
- **GLM-4.6V-Flash 9B** (official FP8/AWQ) — mirrors GLM-4.6V grounding + function-calling behavior for cheap.
- **Ovis2.6-30B-A3B** (~16GB at 4-bit) — native-res NaViT, thinking budget, bbox output.
- **Specialist add-ons worth wiring in regardless of judge choice**: PaddleOCR-VL-1.5 (0.9B) for Korean sign/building text extraction fed to the judge as text; Molmo 2-8B for independent point-based "count all pedestrians/vehicles" cross-checks (verify commercial-use posture of its training data first); VARCO-VISION-2.0-14B if Korean OCR-with-boxes becomes a hard requirement.

**Suggested architecture**: vLLM (TP=4) serving Qwen3.5-122B-A10B-AWQ with xgrammar-guided JSON schema; per-image preprocessing that (1) passes the full screenshot at native res and (2) optionally crops HUD-text and BEV-panel regions as extra images; PaddleOCR-VL sidecar text injected into the prompt for Korean signage; A/B GLM-4.6V-AWQ for bbox-verification prompts. Bench all three finalists on ~50 labeled screenshots (missed/FP/misaligned cases) before locking in — public benchmarks don't cover overlay-verification, and KRETA/KOCRBench numbers for the Feb-2026 models are not yet published.

**Key sources**: QwenLM/Qwen3-VL GitHub + Qwen3-VL Tech Report (arXiv 2511.21631, Nov 2025); HF Qwen/Qwen3.5-27B & Qwen3.5-122B-A10B (Feb 2026); Codersera "Qwen 3.5+3.6+3.7 Max Guide" (2026); HF zai-org/GLM-4.6V + MarkTechPost (2025-12-09) + VentureBeat (Dec 2025); internvl.github.io InternVL3.5 blog (2025-08-26); Datature "Gemma 4 for CV Engineers" + Gemma 4 Tech Report (arXiv 2607.02770, Apr 2026); allenai.org/blog/molmo2 (2025-12-17); HF AIDC-AI/Ovis2.6-30B-A3B (May 2026); HF openbmb/MiniCPM-V-4.6 (May 2026); ComfyUI Wiki DeepSeek-OCR-2 (2026-01-27); Mistral changelog / Serenities AI (Mar 2026); lh99tw.github.io Korean OCR VLM comparison (2026-04-17); KRETA (arXiv 2508.19944); HF NCSOFT/VARCO-VISION-2.0-14B (2025); RedHatAI/Qwen3-VL-32B-Instruct-NVFP4, cyankiwi/GLM-4.6V-AWQ-4bit, QuantTrio/Qwen3-VL-32B-Instruct-AWQ (HF); vLLM RTX 5090 issues #13306/#14452 + vLLM forums (2025-26).
