# Pinning an LLM Inference Server to a GPU Subset + Serving Topology — 8x RTX 5090, mid-2026 Best Practices

**Board-level facts that drive everything below:** RTX 5090 = 32 GB GDDR7, PCIe Gen5 x16, **no NVLink, no MIG, and P2P/DMA is disabled in the driver for GeForce** (hardware supports it; NVIDIA blocks it in software — confirmed for 3090/4090/5090 in NVIDIA/nccl issue #1637 and NVIDIA dev forums, 2025–2026). All inter-GPU traffic staged through host RAM. This makes TP expensive and DP attractive (see §2).

---

## 1. GPU selection mechanisms

### 1.1 Bare metal: `CUDA_VISIBLE_DEVICES`

```bash
nvidia-smi -L
# GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-8932f937-d72c-4106-c12f-20bd9faed9f6)
# ...
```

- **Index form:** `CUDA_VISIBLE_DEVICES=0,1,2,3`. Caveat: CUDA's default enumeration order is `CUDA_DEVICE_ORDER=FASTEST_FIRST`, which need **not** match `nvidia-smi`'s PCI-bus order. On a homogeneous 8x5090 box the two usually coincide, but the only way to *guarantee* CUDA index == nvidia-smi index is `export CUDA_DEVICE_ORDER=PCI_BUS_ID` (NVIDIA CUDA Programming Guide, env-vars appendix; NVIDIA "CUDA Pro Tip: Control GPU Visibility", updated through 2025). Always pair them:
  ```bash
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  ```
- **UUID form (safer):** `CUDA_VISIBLE_DEVICES=GPU-8932f937,GPU-...` — UUIDs from `nvidia-smi -L`; abbreviated prefixes allowed if unique. UUIDs survive reboots, driver updates, and the classic failure mode where **a GPU falling off the bus shifts every index after it** and your "GPU 4–7 experiments" suddenly land on the serving GPUs (NVIDIA docs; ETH D-ITET computing wiki).
- **vLLM-specific caveat (important):** vLLM historically crashed on UUID-form `CUDA_VISIBLE_DEVICES` (`int()` parse error) — vllm-project/vllm issue #32569, opened **Jan 19 2026**, closed via PR #45026, so only very recent vLLM handles UUIDs. Two safe patterns: (a) integer indices + `CUDA_DEVICE_ORDER=PCI_BUS_ID`, or (b) **do the pinning at container level with UUIDs and let vLLM see clean indices 0–3 inside** — this is the recommended approach.
- Inside any correctly pinned environment, the process sees the 4 GPUs re-numbered **0–3**, so `--tensor-parallel-size 4` needs no further device config.

### 1.2 Docker

Three mechanisms, newest last:

```bash
# (a) --gpus device list (nvidia-container-toolkit). Note the quoting — the comma
# must reach docker, hence '"..."':
docker run --gpus '"device=0,1,2,3"' vllm/vllm-openai:latest ...
# UUID form (preferred for production):
docker run --gpus '"device=GPU-uuid1,GPU-uuid2,GPU-uuid3,GPU-uuid4"' ...

# (b) Legacy env form with the NVIDIA runtime:
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=GPU-uuid1,GPU-uuid2,... ...

# (c) CDI — the modern (2026) standard:
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml   # regenerate after driver updates
docker run --device nvidia.com/gpu=0 --device nvidia.com/gpu=1 \
           --device nvidia.com/gpu=2 --device nvidia.com/gpu=3 ubuntu nvidia-smi -L
```

CDI ships in Docker Engine ≥25.0 and is **enabled by default since Docker 28.3.0** (older: enable the `cdi` feature in `/etc/docker/daemon.json`) (Docker docs "Container Device Interface", 2025–2026; NVIDIA Container Toolkit CDI docs). CDI device names can also be UUIDs (`nvidia.com/gpu=GPU-xxxx`). CDI removes the need for the nvidia runtime hook and is what containerd/podman/K8s converge on.

**docker compose:**
```yaml
services:
  vlm:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["GPU-uuid1", "GPU-uuid2", "GPU-uuid3", "GPU-uuid4"]  # or ["0","1","2","3"]
              capabilities: [gpu]
```
`device_ids` and `count` are mutually exclusive; `capabilities` is mandatory (Docker Compose Deploy Specification, 2026).

### 1.3 Kubernetes

- **Classic device plugin:** pod requests `resources.limits: {nvidia.com/gpu: 4}`. **You cannot pin specific physical GPUs per pod** — kubelet+plugin pick devices (the plugin does topology-preferred allocation, but selection is opaque) (NVIDIA/k8s-device-plugin README; NVIDIA forums re: selecting specific MIG/GPU instances). Workarounds people use: MIG resource names (N/A on GeForce), node labeling + one-GPU-flavor-per-node, or a fork like Deepomatic's shared-GPU plugin.
- **The 2026 answer is DRA (Dynamic Resource Allocation)**, GA since Kubernetes v1.34 (Aug 2025): with the NVIDIA DRA driver, a `ResourceClaim` uses CEL selectors over device attributes (index, UUID, memory), so you *can* deterministically target specific GPUs per pod. If this box later joins your KServe/Ray cluster, plan on DRA rather than device-plugin hacks.
- Note KServe/Ray both ultimately consume `nvidia.com/gpu` requests; Ray additionally respects `CUDA_VISIBLE_DEVICES` it sets per worker.

### 1.4 systemd unit (bare-metal production pattern)

```ini
# /etc/systemd/system/vllm-vlm.service
[Service]
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=CUDA_VISIBLE_DEVICES=GPU-uuid1,GPU-uuid2,GPU-uuid3,GPU-uuid4
Environment=VLLM_HOST_IP=127.0.0.1
ExecStart=/opt/vllm/bin/vllm serve Qwen/Qwen3-VL-32B-Instruct --tensor-parallel-size 4 --port 8000
Restart=on-failure
User=vllm
```
Optionally add cgroup-level hard isolation: `DeviceAllow=/dev/nvidia0 rw` … only for the four serving devices (plus `/dev/nvidiactl`, `/dev/nvidia-uvm`) with `DevicePolicy=closed` — env vars are advisory, cgroups are enforced.

### 1.5 Guardrails: compute mode + persistence

```bash
# Only one process may hold a CUDA context on the serving GPUs — a stray
# experiment fails fast instead of silently stealing VRAM:
sudo nvidia-smi -i 0,1,2,3 -c EXCLUSIVE_PROCESS      # 0=DEFAULT,1=EXCLUSIVE_PROCESS,2=PROHIBITED
# Persistence: keep driver initialized, avoid multi-second cold-start; daemon form preferred:
sudo systemctl enable --now nvidia-persistenced       # vs legacy nvidia-smi -pm 1
```
Compute mode **resets on reboot** — persist it with a oneshot systemd unit (Microway "nvidia-smi: Control Your GPUs"; NVIDIA driver-persistence docs). EXCLUSIVE_PROCESS is compatible with vLLM TP (one worker process per GPU, one context each). You could set `-c PROHIBITED` on nothing, and leave GPUs 4–7 at DEFAULT for the experimenters.

---

## 2. Topology for one VLM on 4x RTX 5090 (PCIe, no P2P)

### The three options

| | TP=4 (`--tensor-parallel-size 4`) | PP (`--pipeline-parallel-size`) | DP (4 replicas of a 1-GPU model) |
|---|---|---|---|
| Max model (BF16 weights+KV) | ~128 GB pool → 70B-class AWQ/FP8, 32B BF16 comfortably | same pool as TP, combinable (TP2×PP2) | ≤32 GB per replica → ~8–14B BF16 or ~30B AWQ/W4 |
| Interconnect cost | **2 all-reduces per layer, every token** — worst case on P2P-less PCIe (staged through sysmem) | Only activations at stage boundaries, point-to-point — cheapest comms | Zero inter-GPU traffic |
| Latency (single request) | Best *if* interconnect were fast; on this box the all-reduce tax can eat the gain | Worst per-token latency (pipeline bubbles); needs concurrency to fill micro-batches | Best per-request latency for models that fit on one GPU |
| Throughput | Sub-linear; PCIe-only boxes see ~1.4x per added card, not 2x (Will It Run AI vLLM multi-GPU guide, 2026; GIGAGPU DP-vs-TP; databasemart vLLM optimization guide) | Good at high concurrency | Near-linear 4x; "rarely matched by TP when DP is feasible" (GIGAGPU, 2026) |
| Failure blast radius | 1 GPU error kills the whole engine | same | 1 replica dies, 3 keep serving |

**Rule of thumb for this box:** use the *smallest* parallelism that fits model + KV cache. If the VLM fits in 32 GB → **DP=4**. If it needs 2 GPUs → 2x(TP=2) or TP=2 behind DP. Only use TP=4 because the model genuinely needs ~4x32 GB, and then benchmark TP=4 vs TP=2×PP=2 — on P2P-less PCIe, PP often wins throughput (vLLM docs "Parallelism and Scaling", 2026: TP recommended within NVLink nodes, PP across slow interconnects — the same logic applies intra-node here).

### vLLM commands

```bash
# (A) Single big engine, TP=4:
vllm serve Qwen/Qwen2.5-VL-72B-Instruct --tensor-parallel-size 4 \
     --dtype bfloat16 --gpu-memory-utilization 0.90 \
     --limit-mm-per-prompt '{"image": 4}' --port 8000
# vLLM auto-detects missing P2P and falls back (custom all-reduce disabled);
# if NCCL hangs on 5090s, try NCCL_P2P_DISABLE=1 and/or --disable-custom-all-reduce
# (vllm issue #14628 "Multi GPU inference using two RTX 5090s", 2025; vLLM forums 2025-2026).

# (B) TP2 x PP2 hybrid:
vllm serve <model> --tensor-parallel-size 2 --pipeline-parallel-size 2

# (C) DP=4 with vLLM's built-in load balancer (single endpoint, one command):
vllm serve Qwen/Qwen3-VL-8B-Instruct --data-parallel-size 4 --api-server-count 4 --port 8000

# (C') DP external-LB mode — 4 fully independent ranks, your LB on top:
CUDA_VISIBLE_DEVICES=0 vllm serve $M --data-parallel-size 4 --data-parallel-rank 0 --port 8001 &
CUDA_VISIBLE_DEVICES=1 vllm serve $M --data-parallel-size 4 --data-parallel-rank 1 --port 8002 &
# ... ranks 2,3 on ports 8003/8004
```
(vLLM docs "Data Parallel Deployment", stable, 2026: internal LB = single endpoint, queue-depth-aware; hybrid `--data-parallel-hybrid-lb`; external LB recommended at scale.) For a **dense** model, 4 totally independent `vllm serve` processes (no `--data-parallel-*` at all) behind a LB is equally valid and operationally simplest. `--data-parallel-size` coordination only *matters* for **MoE models**, where DP ranks must sync expert layers every step; add `--enable-expert-parallel` to shard experts (EP) across the DP group instead of TP-ing them (vLLM Data Parallel + Expert Parallel Deployment docs, 2026). Note: MoE + EP is exactly the traffic pattern that suffers most without P2P (smcleod.net driver-patch writeup, Feb 25 2026 — patched P2P gave 10–30%+ throughput on consumer cards, MoE benefiting most; patch requires forked kernel modules + IOMMU passthrough — not recommended for a team box).

**Load balancer options for plain-replica DP:** nginx `upstream { least_conn; server ...:8001; ... }` (vLLM docs "Using Nginx"); **LiteLLM proxy** with 4 entries of the same `model_name` and `routing_strategy: least-busy` — also gives you keys/quotas/logging (docs.litellm.ai routing, 2026); or the **vllm-router** / production-stack router (round-robin/session, 2026 writeups).

### SGLang equivalents

```bash
# TP=4:
python -m sglang.launch_server --model-path Qwen/Qwen3-VL-32B-Instruct --tp-size 4
# DP=4 via the Rust sglang-router (cache-aware LB, co-launch):
python -m sglang_router.launch_server --model-path <8B-VLM> --dp-size 4
# hybrid: --tp-size 2 --dp-size 2
```
SGLang supports TP/PP/EP/DP; the router does cache-aware routing (up to ~92% throughput gain on shared-prefix workloads vs naive RR — LMSYS SGLang v0.4 blog; SGLang router docs 2026). Note one SGLang GitHub datapoint (#20807, 2026): built-in DP ~87% efficiency vs ~100% for fully independent processes — same lesson as vLLM: independent replicas + external LB is the throughput ceiling.

---

## 3. Multi-tenancy on the remaining 4 GPUs

- **Baseline = process/container-level pinning:** every experiment launched with its own `CUDA_VISIBLE_DEVICES` (or `docker run --gpus '"device=..."'`). Combine with `EXCLUSIVE_PROCESS` on the *serving* GPUs so experiments physically cannot land there.
- **Memory-fraction flags when two things must share one GPU:** vLLM `--gpu-memory-utilization 0.45` (default 0.90 — vLLM **preallocates** that fraction for weights+KV, so a default-config vLLM instantly "fills" a GPU); SGLang `--mem-fraction-static`; PyTorch `torch.cuda.set_per_process_memory_fraction()`; TF `TF_FORCE_GPU_ALLOW_GROWTH=true`. These are cooperative, not enforced.
- **MPS (`nvidia-cuda-mps-control`):** real concurrent SM sharing, per-client memory limits (`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`) and SM caps (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`). Cons: **no fault isolation — one client crash can reset the GPU and kill all co-tenants**; no full memory protection (nebuly nos partitioning-modes comparison; Medium/K8s-MPS writeups, 2025). Fine for trusted small experiments, wrong for the production VLM. (Blackwell adds an MPS partitioning feature ("MLOPart"-style low-latency sub-devices) but still without MIG-grade isolation — NVIDIA 2026.)
- **MIG: not available** — GeForce cards have no MIG; that's A100/H100/B-series territory. So your isolation ladder on 5090s is: separate GPUs (best) → MPS with limits → cooperative memory fractions.
- **K8s note:** if this box joins the cluster, device-plugin **time-slicing** (`sharing.timeSlicing.replicas: N` ConfigMap in GPU Operator) advertises N schedulable replicas per GPU but provides **zero memory isolation** — oversubscription only, OOM roulette for LLM workloads (NVIDIA GPU Operator "Time-Slicing GPUs in Kubernetes" docs, 2026). Use it for notebooks, never for inference servers.

---

## 4. Serving stack comparison for a production VLM on this box

| | **vLLM** | **SGLang** | **TensorRT-LLM** | **Ollama / llama.cpp** |
|---|---|---|---|---|
| VLM support | Broadest coverage (Qwen2.5/3-VL, InternVL, Llama-3.2V, Gemma-3, Pixtral, GLM-4V…); recipes maintained per model (vLLM Recipes: Qwen3-VL, 2026) | Very good; Qwen3-VL day-0 with image+video; strong on Qwen/InternVL/Llava families (docs.sglang.io multimodal models, 2026) | Growing but narrower (Qwen2-VL, Llama-3.2 VLM FP8, Mistral-3.1, Phi-4-MM); PyTorch backend is now default which eased VLM onboarding (TRT-LLM release notes 2025–2026) | Ollama new multimodal engine: LLaVA, Llama3.2-V, Qwen-VL, Gemma-3 (Ollama blog "multimodal models"; mljourney 2026); llama.cpp server multimodal OK but quantized/CPU-offload oriented |
| Structured output | `response_format: {type:"json_schema"}` + `guided_json/regex/choice/grammar`; **xgrammar default backend**, near-zero overhead (vLLM structured-outputs docs; Red Hat Developer, Jun 2025) | JSON schema/regex/EBNF, xgrammar default; via OpenAI `response_format` (docs.sglang.io structured outputs) | Guided decoding available on PyTorch backend (xgrammar), less mature | Ollama structured outputs via JSON schema `format` field (Ollama blog, Dec 2024→stable) |
| Multi-image / request | Yes — `--limit-mm-per-prompt '{"image": N}'` | Yes | Model-dependent | Model-dependent, weakest |
| OpenAI-compatible API | `/v1/chat/completions` native | native | `trtllm-serve` native (mintlify TRT-LLM deployment docs, 2026) | `/v1` compat layer + native `/api/generate` |
| Multi-GPU on 4x5090 | TP/PP/DP first-class | TP/DP + router first-class | TP yes, but tuning-heavy | llama.cpp layer-split only; no real TP throughput |
| Ops burden | pip/Docker, minutes | pip/Docker, minutes | Heaviest: engine/config tuning, cold-start measured in tens of minutes for compiled path (Spheron H100 benchmark blog, 2026); best when model is frozen for months | Trivial; single-user scale only (LeetLLM inference-engine comparison, 2026) |
| Recommendation | **Default choice** | Pick if workload has heavy shared-prefix / agentic multi-turn (RadixAttention, ~29% throughput edge on those — LeetLLM 2026) | Only if you later chase last-10% perf on a frozen model | Dev-box convenience on GPUs 4–7, not for the service |

**Vision request wire format (vLLM/SGLang/TRT-LLM identical, OpenAI-style):**
```python
client = OpenAI(base_url="http://host:8000/v1", api_key="x")
r = client.chat.completions.create(model="Qwen/Qwen3-VL-32B-Instruct", messages=[{
  "role": "user",
  "content": [
    {"type": "text", "text": "What's in these images?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},  # base64 form
  ]}],
  response_format={"type": "json_schema", "json_schema": {...}})
```
Practical VLM tip: large images explode token counts (a 4000x3000 PNG ≈ 16k tokens on Qwen-VL); cap with `--mm-processor-kwargs '{"min_pixels":..., "max_pixels":...}'` or resize client-side (HF Qwen2.5-VL deployment discussion, 2025). Ollama's native API instead takes `"images": ["<base64>"]`, but its `/v1` endpoint accepts `image_url` too.

---

## 5. Monitoring

- **dcgm-exporter** (NVIDIA/dcgm-exporter, 2026) as a container on the bare-metal box: `docker run -d --gpus all --cap-add SYS_ADMIN -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:latest`; scrape `:9400/metrics` with Prometheus; Grafana dashboard **ID 12239** ("NVIDIA DCGM Exporter Dashboard", grafana.com). Gives per-GPU (by UUID!) util, VRAM, power, temp, clocks; <2% overhead (Spheron GPU-monitoring guide, 2026). Caveat: `DCGM_FI_PROF_*` profiling metrics are datacenter-GPU features — on GeForce 5090 expect the basic field set; if DCGM balks, fallback exporter `utkuozdemir/nvidia_gpu_exporter` (nvidia-smi-based) works everywhere.
- Add the **engine's own `/metrics`**: vLLM exposes Prometheus metrics (`vllm:num_requests_running/waiting`, TTFT/TPOT histograms, KV-cache usage); SGLang similar. GPU metrics tell you the box is busy; engine metrics tell you *why* (queueing vs KV saturation).
- Wire per-GPU-UUID alerts so a stray experiment allocating on the serving GPUs pages you (belt to the EXCLUSIVE_PROCESS suspenders from §1.5).

---

## TL;DR recipe for this box

1. `CUDA_DEVICE_ORDER=PCI_BUS_ID` everywhere; capture UUIDs with `nvidia-smi -L`; pin by UUID at the Docker/CDI layer, letting vLLM see indices 0–3 (avoids vLLM's UUID-parsing history, issue #32569 Jan 2026).
2. `nvidia-persistenced` on; `EXCLUSIVE_PROCESS` on the 4 serving GPUs (persisted via systemd oneshot).
3. Model fits in 32 GB → **4 independent vLLM replicas + LiteLLM/nginx least_conn** (best throughput+latency+blast-radius on a no-P2P PCIe box). Model needs the pool → `--tensor-parallel-size 4`, and A/B `-tp 2 -pp 2`; expect sub-linear TP scaling and consider `--disable-custom-all-reduce` / `NCCL_P2P_DISABLE=1` if NCCL misbehaves on 5090s.
4. Experiments on GPUs 4–7: per-process `CUDA_VISIBLE_DEVICES` (or containers), MPS only for trusted co-tenancy, no MIG exists on GeForce.
5. dcgm-exporter + Prometheus + Grafana 12239 + vLLM `/metrics`.

**Sources:** NVIDIA CUDA Programming Guide env-vars appendix (2026); NVIDIA "CUDA Pro Tip: CUDA_VISIBLE_DEVICES"; nvidia-smi -L / Microway nvidia-smi guide; NVIDIA Container Toolkit CDI docs (v1.17, 2025-26); Docker Docs CDI + Compose Deploy Spec (2026); NVIDIA/k8s-device-plugin README; NVIDIA GPU Operator time-slicing docs (2026); vLLM docs: Parallelism & Scaling, Data Parallel Deployment, Expert Parallel, Structured Outputs, Nginx LB (stable/latest, 2026); vllm issues #32569 (Jan 2026), #14628; NVIDIA/nccl #1637 (5090 P2P); smcleod.net P2P driver patch (Feb 25 2026); GIGAGPU DP-vs-TP (2026); Will It Run AI vLLM TP guide (2026); databasemart vLLM optimization guide; LMSYS SGLang v0.4 blog (Dec 2024) + docs.sglang.io (multimodal, structured outputs, router, 2026); sglang #20807; TensorRT-LLM docs/release notes + trtllm-serve (2026); Ollama blog (multimodal engine; structured outputs); LeetLLM & Spheron engine comparisons (2026); docs.litellm.ai routing; NVIDIA/dcgm-exporter + Grafana dashboard 12239; nebuly nos MPS/MIG/time-slicing comparison; Red Hat Developer vLLM structured outputs (Jun 2025).
