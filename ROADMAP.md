# Tier A ML Performance Engineering — Learning Roadmap

**Target roles:** Inference Performance Engineer, ML Systems Engineer, GPU Software Engineer at inference-serving companies (Together, Fireworks, Anyscale, Modal, Baseten, Replicate), foundation-model applied-perf teams (Anthropic, OpenAI, Cohere, Databricks Mosaic), and serving-framework project orbits (vLLM, SGLang, TensorRT-LLM, PyTorch).

**Timeline:** ~4 months focused effort (16 weeks), or ~6 months alongside a day job. Assumes the current CUDA foundation (SGEMM ladder complete, WMMA/MMA in progress on L40) as a starting point.

**Guiding principle:** Every phase produces a concrete GitHub artifact. The portfolio is the interview.

**Started:** 2026-08-09 · **Target completion:** ~2026-12-09 (16 weeks focused) / ~2027-02-09 (alongside work)

> Tracking: tick boxes as you go. Keep the "Progress log" at the bottom updated with dates
> and links — it doubles as raw material for the blog posts and interview stories.

---

## Phase 0 — What you already have (don't redo)

- [x] CUDA memory hierarchy, coalescing, occupancy, shared memory tiling
- [x] The SGEMM optimization ladder
- [x] Nsight Compute / nvprof basics
- [x] L40 access for Ada-class experiments

This is roughly the "CUDA fundamentals" tier that most Tier A candidates lack. It stays as the base of every downstream artifact.

---

## Phase 1 — Triton Fluency (4 weeks)

**Deliverable:** A GitHub repo with 4-5 self-authored Triton kernels, each autotuned, correctness-tested, and benchmarked against eager PyTorch and `torch.compile`.

- [ ] Phase 1 deliverable complete

### Concepts to master
- [ ] Block-level programming model: `tl.program_id`, `tl.arange`, masks, `tl.load` / `tl.store` with strides
- [ ] Autotuning: `@triton.autotune`, config search, `key` semantics, prune_configs_by
- [ ] Reductions: `tl.sum`, `tl.max`, online softmax pattern
- [ ] Matmul in Triton: `tl.dot`, accumulator dtypes, tile shapes
- [ ] Debugging: `TRITON_INTERPRET=1`, printing tensors, correctness harness
- [ ] Reading Triton IR when autotune outputs surprise you

### Kernels to build (in this order)
1. [ ] **Fused RMSNorm + quantize** — combines a reduction, a normalization, and an output cast. Great warmup.
2. [ ] **Online softmax** — the FlashAttention prerequisite. Understand the numerical trick.
3. [ ] **FlashAttention-2 forward** — the canonical Triton exercise. Follow the official tutorial then rewrite from scratch.
4. [ ] **W4A16 GEMM prologue** — dequantize INT4 weights into registers, then `tl.dot`. Marlin-style.
5. [ ] **A custom fused epilogue** — GEMM + bias + activation + quantize. Anything real from a paper.

### Resources
- [ ] **Official Triton tutorials** — https://triton-lang.org/main/getting-started/tutorials/index.html. Work through all of them, not just the ones that look interesting.
- [ ] **GPU MODE Lecture 14** — "Practitioner's Guide to Triton" (Umer Adil). The single best video for someone starting from CUDA.
- [ ] **GPU MODE Lecture 29** — "Triton Internals" (Kapil Sharma). Watch after Lecture 14 for the compiler-side mental model.
- [ ] **Sasha Rush — "GPU Puzzles" (Triton edition)** — https://github.com/srush/Triton-Puzzles. Puzzle-based learning; 2-3 evenings.
- [ ] **Umer Adil — "Everything about Triton"** blog series.
- [ ] **Liger Kernel** — https://github.com/linkedin/Liger-Kernel. Production Triton kernels for LLM training. Read the RMSNorm, SwiGLU, and cross-entropy implementations to see how real people structure kernels.
- [ ] **Unsloth kernels** — https://github.com/unslothai/unsloth. Another good production reference for fine-tuning-focused Triton kernels.

---

## Phase 2 — Profiling as a First-Class Skill (3 weeks, partially parallel with Phase 1)

**Deliverable:** A written end-to-end profiling case study on a real open-source model. Take Llama-3-8B or Qwen-2.5-7B inference on the L40, profile with Nsight Systems + Nsight Compute + PyTorch profiler, identify the top 3 bottlenecks by wall time, fix one, and write it up as a blog post.

- [ ] Phase 2 deliverable complete

### Concepts to master
- [ ] **System-level (Nsight Systems / nsys):** kernel launch latency, host-device sync gaps, memcpy overlap, stream utilization, CPU-bound vs GPU-bound identification, cudaMalloc / cudaFree overhead.
- [ ] **Kernel-level (Nsight Compute / ncu):** achieved occupancy vs theoretical, memory throughput (L1/L2/DRAM), warp stall reasons (long scoreboard, short scoreboard, memory dependency), pipe utilization (FMA, ALU, TENSOR).
- [ ] **Roofline model:** arithmetic intensity, memory-bound vs compute-bound, ridge point. Draw one for your kernel by hand once.
- [ ] **PyTorch profiler + Chrome trace:** `torch.profiler.profile`, kernel view, memory view, distributed view. `record_function` for annotation.
- [ ] **CUDA graphs:** when they help (small-launch overhead), when they hurt (dynamic shapes), how to capture and replay.
- [ ] **CUDA streams and async:** overlapping H2D copy with compute, priority streams.

### Resources
- [ ] **GPU MODE Lecture 1** — "How to profile CUDA kernels in PyTorch" (Mark Saroufim).
- [ ] **GPU MODE Lecture 8** — "CUDA Performance Checklist" (Mark Saroufim). Print the checklist and use it every time you profile.
- [ ] **NVIDIA Nsight Compute docs** — https://docs.nvidia.com/nsight-compute/. The "Kernel Profiling Guide" section is the reference.
- [ ] **PyTorch profiler recipes** — https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html.
- [ ] **Horace He — "Making Deep Learning Go Brrrr From First Principles"** — https://horace.io/brrr_intro.html. The best conceptual piece on compute-bound vs memory-bound reasoning.
- [ ] **NVIDIA GTC talks** — Stephen Jones "How CUDA Works" is worth two hours.

---

## Phase 3 — PyTorch Internals + torch.compile (2 weeks)

**Deliverable:** A one-page writeup with 3 mini-benchmarks showing where `torch.compile` wins, where it loses, and how to fix a graph break.

- [ ] Phase 3 deliverable complete

### Concepts to master
- [ ] Eager vs traced vs compiled execution paths
- [ ] Dynamo: what causes graph breaks, `torch._dynamo.explain`
- [ ] TorchInductor: reading the generated Triton, output_code.py inspection
- [ ] Dynamic shapes: `mark_dynamic`, guards, recompilation
- [ ] Custom ops: `torch.library.custom_op`, `register_fake` (for FakeTensor), autograd registration
- [ ] CUDA graphs integration via `mode="reduce-overhead"`
- [ ] `torch.compile` regions vs full-model compilation

### Resources
- [ ] **PyTorch 2.x docs — torch.compile tutorial and troubleshooting** — https://docs.pytorch.org/docs/stable/torch.compiler.html.
- [ ] **GPU MODE Lecture 10** — "Building a CUDA/C++ extension for PyTorch".
- [ ] **Horace He's blog + talks on torch.compile** — his GTC talk and various threads.
- [ ] **PyTorch Developer Podcast** (Ed Yang) — episodes on Dynamo, Inductor, FakeTensor. Excellent internals coverage.
- [ ] **Christian Puhrsch / Driss Guessous kernel work in PyTorch** — read PRs to see how custom kernels get integrated upstream.

---

## Phase 4 — Serving Stack (4 weeks)

**Deliverable:** One merged PR into vLLM, SGLang, or TensorRT-LLM (a bug fix, a benchmark script, a small kernel — size doesn't matter, the fact of the merge does), plus a self-serving deployment of an LLM with measured throughput/latency numbers.

- [ ] Merged PR
- [ ] Deployment with measured throughput/latency

### Concepts to master (theory first)
- [ ] **KV cache mechanics:** why it dominates memory, block-based allocation, sharing/copying semantics
- [ ] **Prefill vs decode:** why prefill is compute-bound and decode is memory-bound, and what that implies for batching
- [ ] **Continuous batching:** in-flight requests, iteration-level scheduling
- [ ] **Chunked prefill:** breaking long prompts to overlap with decode
- [ ] **PagedAttention:** virtual-memory-inspired KV cache (the vLLM contribution)
- [ ] **RadixAttention:** prefix caching via a radix tree (the SGLang contribution)
- [ ] **Speculative decoding:** draft models, EAGLE, Medusa
- [ ] **Disaggregated prefill/decode:** separate GPU pools per phase
- [ ] **Parallelism strategies:** tensor parallel (Megatron-style), pipeline parallel, expert parallel for MoE

### Frameworks to actually use
- [ ] **vLLM** — clone it, run a model, profile a batch, read `csrc/` and the attention backends. Understand the block manager. https://github.com/vllm-project/vllm.
- [ ] **SGLang** — same drill. Focus on the frontend language and RadixAttention. https://github.com/sgl-project/sglang.
- [ ] **TensorRT-LLM** — heavier lift; read the docs and one plugin. Skip unless targeting NVIDIA directly.

### Resources
- [ ] **vLLM paper** — "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., 2023).
- [ ] **SGLang paper** — "SGLang: Efficient Execution of Structured Language Model Programs" (Zheng et al., 2024).
- [ ] **Aleksa Gordić — "Inside vLLM"** blog series. Excellent architectural walkthrough.
- [ ] **GPU MODE Lecture 30+ (various) on vLLM and inference stacks.**
- [ ] **Anyscale / Databricks / NVIDIA engineering blogs** — search their blogs for LLM inference posts; there's a steady stream of high-quality applied content.
- [ ] **Anthropic / OpenAI / Cohere engineering blogs** — occasional posts, worth watching.

---

## Phase 5 — Quantization (2-3 weeks)

**Deliverable:** A benchmark repo where you quantize an open LLM in three formats (INT8 W8A8, INT4 W4A16, FP8) using existing kernels, and measure the throughput / accuracy / memory trade-off with a proper eval harness.

- [ ] Phase 5 deliverable complete

### Concepts to master
- [ ] **Numeric formats:** INT8 (per-tensor/per-channel/per-token scaling), INT4 group-quantized, FP8 E4M3 vs E5M2, blockwise/tile-wise scaling
- [ ] **Weight-only vs weight+activation quant:** when each applies
- [ ] **Post-training vs quantization-aware training:** GPTQ, AWQ, SmoothQuant, QAT trade-offs
- [ ] **Kernel families:** Marlin (W4A16 for Ampere+), Machete (Hopper W4A16), BitBLAS (mixed-precision), CUTLASS FP8 GEMMs
- [ ] **Calibration data and its impact on accuracy**
- [ ] **FP8 training numerics:** why blockwise scaling matters, DeepSeek-V3's fine-grained scaling story

### Resources
- [ ] **GPU MODE Lecture 7** — "Advanced Quantization" (Charles Hernandez).
- [ ] **GPU MODE Lecture 30** — "Quantized Training" (various).
- [ ] **HuggingFace quantization docs** — https://huggingface.co/docs/transformers/main/en/quantization/overview.
- [ ] **AutoAWQ, GPTQModel, llm-compressor** — read the code for the passes you'll use.
- [ ] **Marlin paper** — "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference of Large Language Models" (Frantar et al.).
- [ ] **DeepSeek-V3 technical report** — the FP8 sections are the current state-of-the-art applied FP8 training story.
- [ ] **NVIDIA Transformer Engine** — https://github.com/NVIDIA/TransformerEngine. Read the FP8 recipes.

---

## Phase 6 — Read-Level CUTLASS (2 weeks)

**Deliverable:** One repo commit or notebook where you walk through a real CUTLASS Hopper GEMM, annotate what each phase does, and benchmark the CUTLASS Python bindings on a shape of your choice against cuBLAS.

- [ ] Phase 6 deliverable complete

### Concepts to master (reading, not writing)
- [ ] What CUTLASS and CuTe are and what problems they solve
- [ ] Structure of a modern CUTLASS GEMM: mainloop, epilogue, TMA loads, MMA instructions
- [ ] CuTe layouts at conceptual level — shape, stride, mode, hierarchical layouts
- [ ] Ping-Pong vs Cooperative warp specialization patterns (conceptual)
- [ ] Epilogue Visitor Trees for composable post-GEMM ops
- [ ] When to reach for CUTLASS Python bindings vs write Triton

### Resources
- [ ] **CUTLASS README + examples/** — https://github.com/NVIDIA/cutlass. Read examples 48 (Hopper warp-specialized GEMM) and 61 (Hopper GEMM with EVT epilogue).
- [ ] **Colfax Research CuTe tutorial (matrix transpose)** — the on-ramp to CuTe layout thinking.
- [ ] **PyTorch blog — CUTLASS Ping-Pong deep dive** — https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/.
- [ ] **Kapil Sharma — "Learn CUTLASS the hard way"** (Nov 2025) — recent, practical, exactly the depth needed for read-level fluency.
- [ ] **GPU MODE Lecture 23** — "Tensor Cores" by Vijay Thakkar and Pradeep Ramani (CUTLASS/CuTe authors). Watch even at read-level; the mental model matters.

---

## Cross-Cutting Skills (integrate throughout)

- [ ] **C++/CUDA PyTorch extensions:** `torch.utils.cpp_extension`, pybind11-style bindings, `TORCH_LIBRARY`. GPU MODE Lecture 10 is the reference. You'll write one whenever you want a hand-written CUDA kernel usable from Python.
- [ ] **Distributed basics:** NCCL primitives (all-reduce, all-gather, reduce-scatter, broadcast), tensor parallelism sharding patterns, communication-compute overlap. Not deep — just enough to reason about multi-GPU serving.
- [ ] **Docker / K8s for GPU workloads:** enough to package a serving stack and know why NCCL breaks in weird container configs.
- [ ] **Numerics literacy:** BF16 vs FP16 dynamic range, why accumulate in FP32, when Tensor Core precision matters. The DeepSeek-V3 FP8 story is a good case study.

---

## What You Can Safely Skip for Tier A

Do not spend time on these unless a specific role demands it:
- Writing CUTLASS templates from scratch
- SASS analysis
- PTX beyond reading MMA instructions
- Hopper WGMMA implementation details (concepts fine, code walkthrough fine, hand-writing not needed)
- Blackwell UMMA / tcgen05 (too new, and no L40 access)
- MLIR / TorchInductor internals below the API surface
- Full GPU microarchitecture (SM scheduler scoreboards, register file banking) — nice to have, not required
- Compiler frontends (Triton compiler internals below the tutorial-level view)

---

## Portfolio Checklist (the interview artifact)

By end of Phase 6, GitHub should show:

1. [ ] **SGEMM ladder repo** — polished, with a README, benchmark table, and a short writeup. Table stakes.
2. [ ] **Triton kernel library** — 4-5 kernels, autotuned, tested, benchmarked. Include a README with the roofline for each kernel and where it sits.
3. [ ] **End-to-end profiling case study** — model, methodology, findings, one fix, before/after numbers. Blog-post format.
4. [ ] **Merged PR** into vLLM / SGLang / TensorRT-LLM. Size doesn't matter, the merge does.
5. [ ] **Quantization benchmark repo** — three formats, throughput/accuracy/memory numbers, eval harness.
6. [ ] **cuQuantum portfolio project** — the distinctive angle. Quantum simulation kernels on GPU. Most Tier A candidates look identical; this separates you.

Optional but strong:
- [ ] One blog post per phase (Simon Boehm / Horace He style). Even one good post is a hiring magnet.
- [ ] A KernelBench submission or leaderboard entry.
- [ ] One contribution to Liger Kernel or a similar open-source Triton library.

---

## Community & Ongoing Learning

- **GPU MODE Discord** — https://discord.gg/gpumode. Weekly lectures, active channels for questions. This is the community.
- **GPU MODE lectures repo** — https://github.com/gpu-mode/lectures. Slides and code for every lecture.
- **GPU MODE resource-stream** — https://github.com/gpu-mode/resource-stream. Curated book/paper/blog/video meta-index.
- **Twitter/X:** Tri Dao, Horace He, Vijay Thakkar, Pradeep Ramani, Mark Saroufim, Simon Boehm, Kapil Sharma, Aleksa Gordić, Umer Adil. Not for scrolling — for finding new posts and papers.
- **arXiv daily** — cs.PF, cs.DC, cs.LG systems papers. Skim titles daily.

### Books worth owning
- **Programming Massively Parallel Processors** (Hwu, Kirk, Hajj) — CUDA foundations. You've probably read it; keep as reference.
- **CUDA C Programming Guide** (NVIDIA) — reference, not cover-to-cover.
- **Optional:** *Computer Architecture: A Quantitative Approach* (Hennessy & Patterson) for the memory-hierarchy chapters when you want the deeper picture.

---

## Interview Prep (final 3-4 weeks, overlapping with polish)

Tier A interviews typically include:

1. **Kernel walkthrough:** "Walk me through a kernel from your GitHub." Pick the Triton FA kernel or the SGEMM Vect2D. Know it cold — every choice, every trade-off, the profiling behind it.
2. **Debugging a slow model:** "Here's a slow inference workload. Walk me through how you'd profile it." Answer using the Nsight Systems → Nsight Compute → PyTorch profiler → fix flow from the case study.
3. **Systems-design-style:** "Design an inference server for Llama-3-70B at 10k QPS." Cover parallelism, batching strategy, KV cache, quantization choice, hardware fit.
4. **Concept drills:** roofline model, why decode is memory-bound, what causes graph breaks, how PagedAttention works, when W4A16 makes sense vs FP8.
5. **Coding:** occasionally a LeetCode-style round, occasionally a CUDA/Triton kernel to write live. The existing DSA prep covers the former; the kernel library covers the latter.

The single most differentiating interview signal for Tier A: being able to describe a real profiling story from your own work in specific quantitative terms. "I saw 34% achieved occupancy, the ncu warp-state view showed long-scoreboard stalls dominating, I hypothesized shared-memory bank conflicts, verified with the L1 metrics, refactored the tile layout, went from 1.2 TFLOPS to 3.8 TFLOPS." That sentence, delivered from real work, wins offers.

---

## Progress log

One line per session: date, what you did, the number that moved, and a link if there's an artifact.

| Date | Phase | What happened | Artifact / numbers |
| --- | --- | --- | --- |
| 2026-08-09 | — | Roadmap adopted and committed to the repo | this file |
