# DoolyProf TODOs for Qwen3-Coder-480B on GB200 / GKE

Status as of 2026-07-26. Context: profiling Qwen3-Coder-480B-A35B-Instruct-FP8
for a TP=4 discrete-event simulator, against serving deployment
`no-offloading-tp4-vllm` (image `vllm/vllm-openai:v0.21.0`, GKE namespace
`igw-llm-d`, GB200 a4x nodes with 4 GPUs).

## 1. Expert parallelism (`--enable-expert-parallel`) not replicated  [DEFERRED]

The serving deployment runs vLLM with `--enable-expert-parallel` on top of
TP=4. DoolyProf's fake-TP tracer/profiler only replicates plain TP sharding.
This changes what is being measured for MoE layers:

| | Production (TP=4 + EP) | DoolyProf today (fake TP=4) |
|---|---|---|
| Experts per rank | ~40 of 160, full intermediate width | all 160, 1/4 intermediate width |
| Expert GEMM shapes | tall/narrow (few experts, full width) | short/wide (all experts, sharded width) |
| Token routing | tokens dispatched to the rank owning the expert | every rank computes its shard of every routed expert |
| Collectives around MoE | dispatch/combine pattern (all_gather + scatter or all2all, backend-dependent) | plain all_reduce after the MoE block |

Consequences for simulation fidelity:
- FusedMoE kernel latency is measured with the wrong per-rank expert
  geometry. Total FLOPs per rank are similar, but kernel efficiency differs
  (different tile shapes, different number of expert launches, different
  activation-imbalance behavior under EP where hot experts skew per-rank load).
- The collective plan extracted from the trace (all_reduce, count=125) does
  not match the EP dispatch/combine traffic, so comm time attributed to MoE
  layers is structurally wrong.

How to pick this up later:
1. Tracer: `run-tracer.py` builds `llm_kwargs` explicitly; add
   `enable_expert_parallel=True` there and in
   `VLLMLayerProfiler.__init__` (mirror how `kv_cache_dtype` / `block_size`
   were plumbed in July 2026). Open question: interaction with fake-TP -
   `apply_fake_tp_patches` no-ops the TP collectives and patches
   `get_tensor_model_parallel_world_size`; EP sizing in vLLM derives from the
   EP group, so the patch set likely needs an equivalent
   `get_expert_parallel_*` treatment so each fake rank builds ~40 full-width
   experts instead of 160 sharded ones.
2. Comm: profile the EP dispatch/combine collectives with real TP=4 on one
   node (the comm profiler currently only sweeps all_reduce / all_gather /
   reduce_scatter; the EP path may use different primitives depending on the
   vLLM all2all backend).
3. Validation: compare per-layer MoE time from a production nsys/torch
   profile against the DB numbers for both variants (TP-sharded vs EP) to
   quantify how much this matters before investing further.

Currently acceptable because serving is single-node TP=4 and we compare
end-to-end against real GPU benchmarks first; revisit before trusting
per-layer MoE breakdowns or simulating the TP=8 cross-node EP variant.

## 2. Cross-node TP=8 comm profiling  [DEFERRED]

Compute side works today (fake TP=8 on one GPU, ~60 GB shard). The comm
profiler can only launch a node-local vLLM, so cross-node TP=8 collectives
cannot be measured. Plan: standalone `torchrun --nnodes 2 --nproc-per-node 4`
sweep over the same `COLLECTIVE_SIZE_SPACE` writing to the same
`coll_op_results` schema. Note the `topology` column is currently hardcoded
to `"mew1"` at the `CommProfiler.profile()` call site in
`doolyprof/run/comm_worker.py` - pass a real label (e.g. `nvl-intra` vs
`ib-inter`) to keep datasets distinguishable. Check first whether the 8
GB200 trays share one NVL72 NVLink domain - if so the gap mostly disappears.

## 3. KV cache dtype - validate fp8 vs bf16 impact  [PLUMBED, NEEDS VALIDATION]

`--kv-cache-dtype` and `--block-size` are now plumbed through run-tracer,
run-profiler, Profiler, and VLLMLayerProfiler (July 2026). The production DB
is being captured with `--kv-cache-dtype fp8 --block-size 256` to match
serving. Follow-up: quantify the delta by comparing Attention rows between
the bf16-KV smoke DB (`dooly_qwen480b_smoke2.db`) and the fp8-KV production
DB - fp8 KV halves KV-read bytes, so decode attention at long context should
be up to ~2x faster; confirms whether older bf16-KV data is salvageable.

## 4. DB rows are not keyed by environment details  [OPEN]

Signatures dedup on kernel names + shapes, which implicitly captures backend
and KV dtype, but the DB has no explicit columns for kv_cache_dtype,
block_size, vLLM version, or image digest. Record alongside each capture:
image `vllm/vllm-openai:v0.21.0`, torch 2.11.0+cu130, flashinfer 0.6.8.post1,
`VLLM_USE_DEEP_GEMM=0 TORCHDYNAMO_DISABLE=1` env. Consider adding a
`capture_metadata` table.

## 5. Raw `vllm::moe_forward` op never resolves  [COSMETIC]

On vLLM 0.21 the resolver's dummy invocation of the raw custom op fails
("Expected Optional[Tensor] for shared_experts_input but found int" - the
trace records None args as 0). Harmless: the walk-up fallback profiles the
parent FusedMoE module instead, which is what the simulator wants anyway.
Fix would be in the trace parser's argument reconstruction (map recorded 0
to None for `Tensor?` params of custom ops).

## 6. Upstream the vLLM 0.21 / Blackwell-MoE compat patches  [OPEN]

Local patches worth PRing to dooly-project/doolyprof:
- `tracer/types.py` + `tracer/hooks.py`: `unwrap_taint()` for nested
  DimTaint (crashes on any FP8 MoE model otherwise).
- `tracer/hooks.py`: `_NON_CONFIG_ATTRS` now excludes per-instance identity
  counters (`moe_layer_id` etc.) - without this, 62 identical FusedMoE
  layers profile 62 times.
- `profiler/resolver.py`, `profiler/profiler.py`: `set_forward_context()`
  lost `virtual_engine` in vLLM 0.21.
- `profiler/vllm_layer_profiler.py`: version-adaptive
  `BlockTable.compute_slot_mapping` (GPU Triton kernel API in 0.21,
  `commit_slot_mapping` removed).
- `run/comm_profiler.py`: `gpu_memory_utilization` 0.5 -> 0.85 (real TP=4
  rank of a 480B FP8 model needs ~120 GB weights per GPU).
- kv_cache_dtype / block_size plumbing (all entry points).
- Runtime env workarounds (not code): `VLLM_USE_DEEP_GEMM=0
  TORCHDYNAMO_DISABLE=1` - dynamo trips over the taint tensor subclass in
  DeepGEMM's `per_block_cast_to_fp8`.
