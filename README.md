# DoolyProf

`DoolyProf` is a configuration agnostic and redundancy aware profiler for LLM-inference workloads.
For any deployment configuration (e.g., hardware, attention backend implementation, serving engine, model), `DoolyProf` automatically detects the set of operations or modules present in the inference forward pass, and profiles each operation/module to output a SQLite database.
By detecting duplicate operations or modules across profile runs, `DoolyProf` allows for efficient profiling across different configurations. 
Downstream consumers (e.g., discrete-event simulator, prediction-based schedulers) can use to predict LLM serving performance across the large configuration space with minimal profiling overhead.

Check our pre-print at [Dooly: Configuration-Agnostic, Redundancy-Aware Profiling for LLM Inference Simulation](https://arxiv.org/abs/2605.07985).

## Installation

Requires Python 3.10+, a CUDA-capable GPU, and a compatible PyTorch / vLLM build. 
We test with PyTorch 2.9.1+, vLLM 0.17.1, and CUDA 13.1.

```bash
git clone https://github.com/dooly-project/doolyprof.git
cd doolyprof
pip install -r requirements.txt
```

## Quick start

### One-shot: trace + profile

```bash
python -m doolyprof.run.run-e2e \
    --model meta-llama/Llama-3.1-8B \
    --attention-backend FLASHINFER \
    --max-batch-size 32 \
    --max-seq-len 1024 \
    --tp 1 \
    --db-path ./dooly_data.db
```

This produces `./dooly_data.db`, a SQLite database keyed by `(model, attention_backend, tp_size, dtype)` containing per-operation latency samples.

### Two-stage workflow

If you want to inspect the trace before profiling, or re-profile against an
existing trace:

```bash
# 1. Trace only — captures tensor dimension semantics
python -m doolyprof.run.run-tracer \
    --model meta-llama/Llama-3.1-8B

# 2. Profile using the trace
python -m doolyprof.run.run-profiler \
    --model meta-llama/Llama-3.1-8B \
    --traces ./vllm_traces_tp/traces-FLASHINFER-bfloat16/tp_1/meta-llama_Llama-3_1-8B/*.json \
    --attention-backend FLASHINFER \
    --max-batch-size 32 \
    --max-seq-len 1024 \
    --tp 1 \
    --db-path ./dooly_data.db
```

## Tensor parallelism

You can profile a TP>1 configuration on a single GPU using fake-TP:

```bash
python -m doolyprof.run.run-tracer --model meta-llama/Llama-3.1-70B --tp 4

```

The tracer sets `VLLM_FAKE_TP=8` and patches vLLM's `parallel_state` and collective ops to no-ops, so the model executes on one GPU while reporting TP=4 communication shapes. 

## Repository layout

```
doolyprof/
└── doolyprof/       # python package (import as `doolyprof.X`)
    ├── tracer/      # Taint propagation through PyTorch ops
    ├── profiler/    # Layer profiling and operation capture
    └── run/         # Entry points: run-tracer, run-profiler, run-e2e
```

## Citation
```
@misc{kim2026doolyconfigurationagnosticredundancyawareprofiling,
      title={Dooly: Configuration-Agnostic, Redundancy-Aware Profiling for LLM Inference Simulation}, 
      author={Joon Ha Kim and Geon-Woo Kim and Anoop Rachakonda and Daehyeok Kim},
      year={2026},
      eprint={2605.07985},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2605.07985}, 
}
```
## License
MIT — see [LICENSE](./LICENSE).
