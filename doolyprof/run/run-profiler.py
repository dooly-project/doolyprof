import argparse
import os
import sys
from pathlib import Path
from time import time

# Add project root to path for doolyprof imports
# run-profiler.py -> run/ -> doolyprof/ -> repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Parse --tp and --gpu early BEFORE importing vLLM / torch / CUDA.
# - VLLM_FAKE_TP: vllm_layer_profiler.py applies fake TP patches at module load.
# - CUDA_VISIBLE_DEVICES: must be set before any CUDA init so the main process
#   (which runs _profile_raw_ops directly) and every spawned child / subprocess
#   (resolve_model, _profile_vllm_modules_subprocess, run_comm_profiler) inherit
#   the same GPU selection. Otherwise --gpu only reaches VLLMLayerProfiler and
#   the main process + comm worker silently fall back to GPU 0.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--tp", type=int, default=1)
_pre_parser.add_argument("--gpu", type=str, default="0,1,2,3")
_pre_args, _ = _pre_parser.parse_known_args()
os.environ["VLLM_FAKE_TP"] = str(_pre_args.tp)
os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import multiprocessing as mp
from doolyprof.profiler.model import ModelInfo
from doolyprof.profiler.profiler import Profiler 

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int8": torch.int8,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["meta-llama/Llama-3.1-8B", "BioMistral/BioMistral-7B", 
                 "Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"],
    )
    parser.add_argument(
        "--traces",
        nargs="+",
        default=[
            "./utns_sim/data/vllm_profile/profiler_traces/llama-trace.json",
            "./utns_sim/data/vllm_profile/profiler_traces/mistral-trace.json",
            "./utns_sim/data/vllm_profile/profiler_traces/qwen-trace.json",
            "./utns_sim/data/vllm_profile/profiler_traces/deepseek-trace.json",
        ],
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_MAP.keys(),
        default="bfloat16",
    )
    parser.add_argument(
        "--profile-output",
        default="./vllm_profile/per-model-profile.csv",
    )
    parser.add_argument("--attention-backend", default="FLASHINFER")
    parser.add_argument("--flash-attn-version", type=int, default=2, choices=[2, 3, 4],
                        help="Pin flash-attention version when --attention-backend=FLASH_ATTN. Without this, vLLM auto-picks FA3 on Hopper / FA4 on Blackwell.")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=["fp8", "awq", "gptq"],
                        help="Optional quantization scheme. 'fp8' requires Hopper+ (H100). "
                             "Combined with load_format=dummy, vLLM initializes random "
                             "quantized weights — no quantized checkpoint needed.")
    parser.add_argument("--workload-mode", default="vidur")
    parser.add_argument("--max-batch-size", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--test-counts", type=int, default=1)
    parser.add_argument("--force-prefill-kernel", action="store_true", default=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to SQLite database for persistent storage and deduplication")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device ID to use (default: 0)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Force re-profiling even if signatures already exist in database")
    parser.add_argument("--profile-method", default="cuda_event",
                        choices=["cuda_event", "kineto", "wall_clock"],
                        help="Profiling method for network operations (default: cuda_event)")
    parser.add_argument("--profile-comm", action="store_true",
                        help="Profile communication operations")
    parser.add_argument("--overwrite-comm", action="store_true",
                        help="Force re-profiling of communication operations")
    parser.add_argument("--no-cuda-graph-comm", action="store_true",
                        help="Disable CUDA graph profiling for communication operations")
    parser.add_argument("--comm-timing", choices=["isolated", "amortized"], default="isolated",
                        help="Collective timing mode for the comm profiler: 'isolated' "
                             "(default, per-call synced median) or 'amortized' (measure_iters "
                             "back-to-back between two CUDA events, single sync, divided by iters).")
    return parser.parse_args()


def resolve_model(model, dtype, attention_backend, flash_attn_version, quantization, max_seq_len, result_queue, gpu):
    from doolyprof.profiler.importer import OpImporter
    from doolyprof.profiler.resolver import ModuleResolver
    from doolyprof.profiler.vllm_layer_profiler import VLLMLayerProfiler
    
    importer=OpImporter()

    # Get the minimum of user-specified max_seq_len and model's actual max
    model_max = model.get_max_model_len()
    if model_max:
        max_model_len = min(max_seq_len, model_max)
        print(f"[RESOLVER] Using max_model_len={max_model_len} (min of user's {max_seq_len} and model's {model_max})")
    else:
        max_model_len = max_seq_len
        print(f"[RESOLVER] Using max_model_len={max_model_len} from user config")

    vlp = VLLMLayerProfiler(
        model_name=model.name,
        dtype=str(model.dtype).split(".")[-1],
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
        attn_backend=attention_backend,
        flash_attn_version=flash_attn_version,
        quantization=quantization,
        gpu=gpu,
    )
    resolver = ModuleResolver(
        vlp=vlp,
        importer=importer,
        dtype=dtype,
        taint_registry=getattr(model, 'taint_registry', {}) or {},
    )

    resolved_ops = resolver.resolve(model.used_ops)

    result_queue.put([(module, None) for module, _ in resolved_ops])    

    print("\nClosing vLLM instance...")
    vlp.close()
    print(f"=== Resolution complete for {model.name} ({dtype}) ({attention_backend}) ===\n")

def run_comm_profiler(models, dtype_str, tp_size, db_path, overwrite, profile_method="cuda_event", use_cuda_graph=True, comm_timing="isolated"):
    """Run comm profiler via subprocess with clean environment (no fake TP)."""
    import subprocess
    import pickle
    import tempfile

    print(f"\n{'='*60}")
    print(f"Launching comm profiler via subprocess (REAL TP={tp_size})")
    print(f"Profile method: {profile_method.upper()}")
    print(f"CUDA Graph: {'ENABLED' if use_cuda_graph else 'DISABLED'}")
    print(f"Clean environment WITHOUT VLLM_FAKE_TP")
    print(f"{'='*60}\n")

    # Extract collective ops from first model
    model_name = models[0].name
    collective_ops = models[0].collective_ops

    # Save collective_ops to temp file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        pickle.dump(collective_ops, f)
        collective_ops_path = f.name

    # Create clean environment WITHOUT VLLM_FAKE_TP
    clean_env = os.environ.copy()
    clean_env.pop('VLLM_FAKE_TP', None)

    try:
        # Choose worker based on profile method
        if profile_method in ["kineto", "wall_clock"]:
            # Use the KINETO-enabled worker for advanced profiling methods
            comm_worker_path = os.path.join(os.path.dirname(__file__), 'comm_worker_kineto.py')
        else:
            # Use the original worker for cuda_event method
            comm_worker_path = os.path.join(os.path.dirname(__file__), 'comm_worker.py')
            if not use_cuda_graph:
                cmd.append('--no-cuda-graph')

        cmd = [
                sys.executable, comm_worker_path,
                '--model', model_name,
                '--collective-ops-pkl', collective_ops_path,
                '--dtype', dtype_str,
                '--tp-size', str(tp_size),
                '--db-path', db_path,
            ]

        # Add profile method for kineto worker
        if profile_method in ["kineto", "wall_clock"]:
            cmd.extend(['--profile-method', profile_method])

        if overwrite:
            cmd.append('--overwrite')

        # Amortized timing option is implemented in comm_worker.py (cuda_event path).
        if profile_method not in ["kineto", "wall_clock"]:
            cmd.extend(['--comm-timing', comm_timing])

        result = subprocess.run(
            cmd,
            env=clean_env,
            check=False
        )

        if result.returncode != 0:
            print(f"\n✗ Comm profiling failed with exit code {result.returncode}")
            sys.exit(result.returncode)
        else:
            print(f"\n✓ Comm profiling ({profile_method}) completed successfully")
    finally:
        # Cleanup temp file
        os.unlink(collective_ops_path)


def main() -> None:
    args = parse_args()
    if len(args.models) != len(args.traces):
        raise SystemExit("--models and --traces must have the same length")

    dtype = DTYPE_MAP[args.dtype]
    models = [
        ModelInfo(name, trace_path, dtype)
        for name, trace_path in zip(args.models, args.traces)
    ]

    mp.set_start_method("spawn", force=True)

    result_queue = mp.Queue()

    for i, model in enumerate(models):
        print(f"[RUNNER] Starting RESOLVER for {model.name}...")
        p = mp.Process(target=resolve_model, args=(model, dtype, args.attention_backend, args.flash_attn_version, args.quantization, args.max_seq_len, result_queue, args.gpu))
        p.start()
        model.to_profile_resolved = result_queue.get()
        p.join()

        if p.exitcode != 0:
            print(f"Run {i + 1} failed with exit code {p.exitcode}")
            
    # Compare modules across models
    models[0].compare_with_many(models[1:], db_path=args.db_path, overwrite=args.overwrite)
    
    print("Profiling Plan (Computation): ")
    
    for model in models:
        print(f"Model: {model.name}\n Modules to Profile: {len(model.to_profile)}")
        for module in model.to_profile:
            print(f"  - {module.module_name}, {module.operation_name}")
        print()

    profile_output = Path(args.profile_output)
    profile_output.parent.mkdir(parents=True, exist_ok=True)

    profiler = Profiler(
        output_path=str(profile_output),
        workload_mode=args.workload_mode,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        test_counts=args.test_counts,
        force_prefill_kernel=args.force_prefill_kernel,
        attention_backend=args.attention_backend,
        flash_attn_version=args.flash_attn_version,
        quantization=args.quantization,
        world_config={"tp": args.tp, "pp": 1},
        dtype=dtype,
        db_path=args.db_path,
        gpu=args.gpu,
    )
    
    import time
    start_time = time.perf_counter()
    profiler.profile(models)
   
    # limit model to profile by one
    if models[0].collective_ops and len(models) == 1 and args.profile_comm:
        # Run comm profiling in separate process (no fake TP)
        run_comm_profiler(
            models=models,
            dtype_str=args.dtype,
            tp_size=args.tp,
            db_path=args.db_path,
            overwrite=args.overwrite_comm,
            profile_method=args.profile_method,
            use_cuda_graph=not args.no_cuda_graph_comm,  # Default True unless disabled
            comm_timing=args.comm_timing,
        )
    
    elapsed_s = time.perf_counter() - start_time    
    print(f"Profiling completed in {elapsed_s:.2f}s")

if __name__ == "__main__":
    main()
