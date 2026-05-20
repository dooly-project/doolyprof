"""
Worker module for communication profiling with KINETO support.
"""

import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def comm_profiler_worker(model_name, collective_ops, dtype_str, tp_size, db_path,
                         profile_method="kineto", overwrite=False):
    """
    Worker function for comm profiling with selectable profiling method.
    """
    # DEBUG: Verify no fake TP
    fake_tp_val = os.environ.get('VLLM_FAKE_TP', 'NOT SET')
    print(f"[WORKER] VLLM_FAKE_TP: {fake_tp_val}")

    if fake_tp_val != 'NOT SET':
        raise RuntimeError(f"VLLM_FAKE_TP is still set to {fake_tp_val}! Cannot profile comm ops with fake TP.")

    # Initialize LLM to set up distributed environment
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM

    print(f"[WORKER] Initializing LLM for TP={tp_size}...")
    llm = LLM(
        model=model_name,
        dtype=dtype_str,
        enforce_eager=True,
        load_format="dummy",  # Fast initialization without loading real weights
        gpu_memory_utilization=0.5,  # Conservative for profiling
        max_model_len=2048,  # Small model_len sufficient for comm profiling
        tensor_parallel_size=tp_size,  # REAL tensor parallelism
    )
    print(f"[WORKER] LLM initialized successfully")

    # Import the KINETO-enabled profiler
    from doolyprof.run.comm_profiler_kineto import CommProfiler, CommProfilingConfig, ProfileMethod

    DTYPE_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    # Map string to ProfileMethod
    method_map = {
        "kineto": ProfileMethod.KINETO,
        "cuda_event": ProfileMethod.CUDA_EVENT,
        "wall_clock": ProfileMethod.WALL_CLOCK,
    }

    selected_method = method_map.get(profile_method.lower(), ProfileMethod.KINETO)
    print(f"[WORKER] Using profiling method: {selected_method}")

    comm_config = CommProfilingConfig(
        max_size=128 * 1024 * 1024,
        dtype=DTYPE_MAP[dtype_str],
        warmup_iters=5,
        measure_iters=3,
        batch_size=10,  # Like Vidur
        profile_method=selected_method,
    )

    collective_backend = "NCCL"
    comm_profiler = CommProfiler(config=comm_config, db_path=db_path)
    comm_profiler.make_db()
    comm_profiler.plan(collective_ops)

    # Check if we should profile based on existing data
    if comm_profiler.should_profile(tp_size, overwrite=overwrite):
        comm_profiler.profile(model_name, dtype_str, comm_config, tp_size, collective_backend, "mew1", llm)
        comm_profiler.save_to_db(tp_size=tp_size, overwrite=overwrite)
    else:
        print(f"[COMM] Skipping profiling - all operations already have data with {selected_method} and overwrite=False")

    # Clean up LLM
    print(f"[WORKER] Cleaning up LLM...")
    del llm

if __name__ == '__main__':
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--collective-ops-pkl', required=True)
    parser.add_argument('--dtype', required=True)
    parser.add_argument('--tp-size', type=int, required=True)
    parser.add_argument('--db-path', required=True)
    parser.add_argument('--profile-method', default='kineto',
                       choices=['kineto', 'cuda_event', 'wall_clock'],
                       help="Profiling method to use")
    parser.add_argument('--overwrite', action='store_true',
                       help="Whether to overwrite existing profiling data in the database")
    args = parser.parse_args()

    # Load collective ops from pickle file
    with open(args.collective_ops_pkl, 'rb') as f:
        collective_ops = pickle.load(f)

    # Run worker
    comm_profiler_worker(args.model, collective_ops, args.dtype, args.tp_size,
                        args.db_path, profile_method=args.profile_method,
                        overwrite=args.overwrite)