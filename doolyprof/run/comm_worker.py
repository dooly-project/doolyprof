"""
Worker module for communication profiling.
CRITICAL: This module must NEVER be imported by run-profiler.py at module level,
and must NOT set VLLM_FAKE_TP or import any modules that set it.
"""

import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def comm_profiler_worker(model_name, collective_ops, dtype_str, tp_size, db_path, overwrite=False, profile_send_recv=False, comm_timing="isolated"):
    """
    Worker function for comm profiling in spawned process.
    Runs with REAL TP (no fake TP patches).

    comm_timing: "isolated" (default, per-call synced) or "amortized"
    (back-to-back between two events, single sync). Additive option.
    """
    # DEBUG: Verify no fake TP
    fake_tp_val = os.environ.get('VLLM_FAKE_TP', 'NOT SET')
    print(f"[WORKER] VLLM_FAKE_TP: {fake_tp_val}")

    if fake_tp_val != 'NOT SET':
        raise RuntimeError(f"VLLM_FAKE_TP is still set to {fake_tp_val}! Cannot profile comm ops with fake TP.")

    # Import comm_profiler (which doesn't set VLLM_FAKE_TP)
    from doolyprof.run.comm_profiler import CommProfiler, CommProfilingConfig

    DTYPE_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    comm_config = CommProfilingConfig(
        max_size=128 * 1024 * 1024,
        dtype=DTYPE_MAP[dtype_str],
        warmup_iters=10,
        measure_iters=10,
        comm_timing=comm_timing,
    )

    collective_backend = "NCCL"
    comm_profiler = CommProfiler(config=comm_config, db_path=db_path)
    comm_profiler.make_db()
    comm_profiler.plan(collective_ops)
    if profile_send_recv:
        comm_profiler.profile_send_recv_bool = True

    # Check if we should profile based on existing data
    if comm_profiler.should_profile(tp_size, overwrite=overwrite) or profile_send_recv:
        comm_profiler.profile(model_name, dtype_str, comm_config, tp_size, collective_backend, "mew1")
        comm_profiler.save_to_db(overwrite=overwrite)
    else:
        print("[COMM] Skipping profiling - all operations already have data and overwrite=False")

if __name__ == '__main__':
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--collective-ops-pkl', required=True)
    parser.add_argument('--dtype', required=True)
    parser.add_argument('--tp-size', type=int, required=True)
    parser.add_argument('--db-path', required=True)
    parser.add_argument('--overwrite', action='store_true', help="Whether to overwrite existing profiling data in the database")
    parser.add_argument('--profile-send-recv', action='store_true', help="Profile 2-rank point-to-point send/recv (PP stage boundary)")
    parser.add_argument('--comm-timing', choices=['isolated', 'amortized'], default='isolated',
                        help="Collective timing mode: 'isolated' (default, per-call synced) or "
                             "'amortized' (back-to-back between two events, single sync).")
    args = parser.parse_args()

    # Load collective ops from pickle file
    with open(args.collective_ops_pkl, 'rb') as f:
        collective_ops = pickle.load(f)

    # Run worker
    comm_profiler_worker(args.model, collective_ops, args.dtype, args.tp_size, args.db_path, overwrite=args.overwrite, profile_send_recv=args.profile_send_recv, comm_timing=args.comm_timing)