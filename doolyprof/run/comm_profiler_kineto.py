"""
Communication profiler with KINETO method support (based on Vidur's approach).
"""

import os
import sqlite3
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import defaultdict
import pickle
import hashlib
import numpy as np
import torch
import torch.distributed as dist
from vllm.distributed import tensor_model_parallel_all_reduce, tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_tp_group

# Profiling configuration
WARMUP_ITERS = 5
MEASURE_ITERS = 3
BATCH_SIZE = 10  # Number of operations to run in batch (like Vidur)

class ProfileMethod:
    CUDA_EVENT = "cuda_event"
    KINETO = "kineto"
    WALL_CLOCK = "wall_clock"

@dataclass
class CommProfilingConfig:
    warmup_iters: int = 5
    measure_iters: int = 5  # 5 iterations per size
    batch_size: int = 10  # Operations per measurement
    dtype: torch.dtype = torch.bfloat16
    max_seq_len: int = 8192
    max_size: int = None
    profile_method: str = ProfileMethod.KINETO  # Default to KINETO like Vidur

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = torch.bfloat16
        if self.max_size is None:
            self.max_size = self.max_seq_len * 1024 * 1024

    def get_collective_sizes_to_profile(self) -> List[int]:
        COLLECTIVE_SIZE_SPACE = (
            list(range(1024, 512 * 1024 + 1, 4 * 1024))                                 # 1K to 512K, step 4K
            + list(range(512 * 1024, 8 * 1024 * 1024 + 1, 16 * 1024))                   # 512K to 8M, step 16K
            + list(range(8 * 1024 * 1024, 64 * 1024 * 1024 + 1, 64 * 1024))             # 8M to 64M, step 64K
            + list(range(64 * 1024 * 1024 + 1, 512 * 1024 * 1024 + 1, 256 * 1024))      # 64M to 512M, step 256K
        )
        sizes_to_profile = []
        for size in COLLECTIVE_SIZE_SPACE:
            if size <= self.max_size:
                sizes_to_profile.append(size)
            else:
                break
        return sizes_to_profile

# ============================================================================
# STANDALONE WORKER FUNCTION (called via collective_rpc)
# ============================================================================
def _run_kineto_profiling_on_worker_standalone(
    worker,
    config_dict: Dict[str, Any],
    plan_info: Dict[str, Dict[str, Any]],
    profile_method: str,
    tp_size: int,
    backend: str,
    topology: str,
) -> List[Dict]:
    """
    Standalone worker function for kineto profiling (not a class method).
    This runs inside each worker process via collective_rpc.
    """
    # Get TP group (should be initialized in worker process)
    from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
    tp_group = get_tp_group()
    tp_world_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    # Reconstruct config from dict
    dtype = config_dict['dtype']
    warmup_iters = config_dict['warmup_iters']
    measure_iters = config_dict['measure_iters']
    batch_size = config_dict['batch_size']
    max_size = config_dict['max_size']

    # Get sizes to profile
    COLLECTIVE_SIZE_SPACE = (
        list(range(1024, 512 * 1024 + 1, 4 * 1024))                                 # 1K to 512K, step 4K
        + list(range(512 * 1024, 8 * 1024 * 1024 + 1, 16 * 1024))                   # 512K to 8M, step 16K
        + list(range(8 * 1024 * 1024, 64 * 1024 * 1024 + 1, 64 * 1024))             # 8M to 64M, step 64K
        + list(range(64 * 1024 * 1024 + 1, 512 * 1024 * 1024 + 1, 256 * 1024))      # 64M to 512M, step 256K
    )
    sizes_to_profile = [size for size in COLLECTIVE_SIZE_SPACE if size <= max_size]

    device = torch.device("cuda")
    dtype_str = str(dtype).split('.')[-1]

    results = []

    # Profile all_reduce if requested
    if "all_reduce" in plan_info:
        print(f"[WORKER {tp_rank}] Profiling all_reduce with {profile_method}...")
        for size in sizes_to_profile:
            # Calculate number of elements
            DTYPE_TO_BYTES = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.int64: 8, torch.int32: 4, torch.int8: 1}
            num_elements = size // DTYPE_TO_BYTES[dtype]
            tensor = torch.randn(num_elements, dtype=dtype, device=device)

            # Warmup
            for _ in range(warmup_iters):
                _ = tp_group.all_reduce(tensor)
                torch.cuda.synchronize()

            latencies = []

            if profile_method == ProfileMethod.KINETO:
                # KINETO method with batching
                profiler = KinetoProfiler(filter_str="nccl")
                for _ in range(measure_iters):
                    latency = profiler.measure_operation(
                        lambda: tp_group.all_reduce(tensor),
                        batch_size=batch_size
                    )
                    latencies.append(latency)

            elif profile_method == ProfileMethod.CUDA_EVENT:
                # CUDA event method
                for _ in range(measure_iters):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    torch.cuda.synchronize()
                    start_event.record()

                    # Run batch_size operations
                    for _ in range(batch_size):
                        _ = tp_group.all_reduce(tensor)

                    end_event.record()
                    torch.cuda.synchronize()

                    # Get time per operation
                    gpu_time_ms = start_event.elapsed_time(end_event) / batch_size
                    latencies.append(gpu_time_ms)

            else:  # WALL_CLOCK
                # Wall clock method
                import time
                for _ in range(measure_iters):
                    torch.cuda.synchronize()
                    start = time.perf_counter()

                    # Run batch_size operations
                    for _ in range(batch_size):
                        _ = tp_group.all_reduce(tensor)

                    torch.cuda.synchronize()
                    end = time.perf_counter()

                    # Get time per operation in milliseconds
                    time_per_op_ms = ((end - start) * 1000) / batch_size
                    latencies.append(time_per_op_ms)

            # Take median of all 5 iterations (each iteration already has mean of 7 stable kernels)
            latency = np.median(latencies)

            results.append({
                "operation": "all_reduce",
                "topology": topology,
                "backend": backend,
                "tp_size": tp_size,
                "size_bytes": size,
                "dtype": dtype_str,
                "dim_param": None,
                "num_dims": 2,
                "latency_ms": latency
            })

    # Profile all_gather if requested
    if "all_gather" in plan_info:
        print(f"[WORKER {tp_rank}] Profiling all_gather with {profile_method}...")
        for size in sizes_to_profile:
            for dim_param in [-1]:  # Only profiling dim=-1 for now
                # Calculate number of elements
                DTYPE_TO_BYTES = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.int64: 8, torch.int32: 4, torch.int8: 1}
                num_elements = size // DTYPE_TO_BYTES[dtype]

                # Create tensor with proper shape for all_gather
                if dim_param == -1:
                    tensor_dim = (num_elements,)
                else:
                    tensor_dim = (1, num_elements)

                tensor = torch.randn(tensor_dim, dtype=dtype, device=device)

                # Warmup
                for _ in range(warmup_iters):
                    _ = tp_group.all_gather(tensor, dim=dim_param)
                    torch.cuda.synchronize()

                latencies = []

                if profile_method == ProfileMethod.KINETO:
                    # KINETO method with batching
                    profiler = KinetoProfiler(filter_str="nccl")
                    for _ in range(measure_iters):
                        latency = profiler.measure_operation(
                            lambda: tp_group.all_gather(tensor, dim=dim_param),
                            batch_size=batch_size
                        )
                        latencies.append(latency)

                elif profile_method == ProfileMethod.CUDA_EVENT:
                    # CUDA event method
                    for _ in range(measure_iters):
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)

                        torch.cuda.synchronize()
                        start_event.record()

                        # Run batch_size operations
                        for _ in range(batch_size):
                            _ = tp_group.all_gather(tensor, dim=dim_param)

                        end_event.record()
                        torch.cuda.synchronize()

                        # Get time per operation
                        gpu_time_ms = start_event.elapsed_time(end_event) / batch_size
                        latencies.append(gpu_time_ms)

                # Skip first 3 measurements and take median of remaining
                stable_latencies = latencies[3:] if len(latencies) > 3 else latencies
                latency = np.median(stable_latencies)

                results.append({
                    "operation": "all_gather",
                    "topology": topology,
                    "backend": backend,
                    "tp_size": tp_size,
                    "size_bytes": size,
                    "dtype": dtype_str,
                    "dim_param": dim_param,
                    "num_dims": len(tensor_dim),
                    "latency_ms": latency
                })

    return results


class KinetoProfiler:
    """Profiler using PyTorch's KINETO trace analysis."""

    def __init__(self, filter_str: str = "nccl", aggregation_fn=np.median):
        self.filter_str = filter_str
        self.aggregation_fn = aggregation_fn
        self.trace_results = []
        self.measurement_count = 0  # Track measurements for trace output

    def measure_operation(self, op_fn, batch_size: int = 1):
        """Measure an operation using KINETO profiling."""

        current_result = None  # Local variable for this call only

        def handle_trace(prof):
            nonlocal current_result
            # Export trace to temporary file and read JSON
            import tempfile
            import json
            import os

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                trace_file = f.name

            try:
                # Export the trace
                prof.export_chrome_trace(trace_file)

                # Read and parse the JSON
                with open(trace_file, 'r') as f:
                    trace_data = json.load(f)

                # Find kernel events and sum their durations
                total_kernel_time_us = 0
                kernel_count = 0
                included_kernel_count = 0

                if 'traceEvents' in trace_data:
                    events = trace_data['traceEvents']
                    kernel_times = []  # Store individual kernel durations

                    for event in events:
                        if event.get('cat') == 'kernel':
                            duration = event.get('dur', 0)  # Duration in microseconds
                            kernel_times.append(duration)
                            kernel_count += 1

                    # Skip first 3 kernel events, take mean of remaining 7
                    if len(kernel_times) >= 10:
                        stable_kernels = kernel_times[3:10]  # Take kernels 4-10 (7 kernels)
                        if len(stable_kernels) > 0:
                            median_kernel_time_us = np.median(stable_kernels)
                            time_per_op_ms = median_kernel_time_us / 1000.0  # Convert to ms
                            current_result = time_per_op_ms
                            print(f"Median of {len(stable_kernels)} kernels: {time_per_op_ms:.6f}ms")
                        else:
                            raise ValueError("Not enough stable kernel events found!")
                    else:
                        raise ValueError(f"Expected 10 kernel events but found {len(kernel_times)}!")
                else:
                    raise ValueError("No kernel events found in KINETO trace! Check if the filter is correct and if the operation is being captured.")
            finally:
                # Clean up temporary file
                if os.path.exists(trace_file):
                    os.unlink(trace_file)


        # Normal profiling without trace export
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=handle_trace,
        ) as prof:
            for _ in range(batch_size):
                op_fn()
            torch.cuda.synchronize()

        self.measurement_count += 1

        # Return the result from this measurement
        return current_result

class CommProfiler:
    def __init__(self, config: CommProfilingConfig, db_path):
        self.config = config
        self.db_path = db_path
        self.profile_method = config.profile_method

    def make_db(self):
        # create table in db
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coll_op_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                topology TEXT NOT NULL,
                backend TEXT NOT NULL,
                tp_size INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                dtype TEXT,
                dim_param INTEGER,
                num_dims INTEGER,
                latency_ms REAL NOT NULL,
                UNIQUE(operation, topology, backend, dtype, tp_size, size_bytes, dim_param, num_dims)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coll_ops (
                collective_id INTEGER PRIMARY KEY AUTOINCREMENT,
                signature_hash TEXT UNIQUE NOT NULL,
                operation_name TEXT NOT NULL,
                topology TEXT NOT NULL,
                tp_degree INTEGER NOT NULL,
                num_dims INTEGER,
                dim_param INTEGER
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_collectives (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                collective_id INTEGER NOT NULL,
                count INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (model_id) REFERENCES profiled_models(model_id),
                FOREIGN KEY (collective_id) REFERENCES coll_ops(collective_id),
                UNIQUE(model_id, collective_id)
               )
            """)

        conn.commit()
        conn.close()

        print(f"✓ Database initialized at {self.db_path}")

    def profile_all_reduce(self, tp_group, sizes, warmup_iters, measure_iters, batch_size, dtype_str):
        """Profile all_reduce operation using selected method."""
        results = []

        for size in sizes:
            # Calculate number of elements
            num_elements = size // 2  # bfloat16 is 2 bytes
            tensor = torch.randn(num_elements, dtype=self.config.dtype).cuda()

            # Warmup
            for _ in range(warmup_iters):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()

            latencies = []

            if self.profile_method == ProfileMethod.KINETO:
                # KINETO method with batching
                profiler = KinetoProfiler(filter_str="nccl")

                for _ in range(measure_iters):
                    latency = profiler.measure_operation(
                        lambda: tp_group.all_reduce(tensor),
                        batch_size=batch_size
                    )
                    latencies.append(latency)

            else:
                raise ValueError(f"Unsupported profiling method: {self.profile_method}")


            # Take median of all measurements
            latency = np.median(latencies)

            results.append({
                "operation": "all_reduce",
                "size_bytes": size,
                "dtype": dtype_str,
                "latency_ms": latency,
                "dim_param": None,
                "num_dims": 2,
            })

            print(f"all_reduce size={size:,} bytes: {latency:.3f}ms ({self.profile_method})")

        return results

    def profile_all_gather(self, tp_group, sizes, warmup_iters, measure_iters, batch_size, dtype_str):
        """Profile all_gather operation using selected method."""
        results = []

        for size in sizes:
            for dim_param in [-1]:  # Only profiling dim=-1 for now
                # Calculate number of elements
                num_elements = size // 2  # bfloat16 is 2 bytes

                # Create tensor with proper shape for all_gather
                if dim_param == -1:
                    tensor_dim = (num_elements,)
                else:
                    tensor_dim = (1, num_elements)

                tensor = torch.randn(tensor_dim, dtype=self.config.dtype).cuda()

                # Warmup
                for _ in range(warmup_iters):
                    _ = tp_group.all_gather(tensor, dim=dim_param)
                    torch.cuda.synchronize()

                latencies = []

                if self.profile_method == ProfileMethod.KINETO:
                    # KINETO method with batching
                    profiler = KinetoProfiler(filter_str="nccl")

                    for _ in range(measure_iters):
                        latency = profiler.measure_operation(
                            lambda: tp_group.all_gather(tensor, dim=dim_param),
                            batch_size=batch_size
                        )
                        latencies.append(latency)

                elif self.profile_method == ProfileMethod.CUDA_EVENT:
                    # CUDA event method
                    for _ in range(measure_iters):
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)

                        torch.cuda.synchronize()
                        start_event.record()

                        # Run batch_size operations
                        for _ in range(batch_size):
                            _ = tp_group.all_gather(tensor, dim=dim_param)

                        end_event.record()
                        torch.cuda.synchronize()

                        # Get time per operation
                        gpu_time_ms = start_event.elapsed_time(end_event) / batch_size
                        latencies.append(gpu_time_ms)

                else:  # WALL_CLOCK
                    # Wall clock method
                    for _ in range(measure_iters):
                        torch.cuda.synchronize()
                        start = time.perf_counter()

                        # Run batch_size operations
                        for _ in range(batch_size):
                            _ = tp_group.all_gather(tensor, dim=dim_param)

                        torch.cuda.synchronize()
                        end = time.perf_counter()

                        # Get time per operation in milliseconds
                        time_per_op_ms = ((end - start) * 1000) / batch_size
                        latencies.append(time_per_op_ms)

                # Skip first 3 measurements and take median of remaining
                stable_latencies = latencies[3:] if len(latencies) > 3 else latencies
                latency = np.median(stable_latencies)

                results.append({
                    "operation": "all_gather",
                    "size_bytes": size,
                    "dtype": dtype_str,
                    "latency_ms": latency,
                    "dim_param": dim_param,
                    "num_dims": 2,
                    "profile_method": self.profile_method,
                    "batch_size": batch_size,
                })

                print(f"all_gather size={size:,} bytes, dim={dim_param}: {latency:.3f}ms ({self.profile_method})")

        return results

    def plan(self, collective_ops):
        """Plan which collective operations to profile."""
        self.plan_info = {}
        for op in collective_ops:
            op_name = op.operation_name
            if op_name not in self.plan_info:
                self.plan_info[op_name] = []
            self.plan_info[op_name].append(op)

        print(f"[COMM] Planning to profile: {list(self.plan_info.keys())}")

    def should_profile(self, tp_size, overwrite=False):
        """Check if we should profile based on existing data and overwrite flag."""
        if overwrite:
            print(f"[COMM] Overwrite mode enabled - will re-profile all operations with {self.profile_method}")
            return True

        if not hasattr(self, 'plan_info') or not self.plan_info:
            return False

        # Check if any planned operations need profiling with this method
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        needs_profiling = False
        for op_name in self.plan_info.keys():
            # Check if this operation already has data
            existing_count = cursor.execute("""
                SELECT COUNT(*) FROM coll_op_results
                WHERE operation = ? AND topology = ? AND tp_size = ?
            """, (op_name, "mew1", tp_size)).fetchone()[0]

            if existing_count == 0:
                needs_profiling = True
                print(f"[COMM] {op_name} needs profiling with {self.profile_method} (no existing data)")
            else:
                print(f"[COMM] {op_name} already has {existing_count} measurements with {self.profile_method}")

        conn.close()
        return needs_profiling

    def profile(self, model_name, dtype_str, config, tp_size, backend, topology, llm):
        """Run the profiling with selected method using collective_rpc."""
        print(f"\n[COMM] Starting profiling with method: {self.profile_method}")
        print(f"[COMM] Batch size: {config.batch_size} operations per measurement")

        # Convert config to dict for serialization
        config_dict = {
            'dtype': config.dtype,
            'warmup_iters': config.warmup_iters,
            'measure_iters': config.measure_iters,
            'batch_size': config.batch_size,
            'max_size': config.max_size,
        }

        # Call standalone function (not instance method) to avoid serialization issues
        results = llm.llm_engine.collective_rpc(
            method=_run_kineto_profiling_on_worker_standalone,
            timeout=None,
            args=(config_dict, self.plan_info, self.profile_method, tp_size, backend, topology),
            kwargs=None
        )

        # collective_rpc returns results from all workers, we just need results from one worker
        # (all workers run the same profiling code and should return identical results)
        if results:
            self.results = results[0]  # Take results from first worker
        else:
            self.results = []

        print(f"\n[COMM] Profiling complete. Collected {len(self.results)} measurements with {self.profile_method}")

    def save_to_db(self, tp_size=None, backend="NCCL", topology="mew1", overwrite=False):
        """Save profiling results to database."""
        if not hasattr(self, 'results') or not self.results:
            print("[COMM] No results to save")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # If overwrite, delete existing entries with this profile method
        if overwrite:
            for result in self.results:
                cursor.execute("""
                    DELETE FROM coll_op_results
                    WHERE operation = ? AND topology = ? AND tp_size = ?
                    AND size_bytes = ? AND dim_param IS ? AND num_dims = ?
                """, (
                    result['operation'], topology, tp_size or 4,
                    result['size_bytes'], result.get('dim_param'), result.get('num_dims', 2)
                ))

        # Insert new results
        for result in self.results:
            try:
                cursor.execute("""
                    INSERT INTO coll_op_results
                    (operation, topology, backend, tp_size, size_bytes, dtype,
                     dim_param, num_dims, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result['operation'], topology, backend, tp_size or 4,
                    result['size_bytes'], result['dtype'],
                    result.get('dim_param'), result.get('num_dims', 2),
                    result['latency_ms']
                ))
            except sqlite3.IntegrityError:
                # Update existing entry
                cursor.execute("""
                    UPDATE coll_op_results
                    SET latency_ms = ?
                    WHERE operation = ? AND topology = ? AND tp_size = ?
                    AND size_bytes = ? AND dim_param IS ? AND num_dims = ?
                """, (
                    result['latency_ms'],
                    result['operation'], topology, tp_size or 4,
                    result['size_bytes'], result.get('dim_param'), result.get('num_dims', 2)
                ))

        conn.commit()
        conn.close()

        print(f"✓ Saved {len(self.results)} measurements to database with {self.profile_method} method")

if __name__ == "__main__":
    # Test script
    print("Communication Profiler with KINETO support")
    print("Use comm_worker.py to run actual profiling")