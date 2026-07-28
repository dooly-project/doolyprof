import argparse
import os
import sys
import torch
import time
import numpy as np
from typing import List, Dict, Tuple, Any, Literal
import sqlite3
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM
from vllm.distributed.parallel_state import get_tp_group

# Dtype to bytes mapping
DTYPE_TO_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int8: 1,
}


def _time_collective(op_fn, warmup_iters: int, measure_iters: int,
                     comm_timing: str = "isolated") -> float:
    """Time a collective ``op_fn()`` (no-arg callable) and return latency in ms.

    Two modes (default preserves the original per-call ISOLATED behavior):

    - ``"isolated"``: each iteration does sync -> record start -> ONE collective
      -> record end -> sync; returns the MEDIAN per-call GPU time. Every call
      pays full NCCL kernel startup-from-idle, which overestimates the cost of
      collectives that actually run back-to-back inside the model forward.

    - ``"amortized"``: record start -> run ``measure_iters`` collectives
      back-to-back -> record end -> ONE sync at the end; returns total / iters.
      Kernels queue back-to-back (no per-call idle gap), closer to how the model
      forward issues them. Additive experiment-only option.
    """
    # Warmup (identical for both modes)
    for _ in range(warmup_iters):
        op_fn()
        torch.cuda.synchronize()

    if comm_timing == "amortized":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(measure_iters):
            op_fn()
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / measure_iters

    # Default: isolated per-call timing, median across iters.
    latencies = []
    for _ in range(measure_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        op_fn()
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    return float(np.median(latencies))


# ============================================================================
# STANDALONE WORKER FUNCTION (called via collective_rpc)
# ============================================================================
def _run_profiling_on_worker_standalone(
    worker,
    config_dict: Dict[str, Any],  # CommProfilingConfig as dict
    plan_info: Dict[str, Dict[str, Any]],
    profile_flags: Dict[str, bool],
    tp_size: int,
    backend: str,
    topology: str,
) -> List[Dict]:
    """
    Standalone worker function for profiling (not a class method).
    This runs inside each worker process via collective_rpc.
    """
    # Get TP group (should be initialized in worker process)
    from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
    tp_group = get_tp_group()
    tp_world_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    device = torch.device("cuda")

    # Reconstruct config from dict
    dtype = config_dict['dtype']
    warmup_iters = config_dict['warmup_iters']
    measure_iters = config_dict['measure_iters']
    max_size = config_dict['max_size']
    # Timing mode: "isolated" (default, per-call) or "amortized" (back-to-back).
    comm_timing = config_dict.get('comm_timing', 'isolated')
    print(f"[DEBUG Worker] comm_timing mode = {comm_timing}")

    # DEBUG: Check TP setup
    print(f"[DEBUG Worker] TP rank={tp_rank}/{tp_world_size}, dtype={dtype}")

    # Generate sweep sizes
    COLLECTIVE_SIZE_SPACE = (
        list(range(1024, 512 * 1024 + 1, 4 * 1024))
        + list(range(512 * 1024, 8 * 1024 * 1024 + 1, 16 * 1024))
        + list(range(8 * 1024 * 1024, 64 * 1024 * 1024 + 1, 64 * 1024))
        + list(range(64 * 1024 * 1024 + 1, 512 * 1024 * 1024 + 1, 256 * 1024))
    )
    sizes = [s for s in COLLECTIVE_SIZE_SPACE if s <= max_size]

    results = []
    dtype_str = str(dtype).split('.')[-1]

    # Profile all_reduce
    if profile_flags.get('all_reduce', False):
        print("Profiling all reduce...")

        for size in sizes:
            try:
                # Create test tensor
                tensor = torch.randn(size, dtype=dtype, device=device)
                size_bytes = tensor.numel() * DTYPE_TO_BYTES[dtype]

                latency = _time_collective(
                    lambda: tp_group.all_reduce(tensor),
                    warmup_iters, measure_iters, comm_timing,
                )

                results.append({
                    'operation': 'all_reduce',
                    'topology': topology,
                    'backend': backend,
                    'tp_size': tp_size,
                    'size_bytes': size_bytes,
                    'dtype': dtype_str,
                    'dim_param': plan_info['all_reduce']['dim_param'],
                    # 'num_dims': len([size]),
                    'num_dims': plan_info['all_reduce']['dim_count'],
                    'latency_ms': latency
                })
            except Exception as e:
                raise ValueError(f"Error profiling all_reduce (size={size}): {e}")

        print(f"✓ Completed all_reduce profiling")

    # Profile all_gather
    if profile_flags.get('all_gather', False):
        print("Profiling all gather...")
        dim_param = plan_info['all_gather']['dim_param']
        original_shape = plan_info['all_gather']['tensor_shape']
        tensor_taints = plan_info['all_gather']['tensor_taints']
        model_config_dims = plan_info['all_gather']['model_config_dims']

        skipped_count = 0
        for size in sizes:
            workload = size // model_config_dims
            if workload == 0:
                skipped_count += 1
                continue

            # Reconstruct tensor shape
            tensor_dim = []
            for dim_size, taint in zip(original_shape, tensor_taints):
                taint_str = str(taint)
                is_variable_dim = 'NUM_TOK' in taint_str or 'NUM_REQUEST' in taint_str
                tensor_dim.append(workload if is_variable_dim else dim_size)
            tensor_dim = tuple(tensor_dim)

            try:
                # Create test tensor
                tensor = torch.randn(tensor_dim, dtype=dtype, device=device)
                size_bytes = tensor.numel() * DTYPE_TO_BYTES[dtype]

                latency = _time_collective(
                    lambda: tp_group.all_gather(tensor, dim=dim_param),
                    warmup_iters, measure_iters, comm_timing,
                )

                results.append({
                    'operation': 'all_gather',
                    'topology': topology,
                    'backend': backend,
                    'tp_size': tp_size,
                    'size_bytes': size_bytes,
                    'dtype': dtype_str,
                    'dim_param': dim_param,
                    'num_dims': len(tensor_dim),
                    'latency_ms': latency
                })
            except Exception as e:
                raise ValueError(f"Error profiling all_gather (size={size}): {e}")

        print(f"✓ Completed all_gather profiling (skipped {skipped_count} sizes)")

    # Profile reduce_scatter
    if profile_flags.get('reduce_scatter', False):
        print("Profiling reduce_scatter...")
        dim_param = plan_info['reduce_scatter']['dim_param']
        original_shape = plan_info['reduce_scatter']['tensor_shape']
        tensor_taints = plan_info['reduce_scatter']['tensor_taints']
        model_config_dims = plan_info['reduce_scatter']['model_config_dims']

        for size in sizes:
            workload = size // model_config_dims
            if workload == 0:
                continue

            # Reconstruct tensor shape
            tensor_dim = []
            for dim_size, taint in zip(original_shape, tensor_taints):
                taint_str = str(taint)
                is_variable_dim = 'NUM_TOK' in taint_str or 'NUM_REQUEST' in taint_str
                tensor_dim.append(workload if is_variable_dim else dim_size)
            tensor_dim = tuple(tensor_dim)

            try:
                # Create test tensor
                tensor = torch.randn(tensor_dim, dtype=dtype, device=device)
                size_bytes = tensor.numel() * DTYPE_TO_BYTES[dtype]

                latency = _time_collective(
                    lambda: tp_group.reduce_scatter(tensor, dim=dim_param),
                    warmup_iters, measure_iters, comm_timing,
                )

                dim0 = tensor_dim[0] if len(tensor_dim) > 0 else None
                dim1 = tensor_dim[1] if len(tensor_dim) > 1 else None

                results.append({
                    'operation': 'reduce_scatter',
                    'topology': topology,
                    'backend': backend,
                    'tp_size': tp_size,
                    'size_bytes': size_bytes,
                    'dtype': dtype_str,
                    'dim_param': dim_param,
                    'num_dims': len(tensor_dim),
                    'latency_ms': latency
                })
            except Exception as e:
                raise ValueError(f"Error profiling reduce_scatter (size={size}): {e}")

        print(f"✓ Completed reduce_scatter profiling")

    # Profile send/recv (pipeline-parallel stage boundary), through vLLM's group
    # coordinator -- same path as all_reduce above, identical to what PP pays.
    if profile_flags.get('send_recv', False):
        if tp_world_size != 2:
            print(f"[WARN] Skipping send_recv: requires exactly 2 ranks, got {tp_world_size}")
        else:
            print("Profiling send_recv...")
            for size in sizes:
                try:
                    tensor = torch.randn(size, dtype=dtype, device=device)
                    size_bytes = tensor.numel() * DTYPE_TO_BYTES[dtype]

                    def _p2p():
                        # dst/src are LOCAL ranks within the 2-rank group
                        if tp_rank == 0:
                            tp_group.send(tensor, dst=1)
                        else:
                            tp_group.recv(tensor.size(), dtype, src=0)

                    latency = _time_collective(
                        _p2p, warmup_iters, measure_iters, comm_timing,
                    )

                    results.append({
                        'operation': 'send_recv',
                        'topology': topology,
                        'backend': backend,
                        'tp_size': tp_size,
                        'size_bytes': size_bytes,
                        'dtype': dtype_str,
                        'dim_param': None,
                        'num_dims': 1,
                        'latency_ms': latency
                    })
                except Exception as e:
                    raise ValueError(f"Error profiling send_recv (size={size}): {e}")

            print(f"✓ Completed send_recv profiling")

    # Profile all-to-all (expert-parallel dispatch/combine). The group coordinator has
    # no all_to_all method, so use torch.distributed.all_to_all_single on the group's
    # NCCL process group. Keyed by per-rank message size, like the other collectives.
    if profile_flags.get('all_to_all', False):
        if tp_world_size < 2:
            print(f"[WARN] Skipping all_to_all: requires >= 2 ranks, got {tp_world_size}")
        else:
            import torch.distributed as dist
            print("Profiling all_to_all...")
            group = tp_group.device_group
            for size in sizes:
                try:
                    # all_to_all_single requires the buffer be divisible by world_size
                    n = (size // tp_world_size) * tp_world_size
                    if n == 0:
                        continue
                    inp = torch.randn(n, dtype=dtype, device=device)
                    out = torch.empty_like(inp)
                    size_bytes = inp.numel() * DTYPE_TO_BYTES[dtype]

                    def _a2a():
                        dist.all_to_all_single(out, inp, group=group)

                    latency = _time_collective(
                        _a2a, warmup_iters, measure_iters, comm_timing,
                    )

                    results.append({
                        'operation': 'all_to_all',
                        'topology': topology,
                        'backend': backend,
                        'tp_size': tp_size,
                        'size_bytes': size_bytes,
                        'dtype': dtype_str,
                        'dim_param': None,
                        'num_dims': 1,
                        'latency_ms': latency
                    })
                except Exception as e:
                    raise ValueError(f"Error profiling all_to_all (size={size}): {e}")

            print(f"✓ Completed all_to_all profiling")

    return results


@dataclass
class CommProfilingConfig:
    """Configuration for communication profiling sweep (following Vidur's approach)"""
    dtype: torch.dtype = torch.bfloat16
    warmup_iters: int = 10
    measure_iters: int = 10
    max_req_count: int = 256
    max_seq_len: int = 8192
    max_size: int = max_seq_len * 1024 * 1024
    # Timing mode for collectives: "isolated" (default, per-call synced median)
    # or "amortized" (measure_iters back-to-back between two events, single
    # sync, divided by iters). Additive experiment-only option.
    comm_timing: str = "isolated"
    
    def __post_init__(self):
        if self.dtype is None:
            self.dtype = torch.bfloat16

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

class CommProfiler:
    def __init__(self, config: CommProfilingConfig, db_path):
        self.config = config
        self.db_path = db_path

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

    def should_profile(self, tp_size, overwrite=False):
        """Check if we should profile based on existing data and overwrite flag."""
        if overwrite:
            print("[COMM] Overwrite mode enabled - will re-profile all operations")
            return True

        if not hasattr(self, 'plan_info') or not self.plan_info:
            return False

        # Check if any planned operations need profiling
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
                print(f"[COMM] {op_name} needs profiling (no existing data)")
            else:
                print(f"[COMM] {op_name} already profiled ({existing_count} measurements), skipping")

        conn.close()
        return needs_profiling

    def plan(self, collectives: List[Any]):
        from doolyprof.profiler.finput import ProfileInput

        # extract input shape, dim0, dim1
        self.plan_info: Dict[str, Dict[str, Any]]= {}
        
        for collective in collectives:
            tensors: List[ProfileInput] = collective.inputs
            tensor_shape: List[Any] = [tensor.actual_shape for tensor in tensors]
            tensor_taints: List[Any] = [tensor.dimensions for tensor in tensors]
            args_str = collective.hierarchy.anchor.args.get('comm_args', None)

            # Parse comm_args (stored as string like "-1" or "0")
            dim_param = None
            if args_str is not None:
                try:
                    dim_param = int(args_str)
                except ValueError:
                    print(f"Warning: Could not parse comm_args '{args_str}' as integer")

            # collective operations only have single tensor input
            model_config_idxs = tensors[0].get_model_config_dims()
            model_dims = 1
            for model_config_idx in model_config_idxs:
                model_dims *= tensor_shape[0][model_config_idx]

            print(model_dims)

            self.plan_info[collective.operation_name] = {
                'tensor_shape': tensor_shape[0],
                'tensor_taints': tensor_taints[0],
                'taint_to_shape': dict(zip(tensor_taints[0], tensor_shape[0])),
                'model_config_dims': model_dims,
                'dim_count': len(tensor_shape[0]),
                'dim_param': dim_param  # Store as integer, not string
            }
            print([tensor.get_model_config_dims() for tensor in tensors])
            
        self.profile_all_reduce_bool: bool = True if self.plan_info.get('all_reduce', None) else False
        self.profile_all_gather_bool: bool = True if self.plan_info.get('all_gather', None) else False
        self.profile_reduce_scatter_bool: bool = True if self.plan_info.get('reduce_scatter', None) else False
        self.profile_send_recv_bool: bool = False  # enabled by the worker for PP profiling

    def profile(
        self,
        model: str,
        dtype: Literal["auto", "half", "float16", "bfloat16", "float", "float32"],
        config: CommProfilingConfig,
        tp_size: int,
        backend: str,
        topology: str
    ) -> List[Dict]:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        llm = LLM(
            model=model,
            dtype=dtype,  # Model dtype for initialization
            enforce_eager=True,
            load_format="dummy",  # Fast initialization without loading real weights
            gpu_memory_utilization=0.5,  # Conservative for profiling
            max_model_len=2048,  # Small model_len sufficient for comm profiling
            tensor_parallel_size=tp_size,  # REAL tensor parallelism
        )

        print(f"\n=== Communication Profiling Sweep ===")
        print(f"Model: {model}")
        print(f"TP Size: {tp_size}")
        print(f"Backend: {backend}")
        print(f"Topology: {topology}")
        print(f"Max size: {config.max_size} elements")

        # Generate sweep sizes for display
        sizes = config.get_collective_sizes_to_profile()
        print(f"Size range: {sizes[0]} to {sizes[-1]} elements")
        print(f"Example sizes: {sizes[:5]}... {sizes[-5:]}")
        print()

        # Convert config to dict for serialization across RPC boundary
        config_dict = {
            'dtype': config.dtype,
            'warmup_iters': config.warmup_iters,
            'measure_iters': config.measure_iters,
            'max_size': config.max_size,
            'comm_timing': getattr(config, 'comm_timing', 'isolated'),
        }

        # Prepare profile flags
        profile_flags = {
            'all_reduce': self.profile_all_reduce_bool,
            'all_gather': self.profile_all_gather_bool,
            'reduce_scatter': self.profile_reduce_scatter_bool,
            'send_recv': getattr(self, 'profile_send_recv_bool', False),
        }

        # Call standalone function (not instance method) to avoid serialization issues
        results = llm.llm_engine.collective_rpc(
            method=_run_profiling_on_worker_standalone,
            timeout=None,
            args=(config_dict, self.plan_info, profile_flags, tp_size, backend, topology),
            kwargs=None
        )

        # collective_rpc returns results from all workers, we just need results from one worker
        # (all workers run the same profiling code and should return identical results)
        if results:
            self.results = results[0]  # Take results from first worker
            
        print(f"✓ Completed {len(self.results)} profiling measurements")

    def save_to_db(self, overwrite=True):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        unique_ops = set()

        for result in self.results:
            # Check if this configuration already exists
            existing = cursor.execute("""
                SELECT id FROM coll_op_results
                WHERE operation = ? AND topology = ? AND backend = ?
                AND tp_size = ? AND size_bytes = ? AND dtype = ?
                AND dim_param = ? AND num_dims = ?
            """, (
                result['operation'], result['topology'], result['backend'],
                result['tp_size'], result['size_bytes'], result['dtype'],
                result['dim_param'], result['num_dims']
            )).fetchone()

            if existing and overwrite:
                # Update existing result
                cursor.execute("""
                    UPDATE coll_op_results
                    SET latency_ms = ?
                    WHERE id = ?
                """, (result['latency_ms'], existing[0]))
            elif not existing:
                # Insert new result
                cursor.execute("""
                    INSERT INTO coll_op_results
                    (operation, topology, backend, tp_size, size_bytes, dtype, dim_param, num_dims, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                result['operation'],
                result['topology'],
                result['backend'],
                result['tp_size'],
                result['size_bytes'],
                result['dtype'],
                result['dim_param'],
                result['num_dims'],
                result['latency_ms']
            ))
            
            # track unique collective operations for coll_ops table
            unique_ops.add((
                result['operation'],
                result['topology'], 
                result['tp_size'],
                result['dim_param'],
                result['num_dims']
            ))
        
        # insert unique collective operations into coll_ops table
        for op_name, topology, tp_degree, dim_param, num_dims in unique_ops:
            # Generate signature
            import hashlib
            sig_data = f"{op_name}|{topology}|{tp_degree}|{dim_param}|{num_dims}"
            sig_hash = hashlib.sha256(sig_data.encode()).hexdigest()
            
            cursor.execute("""
                INSERT OR IGNORE INTO coll_ops 
                (signature_hash, operation_name, topology, tp_degree, dim_param, num_dims)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (sig_hash, op_name, topology, tp_degree, dim_param, num_dims))
        
        conn.commit()
        conn.close()