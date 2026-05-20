import os
import socket
import sqlite3
import json
import csv
import time
from datetime import datetime
from pathlib import Path
import torch
import torch.distributed as dist
import traceback
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from collections import defaultdict
from tqdm import tqdm
from vllm.distributed import parallel_state
import multiprocessing as mp

def find_free_port() -> int:
    """Find a free port by binding to port 0 and letting the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

from doolyprof.profiler.vllm_layer_profiler import VLLMLayerProfiler, make_dict 
from doolyprof.profiler.model import ModelInfo as ProfileModel
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from doolyprof.profiler.importer import OpImporter
from doolyprof.profiler.module import ModuleInfo
from doolyprof.profiler.input_generator import InputGenerator 
from doolyprof.profiler.resolver import ModuleResolver
from doolyprof.profiler.utils.generate_batch import AttentionBatchConfig


def wrap_list_args(op_packet, op_name: str, inputs: List) -> List:
    """Wrap inputs that should be lists according to PyTorch schema."""
    from torch._ops import OpOverloadPacket

    if not isinstance(op_packet, OpOverloadPacket):
        return inputs

    # Get specific overload — same best-match logic as prepare_op_args.
    overload_name = op_name.split('.')[-1] if '.' in op_name else None
    if overload_name and hasattr(op_packet, overload_name):
        op_overload = getattr(op_packet, overload_name)
    else:
        best_overload = None
        best_score = float('inf')
        for attr_name in dir(op_packet):
            try:
                candidate = getattr(op_packet, attr_name)
                if not hasattr(candidate, '_schema'):
                    continue
                positional_count = sum(
                    1 for a in candidate._schema.arguments if not a.kwarg_only
                )
                score = abs(positional_count - len(inputs))
                if score < best_score:
                    best_score = score
                    best_overload = candidate
            except Exception:
                continue
        if best_overload is None:
            return inputs
        op_overload = best_overload

    schema = op_overload._schema
    wrapped = list(inputs)

    for i, arg in enumerate(schema.arguments):
        if i >= len(wrapped):
            break
        arg_type = str(arg.type)
        # Wrap list arguments (List[...] or type[] patterns)
        if ('List[' in arg_type or '[]' in arg_type) and not isinstance(wrapped[i], list):
            wrapped[i] = [wrapped[i]]

    return wrapped


def prepare_op_args(op_packet, op_name: str, inputs: List) -> Tuple[List, Dict]:
    """
    Prepare inputs for PyTorch op call by:
    1. Wrapping inputs that should be lists according to schema
    2. Separating keyword-only arguments from positional arguments

    Returns:
        Tuple of (positional_args, keyword_args)
    """
    from torch._ops import OpOverloadPacket

    if not isinstance(op_packet, OpOverloadPacket):
        return inputs, {}

    # Get specific overload (e.g., aten::add.Tensor).
    # When the op name includes an explicit overload suffix (e.g. "aten::mean.dim"),
    # use it directly.  When there is no suffix (e.g. "aten::mean"), try every
    # available overload and pick the one whose number of positional (non-kwarg-only)
    # arguments best matches the number of inputs we have.  This prevents
    # misrouting inputs through the wrong schema (e.g. feeding [-1] into dtype).
    overload_name = op_name.split('.')[-1] if '.' in op_name else None

    if overload_name and hasattr(op_packet, overload_name):
        op_overload = getattr(op_packet, overload_name)
    else:
        # No explicit overload; score every overload by how closely its
        # positional-arg count matches len(inputs).
        best_overload = None
        best_score = float('inf')
        for attr_name in dir(op_packet):
            try:
                candidate = getattr(op_packet, attr_name)
                if not hasattr(candidate, '_schema'):
                    continue
                positional_count = sum(
                    1 for a in candidate._schema.arguments if not a.kwarg_only
                )
                score = abs(positional_count - len(inputs))
                if score < best_score:
                    best_score = score
                    best_overload = candidate
            except Exception:
                continue
        if best_overload is None:
            return inputs, {}
        op_overload = best_overload

    schema = op_overload._schema
    positional_args = []
    keyword_args = {}

    for i, arg in enumerate(schema.arguments):
        if i >= len(inputs):
            break

        value = inputs[i]
        arg_type = str(arg.type)

        # Wrap list arguments (List[...] or type[] patterns) - but not if already a list
        if ('List[' in arg_type or '[]' in arg_type) and not isinstance(value, list):
            value = [value]

        # Separate keyword-only arguments
        if arg.kwarg_only:
            keyword_args[arg.name] = value
        else:
            positional_args.append(value)

    # Fix shape-list arguments (e.g. reshape/view second arg) whose concrete
    # values captured from the trace don't match the dummy-run tensor size.
    # When the primary input is a Tensor, check each list arg that looks like
    # a shape spec (all ints, possibly with -1).  If its concrete product
    # doesn't divide the tensor's total elements evenly, replace offending
    # values with -1 so the op can infer the right size itself.
    primary_tensor = next(
        (a for a in positional_args if isinstance(a, torch.Tensor)), None
    )
    if primary_tensor is not None:
        total_elements = primary_tensor.numel()
        for idx, val in enumerate(positional_args):
            if not isinstance(val, list):
                continue
            if not all(isinstance(v, int) for v in val):
                continue
            concrete_vals = [v for v in val if v != -1]
            if not concrete_vals:
                continue
            product = 1
            for v in concrete_vals:
                product *= abs(v)
            if product == 0 or total_elements % product != 0:
                # Replace positive concrete values that are incompatible with -1
                positional_args[idx] = [-1 if v > 0 else v for v in val]

    return positional_args, keyword_args


class Profiler:
    def __init__(
        self,
        output_path: str,
        workload_mode: str = "vidur",
        attention_backend: str = "TRITON_ATTN",
        flash_attn_version: int = 2,
        quantization: Optional[str] = None,
        max_batch_size: int = 5,
        max_seq_len: int = 512,
        test_counts: int = 10,
        force_prefill_kernel: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        world_config: Optional[Dict[str, int]] = None,
        db_path: Optional[str] = None,
        gpu: str = "0",
    ):
        self.importer = OpImporter()
        self.module_caller = None
        self.output_path = output_path
        self.db_path = db_path
        self.world_config = world_config
        self.tp_config = world_config.get("tp", 1)
        self.pp_config = world_config.get("pp", 1)
        self.gpu = gpu

        # Initialize database if db_path provided
        if self.db_path:
            self._init_db()

        # Per-operation wall-time log (CSV, one row per profiled module).
        # When running parallel shards across GPUs, each shard sets
        # DOOLY_TIMING_SHARD (e.g. "gpu0") so the file name is unique and
        # workers don't race on the same CSV.
        shard = os.environ.get("DOOLY_TIMING_SHARD", "").strip()
        suffix = f"_{shard}" if shard else ""
        self.timing_log_path: Optional[str] = None
        if self.db_path:
            self.timing_log_path = str(
                Path(self.db_path).with_name(f"profile_timings{suffix}.csv")
            )
        else:
            self.timing_log_path = str(
                Path(output_path).parent / f"profile_timings{suffix}.csv"
            )
        self._init_timing_log()

        self.dtype = dtype

        self.workload_mode = workload_mode
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.attention_backend = attention_backend
        self.flash_attn_version = flash_attn_version
        self.quantization = quantization

        output_dir = str(Path(output_path).parent)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.attention_config = {
            "batch_size": max_batch_size,
            "mode": workload_mode,
            "test_counts": test_counts,
            "output_dir": output_dir,
            "use_decode_kernel": False,
            "req_size": max_seq_len,
            "quiet_mode": True,
            "force_prefill_kernel": force_prefill_kernel,
            "num_runs": 5,
            "decode_repeat": 10,
            "random_pages": True,
        }

        self.attention_profiler: Optional[VLLMLayerProfiler] = None
        self._global_kv_cache: Optional[torch.Tensor] = None
        self._global_kv_cache_spec: Optional[Tuple[int, int, int, int]] = None

    def _profile_with_batched_cuda_graph(
        self,
        run_fn: Callable,
        warmup_runs: int = 5,
        profile_runs: int = 20,
        iterations_per_graph: int = 10
    ) -> List[float]:
        """
        Profile using batched CUDA graph approach for minimal CPU overhead.

        Captures N consecutive kernel launches in a single CUDA graph,
        then measures the batch execution time and divides by N.

        This eliminates:
        - CPU loop overhead between iterations
        - graph.replay() call overhead (only 1 call per batch)
        - GPU idle time between operations

        Args:
            run_fn: Callable that executes the operation to profile
            warmup_runs: Number of warmup iterations
            profile_runs: Total number of profiling measurements desired
            iterations_per_graph: Number of operations to batch in graph

        Returns:
            List of per-operation timing measurements in milliseconds
        """
        # Warmup
        for _ in range(warmup_runs):
            run_fn()
        torch.cuda.synchronize()

        # Capture the graph with N consecutive kernel launches
        graph = torch.cuda.CUDAGraph()

        try:
            with torch.cuda.graph(graph):
                # Capture N operations back-to-back (no events inside graph)
                for _ in range(iterations_per_graph):
                    run_fn()
        except Exception as e:
            # CUDA graph capture failed - fall back to standard profiling
            print(f"[PROFILER] CUDA Graph capture failed ({e}), using standard profiling")
            return self._profile_standard(run_fn, warmup_runs, profile_runs)

        # Measure the graph replay time (events OUTSIDE graph)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        batch_timings = []
        num_replays = max(1, profile_runs // iterations_per_graph)

        for _ in range(num_replays):
            start_event.record()
            graph.replay()  # Replays all N operations at once
            end_event.record()
            torch.cuda.synchronize()

            batch_time = start_event.elapsed_time(end_event)
            batch_timings.append(batch_time)

        # Compute per-operation timings
        # Each batch_time is for iterations_per_graph operations
        per_op_timings = [t / iterations_per_graph for t in batch_timings]

        return per_op_timings

    def _profile_standard(
        self,
        run_fn: Callable,
        warmup_runs: int = 5,
        profile_runs: int = 20
    ) -> List[float]:
        """
        Standard profiling with CUDA events (fallback method).

        Sequential event recording with sync after each iteration.
        Used when CUDA graphs fail or are not applicable.
        """
        # Warmup
        for _ in range(warmup_runs):
            run_fn()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        run_results = []
        for _ in range(profile_runs):
            start_event.record()
            run_fn()
            end_event.record()
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
            run_results.append(latency_ms)

        return run_results

    def _profile_gpu_busy(
        self,
        run_fn: Callable,
        warmup_runs: int = 5,
        profile_runs: int = 20
    ) -> List[float]:
        """
        GPU busy profiling method for minimal CPU overhead and accurate GPU timing.

        This method keeps the GPU busy with background work while the CPU queues
        up all event recordings in advance. This eliminates CPU scheduling delays
        between kernel launches and provides more accurate GPU-only timing.

        Benefits:
        - Eliminates CPU scheduling overhead between iterations
        - GPU stays busy, avoiding idle time between kernels
        - More accurate timing for fast operations
        - Reduces measurement artifacts from CPU-GPU synchronization

        Args:
            run_fn: Callable that executes the operation to profile
            warmup_runs: Number of warmup iterations
            profile_runs: Number of profiling measurements

        Returns:
            List of timing measurements in milliseconds
        """
        # Warmup
        for _ in range(warmup_runs):
            run_fn()
        torch.cuda.synchronize()

        # Create background work tensors (large enough to keep GPU busy)
        busy_tensor1 = torch.randn(4096, 4096, dtype=torch.float32, device=torch.device("cuda:0"))
        busy_tensor2 = torch.randn(4096, 4096, dtype=torch.float32, device=torch.device("cuda:0"))

        # Create all events upfront
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(profile_runs)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(profile_runs)]

        # Start GPU busy work (non-blocking on CPU)
        busy_event = torch.cuda.Event()
        for _ in range(3):  # Keep GPU busy with background matrix multiplications
            torch.matmul(busy_tensor1, busy_tensor2)
        busy_event.record()

        # While GPU is busy, CPU queues up all the event recordings
        for i in range(profile_runs):
            start_events[i].record()
            run_fn()
            end_events[i].record()

        # Wait for everything to complete
        torch.cuda.synchronize()

        # Collect all timings
        timings = []
        for i in range(profile_runs):
            timings.append(start_events[i].elapsed_time(end_events[i]))

        return timings

    def _init_db(self) -> None:
        """Initialize SQLite database with required tables."""
        # Create parent directory if needed
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            Path(db_dir).mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signatures (
                operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                signature_hash TEXT UNIQUE NOT NULL,
                operation_name TEXT NOT NULL,
                signature_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signature_hash TEXT NOT NULL,
                mean_latency_ms REAL,
                median_latency_ms REAL,
                max_latency_ms REAL,
                min_latency_ms REAL,
                std_latency_ms REAL,
                num_tokens INTEGER,
                num_requests INTEGER,
                is_context BOOLEAN NULL,
                is_prefill BOOLEAN NULL,
                kv_cache_size INTEGER NULL,
                prefill_chunk_size INTEGER NULL,
                num_tensor_parallel_workers INTEGER,
                input_shapes TEXT, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signature_hash) REFERENCES signatures(signature_hash)
            )
        """)
        # Composite index covering the upsert WHERE-clause in _write_results.
        # Without this, the per-row "does this exact (sig, workload_params) row
        # already exist?" SELECT degrades to a full scan, making the write
        # phase O(N²) in result-row count. With it: O(N log N).
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_lookup
            ON results (signature_hash, num_tokens, num_requests,
                        is_context, is_prefill, kv_cache_size,
                        prefill_chunk_size, num_tensor_parallel_workers)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiled_models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_family TEXT NOT NULL,
                backend TEXT NOT NULL,
                tp_degree INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Enforce uniqueness on the (model, backend, tp, dtype) tuple. Must be
        # an INDEX rather than an inline UNIQUE constraint so it can be added
        # to existing DBs without recreating the table; existing rows missing
        # a dtype value are backfilled from a one-shot ALTER TABLE migration.
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS ix_profiled_models_unique
                ON profiled_models(model_name, backend, tp_degree, dtype)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                operation_id INTEGER NOT NULL,
                count INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES profiled_models(model_id),
                FOREIGN KEY (operation_id) REFERENCES signatures(operation_id),
                UNIQUE(model_id, operation_id, count)
            )
        """)
        conn.execute("""
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
        conn.execute("""
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_collective_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                collective_id INTEGER NOT NULL,
                batch_param TEXT NOT NULL,
                model_config_dim INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES profiled_models(model_id),
                FOREIGN KEY (collective_id) REFERENCES coll_ops(collective_id),
                UNIQUE(model_id, collective_id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_collective_configs_lookup
            ON model_collective_configs(model_id, collective_id)
        """)
        conn.execute("""
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
        conn.commit()
        conn.close()
        print(f"[PROFILER] Initialized database at {self.db_path}")

    # ------------------------------------------------------------------
    # Per-operation timing log
    # ------------------------------------------------------------------

    TIMING_LOG_FIELDS = (
        "timestamp", "model_name", "backend", "tp", "phase",
        "operation_name", "module_name", "signature_hash",
        "n_configs", "n_results", "wall_time_s",
    )

    def _init_timing_log(self) -> None:
        """Ensure the timing-log CSV exists with a header."""
        if not self.timing_log_path:
            return
        path = Path(self.timing_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.TIMING_LOG_FIELDS)

    def _append_timing(
        self,
        model_name: str,
        phase: str,
        operation_name: str,
        module_name: Optional[str],
        signature_hash: Optional[str],
        n_configs: int,
        n_results: int,
        wall_time_s: float,
    ) -> None:
        """Append one timing row. Safe to call from parent or subprocess."""
        if not self.timing_log_path:
            return
        row = [
            datetime.utcnow().isoformat(timespec="seconds"),
            model_name,
            self.attention_backend,
            self.tp_config,
            phase,
            operation_name,
            module_name or "",
            signature_hash or "",
            n_configs,
            n_results,
            f"{wall_time_s:.4f}",
        ]
        with open(self.timing_log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    # main profiling function
    def profile(self, models_to_profile: List[ProfileModel]) -> None:
        results, attention_results, attention_kernel_infos = self._profile_models(models_to_profile)
    
        if self.db_path:
            self.write_to_models_db(models_to_profile)
            self.write_to_results_db(results)
        else:
            self.write_csv(results, attention_results, attention_kernel_infos)

    def _profile_models(self, models: List[ProfileModel]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        all_results: List[Dict[str, Any]] = []
        attention_results: List[Dict[str, Any]] = []
        attention_kernel_infos: List[Dict[str, Any]] = []

        result_queue = mp.Queue()

        for model in models:
            modules = model.to_profile or []
            if not modules:
                continue
            print(f"\n[PROFILER] Profiling model: {model.name}")

            # Use minimum of user-specified max_seq_len and model's max_model_len
            # This respects user input as upper bound while also respecting model limits
            # (following Vidur's approach)
            model_max = model.get_max_model_len()
            if model_max:
                self.max_model_len = min(self.max_seq_len, model_max)
                print(f"[PROFILER] Using max_model_len={self.max_model_len} (min of user's {self.max_seq_len} and model's {model_max})")
            else:
                self.max_model_len = self.max_seq_len
                print(f"[PROFILER] Using max_model_len={self.max_model_len} from user config")

            print(f"\n[PROFILER] Profiling at TP={self.tp_config}")

            # Partition modules into raw ops vs vLLM modules
            # to_profile_resolved contains 4-tuples: (module, callable, sig_hash, sig_json)
            raw_ops = [(m, c, h, j) for m, c, h, j in model.to_profile_resolved if not m.is_module]
            vllm_modules = [(m, c, h, j) for m, c, h, j in model.to_profile_resolved if m.is_module]

            # Phase 1: Profile raw ops in main process (no vLLM needed - maximum GPU memory)
            if raw_ops:
                print(f"\n[PROFILER] Phase 1: Profiling {len(raw_ops)} raw ops (no vLLM)")
                raw_results = self._profile_raw_ops(model, raw_ops)
                all_results.extend(raw_results)

            # Phase 2: Profile vLLM modules in subprocess (needs vLLM context)
            if vllm_modules:
                print(f"\n[PROFILER] Phase 2: Profiling {len(vllm_modules)} vLLM modules")
                # Debug: Check VLLM_FAKE_TP before spawning subprocess
                import os
                fake_tp_parent = os.environ.get("VLLM_FAKE_TP", "Not Set")
                print(f"[PROFILER PARENT] VLLM_FAKE_TP before subprocess: {fake_tp_parent}")

                p = mp.Process(
                    target=self._profile_vllm_modules_subprocess,
                    args=(model, vllm_modules, result_queue)
                )
                p.start()

                # Get results BEFORE join to avoid deadlock
                vllm_results = result_queue.get()
                all_results.extend(vllm_results)

                p.join()

                if p.exitcode != 0:
                    print(f"[PROFILER] WARNING: Subprocess exited with code {p.exitcode}")

            print("="*20)

        return all_results, attention_results, attention_kernel_infos

    def _profile_raw_ops(
        self,
        model: ProfileModel,
        raw_ops: List[Tuple[ModuleInfo, Callable, str, str]]
    ) -> List[Dict[str, Any]]:
        """Profile raw ops (is_module=False) in main process without vLLM.

        Raw ops don't need VLLMLayerProfiler since they just call PyTorch/vLLM
        ops directly via OpImporter. This gives them maximum GPU memory.

        Note: Callables are None from resolution phase (run-profiler.py line 89)
        because they can't cross process boundaries. We re-import them here.
        """
        all_results = []

        pbar = tqdm(raw_ops, desc=f"Profiling raw ops (TP={self.tp_config})", unit="op")

        for i, (module, _, sig_hash, sig_json) in enumerate(pbar):
            pbar.set_description(f"Raw ops [{i+1}/{len(raw_ops)}]")
            tqdm.write(f"[{i+1}/{len(raw_ops)}] Profiling raw op: {module.operation_name} for {module.module_name}")

            # Import the callable (resolution phase sets it to None)
            callable_obj, msg = self.importer.import_op(op_full_name=module.operation_name)
            if callable_obj is None:
                tqdm.write(f"  Could not import: {msg}")
                continue

            torch.cuda.empty_cache()

            _t_start = time.perf_counter()
            module_results = self._profile_single_raw_op(
                module, model, callable_obj,
                max_model_len=self.max_model_len,
                sig_hash=sig_hash,
                sig_json=sig_json,
            )
            _wall_s = time.perf_counter() - _t_start

            n_results = len(module_results) if module_results else 0
            self._append_timing(
                model_name=model.name,
                phase="raw_op",
                operation_name=module.operation_name,
                module_name=module.module_name,
                signature_hash=sig_hash,
                n_configs=n_results,
                n_results=n_results,
                wall_time_s=_wall_s,
            )

            if module_results:
                all_results.extend(module_results)
                tqdm.write(f"  Profiled {len(module_results)} configs in {_wall_s:.2f}s")

        print(f"[PROFILER] Finished profiling {len(all_results)} raw op configs")
        return all_results

    def _profile_vllm_modules_subprocess(
        self,
        model: ProfileModel,
        vllm_modules: List[Tuple[ModuleInfo, Callable, str, str]],
        results: mp.Queue
    ):
        """Profile vLLM modules (is_module=True) with full vLLM context.

        This runs in a subprocess to ensure GPU memory is fully freed when done.
        """
        # Debug: Check if VLLM_FAKE_TP is set in subprocess
        import os
        fake_tp = os.environ.get("VLLM_FAKE_TP", "Not Set")
        print(f"\n[PROFILER SUBPROCESS] VLLM_FAKE_TP env var: {fake_tp}")
        print(f"[PROFILER SUBPROCESS] World config TP: {self.world_config.get('tp', 1)}")

        # Initialize VLLMLayerProfiler
        self.module_caller = VLLMLayerProfiler(
            model_name=model.name,
            dtype=str(self.dtype).split(".")[-1],
            enforce_eager=True,
            gpu_memory_utilization=0.9,
            max_model_len=self.max_model_len,
            attn_backend=self.attention_backend,
            flash_attn_version=self.flash_attn_version,
            quantization=self.quantization,
            gpu=self.gpu,
        )

        # Re-resolve callables with loaded model. Pass the taint_registry
        # that was captured at trace time (stored on ModelInfo by TraceParser)
        # so the resolver can recover NUM_REQS when the module's direct
        # tensor inputs don't expose that dim — matches what
        # run-profiler.py does at Phase 1.
        resolver = ModuleResolver(
            vlp=self.module_caller,
            importer=self.importer,
            dtype=self.dtype,
            taint_registry=getattr(model, 'taint_registry', {}) or {},
        )

        resolved_modules = []
        for module, _, sig_hash, sig_json in vllm_modules:
            callable_obj = resolver.try_import(module)
            resolved_modules.append((module, callable_obj, sig_hash, sig_json))

        all_results = []

        pbar = tqdm(resolved_modules, desc=f"Profiling modules (TP={self.tp_config})", unit="module")

        for i, (module, callable_obj, sig_hash, sig_json) in enumerate(pbar):
            pbar.set_description(f"Modules [{i+1}/{len(resolved_modules)}]")
            tqdm.write(f"[{i+1}/{len(resolved_modules)}] Profiling {module.module_name} - ({module.operation_name})")

            torch.cuda.empty_cache()

            _t_start = time.perf_counter()
            module_results = self._profile_single_module_for_model(
                module, model, callable_obj,
                max_model_len=self.max_model_len,
                sig_hash=sig_hash,
                sig_json=sig_json,
            )
            _wall_s = time.perf_counter() - _t_start

            n_results = len(module_results) if module_results else 0
            self._append_timing(
                model_name=model.name,
                phase="vllm_module",
                operation_name=module.operation_name,
                module_name=module.module_name,
                signature_hash=sig_hash,
                n_configs=n_results,
                n_results=n_results,
                wall_time_s=_wall_s,
            )

            if module_results:
                all_results.extend(module_results)
                tqdm.write(f"  Profiled {len(module_results)} configs in {_wall_s:.2f}s")

        self.module_caller.close()

        # Put all results at once to avoid queue deadlock
        results.put(all_results)
        print(f"[PROFILER] Finished profiling {len(all_results)} module configs")

    def _profile_single_raw_op(
        self,
        module: ModuleInfo,
        model: ProfileModel,
        callable_obj: Callable,
        max_model_len: Optional[int] = None,
        sig_hash: Optional[str] = None,
        sig_json: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Profile a single raw op (is_module=False) without vLLM context."""
        try:
            limit = max_model_len if max_model_len is not None else self.max_seq_len

            generator = InputGenerator(
                module=module,
                max_num_request=self.max_batch_size,
                max_num_token=limit,
                test_counts=self.attention_config["test_counts"],
                dtype=self.dtype,
            )

            workload_configs = generator.prepare_inputs(
                mode=self.workload_mode,
                use_batch_config=False  # Raw ops don't use batch config
            )

            # Sort by size descending so large configs fail early if OOM
            def get_config_size(config):
                return config.get('num_tokens', 0) * config.get('num_requests', 1)
            workload_configs = sorted(workload_configs, key=get_config_size, reverse=True)

            results = []
            pbar_workload_configs = tqdm(
                workload_configs,
                desc=f"  Configs for {module.operation_name[:30]}",
                unit="config",
                leave=False,
            )

            for workload_config in pbar_workload_configs:
                # Update progress bar with current config info
                pbar_workload_configs.set_description(
                    f"  Configs for {module.operation_name[:30]} "
                    f"(tokens: {workload_config.get('num_tokens', '?')}, reqs: {workload_config.get('num_requests', '?')})"
                )

                inputs, workload_params = generator._create_tensor_inputs_from_params(workload_config)

                if not inputs:
                    continue

                args, kwargs = prepare_op_args(callable_obj, module.operation_name, inputs)


                @torch.inference_mode()
                def run_callable():
                    # Reconstruct kwargs from arg_name if present (for modules called with keyword args)
                    has_named = any(
                        getattr(pi, 'arg_name', None) is not None
                        for pi in module.inputs
                    )
                    if has_named and module.is_module:
                        # Re-split: positional inputs have no arg_name
                        positional_tensors = [
                            tensor for pi, tensor in zip(module.inputs, inputs)
                            if getattr(pi, 'arg_name', None) is None
                        ]
                        kwarg_tensors = {
                            pi.arg_name: tensor
                            for pi, tensor in zip(module.inputs, inputs)
                            if getattr(pi, 'arg_name', None) is not None
                        }
                        callable_obj(*positional_tensors, **kwarg_tensors)
                    else:
                        callable_obj(*args, **kwargs)

                try:
                    # use batched CUDA graph profiling for minimal CPU overhead
                    run_results = self._profile_with_batched_cuda_graph(
                        run_fn=run_callable,
                        warmup_runs=5,
                        profile_runs=20,
                        iterations_per_graph=10
                    )

                    # Use original inputs for shape tracking (not args which may have list wrapping)
                    all_inputs = inputs
                    result = {
                        "signature_hash": sig_hash,
                        "signature_json": sig_json,  # for signatures table only
                        "module_name": module.module_name,
                        "operation_name": module.operation_name,
                        "mean_latency_ms": sum(run_results) / len(run_results),
                        "median_latency_ms": sorted(run_results)[len(run_results)//2],
                        "max_latency_ms": max(run_results),
                        "min_latency_ms": min(run_results),
                        "std_latency_ms": (sum((x - sum(run_results)/len(run_results))**2 for x in run_results) / len(run_results))**0.5,
                        "input_shapes": [list(i.shape) if hasattr(i, "shape") else str(i) for i in all_inputs],
                        "workload_params": workload_params,
                        "num_tokens": workload_params.get('num_tokens'),
                        "num_requests": workload_params.get('num_requests'),
                        "num_tensor_parallel_workers": self.tp_config,
                    }
                    results.append(result)

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" in str(e).lower():
                        print(f"[PROFILER] OOM for {module.operation_name}, skipping...")
                    else:
                        print(f"[PROFILER] Error profiling {module.operation_name}: {e}")
                    torch.cuda.empty_cache()
                    continue

                del inputs, args, kwargs
                torch.cuda.empty_cache()

            pbar_workload_configs.close()
            return results if results else None

        except Exception as e:
            print(f"[PROFILER] Error profiling {module.operation_name}: {e}")
            return None

    def _profile_single_module_for_model(
        self,
        module: ModuleInfo,
        model: ProfileModel,
        callable_obj: Callable,
        # model_params: Dict[str, Any],
        max_model_len: Optional[int] = None,
        sig_hash: Optional[str] = None,
        sig_json: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        try:
            # Adjust model params for tensor parallelism
            num_kv_heads = model.num_kv_heads
            head_dim = model.head_dim

            # Use max_model_len as the limit for token counts
            # (Vidur also limits to model's max_position_embeddings, not batch * seq_len)
            limit = max_model_len if max_model_len is not None else self.max_seq_len

            layer_name = None
            if module.is_module and self.module_caller is not None:
                name_to_module = make_dict(self.module_caller.model)
                # Prefer the resolver's kernel-matched path when present.
                # Without this, get_layer_name_by_module defaults to paths[0]
                # (e.g. layer 0) for every Attention variant, so the second
                # variant's submodule at layer 1 is profiled with layer 0's
                # forward_context → KeyError on self.layer_name lookup.
                resolved_path = getattr(module, '_resolved_path', None)
                if resolved_path is not None:
                    layer_name = self.module_caller.get_layer_name_by_module(
                        module.operation_name, {module.operation_name: [resolved_path]}
                    ) or resolved_path
                else:
                    layer_name = self.module_caller.get_layer_name_by_module(module.operation_name, name_to_module)

            generator = InputGenerator(
                module=module,
                max_num_request=self.max_batch_size,
                max_num_token=limit,
                test_counts=self.attention_config["test_counts"],
                dtype=self.dtype,
            )

            # Let the InputGenerator classify the op and pick the right sweep.
            # Previously forced use_batch_config=True for any registered layer,
            # which over-swept MoE (FusedMoE / SharedFusedMoE) on kv_cache_size
            # / prefill_chunk_size axes that don't affect MoE latency.
            workload_configs = generator.prepare_inputs(mode=self.workload_mode)

            # Sort workload configs by size descending so large configs fail early if OOM
            def get_config_size(config):
                if isinstance(config, AttentionBatchConfig):
                    return config.batch_size * (config.prefill_chunk_size + config.kv_cache_size)
                else:
                    return config.get('num_tokens', 0) * config.get('num_requests', 1)

            workload_configs = sorted(workload_configs, key=get_config_size, reverse=True)

            results = []
            config_idx = 0
            pbar_workload_configs = tqdm(
                workload_configs,
                desc=f"  Configs for {module.module_name[:25]}",
                unit="config",
                leave=False,
            )

            for workload_config in pbar_workload_configs:
                # workload_config is AttentionBatchConfig for attention, Dict[str, int] for non-attention
                if isinstance(workload_config, AttentionBatchConfig):
                    pbar_workload_configs.set_description(
                        f"  Configs for {module.module_name[:25]} "
                        f"(Batch: {workload_config.batch_size}, Prefill: {workload_config.prefill_chunk_size}, KV: {workload_config.kv_cache_size})"
                    )
                else:
                    pbar_workload_configs.set_description(
                        f"  Configs for {module.module_name[:25]} "
                        f"(tokens: {workload_config.get('num_tokens', '?')}, reqs: {workload_config.get('num_requests', '?')})"
                    )

                # _create_tensor_inputs_from_params now accepts AttentionBatchConfig or Dict[str, int]
                # and returns (inputs, workload_params)
                inputs, workload_params = generator._create_tensor_inputs_from_params(workload_config)

                if not inputs:
                    # Invalid config, skip
                    continue

                # Prepare inputs: wrap list args and separate keyword-only args
                args, kwargs = prepare_op_args(callable_obj, module.operation_name, inputs)

                forward_context = None
                slot_mapping_dict = None
                num_tokens_for_context = None

                if layer_name and isinstance(workload_config, AttentionBatchConfig):
                    batch_config = AttentionBatchConfig(
                        batch_size=workload_config.batch_size,
                        prefill_chunk_size=workload_config.prefill_chunk_size,
                        kv_cache_size=workload_config.kv_cache_size,
                        is_prefill=workload_config.is_prefill,
                    )
                    try:
                        result = self.module_caller.build_layer_metadata(layer_name, batch_config)
                        if result is None:
                            # Skip this config - exceeds available KV cache blocks
                            continue
                        layer_metadata, _ = result
                        forward_context = {layer_name: layer_metadata}

                        # Get slot_mapping from CommonAttentionMetadata for KV cache updates
                        common_meta = self.module_caller._build_common_attention_metadata(batch_config, kv_cache_gid=0)
                        if common_meta is not None:
                            slot_mapping_dict = {layer_name: common_meta.slot_mapping}
                            num_tokens_for_context = common_meta.num_actual_tokens
                    except Exception as e:
                        # Metadata build failed — fall back to an empty forward_context
                        # so vLLM's invariant holds (any module forward expects a
                        # context set). Matches resolver.py behaviour.
                        print(f"[PROFILER] Warning: Could not build metadata (ok for non-attention ops): {e}")
                        forward_context = {layer_name: None}
                        slot_mapping_dict = None
                        num_tokens_for_context = None
                elif layer_name:
                    # Registered layer with a non-AttentionBatchConfig sweep (e.g.
                    # MoE using the 1-D num_tokens dict path). Metadata isn't
                    # applicable, but the module still calls get_forward_context()
                    # internally and will raise without one. Provide an empty
                    # context so the kernel can dispatch.
                    forward_context = {layer_name: None}

                @torch.inference_mode()
                def run_callable():
                    # Reconstruct kwargs from arg_name if present (for modules called with keyword args)
                    has_named = any(
                        getattr(pi, 'arg_name', None) is not None
                        for pi in module.inputs
                    )
                    if has_named and module.is_module:
                        # Re-split: positional inputs have no arg_name
                        positional_tensors = [
                            tensor for pi, tensor in zip(module.inputs, inputs)
                            if getattr(pi, 'arg_name', None) is None
                        ]
                        kwarg_tensors = {
                            pi.arg_name: tensor
                            for pi, tensor in zip(module.inputs, inputs)
                            if getattr(pi, 'arg_name', None) is not None
                        }
                        callable_obj(*positional_tensors, **kwarg_tensors)
                    else:
                        callable_obj(*args, **kwargs)

                @torch.inference_mode()
                def run_callable_with_context():
                    # Reconstruct kwargs from arg_name if present (for modules called with keyword args)
                    has_named = any(
                        getattr(pi, 'arg_name', None) is not None
                        for pi in module.inputs
                    )
                    if has_named and module.is_module:
                        # Re-split: positional inputs have no arg_name
                        positional_tensors = [
                            tensor for pi, tensor in zip(module.inputs, inputs)
                            if getattr(pi, 'arg_name', None) is None
                        ]
                        kwarg_tensors = {
                            pi.arg_name: tensor
                            for pi, tensor in zip(module.inputs, inputs)
                            if getattr(pi, 'arg_name', None) is not None
                        }
                        with set_current_vllm_config(self.module_caller.vllm_config):
                            with set_forward_context(
                                forward_context,
                                self.module_caller.vllm_config,
                                virtual_engine=0,
                                num_tokens=num_tokens_for_context,
                                slot_mapping=slot_mapping_dict
                            ):
                                callable_obj(*positional_tensors, **kwarg_tensors)
                    else:
                        with set_current_vllm_config(self.module_caller.vllm_config):
                            with set_forward_context(
                                forward_context,
                                self.module_caller.vllm_config,
                                virtual_engine=0,
                                num_tokens=num_tokens_for_context,
                                slot_mapping=slot_mapping_dict
                            ):
                                callable_obj(*args, **kwargs)

                run_fn = run_callable_with_context if forward_context else run_callable

                try:
                    # For Attention modules, use gpu_busy method for accurate timing
                    if "Attention" in module.operation_name:
                        run_results = self._profile_gpu_busy(
                            run_fn=run_fn,
                            warmup_runs=5,
                            profile_runs=20
                        )
                    else:
                        # Use batched CUDA graph profiling for other operations
                        run_results = self._profile_with_batched_cuda_graph(
                            run_fn=run_fn,
                            warmup_runs=5,
                            profile_runs=20,
                            iterations_per_graph=10
                        )

                    # Use original inputs for shape tracking (before prepare_op_args wraps list args)
                    all_inputs = inputs
                    result = {
                        "signature_hash": sig_hash,
                        "signature_json": sig_json,  # for signatures table only
                        "module_name": module.module_name,
                        "operation_name": module.operation_name,
                        "num_tensor_parallel_workers": self.tp_config,
                        "mean_latency_ms": sum(run_results) / len(run_results),
                        "median_latency_ms": sorted(run_results)[len(run_results)//2],
                        "max_latency_ms": max(run_results),
                        "min_latency_ms": min(run_results),
                        "std_latency_ms": (sum((x - sum(run_results)/len(run_results))**2 for x in run_results) / len(run_results))**0.5,
                        "input_shapes": [
                            list(i.shape) if hasattr(i, "shape") else str(i) for i in all_inputs
                        ],
                        "workload_params": workload_params,
                        "num_tokens": workload_params.get('num_tokens'),
                        "num_requests": workload_params.get('num_requests'),
                    }
                    results.append(result)

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" in str(e).lower() or "CUDA" in str(e):
                        print(f"[PROFILER] OOM for {module.module_name} with config {workload_params}, skipping...")
                    else:
                        print(f"[PROFILER] Error for {module.module_name}: {e}")
                    # Clean up and continue
                    torch.cuda.empty_cache()
                    continue

                # Clean up after successful profiling
                del inputs, args, kwargs
                torch.cuda.empty_cache()
                config_idx += 1

            pbar_workload_configs.close()

            if results:
                print(f"[PROFILER] Successfully profiled {module.module_name}.")
                return results

            return None

        except Exception as e:
            print(f"[PROFILER] Error profiling {module.module_name}: {e}")
            traceback.print_exc()
            return None
        
    def write_csv(self, results: List[Dict[str, Any]], attention_results: List[Dict[str, Any]], attention_kernel_infos: List[Dict[str, Any]]) -> None:
        with open(self.output_path, "w") as f:
            f.write(
                "module_name,operation_name,num_tensor_parallel_workers,"
                "num_tokens,num_requests,"
                "mean_latency_ms,median_latency_ms,max_latency_ms,min_latency_ms,std_latency_ms,"
                "input_shapes,workload_params\n"
            )
            for result in results:
                workload_params_str = str(result.get("workload_params", {}))
                num_tokens = result.get("num_tokens", "")
                num_requests = result.get("num_requests", "")
                f.write(
                    f"{result['module_name']},{result['operation_name']},{self.tp_config},"
                    f"{num_tokens},{num_requests},"
                    f"{result['mean_latency_ms']:.4f},{result['median_latency_ms']:.4f},"
                    f"{result['max_latency_ms']:.4f},{result['min_latency_ms']:.4f},{result['std_latency_ms']:.4f},"
                    f"\"{result['input_shapes']}\",\"{workload_params_str}\"\n"
                )
        if attention_results:
            self.attention_profiler._export_estimator_features(
                attention_results,
                attention_kernel_infos,
                csv_filename=os.path.join(self.attention_config["output_dir"], "attention_profile.csv"),
            )
        print(f"[PROFILER] Profiling complete. Results saved to {self.output_path}")

    def write_to_models_db(self, models: List[ProfileModel]) -> None:
        """Write model names to SQLite database."""
        if not self.db_path:
            print("[PROFILER] No database path specified, skipping models database write")
            return

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("BEGIN")

            # dtype is stored on profiled_models so multi-dtype experiments
            # don't ghost-collide on the (model, backend, tp) tuple.
            dtype_str = str(self.dtype).split(".")[-1]
            for model in models:
                # Check if this model configuration already exists
                cursor = conn.execute(
                    """SELECT model_id FROM profiled_models
                       WHERE model_name = ? AND model_family = ? AND backend = ? AND tp_degree = ? AND dtype = ?""",
                    (model.name, model.model_family, self.attention_backend, self.tp_config, dtype_str)
                )
                existing = cursor.fetchone()

                if existing:
                    # Model already exists, use existing model_id
                    model_id = existing[0]
                    print(f"[PROFILER] Model {model.name} (backend={self.attention_backend}, tp={self.tp_config}, dtype={dtype_str}) already exists (model_id={model_id})")
                else:
                    # Insert new model and get the auto-generated model_id
                    cursor = conn.execute(
                        """INSERT INTO profiled_models (model_name, model_family, backend, tp_degree, dtype) VALUES (?, ?, ?, ?, ?)""",
                        (model.name, model.model_family, self.attention_backend, self.tp_config, dtype_str)
                    )
                    model_id = cursor.lastrowid
                    print(f"[PROFILER] Created new model entry (model_id={model_id}, dtype={dtype_str})")

                # Use all_resolved_ops which contains ALL operations this model uses
                # (not just the ones that need profiling).
                #
                # Multiple all_resolved_ops entries may share the same sig_hash
                # when trace-dedup over-splits by sibling-kernel context
                # (e.g. two GemmaRMSNorm "bands" whose inner aten::mul kernel
                # is identical; the comparator collapses them back to one
                # sig_hash for cpu_op anchors). Sum their counts before the
                # INSERT loop so model_operations reflects the true total
                # instead of the last iteration's count overwriting earlier
                # ones. Module-level anchors whose sig_hash already
                # distinguishes variants (e.g. SWA-split Attention) stay as
                # separate keys here and still land in their own rows.
                counts_per_sig: Dict[str, int] = defaultdict(int)
                templates_per_sig: Dict[str, Tuple[Any, str]] = {}
                for module, callable_obj, sig_hash, sig_json in model.all_resolved_ops:
                    if not sig_hash:
                        continue
                    counts_per_sig[sig_hash] += getattr(module, 'count', 1)
                    if sig_hash not in templates_per_sig:
                        templates_per_sig[sig_hash] = (module, sig_json)

                for sig_hash, total_count in counts_per_sig.items():
                    module, sig_json = templates_per_sig[sig_hash]

                    cursor = conn.execute(
                        "SELECT operation_id FROM signatures WHERE signature_hash = ?",
                        (sig_hash,)
                    )
                    existing = cursor.fetchone()

                    if existing:
                        operation_id = existing[0]
                    else:
                        cursor = conn.execute(
                            """INSERT INTO signatures (signature_hash, operation_name, signature_json)
                               VALUES (?, ?, ?)""",
                            (sig_hash, module.operation_name, sig_json)
                        )
                        operation_id = cursor.lastrowid

                    existing = conn.execute(
                        """SELECT id FROM model_operations
                           WHERE model_id = ? AND operation_id = ?""",
                        (model_id, operation_id)
                    ).fetchone()

                    if existing:
                        conn.execute(
                            """UPDATE model_operations
                               SET count = ?
                               WHERE model_id = ? AND operation_id = ?""",
                            (total_count, model_id, operation_id)
                        )
                    else:
                        conn.execute(
                            """INSERT INTO model_operations (model_id, operation_id, count)
                               VALUES (?, ?, ?)""",
                            (model_id, operation_id, total_count)
                        )

                # Write collective operations
                import hashlib
                topology = os.environ.get("DOOLY_TOPOLOGY", "mew1")  # Get from environment or default

                # Aggregate counts for each unique collective operation signature
                # Key: (operation_name, dim_param, num_dims), Value: total_count
                collective_counts = defaultdict(int)
                collective_metadata = {}  # Store first occurrence metadata

                for collective_op in model.collective_ops:
                    # Extract dim_param from trace
                    dim_param_str = collective_op.hierarchy.anchor.args.get('comm_args')
                    dim_param = int(dim_param_str) if dim_param_str else None

                    # Get num_dims from tensor shape
                    tensor_shape = collective_op.inputs[0].actual_shape
                    num_dims = len(tensor_shape)

                    # Create key for aggregation
                    key = (collective_op.operation_name, dim_param, num_dims)

                    # Aggregate count
                    count = getattr(collective_op, 'count', 1)
                    collective_counts[key] += count

                    # Store metadata from first occurrence
                    if key not in collective_metadata:
                        # Extract MODEL_CONFIG dimension
                        model_config_idxs = collective_op.inputs[0].get_model_config_dims()
                        model_config_dim = 1
                        for idx in model_config_idxs:
                            model_config_dim *= tensor_shape[idx]

                        # Determine batch parameter based on tensor taints
                        # NUM_TOKS typically appears in first dimension for all_reduce
                        # NUM_REQS typically appears in first dimension for all_gather
                        tensor_taints = collective_op.inputs[0].dimensions
                        batch_param = "NUM_TOKS" if "NUM_TOKS" in str(tensor_taints) else "NUM_REQS"

                        collective_metadata[key] = {
                            'operation_name': collective_op.operation_name,
                            'dim_param': dim_param,
                            'num_dims': num_dims,
                            'model_config_dim': model_config_dim,
                            'batch_param': batch_param
                        }

                # Now insert aggregated collectives
                for key, total_count in collective_counts.items():
                    op_name, dim_param, num_dims = key

                    # Generate signature (MUST match comm_profiler logic)
                    sig_data = f"{op_name}|{topology}|{self.tp_config}|{dim_param}|{num_dims}"
                    sig_hash = hashlib.sha256(sig_data.encode()).hexdigest()

                    # Check if collective exists in coll_ops
                    cursor = conn.execute(
                        "SELECT collective_id FROM coll_ops WHERE signature_hash = ?",
                        (sig_hash,)
                    )
                    existing = cursor.fetchone()

                    if existing:
                        collective_id = existing[0]
                    else:
                        # Insert into coll_ops (comm_profiler will populate profiling data later)
                        cursor = conn.execute("""
                            INSERT INTO coll_ops
                            (signature_hash, operation_name, topology, tp_degree, dim_param, num_dims)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (sig_hash, op_name, topology, self.tp_config, dim_param, num_dims))
                        collective_id = cursor.lastrowid

                    # Link model to collective with AGGREGATED count
                    # Check if exists, then update or insert
                    existing = conn.execute("""
                        SELECT id FROM model_collectives
                        WHERE model_id = ? AND collective_id = ?
                    """, (model_id, collective_id)).fetchone()

                    if existing:
                        conn.execute("""
                            UPDATE model_collectives
                            SET count = ?
                            WHERE model_id = ? AND collective_id = ?
                        """, (total_count, model_id, collective_id))
                    else:
                        conn.execute("""
                            INSERT INTO model_collectives (model_id, collective_id, count)
                            VALUES (?, ?, ?)
                        """, (model_id, collective_id, total_count))

                    # Store model-specific collective configuration
                    # Check if exists, then update or insert
                    metadata = collective_metadata[key]
                    existing = conn.execute("""
                        SELECT id FROM model_collective_configs
                        WHERE model_id = ? AND collective_id = ?
                    """, (model_id, collective_id)).fetchone()

                    if existing:
                        conn.execute("""
                            UPDATE model_collective_configs
                            SET batch_param = ?, model_config_dim = ?
                            WHERE model_id = ? AND collective_id = ?
                        """, (metadata['batch_param'], metadata['model_config_dim'], model_id, collective_id))
                    else:
                        conn.execute("""
                            INSERT INTO model_collective_configs
                            (model_id, collective_id, batch_param, model_config_dim)
                            VALUES (?, ?, ?, ?)
                        """, (model_id, collective_id, metadata['batch_param'], metadata['model_config_dim']))

                if collective_counts:
                    print(f"[PROFILER] Wrote {len(collective_counts)} unique collective operations for {model.name}:")
                    for key, total_count in collective_counts.items():
                        op_name, dim_param, num_dims = key
                        print(f"  - {op_name} (dim={dim_param}, ndims={num_dims}): count={total_count}")

            conn.execute("COMMIT")
            print(f"[PROFILER] Wrote {len(models)} models to database at {self.db_path}")

        except Exception as e:
            conn.execute("ROLLBACK")
            print(f"[PROFILER] Error writing models to database: {e}")
            raise
        finally:
            conn.close()

    def write_to_results_db(self, results: List[Dict[str, Any]]) -> None:
        """Write profiling results to SQLite database."""
        if not self.db_path:
            print("[PROFILER] No database path specified, skipping database write")
            return

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("BEGIN")

            for result in results:
                sig_hash = result.get('signature_hash')
                if not sig_hash:
                    continue

                # Check if signature exists before inserting (to avoid incrementing autoincrement)
                cursor = conn.execute(
                    "SELECT operation_id FROM signatures WHERE signature_hash = ?",
                    (sig_hash,)
                )
                if not cursor.fetchone():
                    # Only insert if it doesn't exist
                    conn.execute(
                        """INSERT INTO signatures (signature_hash, operation_name, signature_json)
                           VALUES (?, ?, ?)""",
                        (sig_hash, result['operation_name'], result.get('signature_json'))
                    )

                workload_params = result.get('workload_params', {})
                num_tokens = workload_params.get('num_tokens')
                num_requests = workload_params.get('num_requests')
                is_prefill = workload_params.get('is_prefill', None)
                prefill_chunk_size = workload_params.get('prefill_chunk_size', None)
                kv_cache_size = workload_params.get('kv_cache_size', None)
                is_context = is_prefill is not None and kv_cache_size is not None and prefill_chunk_size is not None

                # Check if this exact configuration already exists
                tp_workers = result.get('num_tensor_parallel_workers', self.tp_config)
                existing = conn.execute(
                    """SELECT id FROM results
                       WHERE signature_hash = ?
                       AND (num_tokens = ? OR (num_tokens IS NULL AND ? IS NULL))
                       AND (num_requests = ? OR (num_requests IS NULL AND ? IS NULL))
                       AND (is_context = ? OR (is_context IS NULL AND ? IS NULL))
                       AND (is_prefill = ? OR (is_prefill IS NULL AND ? IS NULL))
                       AND (kv_cache_size = ? OR (kv_cache_size IS NULL AND ? IS NULL))
                       AND (prefill_chunk_size = ? OR (prefill_chunk_size IS NULL AND ? IS NULL))
                       AND num_tensor_parallel_workers = ?""",
                    (sig_hash,
                     num_tokens, num_tokens,
                     num_requests, num_requests,
                     is_context, is_context,
                     is_prefill, is_prefill,
                     kv_cache_size, kv_cache_size,
                     prefill_chunk_size, prefill_chunk_size,
                     tp_workers)
                ).fetchone()

                if existing:
                    # Update existing result with new measurements
                    conn.execute(
                        """UPDATE results
                           SET mean_latency_ms = ?, median_latency_ms = ?,
                               max_latency_ms = ?, min_latency_ms = ?, std_latency_ms = ?,
                               input_shapes = ?, created_at = CURRENT_TIMESTAMP
                           WHERE id = ?""",
                        (
                            result.get('mean_latency_ms'),
                            result.get('median_latency_ms'),
                            result.get('max_latency_ms'),
                            result.get('min_latency_ms'),
                            result.get('std_latency_ms'),
                            json.dumps(result.get('input_shapes', [])),
                            existing[0]
                        )
                    )
                else:
                    # Insert new result
                    conn.execute(
                        """INSERT INTO results (
                            signature_hash,
                            mean_latency_ms, median_latency_ms, max_latency_ms, min_latency_ms, std_latency_ms,
                            num_tokens, num_requests,
                            is_context, is_prefill, kv_cache_size, prefill_chunk_size,
                            num_tensor_parallel_workers,
                            input_shapes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            sig_hash,
                            result.get('mean_latency_ms'),
                            result.get('median_latency_ms'),
                            result.get('max_latency_ms'),
                            result.get('min_latency_ms'),
                            result.get('std_latency_ms'),
                            num_tokens,
                            num_requests,
                            is_context,
                            is_prefill,
                            kv_cache_size,
                            prefill_chunk_size,
                            tp_workers,
                            json.dumps(result.get('input_shapes', [])),
                        )
                    )

            conn.execute("COMMIT")
            print(f"[PROFILER] Wrote {len(results)} results to database at {self.db_path}")

        except Exception as e:
            conn.execute("ROLLBACK")
            print(f"[PROFILER] Error writing to database: {e}")
            raise
        finally:
            conn.close()

    def _init_vllm_distributed_environment(self) -> None:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["VLLM_USE_V1"] = "0"

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")

        parallel_state.init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="env://",
        )

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    def _get_module_callable(self, module: ModuleInfo) -> Optional[Any]:
        callable_obj = None
        
        if module.is_module:
            name_to_module = make_dict(self.module_caller.model)
            anchor_name = module.anchor.name
            lookup_name = anchor_name
            instance_idx = 0
            if '_' in anchor_name and anchor_name.rsplit('_', 1)[-1].isdigit():
                lookup_name = anchor_name.rsplit('_', 1)[0]
                instance_idx = int(anchor_name.rsplit('_', 1)[1])
            paths = name_to_module.get(lookup_name)
            callable_path = paths[min(instance_idx, len(paths) - 1)] if paths else None

            if callable_path:
                print(f"[PROFILER] Module path: {callable_path} (from anchor '{anchor_name}' -> lookup '{lookup_name}[{instance_idx}]')")
                callable_obj = self.module_caller.model.get_submodule(callable_path)
            else:
                print(f"[PROFILER] Module not found. Anchor: '{anchor_name}', Lookup: '{lookup_name}', Keys: {list(name_to_module.keys())}")
                return None
        
        else:
            callable_obj, _ = self.importer.import_op(op_full_name=module.operation_name)

            if callable_obj is None:
                print(f"[PROFILER] Could not import module: {module.module_name} for operation {module.operation_name}")
                return None

        return callable_obj
