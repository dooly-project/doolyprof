"""
Fake Tensor Parallelism (TP) for single-GPU profiling.

Patches vLLM's parallel_state, distributed communication, and linear layer
modules so that models behave as if running under TP > 1, while actually
executing on a single GPU.

IMPORTANT: This module must be imported BEFORE any vLLM modules so that
all cached references to parallel_state functions see the patched versions.

Usage:
    # In your entry-point script, set the env var before importing anything else:
    os.environ["VLLM_FAKE_TP"] = "2"

    # Then import this module to apply patches:
    from doolyprof.profiler.utils.fake_tp import apply_fake_tp_patches, get_fake_tp_size
"""

import os
import torch
from torch.profiler import record_function
from doolyprof.tracer.tensor import TaintedTensor
from doolyprof.tracer.types import TaintedInt, TaintedShape

def format_tensor_shape(tensor):
    if isinstance(tensor, TaintedTensor):
        return tensor.taint_str_with_history()
    elif isinstance(tensor, torch.Tensor):
        dims = [f"?({d})" for d in tensor.shape]
        return "[" + ", ".join(dims) + "]"
    return None

def get_fake_tp_size() -> int:
    """Return the fake TP size from the VLLM_FAKE_TP environment variable."""
    return int(os.environ.get("VLLM_FAKE_TP", "1"))

def apply_fake_tp_patches(tp_size: int) -> None:
    """
    Patch vLLM internals to simulate tensor parallelism on a single GPU.

    Patches three areas:
    1. parallel_state — world size, rank, and TP group
    2. distributed — all-reduce/all-gather become no-ops
    3. linear layers — world size and rank for column/row parallel

    Args:
        tp_size: The simulated tensor parallel world size (e.g. 2, 4, 8).
    """
    import vllm.distributed.parallel_state as ps
    from vllm import distributed
    from vllm.model_executor.layers import linear
    from vllm.distributed import communication_op

    # --- Fake TP group ---------------------------------------------------
    class FakeTPGroup:
        _COMM_OPS = {'all_reduce', 'all_gather', 'reduce_scatter', 'gather', 'broadcast_tensor_dict'}

        world_size = tp_size
        rank_in_group = 0
        local_rank = 0
        rank = 0
        device = torch.device("cuda")
        cpu_group = None
        device_group = None

        def __getattribute__(self, name):
            # Get the actual attribute (avoid infinite recursion)
            attr = object.__getattribute__(self, name)

            # Don't wrap special attributes
            if name.startswith('_') or name in {'world_size', 'rank_in_group', 'local_rank', 'rank', 'device', 'cpu_group', 'device_group'}:
                return attr

            # Check if it's a communication operation
            comm_ops = object.__getattribute__(self, '_COMM_OPS')

            # If it's a callable comm op, wrap it with logging
            if name in comm_ops and callable(attr):
                return self._wrap_comm_call(name, attr)

            return attr
    
        def _wrap_comm_call(self, op_name: str, method):
            def wrapper(*args, **kwargs):
                # Build annotation (matches API: format)
                tainted_ints=[]
                plain_ints=[]
                input_shapes=[]

                def collect_from(obj, param_name=None):
                    if isinstance(obj, TaintedInt):
                        if obj.taint is not None:
                            tainted_ints.append(f"{obj.taint}={obj.value}")
                    elif isinstance(obj, int):
                        # Capture plain ints with parameter name if available
                        if param_name:
                            plain_ints.append(f"{param_name}={obj}")
                        else:
                            plain_ints.append(str(obj))
                    elif isinstance(obj, torch.Tensor):
                        shape_str = format_tensor_shape(obj)
                        if shape_str:
                            input_shapes.append(shape_str)
                    elif isinstance(obj, (list, tuple)) and not isinstance(obj, TaintedShape):
                        for item in obj:
                            collect_from(item)
                    elif isinstance(obj, dict):
                        for k, v in obj.items():
                            collect_from(v, param_name=k)

                # Collect from positional args
                collect_from(args)

                # Collect from keyword args with parameter names
                for key, value in kwargs.items():
                    collect_from(value, param_name=key)

                taint_str = " | ".join(input_shapes) if input_shapes else "no_inputs"
                annotated_name = f"COMM: {op_name} IN:[{taint_str}]"

                # Add ARGS: tainted ints first, then plain ints, then tp_size
                args_list = []
                args_list.extend(tainted_ints)
                args_list.extend(plain_ints)
                # args_list.append(f"tp={self.world_size}")

                if args_list:
                    annotated_name += " ARGS:[" + ", ".join(args_list) + "]"

                with record_function(annotated_name):
                    return method(*args, **kwargs)

            return wrapper

        def all_reduce(self, x):
            # with record_function(f"COMM: vllm::all_reduce IN:[{format_tensor_shape(x)}])"):
            return ps.all_reduce_fake(
                tensor=x, group_name="fake_reduce"
            )

        def all_gather(self, x, dim):
            # with record_function(f"COMM: vllm::all_gather IN:[{format_tensor_shape(x)}, dim={dim})]"):
            return ps.all_gather_fake(
                tensor=x, dim=dim,
                world_size=self.world_size,
                group_name="fake_gather",
            )

        def reduce_scatter(self, tensor, dim, world_size, group_name):
            # with record_function(f"COMM: vllm::reduce_scatter(tensor={format_tensor_shape(tensor)}, dim={dim}), world_size={world_size}, group_name={group_name}"):
            return ps.reduce_scatter_fake(
                tensor=tensor, dim=dim,
            world_size=world_size,
            group_name=group_name,
        )
                
        def gather(self, tensor, dst=0, dim=-1):
            # No-op in fake TP
            return tensor

        def broadcast_tensor_dict(self, tensor_dict=None, src=0, group=None, metadata_group=None):
            # No-op in fake TP
            return tensor_dict if tensor_dict is not None else {}
        
    fake_group = FakeTPGroup()

    # --- parallel_state --------------------------------------------------
    # TP group functions
    ps.get_tp_group = lambda: fake_group
    communication_op.get_tp_group = lambda: fake_group
    ps.get_tensor_model_parallel_world_size = lambda: tp_size
    ps.get_tensor_model_parallel_rank = lambda: 0

    # Pipeline parallel functions (use defaults for single-GPU)
    ps.get_pipeline_model_parallel_world_size = lambda: 1
    ps.get_pipeline_model_parallel_rank = lambda: 0

    # Data parallel functions (use defaults for single-GPU)
    ps.get_data_parallel_world_size = lambda: 1
    ps.get_data_parallel_rank = lambda: 0

    # Context parallel functions (use defaults)
    ps.get_prefill_context_model_parallel_world_size = lambda: 1
    ps.get_prefill_context_model_parallel_rank = lambda: 0
    ps.get_decode_context_model_parallel_world_size = lambda: 1
    ps.get_decode_context_model_parallel_rank = lambda: 0
    
    try:
        from vllm.config.model import ModelConfig
        # Store original method
        original_get_num_attention_heads = ModelConfig.get_num_attention_heads

        def fake_get_num_attention_heads(self, parallel_config):
            """Patched version that uses fake TP size instead of real parallel_config."""
            num_heads = self.model_arch_config.total_num_attention_heads
            result = num_heads // tp_size  # Use fake TP size instead of parallel_config.tensor_parallel_size
            print(f"[FakeTP] get_num_attention_heads: {num_heads} // {tp_size} = {result} (vs original: {num_heads // parallel_config.tensor_parallel_size})")
            return result

        # Apply the patch
        ModelConfig.get_num_attention_heads = fake_get_num_attention_heads

        print(f"[FakeTP] Patched ModelConfig.get_num_attention_heads to use TP={tp_size}")

    except ImportError:
        print(f"[FakeTP] Warning: Could not patch ModelConfig.get_num_attention_heads")
    
    try:
        from vllm.config.model import ModelConfig
        # Store original method
        original_get_num_kv_heads = ModelConfig.get_num_kv_heads

        def fake_get_num_kv_heads(self, parallel_config):
            """Patched version that uses fake TP size instead of real parallel_config.

            Mirrors vLLM's own semantics: MLA collapses to one KV head during decode,
            and KV heads are replicated rather than split when the total is smaller
            than the TP degree, so every rank keeps at least one.

            Needed because some backends read the KV head count from ModelConfig
            rather than from the attention layer. TritonAttentionMetadataBuilder
            computes seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_heads_kv, so an
            unsharded count makes that threshold too small and selects the 2D kernel
            at batch sizes where a real TP rank would select the 3D kernel.
            """
            if self.use_mla:
                # When using MLA during decode it becomes MQA
                return 1
            total_num_kv_heads = self.get_total_num_kv_heads()
            result = max(1, total_num_kv_heads // tp_size)  # fake TP size, not parallel_config
            print(f"[FakeTP] get_num_kv_heads: {total_num_kv_heads} // {tp_size} = {result} (vs original: {max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)})")
            return result

        # Apply the patch
        ModelConfig.get_num_kv_heads = fake_get_num_kv_heads

        print(f"[FakeTP] Patched ModelConfig.get_num_kv_heads to use TP={tp_size}")

    except ImportError:
        print(f"[FakeTP] Warning: Could not patch ModelConfig.get_num_kv_heads")


    # MIGHT NOT NEED THIS ANYMORE:
    # # --- distributed communication (no-ops) ------------------------------
    # distributed.tensor_model_parallel_all_reduce = lambda x: x
    # distributed.tensor_model_parallel_all_gather = lambda x: x

    # # --- linear layers ---------------------------------------------------
    # linear.get_tensor_model_parallel_world_size = lambda: tp_size
    # linear.get_tensor_model_parallel_rank = lambda: 0


    print(f"[FakeTP] Patches applied: TP={tp_size}")
