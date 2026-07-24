"""
VLLM Layer Profiler - Profile specific layers/modules using vLLM's full setup.

This profiler:
1. Initializes vLLM's LLM to set up everything (model, caches, metadata builders)
2. Extracts specific layers and their already-bound state caches
3. Uses vLLM's existing metadata builders to create layer-specific metadata
4. Allows profiling full layers OR specific sub-operations (conv1d, SSM, projections)

What we reuse from vLLM:
- Model loading and weight initialization
- State cache allocation and binding
- Layer-specific metadata builders (Mamba2AttentionMetadataBuilder, etc.)
- Forward context setup

What we construct ourselves:
- CommonAttentionMetadata (simple dataclass with batch info)
"""

import os
# IMPORTANT: Disable vLLM v1 multiprocessing BEFORE importing vLLM
# This allows us to access the model directly in the main process

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE"] = str(1024 * 1024 * 1024)

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Literal, Union
from dataclasses import dataclass
from tqdm import tqdm
from contextlib import contextmanager, nullcontext

from doolyprof.profiler.utils.generate_batch import AttentionBatchConfig

# Apply fake TP patches BEFORE importing vLLM so all modules see patched
# parallel_state at import time. Set VLLM_FAKE_TP env var before importing
# this module (e.g. in run-profiler.py via --tp flag).
from doolyprof.profiler.utils.fake_tp import apply_fake_tp_patches, get_fake_tp_size

_FAKE_TP_SIZE = get_fake_tp_size()
if _FAKE_TP_SIZE > 1:
    apply_fake_tp_patches(_FAKE_TP_SIZE)

from vllm import LLM
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backend import CommonAttentionMetadata


class VLLMLayerProfiler:
    """
    Profile specific layers using vLLM's full infrastructure.

    This class leverages vLLM's LLM to handle all setup:
    - Model loading and weight initialization
    - State cache allocation and binding
    - Metadata builder initialization

    Then allows targeted profiling of specific layers or sub-operations.
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        enforce_eager: bool = True,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        attn_backend: str = "FLASHINFER",
        flash_attn_version: int = 2,
        quantization: Optional[str] = None,
        gpu: str = "0"
    ):
        """
        Initialize the profiler by creating a vLLM LLM instance.

        Args:
            model_name: HuggingFace model name
            dtype: Data type ("float16" or "bfloat16")
            enforce_eager: Disable CUDA graphs for easier profiling
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
            quantization: Optional quantization scheme (e.g. "fp8", "awq", "gptq").
                When set with load_format="dummy", vLLM initializes random
                quantized weights — no need to download a quantized checkpoint.
        """
        self.model_name = model_name
        self.dtype = dtype
        self.torch_dtype = getattr(torch, dtype)

        # Set GPU visibility so vLLM initializes on the correct device.
        # After CUDA_VISIBLE_DEVICES is set, the chosen GPU is remapped to
        # index 0 within this process, so we always use cuda:0 locally.
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        self.device = torch.device("cuda:0")

        print(f"[VLLMLayerProfiler] Initializing vLLM with model: {model_name}")

        # Fake TP size (patches applied at module level, see top of file)
        self.tp_size = _FAKE_TP_SIZE

        # Let vLLM handle ALL setup
        llm_kwargs = dict(
            model=model_name,
            dtype=dtype,
            enforce_eager=enforce_eager,
            load_format="dummy",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            attention_backend=attn_backend,
            tensor_parallel_size=1,  # Single process, fake TP patches handle sharding
            trust_remote_code=True,
        )
        if attn_backend == "FLASH_ATTN":
            llm_kwargs["attention_config"] = {"flash_attn_version": flash_attn_version}
        if quantization:
            llm_kwargs["quantization"] = quantization
        self.llm = LLM(**llm_kwargs)

        # Extract internals
        self.model_runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
        self.model = self.model_runner.model
        self.vllm_config = self.model_runner.vllm_config
        self.kv_cache_config = self.model_runner.kv_cache_config

        # Get attention groups (contains metadata builders)
        self.attn_groups = self.model_runner.attn_groups

        # Use num_blocks from vLLM's KV cache config
        # vLLM already calculates this based on available GPU memory
        self.actual_num_blocks = self.kv_cache_config.num_blocks

    def _get_actual_num_blocks(self) -> int:
        """
        Get the actual number of blocks from the KV cache tensor.

        The kv_cache_config.num_blocks may be larger than the actual allocated
        KV cache size. This method inspects the actual KV cache tensor to get
        the true number of blocks.
        """
        # Get forward context which has attention layers with bound kv_cache
        forward_context = self.vllm_config.compilation_config.static_forward_context

        # print(f"forward_context: {forward_context}")

        if not forward_context:
            # Fallback to config value
            print(f"[VLLMLayerProfiler] No forward context found, using config value")
            return self.kv_cache_config.num_blocks

        # Get any attention layer's kv_cache to check its shape
        for layer_name, attn_layer in forward_context.items():
            if hasattr(attn_layer, 'kv_cache') and attn_layer.kv_cache:
                kv_cache = attn_layer.kv_cache[0]  # [0] for virtual_engine
                # TODO: make this solution more general and check whether we need to use this method at all
                if kv_cache is not None and hasattr(kv_cache, 'shape'):
                    # KV cache shape is typically [2, num_blocks, block_size, num_kv_heads, head_size]
                    # or similar depending on the backend
                    # print(f"kv_cache.shape: {kv_cache.shape}")
                    if len(kv_cache.shape) >= 2:
                        print(f"KV CACHE SHAPE: {kv_cache.shape}")
                        return kv_cache.shape[1]

        # Fallback to config value
        # print(f"[VLLMLayerProfiler] No KV cache found, using config value")
        return self.kv_cache_config.num_blocks

    def get_all_layer_names(self) -> List[str]:
        """Get all layer names registered in attn_groups."""
        all_names = []
        for kv_cache_gid, attn_group_list in enumerate(self.attn_groups):
            for attn_group in attn_group_list:
                all_names.extend(attn_group.layer_names)
        return all_names

    def get_layer_name_by_index(self, layer_idx: int, layer_type: str = "attention") -> str:
        """
        Get the actual layer name from attn_groups based on layer index.

        vLLM stores layer names like "model.layers.0.self_attn" in attn_groups.
        """
        all_names = self.get_all_layer_names()

        # Pattern to match: ".{layer_idx}." in the name
        search_pattern = f".{layer_idx}."

        # First try to find exact match with layer type
        for name in all_names:
            if search_pattern in name:
                if layer_type == "attention" and "attn" in name.lower():
                    return name
                elif layer_type == "mamba" and ("mamba" in name.lower() or "mixer" in name.lower()):
                    return name

        # Fallback: just match by index
        for name in all_names:
            if search_pattern in name:
                return name

        raise ValueError(f"Cannot find layer {layer_idx} (type={layer_type}) in attn_groups. "
                         f"Available: {all_names[:5]}...")

    def get_layer_name_by_module(self, operation_name: str, name_to_module: Dict[str, List[str]]) -> Optional[str]:
        """Find the layer name in attn_groups that matches this module operation.

        Works for ANY module type (attention, mamba, moe, etc.) - uses attn_groups
        as the source of truth for which modules need forward context.
        """
        import re

        module_class = operation_name
        instance_idx = 0
        if '_' in module_class and module_class.rsplit('_', 1)[-1].isdigit():
            instance_idx = int(module_class.rsplit('_', 1)[1])
            module_class = module_class.rsplit('_', 1)[0]

        paths = name_to_module.get(module_class)
        if not paths:
            return None
        # Pick the instance by index; clamp to valid range
        callable_path = paths[min(instance_idx, len(paths) - 1)]

        all_layer_names = self.get_all_layer_names()

        if callable_path in all_layer_names:
            return callable_path

        for name in all_layer_names:
            if callable_path in name or name in callable_path:
                return name
            match = re.search(r'\.(\d+)\.', callable_path)
            if match:
                layer_idx = match.group(0)
                if layer_idx in name:
                    return name

        return None

    def _find_metadata_builder_for_layer(self, layer_name: str):
        """
        Find the metadata builder that handles a specific layer.

        Returns:
            Tuple of (builder, attn_group, kv_cache_group_id)
        """
        for kv_cache_gid, attn_group_list in enumerate(self.attn_groups):
            for attn_gid, attn_group in enumerate(attn_group_list):
                if layer_name in attn_group.layer_names:
                    builder = attn_group.get_metadata_builder(0)
                    return builder, attn_group, kv_cache_gid
        raise ValueError(f"No metadata builder found for layer: {layer_name}")

    def _get_or_create_block_table(
        self,
        batch_size: int,
        blocks_per_seq: int,
        num_tokens: int,
        block_size: int,
        kv_cache_gid: int = 0,
    ):
        """
        Get or create a persistent BlockTable for the given kv_cache group.
        The BlockTable is created once and reused across batch configs.
        """
        from vllm.v1.worker.block_table import BlockTable

        # Create cache key for this kv_cache group
        cache_key = kv_cache_gid

        # Lazily initialize the block table cache
        if not hasattr(self, '_block_table_cache'):
            self._block_table_cache = {}
            self._rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility

        # Use actual number of blocks from KV cache tensor
        max_num_blocks = self.actual_num_blocks

        if cache_key not in self._block_table_cache:
            # Create a BlockTable with maximum capacity
            # Use max possible values to avoid reallocation
            max_batch_size = 256  # Conservative max
            max_blocks_per_req = max_num_blocks  # Max possible blocks per sequence
            max_batched_tokens = max_batch_size * 4096  # Conservative estimate

            bt = BlockTable(
                block_size=block_size,
                max_num_reqs=max_batch_size,
                max_num_blocks_per_req=max_blocks_per_req,
                max_num_batched_tokens=max_batched_tokens,
                pin_memory=True,
                device=self.device,
                kernel_block_size=block_size,
                cp_kv_cache_interleave_size=1,
            )
            self._block_table_cache[cache_key] = bt

        return self._block_table_cache[cache_key]

    def _build_common_attention_metadata(
        self,
        batch_config: AttentionBatchConfig,
        kv_cache_gid: int = 0,
    ) -> Optional[CommonAttentionMetadata]:
        """
        Build CommonAttentionMetadata for attention profiling.

        Returns None if the batch config exceeds available KV cache blocks.
        """
        batch_size = batch_config.batch_size
        is_prefill = batch_config.is_prefill

        if is_prefill:
            new_tokens = batch_config.prefill_chunk_size
            kv_cache_size = batch_config.kv_cache_size
            total_seq_len = new_tokens + kv_cache_size
            tokens_per_seq = new_tokens
        else:
            new_tokens = 1
            kv_cache_size = batch_config.kv_cache_size
            total_seq_len = kv_cache_size + 1
            tokens_per_seq = 1

        num_tokens = batch_size * tokens_per_seq

        kv_cache_group = self.kv_cache_config.kv_cache_groups[kv_cache_gid]
        kv_cache_spec = kv_cache_group.kv_cache_spec

        block_size = getattr(kv_cache_spec, 'block_size', 512)
        # Use actual number of blocks from KV cache tensor, not config
        max_num_blocks = self.actual_num_blocks

        # Calculate blocks needed per sequence
        blocks_per_seq = (total_seq_len + block_size - 1) // block_size
        blocks_per_seq = max(1, blocks_per_seq)

        # Check if batch config exceeds available KV cache blocks
        total_blocks_needed = batch_size * blocks_per_seq
        if total_blocks_needed > max_num_blocks:
            # Skip this config - not enough KV cache blocks available
            return None

        # Build query_start_loc
        query_lens = [tokens_per_seq] * batch_size
        query_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        for i, q_len in enumerate(query_lens):
            query_start_loc[i + 1] = query_start_loc[i] + q_len

        seq_lens_list = [total_seq_len] * batch_size
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=self.device)

        # Get or create persistent BlockTable
        bt = self._get_or_create_block_table(
            batch_size=batch_size,
            blocks_per_seq=blocks_per_seq,
            num_tokens=num_tokens,
            block_size=block_size,
            kv_cache_gid=kv_cache_gid,
        )

        # Clear any previous state
        bt.clear()

        # Use random block IDs to simulate real vLLM behavior
        # Ensure block IDs are within valid range [0, max_num_blocks)
        for seq_idx in range(batch_size):
            random_block_ids = self._rng.integers(
                low=0, high=max_num_blocks, size=blocks_per_seq,
            ).tolist()
            bt.add_row(random_block_ids, row_idx=seq_idx)

        start_pos = kv_cache_size
        req_indices = np.repeat(np.arange(batch_size, dtype=np.int32), tokens_per_seq)
        positions = np.tile(
            np.arange(start_pos, start_pos + tokens_per_seq, dtype=np.int32),
            batch_size,
        )
        if hasattr(bt, "commit_slot_mapping"):
            # vLLM <= 0.17: CPU-side slot mapping, committed to GPU afterwards
            bt.compute_slot_mapping(req_indices, positions)
            bt.commit_block_table(batch_size)
            bt.commit_slot_mapping(num_tokens)
        else:
            # vLLM >= 0.21: compute_slot_mapping(num_reqs, query_start_loc, positions)
            # is a Triton kernel reading block_table.gpu and writing
            # slot_mapping.gpu directly, so the block table must be on GPU first.
            bt.commit_block_table(batch_size)
            positions_gpu = torch.from_numpy(positions).to(
                device=self.device, dtype=torch.int64
            )
            bt.compute_slot_mapping(batch_size, query_start_loc, positions_gpu)

        # retrieve first batch_size rows (existing + new)
        # each containing block IDs spanning entire KV Sequence
        block_tables = bt.block_table.gpu[:batch_size]

        # extract mapping for new tokens being processed
        # maps positions to their corresponding KV cache slots
        slot_mapping = bt.slot_mapping.gpu[:num_tokens]

        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=seq_lens,
            num_reqs=batch_size,
            num_actual_tokens=num_tokens,
            max_query_len=tokens_per_seq,
            max_seq_len=total_seq_len,
            block_table_tensor=block_tables,
            slot_mapping=slot_mapping,
            causal=True,
        )

    def build_layer_metadata(
        self,
        layer_name: str,
        batch_config: AttentionBatchConfig,
    ) -> Optional[Tuple[Any, Any]]:
        """
        Build metadata for a specific layer using vLLM's metadata builder.

        We construct CommonAttentionMetadata (simple dataclass), then use
        vLLM's layer-specific builder for the complex part.

        Returns:
            Tuple of (layer_metadata, metadata_builder), or None if the batch
            config exceeds available KV cache blocks.
        """
        builder, attn_group, kv_cache_gid = self._find_metadata_builder_for_layer(layer_name)

        # print(f"[DEBUG] layer_name: {layer_name}, builder: {builder}, attn_group: {attn_group}, kv_cache_gid: {kv_cache_gid}")

        # We construct CommonAttentionMetadata (simple batch info)
        common_meta = self._build_common_attention_metadata(batch_config, kv_cache_gid)

        # Return None if batch config exceeds available KV cache blocks
        if common_meta is None:
            return None

        # vLLM's builder handles the complex layer-specific metadata
        with set_current_vllm_config(self.vllm_config):
            with torch.inference_mode():
                layer_metadata = builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_meta,
                )

        return layer_metadata, builder

    def close(self):
        import gc
        from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

        try:
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception as e:
            print(f"[VLLMLayerProfiler] Warning during parallel cleanup: {e}")

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception as e:
            print(f"[VLLMLayerProfiler] Warning during engine shutdown: {e}")

        del self.llm

        gc.collect()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def make_dict(model):
    """Map each module class name to an ordered list of all its paths in the model.

    named_modules() visits in depth-first order, which matches the _N suffix
    that the tracer assigns to repeated class instances (e.g. GemmaRMSNorm_0,
    GemmaRMSNorm_1, ...). Storing all paths lets callers pick the correct
    instance by index rather than always returning the first one.
    """
    ret = {}
    for call_path, module in model.named_modules():
        module_name = type(module).__name__
        ret.setdefault(module_name, []).append(call_path)
    return ret

def _run_profiler_in_subprocess(model_name: str, dtype: str, attn_backend: str):
    """Target function for subprocess - all CUDA work happens here."""
    import torch
    print(f"\n=== Run {model_name} ({dtype}) (subprocess PID: {os.getpid()}) ===")

    vlp = VLLMLayerProfiler(
        model_name=model_name,
        dtype=dtype,
        gpu_memory_utilization=0.9,
        attn_backend=attn_backend,
    )

    # Do your profiling work here...

    print("\nClosing vLLM instance...")
    vlp.close()
    print(f"=== Run {model_name} ({dtype}) ({attn_backend}) complete ===\n")

def run_profiler(model_name: str, dtype: str, attn_backend: str):
    # All CUDA work happens in subprocess
    p = mp.Process(target=_run_profiler_in_subprocess, args=(model_name, dtype, attn_backend))
    p.start()
    p.join()  # Wait for subprocess to finish - GPU memory fully freed on exit



if __name__ == "__main__":
    
    test_vlp = VLLMLayerProfiler(
        model_name="facebook/opt-125m",
        dtype="bfloat16",
        gpu_memory_utilization=0.2,
        attn_backend="FLASHINFER",
    )

    print(f"Actual num blocks: {test_vlp.actual_num_blocks}")


    
    
    # import multiprocessing as mp

    # mp.set_start_method("spawn", force=True)

    # for i in range(3):
    #     # All CUDA work happens in subprocess
    #     p = mp.Process(target=_run_profiler_in_subprocess, args=(i,))
    #     p.start()
    #     p.join()  # Wait for subprocess to finish - GPU memory fully freed on exit

    #     if p.exitcode != 0:
    #         print(f"Run {i + 1} failed with exit code {p.exitcode}")

    # print("All runs complete!")
    
