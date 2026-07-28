from doolyprof.profiler.module import ModuleInfo
from typing import List, Dict, Any, Tuple, Optional, Union
import torch

from doolyprof.profiler.utils.generate_batch import (generate_vidur_inputs, generate_vidur_num_tokens, AttentionBatchConfig, get_attention_batch_sizes_to_profile)

def is_attention_operation(module: ModuleInfo) -> bool:
    attention_ops = [
        "attention", "Attention", "atten", "Atten", "attn", "Attn"
    ]
    for op in attention_ops:
        if op in module.module_name:
            return True
    return False

def classify_op(module: ModuleInfo) -> str:
    """Return the op's profile class:

        'attention'  -> 4-D AttentionBatchConfig sweep (num_tokens, batch_size,
                        kv_cache_size, prefill_chunk_size)
        'moe'        -> 1-D num_tokens sweep (FusedMoE, SharedFusedMoE, ...)
        'mamba'      -> reserved for state-space layers; user must define
        'ssm'        -> reserved
        'stateless'  -> default num_tokens / num_requests sweep (linear, rms_norm,
                        embedding, rotary, silu, argmax, ...)

    Extend by adding a branch here + a matching `_prepare_*_inputs` method in
    InputGenerator.
    """
    name = module.module_name or ""
    if is_attention_operation(module):
        return "attention"
    # Match both "MoE" (FusedMoE, SharedFusedMoE, MixtralMoE, DeepseekV2MoE) and
    # "Moe" (Qwen2MoeSparseMoeBlock, ...SparseMoeBlock) so MoE wrappers are not
    # misclassified as 'stateless'.
    for tok in ("FusedMoE", "SharedFusedMoE", "MoE", "Moe"):
        if tok in name:
            return "moe"
    for tok in ("Mamba", "MambaMixer"):
        if tok in name:
            return "mamba"
    if "SSM" in name:
        return "ssm"
    return "stateless"

class InputGenerator:
    # for a given ModuleInfo, generate the inputs needed to profile
    def __init__(self, module: ModuleInfo, max_num_request: int=16,
                 max_num_token: int=1024, min_num_requests:int=1, test_counts: int=10,
                 dtype: torch.dtype = torch.bfloat16,
                 moe_top_k: Optional[int] = None, moe_num_experts: Optional[int] = None):
        self.module = module
        self.max_num_request = max_num_request
        self.max_num_token = max_num_token
        self.test_counts = test_counts
        self.batches = []
        self.seen_batches = set()
        self.dtype = dtype
        self.min_num_requests = min_num_requests
        # MoE 2-D sweep params, set by the profiler from the live FusedMoE module
        # (input_generator only has the ModuleInfo, not the live module/config).
        self.moe_top_k = moe_top_k
        self.moe_num_experts = moe_num_experts

        self.dtype_map = {
            "float32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "c10::BFloat16": torch.bfloat16,
            "long": torch.long,
            "long int": torch.long,
            "int": torch.int32,
            "int32": torch.int32,
            "int64": torch.long,
            "bool": torch.bool,
            "double": torch.float64,
            "trace_dtype": None,
            # torch.* prefixed variants (from MODULE DTYPES section)
            "torch.float32": torch.float32,
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.int32": torch.int32,
            "torch.int64": torch.long,
            "torch.long": torch.long,
            "torch.bool": torch.bool,
            "torch.float64": torch.float64,
        }

    def _parse_scalar_value(self, scalar_str: str) -> Any:
        """
        Parse a scalar value string into its appropriate Python type.
        Handles: floats, ints, bools, lists (e.g., "[4096]"), and strings.
        """
        import ast

        # Try parsing as a Python literal (handles lists, ints, floats, bools, tuples)
        try:
            value = ast.literal_eval(scalar_str)
            return value
        except (ValueError, SyntaxError):
            pass

        # Try float (handles scientific notation like "1.0e-05")
        try:
            value = float(scalar_str)
            # Check if it's actually an int
            if value == int(value) and 'e' not in scalar_str.lower() and '.' not in scalar_str:
                return int(value)
            return value
        except ValueError:
            pass

        # Try bool
        if scalar_str.lower() in ('true', 'false'):
            return scalar_str.lower() == 'true'

        # Keep as string
        return scalar_str

    def _create_tensor_inputs_from_params(
        self, workload_config: Union[AttentionBatchConfig, Dict[str, int], List[int]]
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Create tensor inputs using ProfileInput structure.

        Args:
            workload_config: Either an AttentionBatchConfig, a dict with 'num_tokens'/'num_requests',
                           or a List[int] (legacy format from resolver)

        Returns:
            Tuple of (input_tensors, workload_params) where workload_params contains the
            actual num_tokens and num_requests values used
        """
        inputs = []
        trace_dtype = self.dtype

        from doolyprof.profiler.taint import DimensionType

        # Extract num_tokens and num_requests from workload_config
        if isinstance(workload_config, AttentionBatchConfig):
            num_tokens = workload_config.num_tokens
            num_requests = workload_config.batch_size
            workload_params = {
                'num_tokens': num_tokens,
                'num_requests': num_requests,
                'is_prefill': workload_config.is_prefill,
                'prefill_chunk_size': workload_config.prefill_chunk_size,
                'kv_cache_size': workload_config.kv_cache_size
            }
        elif isinstance(workload_config, list):
            # Legacy List[int] format from resolver
            # First element is num_tokens, length is num_requests
            num_tokens = workload_config[0] if workload_config else 1
            num_requests = len(workload_config)
            workload_params = {
                'num_tokens': num_tokens,
                'num_requests': num_requests
            }
        else:
            # Dict format from non-attention operations
            num_tokens = workload_config.get('num_tokens', 1)
            num_requests = workload_config.get('num_requests', 1)
            workload_params = {
                'num_tokens': num_tokens,
                'num_requests': num_requests
            }

        for input_idx, profile_input in enumerate(self.module.inputs):
            if profile_input.type == "Tensor":
                # Build shape: replace workload dimensions with sweep values
                shape = []
                num_tokens_dim_indices = profile_input.get_toks_dim()
                num_requests_dim_indices = profile_input.get_reqs_dim()
                mix_dim_indices = profile_input.get_mix_dims()

                for dim_idx, actual_dim_value in enumerate(profile_input.actual_shape):
                    if dim_idx in num_tokens_dim_indices:
                        # NUM_TOKENS dimension - use num_tokens from workload config
                        shape.append(num_tokens)
                    elif dim_idx in num_requests_dim_indices:
                        # NUM_REQUESTS dimension - use num_requests from workload config
                        shape.append(num_requests)
                    elif dim_idx in mix_dim_indices:
                        # MIX dimension - calculate size from components
                        components = profile_input.dimension_components[dim_idx] if profile_input.dimension_components else None
                        if components:
                            new_size = 1
                            for taint_name, qty in components.items():
                                if 'NUM_TOK' in taint_name:
                                    new_size *= num_tokens
                                elif 'NUM_REQ' in taint_name:
                                    new_size *= num_requests
                                else:
                                    # MODEL_CONFIG or other - use original quantity
                                    new_size *= qty
                            shape.append(new_size)
                        else:
                            # MIX without components - fallback to fixed value (shouldn't happen)
                            shape.append(actual_dim_value)
                    else:
                        # MODEL_CONFIG dimension - keep fixed value from trace
                        shape.append(actual_dim_value)

                # Validate shape. A dim of 0 typically means the original call
                # had a None here (e.g. optional Tensor args like output_scale /
                # kv_cache_dummy_dep in unified_attention_with_output). Skip
                # that single input by substituting None — preserves positional
                # order for downstream args/kwargs packing — instead of
                # aborting the whole input list.
                if not all(isinstance(d, int) and d > 0 for d in shape):
                    if any(isinstance(d, int) and d == 0 for d in shape):
                        print(f"[InputGenerator] Shape {shape} has zero dim — treating as None")
                        inputs.append(None)
                        continue
                    print(f"[InputGenerator] Invalid shape {shape}, skipping input")
                    print(f"[InputGenerator]   actual_shape={profile_input.actual_shape}")
                    print(f"[InputGenerator]   dimensions={[d.value for d in profile_input.dimensions]}")
                    print(f"[InputGenerator]   num_requests_dim_indices={num_requests_dim_indices}")
                    print(f"[InputGenerator]   num_tokens_dim_indices={num_tokens_dim_indices}")
                    print(f"[InputGenerator]   num_tokens={num_tokens}, num_requests={num_requests}")
                    return [], workload_params

                # Get dtype
                input_dtype_str = profile_input.dtype
                if input_dtype_str == "trace_dtype":
                    input_dtype = trace_dtype
                else:
                    input_dtype = self.dtype_map.get(input_dtype_str, trace_dtype)

                shape_tuple = tuple(shape)

                # Create tensor
                if input_dtype in (torch.long, torch.int32, torch.int64):
                    if len(shape) == 1:
                        max_val = shape[0] if shape[0] < 10000 else 1000
                    else:
                        max_val = 1000
                    tensor = torch.randint(0, max_val, shape_tuple, dtype=input_dtype, device='cuda')
                else:
                    tensor = torch.randn(shape_tuple, dtype=input_dtype, device='cuda')
                inputs.append(tensor)

            elif profile_input.type == "Scalar":
                # Use scalar_value directly
                if profile_input.scalar_value:
                    value = self._parse_scalar_value(profile_input.scalar_value)
                    inputs.append(value)
                else:
                    # Provide default based on dtype
                    if profile_input.dtype == "string":
                        inputs.append("auto")
                    elif profile_input.dtype == "bool":
                        inputs.append(False)
                    elif profile_input.dtype in ("int", "long"):
                        inputs.append(0)
                    elif profile_input.dtype in ("float", "double"):
                        inputs.append(0.0)
                    # else:
                    #     # Empty scalar with unknown dtype (e.g., "Scalar") represents None
                    #     # This handles Tensor? arguments that were None in the trace
                    #     inputs.append(None)

        return inputs, workload_params

    def _generate_num_requests_sweep(self) -> List[int]:
       return get_attention_batch_sizes_to_profile(min_batch_size=self.min_num_requests, max_batch_size=self.max_num_request) 

    def _generate_num_tokens_sweep(self) -> List[int]:
        return generate_vidur_num_tokens(max_size=self.max_num_token)

    def prepare_inputs(self, mode: str, use_batch_config: bool = None) -> Union[List[AttentionBatchConfig], List[Dict[str, int]]]:
        """Generate workload configurations for profiling.

        Dispatches by op class. Add a new class by extending `classify_op()`
        and implementing a matching `_prepare_<class>_inputs` method here.

        Back-compat: if `use_batch_config` is passed explicitly, honor it
        (True -> attention path, False -> stateless path) so existing callers
        that bypass classification (e.g. profiler's raw-op path) keep working.
        """
        if use_batch_config is True:
            return self._prepare_attention_inputs(mode)
        if use_batch_config is False:
            return self._prepare_stateless_inputs(mode)

        op_class = classify_op(self.module)

        if op_class == "attention":
            return self._prepare_attention_inputs(mode)
        if op_class == "moe":
            return self._prepare_moe_inputs(mode)
        if op_class == "mamba":
            return self._prepare_mamba_inputs(mode)
        if op_class == "ssm":
            return self._prepare_ssm_inputs(mode)
        return self._prepare_stateless_inputs(mode)

    def _prepare_attention_inputs(self, mode: str) -> List[AttentionBatchConfig]:
        if mode != "vidur":
            raise NotImplementedError(f"Mode {mode} is not implemented yet")
        self.batches = generate_vidur_inputs(
            max_seq_len=self.max_num_token,
            min_batch_size=1,
            max_batch_size=self.max_num_request,
            profile_only_prefill=False,
            profile_only_decode=False,
        )
        return self.batches

    def _prepare_moe_inputs(self, mode: str) -> List[Dict[str, int]]:
        """2-D sweep over (num_tokens, activated_experts). MoE-block latency depends
        on the token count and on how many distinct experts are activated (few experts
        with many tokens each vs many experts with few tokens each) — LLMServingSim's
        ExpertCategory. kv_cache_size / prefill_chunk_size don't matter.

        `activated_experts` is a power-of-two grid in [top_k, min(num_experts,
        num_tokens*top_k)]; the profiler forces exactly that many experts to fire
        (see moe_hook.force_moe_routing) when timing each config. If MoE params
        (top_k, num_experts) were not supplied by the profiler, fall back to the 1-D
        num_tokens sweep (routing is then left to the dummy-weight gate).
        """
        num_tokens_sweep = self._generate_num_tokens_sweep()
        if self.moe_top_k is None or self.moe_num_experts is None:
            self.batches = [
                {"num_tokens": n, "num_requests": 1} for n in num_tokens_sweep
            ]
            return self.batches

        from doolyprof.profiler.moe_hook import moe_activated_grid
        configs = []
        for n in num_tokens_sweep:
            for activated in moe_activated_grid(self.moe_top_k, self.moe_num_experts, n):
                configs.append({
                    "num_tokens": n,
                    "num_requests": 1,
                    "activated_experts": activated,
                })
        self.batches = configs
        return self.batches

    def _prepare_mamba_inputs(self, mode: str) -> List[Dict[str, int]]:
        """Reserved for Mamba / SSD state-space layers.

        Mamba depends on (num_tokens, batch_size, prefill_chunk_size, is_prefill)
        — 3-D. Fill in when the first Mamba-based model is evaluated.
        """
        raise NotImplementedError(
            "Mamba profile space not yet defined. Extend _prepare_mamba_inputs "
            "with the (num_tokens, batch_size, prefill_chunk_size, is_prefill) "
            "sweep appropriate for your SSM implementation."
        )

    def _prepare_ssm_inputs(self, mode: str) -> List[Dict[str, int]]:
        raise NotImplementedError(
            "SSM profile space not yet defined. Extend _prepare_ssm_inputs "
            "with the sweep appropriate for your state-space layer."
        )

    def _prepare_stateless_inputs(self, mode: str) -> List[Dict[str, int]]:
        """Default path for shape-driven ops (linear, rms_norm, rotary, silu,
        embedding, argmax, to.dtype, ...). Sweeps over num_tokens and/or
        num_requests depending on what dimension roles the tracer recorded.
        """
        num_tokens_sweep = self._generate_num_tokens_sweep()
        num_requests_sweep = self._generate_num_requests_sweep()

        # Check which dimension types are present in inputs
        has_num_tokens_dim = any(
            inp.get_toks_dim() for inp in self.module.inputs if inp.type == "Tensor"
        )
        has_num_requests_dim = any(
            inp.get_reqs_dim() for inp in self.module.inputs if inp.type == "Tensor"
        )

        # MIX dimensions containing NUM_TOKENS or NUM_REQUESTS (incl. MAX_* variants)
        for inp in self.module.inputs:
            if inp.type == "Tensor" and inp.dimension_components:
                for dim_idx in inp.get_mix_dims():
                    components = inp.dimension_components[dim_idx]
                    if components:
                        for taint_name in components.keys():
                            if 'NUM_TOK' in taint_name:
                                has_num_tokens_dim = True
                            if 'NUM_REQ' in taint_name:
                                has_num_requests_dim = True

        workload_configs = []

        if has_num_tokens_dim and has_num_requests_dim:
            for num_tokens in num_tokens_sweep:
                for num_requests in num_requests_sweep:
                    if num_tokens >= num_requests:
                        workload_configs.append({
                            'num_tokens': num_tokens,
                            'num_requests': num_requests,
                        })
        elif has_num_tokens_dim:
            for num_tokens in num_tokens_sweep:
                workload_configs.append({
                    'num_tokens': num_tokens,
                    'num_requests': 1,
                })
        elif has_num_requests_dim:
            for num_requests in num_requests_sweep:
                workload_configs.append({
                    'num_tokens': num_requests,
                    'num_requests': num_requests,
                })
        else:
            workload_configs.append({'num_tokens': 1, 'num_requests': 1})

        self.batches = workload_configs
        return self.batches