"""
finput - input to function setup for profiling

Provides a simple data structure for representing operation inputs
with dimension categorization (WORKLOAD vs MODEL_CONFIG).
"""

from dataclasses import dataclass
from itertools import zip_longest
from typing import List, Optional, Dict
from doolyprof.profiler.taint import DimensionType, parse_taint_string


@dataclass
class ProfileInput:
    """Input representation for profiling (taint-based)"""
    type: str  # "Tensor" or "Scalar"
    dtype: str
    dimensions: List[DimensionType]  # MODEL_CONFIG or WORKLOAD for each dimension
    actual_shape: List[int]  # Actual dimension values from trace
    scalar_value: Optional[str] = None  # For scalar inputs
    arg_name: Optional[str] = None  # kwarg name if input was passed as keyword arg
    dimension_components: Optional[List[Optional[Dict[str, int]]]] = None  # For MIX dimensions: components per dim

    def get_model_config_dims(self) -> List[int]:
        """Get indices of MODEL_CONFIG dimensions (fixed, don't sweep)"""
        return [i for i, dim_type in enumerate(self.dimensions)
                if dim_type == DimensionType.MODEL_CONFIG]

    def get_reqs_dim(self) -> List[int]:
        """Get indices of NUM_REQS dimensions (sweep these during profiling)"""
        return [i for i, dim_type in enumerate(self.dimensions)
                if dim_type == DimensionType.NUM_REQUESTS]

    def get_toks_dim(self) -> List[int]:
        """Get indices of NUM_TOKS dimensions (sweep these during profiling)"""
        return [i for i, dim_type in enumerate(self.dimensions)
                if dim_type == DimensionType.NUM_TOKENS]

    def get_mix_dims(self) -> List[int]:
        """Get indices of MIX dimensions (need special handling during sweeping)"""
        return [i for i, dim_type in enumerate(self.dimensions)
                if dim_type == DimensionType.MIX]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        d = {
            'type': self.type,
            'dtype': self.dtype,
            'dimensions': [dim.value for dim in self.dimensions],
            'actual_shape': self.actual_shape,
            'scalar_value': self.scalar_value
        }
        if self.arg_name is not None:
            d['arg_name'] = self.arg_name
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'ProfileInput':
        """Create ProfileInput from dictionary"""
        dimensions = [DimensionType(dim) for dim in data.get('dimensions', [])]
        return cls(
            type=data['type'],
            dtype=data['dtype'],
            dimensions=dimensions,
            actual_shape=data['actual_shape'],
            scalar_value=data.get('scalar_value'),
            arg_name=data.get('arg_name')
        )


def extract_inputs_from_taint(
    taint_string: str,
    trace_input_dims: List[List[int]],
    trace_input_dtypes: List[str],
    concrete_inputs: Optional[List[str]] = None,
) -> List[ProfileInput]:
    """Extract ProfileInput list from taint annotation.

    Args:
        taint_string: Full taint string (e.g., "_C::rms_norm IN:[[...]]")
        trace_input_dims: Input dimensions from trace (e.g., [[15, 4096], [15, 4096], [4096], []])
        trace_input_dtypes: Input types from trace (e.g., ["c10::BFloat16", ...])
        concrete_inputs: Concrete scalar values (e.g., ["", "", "", "1.0e-05"])

    Example:
        taint_string = "_C::rms_norm IN:[[?(15), MODEL_CONFIG(4096)] | [?(15), MODEL_CONFIG(4096)] | [MODEL_CONFIG(4096)]]"
        trace_input_dims = [[15, 4096], [15, 4096], [4096], []]
        trace_input_dtypes = ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16", "Scalar"]

        Returns ProfileInput entries with dimensions tagged WORKLOAD vs
        MODEL_CONFIG so the sweep generator knows which axes to vary.
    """
    # Strip KWARGS: section before parsing the taint string, save kwarg names
    kwarg_names = []
    if ' KWARGS:[' in taint_string:
        taint_string, kwargs_part = taint_string.split(' KWARGS:[', 1)
        kwarg_names = [n.strip() for n in kwargs_part.rstrip(']').split(',')]

    # Parse taint string to get dimension types
    taint_inputs = parse_taint_string(taint_string)
    # print(f"[FINPUT] taint_inputs : {taint_inputs}")

    # Handle case where taint parsing failed
    if not taint_inputs:
        raise ValueError(f"Failed to parse taint string: {taint_string}")

    # Build ProfileInput list using zip_longest to handle length mismatches
    # taint_inputs may be shorter than trace data (missing scalars)
    profile_inputs = []

    for i, (taint_input, dims, dtype) in enumerate(zip_longest(
        taint_inputs, trace_input_dims, trace_input_dtypes, fillvalue=None
    )):
        # Skip if we have more taint inputs than trace data (shouldn't happen)
        if dims is None:
            # print(f"[DEBUG @ finput] dims is None, skipping.")
            continue
        
        # Get scalar value from concrete inputs if available
        scalar_val = concrete_inputs[i] if concrete_inputs and i < len(concrete_inputs) else None
        
        if len(dims) == 0:
            # Empty dims from trace - three distinct cases we now keep separate:
            # 1. Python-scalar arg (dtype='Scalar'/'ScalarList'/'Bool'/'bool' or
            #    dtype=='' with a concrete value) — taint tracked it as a value.
            # 2. 0-d tensor (scalar buffer like Gemma's normalizer, RMSNorm eps,
            #    rope constants): dtype is a real tensor dtype
            #    (e.g. 'c10::BFloat16') and Input Dims is []. Must stay Tensor
            #    so the generator emits torch.ones(()) of the same dtype —
            #    otherwise aten::mul_.Tensor etc. can't match a schema.
            # 3. Tensor recovered via taint dims (e.g. aten::index list elems).
            scalar_type_indicators = {'Scalar', 'ScalarList', 'Bool', 'bool'}
            is_trace_scalar = (dtype in scalar_type_indicators or (dtype == '' and scalar_val))
            is_tensor_dtype = bool(dtype) and dtype not in scalar_type_indicators

            if not is_trace_scalar and taint_input is not None and len(taint_input.dimensions) > 0:
                # Case 3: taint has dimension info — use it to create tensor
                dim_types = [dim.dim_type for dim in taint_input.dimensions]
                actual_shape = [dim.value for dim in taint_input.dimensions]
                dim_components = [dim.components for dim in taint_input.dimensions]
                # Default to long for index tensors when dtype is empty
                effective_dtype = dtype if dtype else "long int"
                profile_input = ProfileInput(
                    type="Tensor",
                    dtype=effective_dtype,
                    dimensions=dim_types,
                    actual_shape=actual_shape,
                    dimension_components=dim_components
                )
            elif not is_trace_scalar and is_tensor_dtype:
                # Case 2: 0-d tensor (scalar buffer). Preserve as Tensor with
                # empty shape so InputGenerator's Tensor branch emits a 0-d
                # tensor of matching dtype.
                profile_input = ProfileInput(
                    type="Tensor",
                    dtype=dtype,
                    dimensions=[],
                    actual_shape=[],
                )
            else:
                # Case 1: Python scalar
                profile_input = ProfileInput(
                    type="Scalar",
                    dtype=dtype or "Scalar",
                    dimensions=[],
                    actual_shape=[],
                    scalar_value=scalar_val
                )

        elif taint_input is None:
            # Tensor without taint info (extra input not in taint string)
            # Mark all dims as MODEL_CONFIG (conservative - won't be swept)
            profile_input = ProfileInput(
                type="Tensor",
                dtype=dtype or "float",
                dimensions=[DimensionType.MODEL_CONFIG] * len(dims),
                actual_shape=dims
            )
            
        else:
            # Tensor with taint info - extract dimension types
            dim_types = [dim.dim_type for dim in taint_input.dimensions]
            dim_components = [dim.components for dim in taint_input.dimensions]
            # Handle dimension mismatch (broadcasting case)
            # If taint has fewer dims than trace, the tensor was broadcast
            # Pad leading dimensions with MODEL_CONFIG (conservative)
            if len(dim_types) < len(dims):
                # Broadcasting: taint [N] -> trace [N, D] means we pad leading dims
                # Actually for broadcasting, trailing dims match, so pad at front
                num_missing = len(dims) - len(dim_types)
                dim_types = [DimensionType.MODEL_CONFIG] * num_missing + dim_types
                dim_components = [None] * num_missing + dim_components
            elif len(dim_types) > len(dims):
                # Taint has more dims than trace - unusual, raise error
                raise ValueError(
                    f"Dimension mismatch for input {i}: "
                    f"taint has {len(dim_types)} dimensions but trace has {len(dims)} dimensions"
                )

            profile_input = ProfileInput(
                type="Tensor",
                dtype=dtype,
                dimensions=dim_types,
                actual_shape=dims,
                dimension_components=dim_components
            )
        profile_inputs.append(profile_input)

    # Assign kwarg names to the inputs that came from kwargs.
    # kwarg_names lists names for inputs starting at index (total - len(kwarg_names)).
    if kwarg_names:
        num_positional = len(profile_inputs) - len(kwarg_names)
        for idx, name in enumerate(kwarg_names):
            pos = num_positional + idx
            if 0 <= pos < len(profile_inputs):
                profile_inputs[pos].arg_name = name

    return profile_inputs


if __name__ == "__main__":
    import json

    # Test extract_inputs_from_taint
    taint_string = "_C::fused_add_rms_norm IN:[[?(15), MODEL_CONFIG(4096)] | [?(15), MODEL_CONFIG(4096)] | [MODEL_CONFIG(4096)]]"
    trace_input_dims = [[15, 4096], [15, 4096], [4096], []]
    trace_input_dtypes = ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16", "Scalar"]
    concrete_inputs = ["", "", "", "1.0000000000000001e-05"]

    print(f"Extracting inputs from taint:")
    print(f"  Taint: {taint_string}")
    print(f"  Dims: {trace_input_dims}")
    print(f"  Types: {trace_input_dtypes}")
    print()

    # Handle scalar at end (taint has 3 inputs but trace has 4)
    # Adjust by adding empty input to taint
    taint_string_with_scalar = taint_string.replace("]]", " | []]")

    profile_inputs = extract_inputs_from_taint(
        taint_string=taint_string_with_scalar,
        trace_input_dims=trace_input_dims,
        trace_input_dtypes=trace_input_dtypes,
        concrete_inputs=concrete_inputs
    )

    # print(f"Extracted {len(profile_inputs)} ProfileInputs:")
    for i, inp in enumerate(profile_inputs):
        print(f"\n  Input {i}:")
        print(f"    Type: {inp.type}")
        print(f"    Dtype: {inp.dtype}")

        if inp.type == "Tensor":
            print(f"    Shape: {inp.actual_shape}")
            print(f"    Dimension types:")
            for j, (dim_type, dim_val) in enumerate(zip(inp.dimensions, inp.actual_shape)):
                if dim_type == DimensionType.NUM_TOKS:
                    label = 'NUM_TOKS'
                elif dim_type == DimensionType.NUM_REQS:
                    label = 'NUM_REQS'
                elif dim_type == DimensionType.MODEL_CONFIG:
                    label = 'MODEL_CONFIG'
                else:
                    label = 'UNKOWN'
                print(f"      [{j}] {label}({dim_val})")

            workload_dims = inp.get_workload_dims()
            model_config_dims = inp.get_model_config_dims()
            print(f"    Sweep dimensions (WORKLOAD): {workload_dims}")
            print(f"    Fixed dimensions (MODEL_CONFIG): {model_config_dims}")
        else:
            print(f"    Scalar value: {inp.scalar_value}")
