from typing import Optional, Tuple, List, Union, Callable

from .types import Taint, DimTaint 

__all__ = ['propagate_taints']

def _to_dim_taint(t) -> Optional[DimTaint]:
    """Convert a Taint or DimTaint to DimTaint, preserving None."""
    if t is None:
        return None
    if isinstance(t, DimTaint):
        return t
    return DimTaint.from_taint(t)


def _base_taint(t):
    """Fully unwrap nested DimTaint(s) down to the base Taint (or None).

    A single-level unwrap (t.taint) leaves DimTaint(DimTaint(X)) as DimTaint(X),
    which then compares unequal to a plain DimTaint(X) and raises a spurious
    "multiple different taints" error. register_taint already unwraps fully;
    mirror that here so equal taints compare equal regardless of nesting depth.
    """
    while isinstance(t, DimTaint):
        t = t.taint
    return t

def propagate_taints(
    input_infos: List[Tuple[Tuple[int, ...], Tuple[Optional[Taint], ...]]],
    output_shape: Tuple[int, ...]
) -> Tuple[Optional[DimTaint], ...]:
    """
    Propagate taints using coordinate-shape matching.
    """
    import os
    _debug_propagate = os.environ.get("DEBUG_PROPAGATE_TAINTS", "0") == "1"

    if _debug_propagate:
        print(f"[DEBUG propagate] === propagate_taints called ===")
        print(f"[DEBUG propagate] input_infos: {input_infos}")
        print(f"[DEBUG propagate] output_shape: {output_shape}")

    # Filter out empty tensors (tensors with 0 in their shape)
    # These are typically auxiliary inputs (like empty perm tensors) that shouldn't affect taint propagation
    input_infos = [(shape, taints) for shape, taints in input_infos if 0 not in shape]

    input_dim_taints = {}  
    view_claimed_dims = set()
    slice_claimed_dims = set()

    for shape, taints in input_infos:
        for i, size in enumerate(shape):
            taint = taints[i] if i < len(taints) else None
            if taint is not None:
                # Skip None taints (untainted dimensions)
                taint_val = _base_taint(taint)
                if taint_val is None:
                    continue

                if input_dim_taints.get(size) is None:
                    input_dim_taints[size] = taint
                else:
                    # Check if taints are semantically the same (compare values, not object identity)
                    existing = input_dim_taints[size]
                    existing_val = _base_taint(existing)
                    new_val = _base_taint(taint)

                    # If either is None, keep the non-None one
                    if existing_val is None and new_val is not None:
                        input_dim_taints[size] = taint
                    elif existing_val is not None and new_val is None:
                        pass  # Keep existing non-None
                    elif existing_val != new_val:
                        # simple check added to prevent error when testing whether decode has different call path
                        # if existing_val == Taint("NUM_TOKS") and new_val == Taint("NUM_REQS") or existing_val == Taint("NUM_REQS") and new_val == Taint("NUM_TOKS"):
                        #     print(f"[DEBUG] Allowing NUM_REQ/NUM_TOK overlap for input dimension size {size} due to prefill/decode phase change")
                        # else:   
                        # Only raise error if both are non-None and different
                        raise ValueError(f"Multiple different taints for input dimension size {size}: {input_dim_taints[size]} vs {taint}")

    out_taints: List[Optional[DimTaint]] = [None] * len(output_shape)
    if len(input_infos) > 0 and input_infos[0][0] == output_shape:
        # copy the tuple of taints, converting to DimTaint
        first_taints = input_infos[0][1]
        if len(first_taints) == len(output_shape):
            out_taints = [_to_dim_taint(t) for t in first_taints]


    if _debug_propagate:
        print(f"[DEBUG propagate] input_dim_taints: {input_dim_taints}")
        print(f"[DEBUG propagate] output_shape: {output_shape}")
        print(f"[DEBUG propagate] view_claimed_dims: {view_claimed_dims}")
        print(f"[DEBUG propagate] slice_claimed_dims: {slice_claimed_dims}")
        print(f"[DEBUG propagate] out_taints before coord fallback: {out_taints}")

    # CONSERVATIVE: Direct match using input_dim_taints dictionary
    # Only propagate if output dimension exactly matches an input dimension
    for out_idx, out_size in enumerate(output_shape):
        if out_taints[out_idx] is None:  # Only fill if not already set
            if out_size in input_dim_taints:
                taint_to_use = input_dim_taints[out_size]
                # Check if we somehow have a string instead of a Taint/DimTaint
                if isinstance(taint_to_use, str):
                    print(f"[ERROR] input_dim_taints[{out_size}] is a string: {taint_to_use[:100]}...")
                    # Try to recover
                    import re
                    if 'DimTaint' in taint_to_use:
                        match = re.search(r'DimTaint\((\w+)', taint_to_use)
                        if match:
                            taint_name = match.group(1)
                            taint_to_use = Taint(taint_name)
                    else:
                        taint_to_use = Taint(taint_to_use)

                out_taints[out_idx] = _to_dim_taint(taint_to_use)
                if _debug_propagate:
                    print(f"[DEBUG propagate] Direct match: out_taints[{out_idx}] (size={out_size}) = {taint_to_use}")

    # COMMENTED OUT: Split logic (coordinate matching)
    # for input_size, input_taint in input_dim_taints.items():
    #     # not a model config --> pass
    #     if input_taint is None:
    #         continue
    #
    #     # we check whether the products match the input shape's dimension
    #     # e.g. input: [768], output: [12, 64] --> 12 and 64 receive 768's taint
    #     for i in range(len(output_shape)):
    #         for j in range(i + 1, len(output_shape) + 1):
    #             product_val = 1
    #             for k in range(i, j):
    #                 product_val *= output_shape[k]
    #             if product_val == input_size:
    #                 if _debug_propagate:
    #                     print(f"[DEBUG propagate] Match: input_size={input_size} at range [{i},{j})")
    #                 # Skip if ANY dim in range was claimed by view or slice logic
    #                 if any(k in view_claimed_dims or k in slice_claimed_dims for k in range(i, j)):
    #                     if _debug_propagate:
    #                         print(f"[DEBUG propagate] Skipping - dim claimed by view/slice")
    #                     continue
    #                 for k in range(i, j):
    #                     if output_shape[k] == 1 and input_size != 1:
    #                         if _debug_propagate:
    #                             print(f"[DEBUG propagate] Skipping dim {k} - output is 1")
    #                         continue
    #                     # Check if dimension is untainted (None or DimTaint(None))
    #                     # Don't overwrite if it already has a taint
    #                     is_untainted = (out_taints[k] is None or
    #                                    (isinstance(out_taints[k], DimTaint) and out_taints[k].taint is None))
    #                     if is_untainted:
    #                         if _debug_propagate:
    #                             print(f"[DEBUG propagate] Setting out_taints[{k}] = {input_taint}")
    #                         out_taints[k] = _to_dim_taint(input_taint)

    if _debug_propagate:
        print(f"[DEBUG propagate] out_taints after coord fallback: {out_taints}")

    # COMMENTED OUT: Merge fallback logic
    # # MERGE FALLBACK: Check if any untainted output dimensions are products of input dimensions
    # # This handles cases like (7, 8, 64) -> (7, 512) where 512 = 8 * 64
    # if len(input_infos) == 1:
    #     input_shape, input_taints_tuple = input_infos[0]
    #
    #     for out_idx in range(len(output_shape)):
    #         # Skip if already tainted
    #         if out_taints[out_idx] is not None:
    #             continue
    #
    #         out_size = output_shape[out_idx]
    #
    #         # Try to find a contiguous range of input dimensions that multiply to this output size
    #         for i in range(len(input_shape)):
    #             for j in range(i + 1, len(input_shape) + 1):
    #                 product_val = 1
    #                 for k in range(i, j):
    #                     product_val *= input_shape[k]
    #
    #                 if product_val == out_size and j - i > 1:  # Must merge at least 2 dimensions
    #                     # Found it! Merge the taints from input dimensions [i, j)
    #                     if _debug_propagate:
    #                         print(f"[DEBUG propagate] MERGE: output[{out_idx}] (size={out_size}) = product of input[{i}:{j}]")
    #
    #                     # Collect taints and dimension values from the merged input dimensions
    #                     merged_taints_to_combine = []
    #                     merged_dims_to_combine = []
    #                     for k in range(i, j):
    #                         if k < len(input_taints_tuple) and input_taints_tuple[k] is not None:
    #                             merged_taints_to_combine.append(input_taints_tuple[k])
    #                             merged_dims_to_combine.append(input_shape[k])
    #
    #                     if merged_taints_to_combine:
    #                         # Use TaintedInt multiplication logic to merge taints with actual dimension values
    #                         from .types import TaintedInt
    #                         merged_result = None
    #                         merged_value = 1
    #                         for taint, dim_value in zip(merged_taints_to_combine, merged_dims_to_combine):
    #                             # Extract raw taint from DimTaint if needed
    #                             raw_taint = taint.taint if isinstance(taint, DimTaint) else taint
    #
    #                             if merged_result is None:
    #                                 merged_result = raw_taint
    #                                 merged_value = dim_value
    #                             else:
    #                                 # Merge using TaintedInt multiplication logic with actual dimension values
    #                                 dummy1 = TaintedInt(merged_value, merged_result if not isinstance(merged_result, DimTaint) else merged_result.taint)
    #                                 dummy2 = TaintedInt(dim_value, raw_taint)
    #                                 result = (dummy1 * dummy2)._taint
    #                                 merged_value = merged_value * dim_value
    #                                 # The result is already a DimTaint with merge_history from multiplication
    #                                 merged_result = result
    #
    #                         # merged_result is already a DimTaint if multiplication was used, otherwise wrap it
    #                         if isinstance(merged_result, DimTaint):
    #                             out_taints[out_idx] = merged_result
    #                         else:
    #                             out_taints[out_idx] = DimTaint.from_taint(merged_result) if merged_result is not None else None
    #                         if _debug_propagate:
    #                             print(f"[DEBUG propagate] Merged taints: {merged_taints_to_combine} -> {merged_result}")
    #                         break
    #             if out_taints[out_idx] is not None:
    #                 break

    # REGISTRY LOOKUP: For any remaining None values, try to look them up in the registry
    from .registry import lookup_taint
    for out_idx, out_size in enumerate(output_shape):
        if out_taints[out_idx] is None:
            # Try registry lookup
            registry_info = lookup_taint(out_size)
            if registry_info is not None:
                if isinstance(registry_info, dict):
                    # MIX taint with components
                    out_taints[out_idx] = DimTaint(registry_info['taint'],
                                                   merge_history=registry_info['components'])
                    if _debug_propagate:
                        print(f"[DEBUG propagate] Registry lookup: out_taints[{out_idx}] (size={out_size}) = MIX with {registry_info['components']}")
                else:
                    # Pure taint
                    # Debug what's actually in registry_info
                    if isinstance(registry_info, DimTaint):
                        print(f"[ERROR] Registry returned DimTaint for size {out_size}:")
                        print(f"  registry_info = {registry_info}")
                        print(f"  registry_info.taint = {registry_info.taint}")
                        print(f"  type(registry_info.taint) = {type(registry_info.taint)}")
                        # Use the registry_info directly since it's already a DimTaint
                        out_taints[out_idx] = registry_info
                    else:
                        out_taints[out_idx] = _to_dim_taint(registry_info)
                    if _debug_propagate:
                        print(f"[DEBUG propagate] Registry lookup: out_taints[{out_idx}] (size={out_size}) = {registry_info}")

    if _debug_propagate:
        print(f"[DEBUG propagate] FINAL out_taints: {out_taints}")

    return tuple(out_taints)
