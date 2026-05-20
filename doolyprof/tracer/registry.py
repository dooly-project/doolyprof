"""Global taint registry for value→taint mapping.

This module maintains a global registry that maps dimension values to their semantic taints.
It is populated from two sources:
1. Config tainting: MODEL_CONFIG values from model configurations
2. Reshape operations: MIX values created during reshape with their component decomposition

The registry enables efficient taint lookup when resolving -1 dimensions in reshape operations.
"""

from typing import Optional, Dict, Set
from .types import Taint

__all__ = ['TAINT_REGISTRY', 'register_taint', 'lookup_taint', 'clear_registry']

# Global taint registry: value → taint info
# - For pure taints: value → Taint object
# - For MIX taints: value → {'taint': Taint('MIX'), 'components': {Taint, ...}}
TAINT_REGISTRY: Dict[int | float, any] = {}


def register_taint(value: int | float, taint: Taint, components: Optional[Dict[Taint, int]] = None):
    """Register a value→taint mapping in the global registry.

    Args:
        value: The numeric value (dimension size)
        taint: The taint label (NUM_TOK, MODEL_CONFIG, MIX, etc.) or DimTaint object
       components: For MIX taints, the set of component taints
           Example: {Taint('NUM_TOK'), Taint('MODEL_CONFIG')}
           
        # MIX taint with components
        register_taint(240, Taint('MIX'), components={
            Taint('NUM_TOK'),
            Taint('MODEL_CONFIG')
            })
            
    Examples:
        # Pure taint
        register_taint(5120, Taint('MODEL_CONFIG'))

        # MIX taint with components
        register_taint(240, Taint('MIX'), components={
            Taint('NUM_TOK'),
            Taint('MODEL_CONFIG')
        })

        # DimTaint with merge_history (components extracted automatically)
        register_taint(240, DimTaint(Taint('MIX'), merge_history={...}))

    Raises:
        ValueError: If value already exists in registry with different taint/components

    Note:
        This function always registers taints regardless of the global tracking flag.
        The flag only controls arithmetic operations (_binop), not config/reshape registration.
        This ensures config values and reshape dimension semantics are always captured,
        while preventing false taints from warm-up arithmetic operations.
    """
    if value == 1 or value == 0 or value == -1:
        return

    # If taint is a DimTaint object, extract the underlying taint and components
    from .types import DimTaint
    if isinstance(taint, DimTaint):
        if components is None:
            components = taint.merge_history
        taint = taint.taint

        # Check if we have a nested DimTaint
        if isinstance(taint, DimTaint):
            print(f"[ERROR] Nested DimTaint passed to register_taint for value {value}")
            print(f"  Outer DimTaint.taint = {taint}")
            print(f"  Inner DimTaint.taint = {taint.taint}")
            # Extract the actual Taint from the nested structure
            while isinstance(taint, DimTaint):
                taint = taint.taint

    # Check for conflicts if value already exists in registry
    if value in TAINT_REGISTRY:
        existing = TAINT_REGISTRY[value]
        is_new_mix = components is not None and len(components) > 1
        is_existing_mix = isinstance(existing, dict)

        # Build error message components
        if is_existing_mix:
            existing_str = f"MIX {existing.get('components', {})}"
        else:
            existing_str = str(existing)

        if is_new_mix:
            new_str = f"MIX {components}"
        else:
            new_str = str(taint)

        # Check if they are identical (allow identical re-registration)
        is_identical = False
        if is_existing_mix and is_new_mix:
            is_identical = existing.get('components', {}) == components
        elif not is_existing_mix and not is_new_mix:
            is_identical = existing == taint
           
        # small addition to prevent error when testing whether decode has different call path
        # allow overlap for num_req and num_tok since they are flipped from prefill to decode phase change
        # if not is_identical and ((taint == Taint('NUM_REQS') and existing == Taint('NUM_TOKS')) or (taint == Taint('NUM_TOKS') and existing == Taint('NUM_REQS'))):
        #     print(f"[DEBUG] Allowing NUM_REQ/NUM_TOK overlap for value {value} due to prefill/decode phase change")
        #     TAINT_REGISTRY[value] = taint
        #     return
            
        if not is_identical:
            raise ValueError(
                f"\n{'='*80}\n"
                f"TAINT CONFLICT DETECTED\n"
                f"{'='*80}\n"
                f"Value: {value}\n"
                f"Existing: {existing_str}\n"
                f"Attempting to register as: {new_str}\n"
                f"\n"
                f"This means the same dimension value ({value}) is being used for\n"
                f"different semantic purposes in your model configuration.\n"
                f"\n"
                f"SOLUTIONS:\n"
                f"1. Adjust NUM_TOK or NUM_REQS ranges to avoid overlap with MODEL_CONFIG\n"
                f"2. Use different values for different semantic dimensions\n"
                f"3. Check your config values and input buffer sizes\n"
                f"{'='*80}\n"
            )
        # If identical, silently allow re-registration
        return

    # Register the taint (first time)
    if components and len(components) > 1:
        # MIX taint with component history
        TAINT_REGISTRY[value] = {
            'taint': Taint('MIX'),
            'components': components
        }
    else:
        # Pure taint
        TAINT_REGISTRY[value] = taint

    # print(f"[REGISTRY] Registered value {value} with taint {TAINT_REGISTRY[value]}")

def lookup_taint(value: int | float):
    """Look up taint info for a value in the global registry.

    Args:
        value: The numeric value to look up

    Returns:
        - Taint object for pure values
        - Dict with 'taint' and 'components' keys for MIX values
        - None if not found in registry

    Examples:
        >>> register_taint(240, Taint('MIX'), {Taint('NUM_TOK'), Taint('MODEL_CONFIG')})
        >>> lookup_taint(240)
        {'taint': Taint('MIX'), 'components': {Taint('NUM_TOK'), Taint('MODEL_CONFIG')}}
    """
    return TAINT_REGISTRY.get(value)


def clear_registry():
    """Clear the entire registry.

    Useful for testing or when resetting the taint tracking system.
    """
    TAINT_REGISTRY.clear()
