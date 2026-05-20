"""
Taint tracking for tensor dimension semantics.

Provides TaintedTensor (a torch.Tensor subclass) that propagates
semantic labels ("taints") through PyTorch operations, enabling
automatic profiling annotations that show which config-derived
dimensions each op is working on.
"""

from .types import Taint, DimTaint, TaintedInt, TaintedFloat, TaintedShape
from .propagation import propagate_taints
from .config import create_tainted_config, patch_autoconfig, unpatch_autoconfig, patch_input_buffers, unpatch_input_buffers
from .tensor import TaintedTensor, TaintedParameter
from .hooks import (
    install_hooks, uninstall_hooks,
    patch_int_in_modules, unpatch_int_in_modules,
)
from .registry import TAINT_REGISTRY, register_taint, lookup_taint, clear_registry

# Global flag to enable/disable taint tracking
# When False, arithmetic operations on TaintedInt will not propagate taints
# This is useful to skip taint tracking during warm-up/compilation phases
_TAINT_TRACKING_ENABLED = True

def enable_taint_tracking():
    """Enable taint propagation through arithmetic operations."""
    global _TAINT_TRACKING_ENABLED
    _TAINT_TRACKING_ENABLED = True

def disable_taint_tracking():
    """Disable taint propagation through arithmetic operations.

    This is useful during warm-up/compilation phases where the semantics
    don't match actual execution (e.g., using max_num_reqs as num_tokens).
    """
    global _TAINT_TRACKING_ENABLED
    _TAINT_TRACKING_ENABLED = False

def is_taint_tracking_enabled():
    """Check if taint tracking is currently enabled."""
    return _TAINT_TRACKING_ENABLED

__all__ = [
    # Core types
    'Taint', 'DimTaint', 'TaintedInt', 'TaintedFloat', 'TaintedShape', 'TaintedTensor', 'TaintedParameter',
    'create_tainted_config',
    # Hook management
    'install_hooks', 'uninstall_hooks',
    'patch_autoconfig', 'unpatch_autoconfig',
    'patch_input_buffers', 'unpatch_input_buffers',
    'patch_int_in_modules', 'unpatch_int_in_modules',
    # Propagation (for advanced usage)
    'propagate_taints',
    # Registry
    'TAINT_REGISTRY', 'register_taint', 'lookup_taint', 'clear_registry',
    # Enable/disable tracking
    'enable_taint_tracking', 'disable_taint_tracking', 'is_taint_tracking_enabled',
]
