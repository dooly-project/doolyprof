from dataclasses import dataclass
from typing import Optional
import torch

__all__ = ['Taint', 'DimTaint', 'TaintedInt', 'TaintedFloat', 'TaintedList', 'TaintedShape']


@dataclass(frozen=True)
class Taint:
    name: str
    def __repr__(self):
        return self.name


@dataclass
class DimTaint:
    """Taint metadata for a single dimension, with optional merge history.

    Attributes:
        taint: The display taint (MIX if merge_history has multiple types, otherwise pure)
        merge_history: Optional dict mapping Taint -> quantity for product-aggregation.
                      Example: {Taint('NUM_TOK'): 6, Taint('MODEL_CONFIG'): 40}
                      Represents a dimension semantically composed of 6*40 structure.
    """
    taint: Optional[Taint]
    merge_history: Optional[dict] = None  # dict[Taint, int]

    @property
    def is_definitely_independent(self) -> bool:
        """True if this dimension can be safely swept for profiling."""
        return self.taint is None

    @property
    def has_history(self) -> bool:
        """True if this dimension has merge history."""
        return self.merge_history is not None and len(self.merge_history) > 0

    def display(self, size: int) -> str:
        """Format for logging: MC(128) or ?(7) or MIX(240)"""
        if self.taint:
            return f"{self.taint}({size})"
        return f"?({size})"

    def __repr__(self):
        if self.merge_history:
            # merge_history is {Taint: quantity}, format as "taint:qty"
            history_str = ', '.join(f"{taint}:{qty}" for taint, qty in self.merge_history.items())
            return f"DimTaint({self.taint}, history={{{history_str}}})"
        return f"DimTaint({self.taint})"

    # def get_taint_quantities(self) -> dict:
        # """Extract taint quantities for aggregation.

        # Returns dict[Taint, int] mapping each taint type to its quantity.
        # If merge_history exists, returns it directly.
        # If simple taint exists, returns {taint: 1} (implicit quantity of 1).
        # If no taint, returns empty dict.
        # """
        # if self.merge_history:
        #     return self.merge_history.copy()
        # elif self.taint:
        #     return {self.taint: 1}
        # else:
        #     return {}

    @staticmethod
    def from_taint(taint: Optional[Taint]) -> 'DimTaint':
        """Create a simple DimTaint from a plain Taint (no history)."""
        return DimTaint(taint=taint, merge_history=None)

    @staticmethod
    def from_history(history: dict) -> 'DimTaint':
        """Create a DimTaint from taint quantities.

        Args:
            history: dict[Taint, int] mapping taint types to quantities

        Returns:
            DimTaint with appropriate display taint:
            - None if history is empty
            - Pure taint if only one type in history
            - MIX if multiple types in history
        """
        if not history:
            return DimTaint(taint=None, merge_history=None)

        # Filter out any quantities that are 1 or less for display simplification
        # But keep them in merge_history for future operations
        taint_types = list(history.keys())

        if len(taint_types) == 1:
            # Pure dimension
            taint = taint_types[0]
            return DimTaint(taint=taint, merge_history=history)
        else:
            # Mixed dimension
            return DimTaint(taint=Taint('MIX'), merge_history=history)


def _int_val(x):
    """Extract plain int, avoiding recursion for TaintedInt."""
    return int.__int__(x) if isinstance(x, TaintedInt) else x


def _float_val(x):
    """Extract plain float, avoiding recursion for TaintedFloat."""
    if isinstance(x, TaintedFloat):
        return float.__float__(x)
    if isinstance(x, TaintedInt):
        return int.__int__(x)
    return x


class TaintedList(list):
    """List subclass that carries a Taint label for its length.

    This preserves semantic information about the list's size dimension
    while keeping individual element taints unchanged.

    Example:
        [x] * TaintedInt(263, MAX_NUM_REQS)
        -> TaintedList([x, x, ...], length_taint=MAX_NUM_REQS)
        -> len(result) returns TaintedInt(263, MAX_NUM_REQS)
    """

    def __init__(self, items, length_taint=None):
        """Initialize a TaintedList.

        Args:
            items: Iterable of items to populate the list
            length_taint: Optional Taint object for the list's length dimension
        """
        super().__init__(items)
        self._length_taint = length_taint

    def __len__(self):
        """Return plain length.

        Note: We return plain int here to avoid polluting arithmetic operations.
        The length_taint is preserved in the object and can be accessed via
        the length_taint property when needed (e.g., during np.array conversion).
        """
        return super().__len__()

    @property
    def length_taint(self):
        """Get the length taint."""
        return self._length_taint

    def __repr__(self):
        if self._length_taint:
            return f"TaintedList({list.__repr__(self)}, length_taint={self._length_taint})"
        return f"TaintedList({list.__repr__(self)})"


def resolve_taint_from_operands(self_taint, other_taint, result_val, self_val=None, other_val=None, op_name=None):
    """
    Apply taint propagation rules to determine result taint from two operands.

    Args:
        self_taint: Taint from first operand (can be Taint, DimTaint, or None)
        other_taint: Taint from second operand (can be Taint, DimTaint, or None)
        result_val: The numeric result value (used for registry lookup)
        self_val: Value of first operand (used for multiplication merge_history)
        other_val: Value of second operand (used for multiplication merge_history)
        op_name: Operation name ('mul' for special multiplication handling)

    Returns:
        Resolved taint for the result (Taint, DimTaint, or None)

    Rules:
        1. If both have same base taint (or one is None): preserve non-None taint
        2. If both have different taints but both contain CONFIG: merge to CONFIG
        3. If one is NUM_XX and other contains CONFIG: use MIX
        4. Otherwise: untaint (conservative)

    Special case: Multiplication creates merge_history from operand taints and values.
    """
    def get_base_taint(t):
        return t.taint if isinstance(t, DimTaint) else t

    def get_merge_history_with_value(taint, value):
        """Extract or create merge_history for a taint with its value.

        Ensures all keys in merge_history are base Taint objects (not DimTaint).
        """
        if isinstance(taint, DimTaint) and taint.merge_history:
            # Already has merge_history - normalize keys to base Taint
            normalized = {}
            for t, qty in taint.merge_history.items():
                base_t = get_base_taint(t) if isinstance(t, DimTaint) else t
                normalized[base_t] = qty
            # print()[f"[DEBUG get_merge_history_with_value] Using existing merge_history for taint={taint} with value={value} -> normalized={normalized}"]
            return normalized
        elif taint is not None:
            # Simple taint - create merge_history with the value
            base = get_base_taint(taint)
            # print(f"[DEBUG get_merge_history_with_value] Creating merge_history for taint={taint} with value={value} -> base={base}")
            if base:
                return {base: value}
            
        
        return {}

    # SPECIAL CASE: Multiplication creates merge_history
    if op_name == 'mul' and self_taint and other_taint:
        self_history = get_merge_history_with_value(self_taint, self_val)
        other_history = get_merge_history_with_value(other_taint, other_val)

        # Merge histories: combine components (multiply quantities of same taint)
        combined_history = {}
        for taint, qty in self_history.items():
            combined_history[taint] = combined_history.get(taint, 1) * qty
        for taint, qty in other_history.items():
            combined_history[taint] = combined_history.get(taint, 1) * qty

        # Determine result taint based on combined history
        unique_taints = set(combined_history.keys())
        if len(unique_taints) == 1:
            # All same taint - pure (ensure it's a base Taint, not DimTaint)
            result_taint = list(unique_taints)[0]
            # Safety: unwrap if somehow a DimTaint got in
            if isinstance(result_taint, DimTaint):
                result_taint = result_taint.taint
        else:
            # Mixed taints
            result_taint = Taint('MIX')

        # print(f"[DEBUG resolve_taint_from_operands] Multiplication merge: self_history={self_history}, other_history={other_history} -> combined_history={combined_history}, result_taint={result_taint}")
        
        return DimTaint(result_taint, merge_history=combined_history)

    # Extract base taints
    self_base = get_base_taint(self_taint) if self_taint else None
    other_base = get_base_taint(other_taint) if other_taint else None

    # Rule 1: Same base taint or one is None
    if self_base == other_base or self_taint is None or other_taint is None:
        base_taint = self_base or other_base

        if base_taint:
            # Look up existing taint from registry to preserve merge_history
            from .registry import lookup_taint
            existing = lookup_taint(result_val)
            if existing:
                if isinstance(existing, dict):
                    existing_taint = existing.get('taint')
                    existing_components = existing.get('components')
                    existing_base = get_base_taint(existing_taint) if existing_taint else None
                    if existing_base == base_taint:
                        return DimTaint(existing_taint, merge_history=existing_components)
                    else:
                        return base_taint
                elif isinstance(existing, DimTaint):
                    if existing.taint == base_taint:
                        return existing
                    else:
                        return base_taint
                elif existing == base_taint:
                    return existing
                else:
                    return base_taint
            else:
                return base_taint
        else:
            return None
    else:
        # Different base taints - apply rules
        self_str = str(self_base)
        other_str = str(other_base)

        # Determine base taint using rules
        base_taint = None
        # Rule 2: Both contain CONFIG → CONFIG
        if 'CONFIG' in self_str and 'CONFIG' in other_str:
            base_taint = Taint('MODEL_CONFIG')
        # Rule 3: NUM + CONFIG → MIX
        elif (('NUM' in self_str or 'MIX' in other_str) and 'CONFIG' in other_str) or \
             ('CONFIG' in self_str and ('NUM' in other_str or 'MIX' in other_str)):
            base_taint = Taint('MIX')
        else:
            # NUM_REQ (op) NUM_TOK → untaint (conservative)
            return None

        # Look up in registry to preserve merge_history
        if base_taint:
            from .registry import lookup_taint
            existing = lookup_taint(result_val)
            if existing:
                if isinstance(existing, dict):
                    existing_taint = existing.get('taint')
                    existing_components = existing.get('components')
                    if existing_taint == base_taint:
                        return DimTaint(existing_taint, merge_history=existing_components)
                    else:
                        return base_taint
                elif isinstance(existing, DimTaint):
                    if existing.taint == base_taint:
                        return existing
                    else:
                        return base_taint
                elif existing == base_taint:
                    return existing
                else:
                    return base_taint
            else:
                return base_taint
        else:
            return None


class TaintedInt(int):
    """int subclass that carries a Taint label through arithmetic."""
    # Note: int subclasses don't support nonempty __slots__.

    def __new__(cls, value, taint=None):
        # int.__new__ sets the immutable value; use _int_val to unwrap
        # any existing TaintedInt without triggering our overrides.
        instance = int.__new__(cls, _int_val(value))
        instance._taint = Taint(taint) if isinstance(taint, str) else taint

        # actual_value = int(instance)
        # if actual_value == 4 and instance._taint is None:
        #     import traceback
        #     print(f"\n[TAINTEDINT CREATE DEBUG] Creating TaintedInt(4, None)")
        #     print(f"  Traceback:")
        #     traceback.print_stack(limit=15)

        # Register tainted values only when tracking is enabled
        # This prevents false collisions during warm-up/dummy runs
        # If taint is a DimTaint, register_taint will automatically extract components
        # if instance._taint is not None:
            # from . import is_taint_tracking_enabled
            # if is_taint_tracking_enabled():
                # from .registry import register_taint
                # register_taint(int(instance), instance._taint)

        return instance

    @property
    def value(self): return int.__int__(self)
    @property
    def taint(self): return self._taint
    def __repr__(self):
        v = int.__repr__(self)
        return f"{v}:{self._taint}" if self._taint else v
    def __str__(self):
        return int.__repr__(self)
    def __format__(self, spec):
        return int.__format__(self, spec)

    def __getnewargs_ex__(self):
        return ((int.__int__(self),), {"taint": getattr(self, "_taint", None)})

    def _binop(self, other, op, op_name=None):
        """Compute op(self, other) using plain ints, wrap result as TaintedInt if int.

        Note: This always propagates taints regardless of the global tracking flag.
        The flag only controls registration in register_taint(), not propagation.

        Args:
            other: Other operand
            op: Operation function (lambda a, b: ...)
            op_name: Optional operation name ('mul' for multiplication merge_history)
        """
        a, b = int.__int__(self), _int_val(other)
        val = op(a, b)

        # Apply taint propagation rules using centralized helper
        self_taint = self._taint
        other_taint = other._taint if isinstance(other, (TaintedInt, TaintedFloat)) else None
        
        # print(f"[DEBUG _binop] Operation: {op_name}, self: {a}:{self_taint}, other: {b}:{other_taint} -> result value: {val}")
        taint = resolve_taint_from_operands(self_taint, other_taint, val,
                                           self_val=a, other_val=b, op_name=op_name)

        if isinstance(val, int):
            return TaintedInt(val, taint)
        if isinstance(val, float):
            return TaintedFloat(val, taint)
        return val

        # # OLD INLINE LOGIC (commented out - now using resolve_taint_from_operands):
        # def get_base_taint(t):
        #     return t.taint if isinstance(t, DimTaint) else t
        #
        # if isinstance(val, int):
        #     # Helper to extract base taint (unwrap DimTaint wrapper)
        #
        #     # Determine taint for result
        #     self_taint = self._taint
        #     other_taint = other._taint if isinstance(other, TaintedInt) else None
        #
        #     # Extract base taints for comparison (handles DimTaint with different merge_history)
        #     self_base = get_base_taint(self_taint) if self_taint else None
        #     other_base = get_base_taint(other_taint) if other_taint else None
        #
        #     # Rule 1: If both have same base taint (or one is None), preserve the non-None taint
        #     if self_base == other_base or self_taint is None or other_taint is None:
        #         taint = self_taint or other_taint
        #
        #         # If taint is a DimTaint with merge_history, update it conservatively
        #         if isinstance(taint, DimTaint) and taint.merge_history:
        #             if len(taint.merge_history) == 1:
        #                 # Pure taint: check if value matches, if not update it
        #                 pure_taint, old_val = list(taint.merge_history.items())[0]
        #                 if old_val != val:
        #                     # Value changed - update merge_history to match
        #                     taint = DimTaint(pure_taint, merge_history={pure_taint: val})
        #             else:
        #                 # MIX taint: cannot safely determine how arithmetic affects components
        #                 # Be conservative and drop the taint
        #                 taint = None
        #     else:
        #         # Different base taints, both non-None
        #         self_str = str(self_base)
        #         other_str = str(other_base)
        #
        #         # Determine what the base taint should be using rules
        #         base_taint = None
        #         # Rule 2: If both have different taints but both contain CONFIG, merge to CONFIG
        #         if 'CONFIG' in self_str and 'CONFIG' in other_str:
        #             base_taint = Taint('CONFIG')
        #         # Rule 3: If one is NUM_XX (including MAX_NUM_) and other contains CONFIG, use MIX
        #         elif (('NUM' in self_str or 'MIX' in other_str) and 'CONFIG' in other_str) or \
        #              ('CONFIG' in self_str and ('NUM' in other_str or 'MIX' in other_str)):
        #             base_taint = Taint('MIX')
        #         else:
        #             # Other cases: untaint to be conservative
        #             base_taint = None
        #
        #         # If we determined a base taint, look up existing taint from registry to preserve merge_history
        #         if base_taint:
        #             from .registry import lookup_taint
        #             existing = lookup_taint(val)
        #             if existing:
        #                 # Registry can return: Taint, DimTaint, or dict{'taint': MIX, 'components': {...}}
        #                 if isinstance(existing, dict):
        #                     # MIX taint with components stored as dict - reconstruct DimTaint
        #                     existing_taint = existing.get('taint')
        #                     existing_components = existing.get('components')
        #                     if existing_taint == base_taint:
        #                         # Convert dict to DimTaint for use
        #                         taint = DimTaint(existing_taint, merge_history=existing_components)
        #                     else:
        #                         taint = base_taint  # Base taint mismatch
        #                 elif isinstance(existing, DimTaint):
        #                     # DimTaint object
        #                     if existing.taint == base_taint:
        #                         taint = existing  # Use existing with merge_history
        #                     else:
        #                         taint = base_taint  # Base taint mismatch
        #                 elif existing == base_taint:
        #                     # Plain Taint match
        #                     taint = existing
        #                 else:
        #                     taint = base_taint  # Taint mismatch, use new
        #             else:
        #                 taint = base_taint  # Not in registry, use new
        #         else:
        #             taint = None
        #
        #     # DEBUG: Track binop operations for specific values
        #     # if val in [10372, 10353, 524288, 248320]:
        #     #     import sys
        #     #     print(f"\n[BINOP DEBUG] TaintedInt({val}) from {op.__name__ if hasattr(op, '__name__') else 'op'}", flush=True)
        #     #     print(f"  {a} {op.__name__ if hasattr(op, '__name__') else 'op'} {b} = {val}", flush=True)
        #     #     print(f"  self({a})._taint: {self_taint}", flush=True)
        #     #     print(f"  other({b})._taint: {other_taint}", flush=True)
        #     #     if isinstance(self_taint, DimTaint):
        #     #         print(f"  self_base: {self_base}, merge_history: {self_taint.merge_history}", flush=True)
        #     #     if isinstance(other_taint, DimTaint):
        #     #         print(f"  other_base: {other_base}, merge_history: {other_taint.merge_history}", flush=True)
        #     #     print(f"  result_taint: {taint}", flush=True)
        #     #     if isinstance(taint, DimTaint):
        #     #         print(f"  result merge_history: {taint.merge_history}", flush=True)
        #     #     sys.stdout.flush()
        #
        #     return TaintedInt(val, taint)
        # if isinstance(val, float):
        #     # Same logic for floats - use registry lookup to preserve merge_history
        #     self_taint = self._taint
        #     other_taint = other._taint if isinstance(other, (TaintedInt, TaintedFloat)) else None
        #
        #     # Extract base taints for comparison (handles DimTaint with different merge_history)
        #     self_base = get_base_taint(self_taint) if self_taint else None
        #     other_base = get_base_taint(other_taint) if other_taint else None
        #
        #     # Rule 1: If both have same base taint (or one is None), preserve the non-None taint
        #     if self_base == other_base or self_taint is None or other_taint is None:
        #         # Determine the base taint to use
        #         base_taint = self_base or other_base
        #
        #         if base_taint:
        #             # Look up existing taint from registry to preserve merge_history
        #             from .registry import lookup_taint
        #             existing = lookup_taint(val)
        #             if existing:
        #                 # Registry can return: Taint, DimTaint, or dict{'taint': MIX, 'components': {...}}
        #                 if isinstance(existing, dict):
        #                     # MIX taint with components stored as dict - reconstruct DimTaint
        #                     existing_taint = existing.get('taint')
        #                     existing_components = existing.get('components')
        #                     existing_base = get_base_taint(existing_taint) if existing_taint else None
        #                     if existing_base == base_taint:
        #                         taint = DimTaint(existing_taint, merge_history=existing_components)
        #                     else:
        #                         taint = base_taint  # Base taint mismatch
        #                 elif isinstance(existing, DimTaint):
        #                     # DimTaint object
        #                     if existing.taint == base_taint:
        #                         taint = existing  # Use existing with merge_history
        #                     else:
        #                         taint = base_taint  # Base taint mismatch
        #                 elif existing == base_taint:
        #                     # Plain Taint match
        #                     taint = existing
        #                 else:
        #                     taint = base_taint  # Taint mismatch
        #             else:
        #                 taint = base_taint  # Not in registry, use new
        #         else:
        #             # Both are None
        #             taint = None
        #     else:
        #         # Different base taints, both non-None
        #         self_str = str(self_base)
        #         other_str = str(other_base)
        #
        #         # Determine what the base taint should be using rules
        #         base_taint = None
        #         # Rule 2: If both have different taints but both contain CONFIG, merge to CONFIG
        #         if 'CONFIG' in self_str and 'CONFIG' in other_str:
        #             base_taint = Taint('CONFIG')
        #         # Rule 3: If one is NUM_XX (including MAX_NUM_) and other contains CONFIG, use MIX
        #         elif (('NUM' in self_str or 'MIX' in other_str) and 'CONFIG' in other_str) or \
        #              ('CONFIG' in self_str and ('NUM' in other_str or 'MIX' in other_str)):
        #             base_taint = Taint('MIX')
        #         else:
        #             # Other cases: untaint to be conservative
        #             base_taint = None
        #
        #         # If we determined a base taint, look up existing taint from registry to preserve merge_history
        #         if base_taint:
        #             from .registry import lookup_taint
        #             existing = lookup_taint(val)
        #             if existing:
        #                 # Registry can return: Taint, DimTaint, or dict{'taint': MIX, 'components': {...}}
        #                 if isinstance(existing, dict):
        #                     # MIX taint with components stored as dict - reconstruct DimTaint
        #                     existing_taint = existing.get('taint')
        #                     existing_components = existing.get('components')
        #                     if existing_taint == base_taint:
        #                         # Convert dict to DimTaint for use
        #                         taint = DimTaint(existing_taint, merge_history=existing_components)
        #                     else:
        #                         taint = base_taint  # Base taint mismatch
        #                 elif isinstance(existing, DimTaint):
        #                     # DimTaint object
        #                     if existing.taint == base_taint:
        #                         taint = existing  # Use existing with merge_history
        #                     else:
        #                         taint = base_taint  # Base taint mismatch
        #                 elif existing == base_taint:
        #                     # Plain Taint match
        #                     taint = existing
        #                 else:
        #                     taint = base_taint  # Taint mismatch, use new
        #             else:
        #                 taint = base_taint  # Not in registry, use new
        #         else:
        #             taint = None
        #
        #     return TaintedFloat(val, taint)
        # return val

    def __add__(self, other):
        return self._binop(other, lambda a, b: a + b)
    def __radd__(self, other): return self.__add__(other)
    def __sub__(self, other):
        if isinstance(other, torch.Tensor):
            return int.__int__(self) - other
        return self._binop(other, lambda a, b: a - b)
    def __rsub__(self, other):
        if isinstance(other, torch.Tensor):
            return other - int.__int__(self)
        # Use _binop for proper taint propagation: other - self
        if not isinstance(other, TaintedInt):
            other = TaintedInt(other, None)
        return other._binop(self, lambda a, b: a - b)
    def __mul__(self, other):
        # Handle list replication: TaintedInt(n, taint) * [x]
        if isinstance(other, list):
            result_items = other * int(self)
            result = TaintedList(result_items, length_taint=self._taint)
            return result
        return self._binop(other, lambda a, b: a * b, op_name='mul')

    def __rmul__(self, other):
        # Handle list replication: list * TaintedInt(n, taint)
        if isinstance(other, list):
            result_items = other * int(self)
            result = TaintedList(result_items, length_taint=self._taint)
            return result
        # For numeric multiplication, use __mul__ which has op_name='mul'
        return self.__mul__(other)
    def __floordiv__(self, other):
        return self._binop(other, lambda a, b: a // b)
    def __rfloordiv__(self, other):
        # Use _binop for proper taint propagation: other // self
        if not isinstance(other, TaintedInt):
            other = TaintedInt(other, None)
        return other._binop(self, lambda a, b: a // b)
    def __lt__(self, other):
        return int.__int__(self) < _int_val(other)
    def __le__(self, other):
        return int.__int__(self) <= _int_val(other)
    def __gt__(self, other):
        return int.__int__(self) > _int_val(other)
    def __ge__(self, other):
        return int.__int__(self) >= _int_val(other)
    def __mod__(self, other):
        return self._binop(other, lambda a, b: a % b)
    def __rmod__(self, other):
        # Use _binop for proper taint propagation: other % self
        if not isinstance(other, TaintedInt):
            other = TaintedInt(other, None)
        return other._binop(self, lambda a, b: a % b)
    def __truediv__(self, other):
        val = int.__int__(self) / _int_val(other)
        if isinstance(val, float):
            taint = self._taint or (other._taint if isinstance(other, (TaintedInt, TaintedFloat)) else None)
            return TaintedFloat(val, taint)
        return val
    def __rtruediv__(self, other):
        # Reverse true division: other / self with full taint propagation
        val = _int_val(other) / int.__int__(self)
        if isinstance(val, float):
            self_taint = self._taint
            other_taint = other._taint if isinstance(other, (TaintedInt, TaintedFloat)) else None
            taint = resolve_taint_from_operands(other_taint, self_taint, val)
            return TaintedFloat(val, taint)
        return val
    def __neg__(self):
        return TaintedInt(int.__neg__(self), self._taint)
    def __pos__(self):
        return TaintedInt(int.__pos__(self), self._taint)
    def __abs__(self):
        return TaintedInt(int.__abs__(self), self._taint)
    def __pow__(self, other):
        val = int.__int__(self) ** _int_val(other)
        if isinstance(val, int):
            taint = self._taint or (other._taint if isinstance(other, (TaintedInt, TaintedFloat)) else None)
            return TaintedInt(val, taint)
        if isinstance(val, float):
            taint = self._taint or (other._taint if isinstance(other, (TaintedInt, TaintedFloat)) else None)
            return TaintedFloat(val, taint)
        return val
    def __rpow__(self, other):
        # Use _binop for proper taint propagation: other ** self
        if not isinstance(other, (TaintedInt, TaintedFloat)):
            other = TaintedInt(other, None) if isinstance(other, int) else TaintedFloat(other, None)
        return other._binop(self, lambda a, b: a ** b)

class TaintedFloat(float):
    """float subclass that carries a Taint label through arithmetic."""

    def __new__(cls, value, taint=None):
        instance = float.__new__(cls, _float_val(value))
        instance._taint = Taint(taint) if isinstance(taint, str) else taint

        # Register tainted values only when tracking is enabled
        # This prevents false collisions during warm-up/dummy runs
        # if instance._taint is not None:
        #     from . import is_taint_tracking_enabled
        #     if is_taint_tracking_enabled():
        #         from .registry import register_taint
        #         print(f"[DEBUG @ TaintedFloat] Registering value {float(instance)} with taint {instance._taint}")
        #         register_taint(float(instance), instance._taint)

        return instance

    @property
    def value(self): return float.__float__(self)
    @property
    def taint(self): return self._taint

    def __repr__(self):
        v = float.__repr__(self)
        return f"{v}:{self._taint}" if self._taint else v
    def __str__(self):
        return float.__repr__(self)
    def __format__(self, spec):
        return float.__format__(self, spec)

    def __getnewargs_ex__(self):
        return ((float.__float__(self),), {"taint": getattr(self, "_taint", None)})

    def _binop(self, other, op, op_name=None):
        """Compute op(self, other) using plain floats, with full taint propagation.

        Args:
            other: Other operand
            op: Operation function (lambda a, b: ...)
            op_name: Optional operation name ('mul' for multiplication merge_history)
        """
        a, b = float.__float__(self), _float_val(other)
        val = op(a, b)

        # Apply taint propagation rules
        self_taint = self._taint
        other_taint = other._taint if isinstance(other, (TaintedInt, TaintedFloat)) else None
        taint = resolve_taint_from_operands(self_taint, other_taint, val,
                                           self_val=a, other_val=b, op_name=op_name)

        if isinstance(val, float):
            return TaintedFloat(val, taint)
        if isinstance(val, int):
            return TaintedInt(val, taint)
        return val

    def __add__(self, other): return self._binop(other, lambda a, b: a + b)
    def __radd__(self, other): return self.__add__(other)
    def __sub__(self, other): return self._binop(other, lambda a, b: a - b)
    def __rsub__(self, other):
        # Use _binop for proper taint propagation: other - self
        if not isinstance(other, (TaintedInt, TaintedFloat)):
            other = TaintedFloat(other, None) if isinstance(other, float) else TaintedInt(other, None)
        return other._binop(self, lambda a, b: a - b)
    def __mul__(self, other): return self._binop(other, lambda a, b: a * b, op_name='mul')
    def __rmul__(self, other): return self.__mul__(other)
    def __truediv__(self, other): return self._binop(other, lambda a, b: a / b)
    def __rtruediv__(self, other):
        # Reverse true division: other / self with full taint propagation
        val = _float_val(other) / float.__float__(self)
        if isinstance(val, float):
            self_taint = self._taint
            other_taint = other._taint if isinstance(other, (TaintedInt, TaintedFloat)) else None
            taint = resolve_taint_from_operands(other_taint, self_taint, val)
            return TaintedFloat(val, taint)
        return val
    def __neg__(self): return TaintedFloat(float.__neg__(self), self._taint)
    def __pos__(self): return TaintedFloat(float.__pos__(self), self._taint)
    def __abs__(self): return TaintedFloat(float.__abs__(self), self._taint)
    def __pow__(self, other): return self._binop(other, lambda a, b: a ** b)
    def __rpow__(self, other):
        # Use _binop for proper taint propagation: other ** self
        if not isinstance(other, (TaintedInt, TaintedFloat)):
            other = TaintedFloat(other, None) if isinstance(other, float) else TaintedInt(other, None)
        return other._binop(self, lambda a, b: a ** b)
    def __lt__(self, other): return float.__float__(self) < _float_val(other)
    def __le__(self, other): return float.__float__(self) <= _float_val(other)
    def __gt__(self, other): return float.__float__(self) > _float_val(other)
    def __ge__(self, other): return float.__float__(self) >= _float_val(other)


# when tensor.shape is called, it returns a TaintedShape object.
# each shape is a TaintedInt object.
class TaintedShape(tuple):

    def __new__(cls, sizes, taints):
        instance = super().__new__(cls, sizes)
        instance._taints = taints
        return instance

    # override __getitem__ to return TaintedInt objects instead of raw integers
    def __getitem__(self, idx):
        # if the index is a slice, return a new TaintedShape object with the sliced sizes and taints
        if isinstance(idx, slice):
            return TaintedShape(super().__getitem__(idx), self._taints[idx])
        # otherwise, return a TaintedInt object with the size and taint at the given index
        size = super().__getitem__(idx)
        taint = self._taints[idx] if idx < len(self._taints) else None

        result = TaintedInt(size, taint)
        return result

    # override __iter__ to iterate over TaintedInt objects instead of raw integers
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # override __repr__ to show the taint information
    def __repr__(self):
        parts = []
        for i in range(len(self)):
            size = super().__getitem__(i)
            taint = self._taints[i] if i < len(self._taints) else None
            if taint:
                # For DimTaint, just show the base taint to avoid verbose output
                if isinstance(taint, DimTaint):
                    taint_str = str(taint.taint) if taint.taint else "?"
                else:
                    taint_str = str(taint)
                parts.append(f"{size}:{taint_str}")
            else:
                parts.append(str(size))
        return f"TaintedShape([{', '.join(parts)}])"

    def __add__(self, other):
        if isinstance(other, TaintedShape):
            return TaintedShape(
                tuple.__add__(self, other),
                self._taints + other._taints,
            )
        elif isinstance(other, (tuple, list)):
            new_taints = self._taints + tuple(
                s._taint if isinstance(s, TaintedInt) else None for s in other
            )
            return TaintedShape(tuple.__add__(self, other), new_taints)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (tuple, list)):
            new_taints = tuple(
                s._taint if isinstance(s, TaintedInt) else None for s in other
            ) + self._taints
            return TaintedShape(tuple.__add__(tuple(other), self), new_taints)
        return NotImplemented

    @property
    def taints(self): return self._taints

    # Pickle support: return arguments to pass to __new__
    def __getnewargs__(self):
        return (tuple(self), self._taints)
