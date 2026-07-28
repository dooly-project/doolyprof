import dataclasses
import threading
import torch
from torch.profiler import record_function

from .types import Taint, DimTaint, TaintedInt, TaintedShape
from .propagation import propagate_taints

__all__ = ['TaintedTensor', 'TaintedParameter']

def _get_plain_taint(t) -> Taint:
    """Extract plain Taint from either Taint or DimTaint."""
    if t is None:
        return None
    if isinstance(t, DimTaint):
        return t.taint
    return t


def _dim_taint_from_registry(size):
    from .registry import lookup_taint

    registry_info = lookup_taint(int(size))
    if registry_info is None:
        return None
    if isinstance(registry_info, dict):
        return DimTaint(
            registry_info["taint"],
            merge_history=registry_info["components"],
        )
    if isinstance(registry_info, DimTaint):
        return registry_info
    return DimTaint.from_taint(registry_info)


def _format_dim(size, dim_taint, include_history=False):
    if dim_taint is None or (
        isinstance(dim_taint, DimTaint) and dim_taint.taint is None
    ):
        dim_taint = _dim_taint_from_registry(size) or dim_taint
    if isinstance(dim_taint, DimTaint):
        actual_taint = dim_taint
        while isinstance(actual_taint.taint, DimTaint):
            if actual_taint.taint.has_history and not actual_taint.has_history:
                actual_taint = actual_taint.taint
            else:
                actual_taint = DimTaint(
                    actual_taint.taint.taint,
                    merge_history=actual_taint.taint.merge_history
                    or actual_taint.merge_history,
                )
        if actual_taint.taint is None:
            registry_taint = _dim_taint_from_registry(size)
            if registry_taint is not None:
                return _format_dim(size, registry_taint, include_history)
            if int(size) == 1:
                return "SINGLETON(1)"
        if include_history and actual_taint.has_history:
            history_str = ",".join(
                f"{taint}:{qty}" for taint, qty in actual_taint.merge_history.items()
            )
            return f"{actual_taint.display(size)}[hist:{history_str}]"
        return actual_taint.display(size)
    if dim_taint is not None:
        return f"{dim_taint}({size})"
    if int(size) == 1:
        return "SINGLETON(1)"
    return f"?({size})"


def _get_target_device(args, kwargs):
    """Find target device - prefer CUDA if any tensor is on CUDA."""
    target_device = None
    def find_device(obj):
        nonlocal target_device
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                target_device = obj.device
            elif target_device is None:
                target_device = obj.device
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                find_device(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                find_device(v)
    find_device(args)
    find_device(kwargs)
    return target_device


def _move_to_device(obj, device):
    """Recursively move tensors to target device."""
    if isinstance(obj, torch.Tensor):
        if obj.device != device:
            return obj.to(device)
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


class TaintedTensor(torch.Tensor):
    # Class variable to track if we've seen the first NUM_TOKS:16
    _first_16_seen = False

    @staticmethod
    def __new__(cls, data: torch.Tensor, dim_taints=None):
        # Debug: trace where dimension 9216 comes from
        # if 9216 in data.shape and dim_taints is not None:
        #     from .types import DimTaint, Taint
        #     import traceback
        #     print(f"[TAINTEDTENSOR __new__] Creating tensor with 9216: shape={data.shape}, dim_taints={dim_taints}", flush=True)
            # Print just the most recent few frames to see where this is coming from
            # for line in traceback.format_stack()[-8:-1]:
            #     print(line.rstrip(), flush=True)

        # Debug: trace where dimension 9216 comes from
        # if 9216 in data.shape and dim_taints is not None:
        #     from .types import DimTaint, Taint
        #     for i, size in enumerate(data.shape):
        #         if size == 9216 and i < len(dim_taints):
        #             taint = dim_taints[i]
        #             # Handle both plain Taint and DimTaint
        #             taint_str = None
        #             if isinstance(taint, DimTaint):
        #                 taint_str = str(taint.taint) if taint.taint else None
        #             elif isinstance(taint, Taint):
        #                 taint_str = str(taint)
        #             elif isinstance(taint, str):
        #                 taint_str = taint

        #             import traceback
        #             print(f"\n{'='*80}")
        #             print(f"[DEBUG 9216] Creating TaintedTensor with shape {data.shape}, dim_taints={dim_taints}")
        #             print(f"Dimension {i} has size 9216 with taint: {taint_str}")
        #             print("Stack trace:")
        #             for line in traceback.format_stack()[-12:-1]:  # Last 11 frames
        #                 print(line.rstrip())
        #             print('='*80 + '\n')
        #             break

        # If in inference mode, we need to avoid "Cannot set version_counter for inference tensor" errors
        # Use .data to get the underlying tensor without copying (avoids GPU kernel overhead)
        if torch.is_inference_mode_enabled():
            with torch.inference_mode(False):
                # .data gives the underlying tensor, which is not an inference tensor
                # This avoids the copy_() kernel that was adding tracer overhead
                return torch.Tensor._make_subclass(cls, data.data, False)
        return torch.Tensor._make_subclass(cls, data, data.requires_grad)

    def __setattr__(self, name, value):
        # Intercept assignments to .data to sync taints
        if name == 'data':
            # DEBUG
            # print(f"\n[DATA SETATTR] Assignment to .data")
            # print(f"  self._dim_taints before: {getattr(self, '_dim_taints', None)}")
            # print(f"  self.shape before: {tuple(torch.Tensor.size(self))}")
            # print(f"  value type: {type(value)}")
            if isinstance(value, TaintedTensor):
                # print(f"  value._dim_taints: {value._dim_taints}")
                # print(f"  value.shape: {tuple(value.shape)}")
                # Update our taints to match new data BEFORE the assignment
                # Use object.__setattr__ to avoid recursion
                object.__setattr__(self, '_dim_taints', value._dim_taints)
            else:
                # print(f"  value.shape: {value.shape}")
                # Plain tensor - create None taints
                object.__setattr__(self, '_dim_taints', tuple([None] * value.ndim))
            # Now do the actual assignment
            super().__setattr__(name, value)
            # print(f"  self._dim_taints after: {self._dim_taints}")
            # print(f"  self.shape after: {tuple(torch.Tensor.size(self))}")
            return

        # Avoid recursion by using class-level check for slots.
        # Ensure we don't block standard Tensor attributes.
        if name in dir(torch.Tensor) or name in getattr(self.__class__, '__slots__', []):
             super().__setattr__(name, value)
        else:
             self.__dict__[name] = value

    def __delattr__(self, name):
        # Handle deletion of custom attributes (e.g., weight_loader)
        # vLLM sometimes deletes weight_loader after loading weights
        try:
            d = super().__getattribute__('__dict__')
            if name in d:
                del d[name]
                return
        except AttributeError:
            pass
        # If not in __dict__, try parent class
        try:
            super().__delattr__(name)
        except AttributeError:
            # Silently ignore if attribute doesn't exist
            # This matches Python's behavior for some special attributes
            pass

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            pass

        try:
            d = super().__getattribute__('__dict__')
            if name in d:
                return d[name]
        except AttributeError:
             pass

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, index):
        """
        Handle slicing/indexing operations with proper taint propagation.

        This handles operations like:
        - A[0] - select a single element/row
        - A[:, 3] - slice along dimensions
        - A[1:5, :] - slice ranges
        - A[[1,2,3]] - advanced indexing

        Taint propagation rules:
        - For dimension-preserving slices (e.g., A[1:5]), taints are preserved
        - For dimension-reducing operations (e.g., A[0]), the selected dimension's taint is removed
        - For advanced indexing, we preserve taints where dimensions are preserved
        """
        # Debug: Check if index is tainted (especially for logit_indices_device)
        if hasattr(index, 'shape') and hasattr(index, '__len__'):
            is_tainted = isinstance(index, TaintedTensor)
            index_len = len(index) if hasattr(index, '__len__') else 'N/A'
            if index_len == 263 or (hasattr(self, 'shape') and 263 in self.shape):
                print(f"[DEBUG __getitem__] index type: {type(index)}, is_tainted: {is_tainted}, "
                      f"index length: {index_len}, self.shape: {self.shape}", flush=True)
                if is_tainted:
                    print(f"[DEBUG __getitem__]   index._dim_taints: {index._dim_taints}", flush=True)

        # Let the parent class handle the actual slicing
        # print("CALLED __GETITEM__ with index:", index)

        with torch._C.DisableTorchFunction():
            with torch._C._DisableTorchDispatch():
                result = super().__getitem__(index)

            # result = super().__getitem__(index)

        # If result is not a tensor (e.g., scalar), return as-is
        if not isinstance(result, torch.Tensor):
            return result

        # Handle different indexing patterns
        if not isinstance(index, tuple):
            index = (index,)

        # Track which dimensions are preserved vs removed
        input_dims = len(self.shape)
        output_dims = len(result.shape) if hasattr(result, 'shape') else 0

        # Debug: track slicing operations with dimension 5
        # if 5 in self.shape or (hasattr(result, 'shape') and 5 in result.shape):
        #     print(f"\n[SLICE DEBUG] Slicing operation with dimension 5:", flush=True)
        #     print(f"  input shape: {self.shape}", flush=True)
        #     print(f"  input taints: {self._dim_taints}", flush=True)
        #     print(f"  index: {index}", flush=True)
        #     print(f"  output shape: {result.shape if hasattr(result, 'shape') else 'N/A'}", flush=True)

        # If result is a scalar tensor with 0 dimensions, return plain tensor
        if output_dims == 0:
            return result

        # Build output taints based on the indexing operation
        output_taints = []
        input_idx = 0

        for idx_item in index:
            if isinstance(idx_item, int):
                # Integer index: dimension is removed
                input_idx += 1
            elif isinstance(idx_item, slice):
                # Slice: dimension is preserved (possibly with different size)
                from .types import TaintedInt, Taint, DimTaint
                from .registry import TAINT_REGISTRY

                # Check if this is a full slice (no bounds specified)
                is_full_slice = (idx_item.start is None and
                                idx_item.stop is None and
                                idx_item.step is None)

                # Check if any slice bound is tainted
                # Note: Python's slice object converts TaintedInt to plain int via __index__
                # So we need to check both the TaintedInt case AND the registry
                # IMPORTANT: Skip negative indices - they're Python syntax for "from end", not semantic values!
                slice_taint = None
                for bound in [idx_item.start, idx_item.stop, idx_item.step]:
                    if bound is None or (isinstance(bound, int) and bound < 0):
                        continue
                    # Check if bound is a TaintedInt (shouldn't happen due to __index__, but check anyway)
                    if isinstance(bound, TaintedInt) and bound.taint:
                        slice_taint = bound.taint
                        break
                    # Check if bound is a plain int that exists in the registry
                    elif isinstance(bound, int) and bound in TAINT_REGISTRY:
                        registry_entry = TAINT_REGISTRY[bound]
                        # Handle both direct taint and MIX dict entries
                        if isinstance(registry_entry, dict):
                            slice_taint = registry_entry.get('taint')
                        else:
                            slice_taint = registry_entry
                        # Debug: track registry lookups for 269
                        if bound == 269:
                            print(f"[SLICE REGISTRY DEBUG] Found 269 in registry: {registry_entry}, extracted taint: {slice_taint}", flush=True)
                        if slice_taint:
                            break

                if slice_taint:
                    # Slice bound is tainted - use its taint (e.g., NUM_TOKS from runtime)
                    # This converts MAX_NUM_TOKS buffer → NUM_TOKS actual data
                    output_taints.append(DimTaint(slice_taint, merge_history=None))
                elif is_full_slice:
                    # Full slice [:] - preserve original taint
                    if input_idx < len(self._dim_taints):
                        output_taints.append(self._dim_taints[input_idx])
                    else:
                        output_taints.append(None)
                else:
                    # Partial slice with untainted bounds - remove taint (arbitrary slicing)
                    output_taints.append(None)

                input_idx += 1
            elif idx_item is Ellipsis:
                # Ellipsis: preserve remaining dimensions
                remaining = input_dims - input_idx - sum(1 for i in index[index.index(idx_item)+1:] if i is not Ellipsis)
                for _ in range(remaining):
                    if input_idx < len(self._dim_taints):
                        output_taints.append(self._dim_taints[input_idx])
                    else:
                        output_taints.append(None)
                    input_idx += 1
            elif idx_item is None:
                # None: adds a new dimension of size 1 (like unsqueeze)
                output_taints.append(None)
            elif hasattr(idx_item, '__len__'):
                # List/array indexing: dimension is preserved but may change size
                # Similar to slice bounds, check if the output size (index length) is registered
                from .types import TaintedInt, Taint, DimTaint
                from .registry import TAINT_REGISTRY

                # Get the output size (length of the index array/tensor)
                # Note: len(plain_tensor) returns plain int, not TaintedInt
                output_size = len(idx_item)

                # Check if the output size is registered (similar to how we check slice bounds)
                # This works because registry lookup is by value, not type
                if output_size in TAINT_REGISTRY:
                    registry_entry = TAINT_REGISTRY[output_size]
                    # Handle both direct taint and MIX dict entries
                    if isinstance(registry_entry, dict):
                        index_taint = registry_entry.get('taint')
                    else:
                        index_taint = registry_entry

                    if index_taint:
                        # Use the taint from the registry (e.g., 263 → MAX_NUM_REQS)
                        # print(f"[INDEX TAINT] Indexing produces output size {output_size}, found in registry as {index_taint}", flush=True)
                        output_taints.append(DimTaint(index_taint, merge_history=None))
                        input_idx += 1
                        continue

                # Default: preserve original taint if not found in registry
                if input_idx < len(self._dim_taints):
                    output_taints.append(self._dim_taints[input_idx])
                else:
                    output_taints.append(None)
                input_idx += 1
            else:
                # Colon or other: preserve dimension
                if input_idx < len(self._dim_taints):
                    output_taints.append(self._dim_taints[input_idx])
                else:
                    output_taints.append(None)
                input_idx += 1

        # Handle remaining dimensions not covered by index
        while len(output_taints) < output_dims and input_idx < input_dims:
            if input_idx < len(self._dim_taints):
                output_taints.append(self._dim_taints[input_idx])
            else:
                output_taints.append(None)
            input_idx += 1

        # Ensure we have the right number of taints
        output_taints = output_taints[:output_dims]
        while len(output_taints) < output_dims:
            output_taints.append(None)

        # Debug: show output taints for slicing with dimension 5
        # if 5 in self.shape or (hasattr(result, 'shape') and 5 in result.shape):
        #     print(f"  computed output_taints: {output_taints}", flush=True)

        # Return TaintedTensor with propagated taints
        return TaintedTensor(result, tuple(output_taints))

    def __init__(self, data: torch.Tensor, dim_taints=None):
        from .types import DimTaint, Taint

        # Debug: trace where dimension 269 gets its taints
        # if 269 in data.shape:
        #     print(f"[TAINTEDTENSOR __init__] shape={data.shape}, dim_taints={dim_taints}", flush=True)

        # DEBUG: Check for mismatch between shape and dim_taints for 9216/18432
        # if dim_taints and (9216 in data.shape or 18432 in data.shape):
            # for i, (sz, t) in enumerate(zip(data.shape, dim_taints)):
                # if isinstance(t, DimTaint) and t.merge_history and sz in [9216, 18432]:
                    # history_vals = list(t.merge_history.values())
                    # if history_vals and history_vals[0] != sz:
                        # import traceback
                        # print(f"\n[TAINTEDTENSOR __init__ MISMATCH] Creating tensor with mismatched dim_taints!", flush=True)
                        # print(f"  Shape: {data.shape}", flush=True)
                        # print(f"  Dim {i}: size={sz}, but dim_taints has merge_history={t.merge_history}", flush=True)
                        # print(f"  All dim_taints: {dim_taints}", flush=True)
                        # print(f"  Traceback:", flush=True)
                        # traceback.print_stack(limit=15)

        def normalize_taint(t):
            """Convert any taint to DimTaint for consistency."""
            if t is None:
                return None
            elif isinstance(t, DimTaint):
                return t
            elif isinstance(t, Taint):
                return DimTaint.from_taint(t)
            else:
                # Unknown type, wrap it anyway
                return DimTaint.from_taint(t)

        if dim_taints is None:
            self._dim_taints = tuple([None] * data.ndim)
        else:
            self._dim_taints = tuple(
                normalize_taint(dim_taints[i]) if i < len(dim_taints) else None
                for i in range(data.ndim)
            )

    @property
    def shape(self) -> TaintedShape:
        sizes = torch.Tensor.size(self)
        # Pass full DimTaint objects (with merge_history) to TaintedShape
        # This preserves component metadata for MIX taints

        # DEBUG: Track when shape with 9216 or 18432 is accessed
        # if 9216 in sizes or 18432 in sizes:
            # import traceback
            # print(f"\n[SHAPE PROPERTY] Accessing .shape: sizes={sizes}, _dim_taints={self._dim_taints}", flush=True)
            # print(f"  Traceback:", flush=True)
            # traceback.print_stack(limit=10)

        # DEBUG: Track when shape with 6144 and (1 or 4) is accessed
        # if 6144 in sizes and (1 in sizes or 4 in sizes):
        #     import traceback
            # print(f"\n[SHAPE PROPERTY] Accessing .shape: sizes={sizes}, _dim_taints={self._dim_taints}", flush=True)
            # print(f"  self type: {type(self)}, id: {id(self)}", flush=True)
            # traceback.print_stack(limit=8)

        return TaintedShape(tuple(sizes), self._dim_taints)

    def size(self, dim=None):
        if dim is None:
            return self.shape
        s = torch.Tensor.size(self, dim)
        t = self._dim_taints[dim] if dim < len(self._dim_taints) else None
        # Pass full DimTaint (with merge_history) to TaintedInt
        # register_taint will extract components automatically
        return TaintedInt(s, t)

    @property
    def dim_taints(self):
        """Returns the full DimTaint objects (may include merge history)."""
        return self._dim_taints


    @property
    def T(self):
        """
        Transpose property - reverses all dimensions.
        For 2D tensors, this swaps rows and columns.

        Override the default .T to directly compute the result without
        going through dispatch, similar to __getitem__.
        """
        # Disable dispatch and function to avoid overhead
        with torch._C.DisableTorchFunction():
            with torch._C._DisableTorchDispatch():
                # Call the parent .T to get plain result
                result = super().T

        # For 2D, reverse dims: (0, 1) -> (1, 0)
        # For nD, reverse all: (0, 1, 2, ..., n-1) -> (n-1, ..., 2, 1, 0)
        ndim = self.ndim
        reversed_dims = tuple(range(ndim - 1, -1, -1))

        # Apply taint permutation
        new_taints = tuple(self._dim_taints[d] for d in reversed_dims)
        return TaintedTensor(result, new_taints)

    def is_dim_independent(self, dim: int) -> bool:
        """Check if a dimension is definitely independent (can be swept for profiling)."""
        if dim >= len(self._dim_taints):
            return True  # No taint info = assume independent
        t = self._dim_taints[dim]
        if t is None:
            return True
        if isinstance(t, DimTaint):
            return t.is_definitely_independent
        # Plain Taint means it's dependent
        return False

    def numpy(self):
        """Allow conversion to numpy by stripping taint and preserving memory aliasing."""
        with torch._C.DisableTorchFunction():
            with torch._C._DisableTorchDispatch():
                cpu_t = self.detach().cpu()
                # perform downcast (create vanilla pytorch Tensor view that points to same raw memory)
                plain = torch.Tensor.as_subclass(cpu_t, torch.Tensor) 
                return torch.Tensor.numpy(plain) # returns the numpy alias

    def tolist(self):
        """Allow .tolist() by stripping taint."""
        with torch._C.DisableTorchFunction():
            with torch._C._DisableTorchDispatch():
                cpu_t = self.detach().cpu()
                plain = torch.Tensor.as_subclass(cpu_t, torch.Tensor)
                return plain.tolist()

    def _apply_shape_taints(self, result, shape_arg):
        """Apply taints from TaintedShape to result tensor."""
        if isinstance(shape_arg, TaintedShape) and isinstance(result, TaintedTensor):
            def to_dim_taint(t):
                if t is None:
                    return None
                if isinstance(t, DimTaint):
                    return t
                return DimTaint.from_taint(t)
            result._dim_taints = tuple(
                to_dim_taint(shape_arg._taints[i]) if i < len(shape_arg._taints) else None
                for i in range(result.ndim)
            )
        return result

    def taint_str(self) -> str:
        """Format taint info for display: [MODEL_CONFIG(128), ?(7)]"""
        sizes = torch.Tensor.size(self)
        parts = []
        for i, size in enumerate(sizes):
            t = self._dim_taints[i] if i < len(self._dim_taints) else None
            if t is None:
                parts.append(_format_dim(size, None))
            elif isinstance(t, DimTaint):
                parts.append(_format_dim(size, t))
            else:
                # Plain Taint
                parts.append(f"{t}({size})")
        return f"[{', '.join(parts)}]"

    def taint_str_with_history(self) -> str:
        """Format taint info including merge history for debugging."""
        sizes = torch.Tensor.size(self)
        parts = []
        for i, size in enumerate(sizes):
            t = self._dim_taints[i] if i < len(self._dim_taints) else None
            if t is None:
                parts.append(_format_dim(size, None, include_history=True))
            elif isinstance(t, DimTaint):
                parts.append(_format_dim(size, t, include_history=True))
            else:
                # Debug: This shouldn't happen if normalize_taint worked correctly
                import os
                if os.environ.get('DEBUG_TAINT_STR', '0') == '1':
                    print(f"[DEBUG taint_str_with_history] Unexpected type for taint at index {i}:")
                    print(f"  Type: {type(t)}")
                    print(f"  Value: {t}")
                    print(f"  Is DimTaint: {isinstance(t, DimTaint)}")
                    import traceback
                    traceback.print_stack(limit=5)
                parts.append(f"{t}({size})")
        return f"[{', '.join(parts)}]"

    def __repr__(self):
        return f"{torch.Tensor.__repr__(self)}\nTaints: {self.taint_str()}"

    def __format__(self, format_spec):
        if self.dim() == 0:
            return self.item().__format__(format_spec)
        return object.__format__(self, format_spec)

    # -----------------------------------------------------------------------
    # Dispatch: intercepts all torch ops on TaintedTensors
    # -----------------------------------------------------------------------

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        input_infos = []
        propagation_dict = {}
        primary_tainted_tensor = None

        def collect_from(obj):
            nonlocal primary_tainted_tensor
            if isinstance(obj, TaintedTensor):
                if primary_tainted_tensor is None:
                    primary_tainted_tensor = obj
                sizes = torch.Tensor.size(obj)
                shape = tuple(sizes)
                input_infos.append((shape, obj._dim_taints))
            elif isinstance(obj, torch.Tensor):
                # Include plain tensors so multi-input ops (embedding, gather,
                # index_select) are not mistaken for single-input ops by the
                # view/slice heuristics. Read any _dim_taints re-attached to a
                # weight Parameter so its taints propagate to op outputs.
                sizes = torch.Tensor.size(obj)
                shape = tuple(sizes)
                explicit_taints = getattr(obj, "_dim_taints", None)
                if explicit_taints is None:
                    explicit_taints = tuple(None for _ in range(len(shape)))
                input_infos.append((shape, tuple(explicit_taints)))
            if isinstance(obj, (list, tuple)) and not isinstance(obj, TaintedShape):
                for item in obj:
                    collect_from(item)
            elif isinstance(obj, dict):
                for v in obj.values():
                    collect_from(v)

        collect_from(args)
        collect_from(kwargs)

        # Unwrap to plain Tensors for execution
        def unwrap(x):
            if isinstance(x, TaintedTensor):
                with torch._C._DisableTorchDispatch():
                    return x.detach()
            elif isinstance(x, (list, tuple)):
                return type(x)(unwrap(i) for i in x)
            elif isinstance(x, dict):
                return {k: unwrap(v) for k, v in x.items()}
            return x

        plain_args = unwrap(args)
        plain_kwargs = unwrap(kwargs)

        # Ensure all tensors are on the same device (prefer CUDA over CPU)
        target_device = _get_target_device(plain_args, plain_kwargs)
        if target_device is not None:
            plain_args = _move_to_device(plain_args, target_device)
            plain_kwargs = _move_to_device(plain_kwargs, target_device)

        # Ensure correct CUDA context
        if torch.cuda.is_available() and target_device is not None and target_device.type == 'cuda':
            torch.cuda.set_device(target_device)

        out = func(*plain_args, **plain_kwargs)

        # Special handling for copy_: source taints should propagate to destination
        func_name = func.name() if hasattr(func, 'name') else str(func)
        is_copy = 'copy_' in func_name or 'copy' in func_name
        
        def wrap_output(x):
            if isinstance(x, torch.Tensor) and not isinstance(x, TaintedTensor):
                out_shape = tuple(x.shape)

                # For copy_, use source taints directly (input_infos[1] is source)
                if is_copy and len(input_infos) >= 2:
                    _, source_taints = input_infos[1]
                    def to_dim_taint(t):
                        if t is None:
                            return None
                        if isinstance(t, DimTaint):
                            return t
                        return DimTaint.from_taint(t)
                    out_taints = tuple(
                        to_dim_taint(source_taints[i]) if i < len(source_taints) else None
                        for i in range(len(out_shape))
                    )
                else:
                    # Debug: track operations resulting in shape with 5
                    func_name = func.name() if hasattr(func, 'name') else str(func)
                    # if 5 in out_shape or (len(input_infos) > 0 and any(5 in info[0] for info in input_infos)):
                        # print(f"\n[TENSOR DEBUG] Operation with dimension 5:", flush=True)
                        # print(f"  func: {func_name}", flush=True)
                        # print(f"  output shape: {out_shape}", flush=True)
                        # for i, (in_shape, in_taints) in enumerate(input_infos):
                        #     print(f"  input[{i}] shape: {in_shape}, taints: {in_taints}", flush=True)

                    out_taints = propagate_taints(input_infos, out_shape)

                    # Debug continued: show output taints for dimension 5
                    # if 5 in out_shape:
                        # print(f"  output taints: {out_taints}", flush=True)

                # Check for in-place modification
                # Simplistic check: op ends with '_' and first arg is TaintedTensor
                func_name = func.name() if hasattr(func, 'name') else str(func)
                is_inplace = func_name.endswith('_')
                if is_inplace and len(args) > 0 and isinstance(args[0], TaintedTensor):
                    args[0]._dim_taints = out_taints
                    if propagation_dict:
                         args[0].__dict__.update(propagation_dict)
                    return args[0]

                new_t = TaintedTensor(x, out_taints)
                if propagation_dict:
                     new_t.__dict__.update(propagation_dict)
                return new_t
            return x

        if isinstance(out, torch.Tensor):
            return wrap_output(out)
        elif isinstance(out, (list, tuple)):
            return type(out)(wrap_output(o) for o in out)
        else:
            return out

    # -----------------------------------------------------------------------
    # Profiler-annotated dispatch wrapper
    # -----------------------------------------------------------------------

    _reentrancy_guard = threading.local() # prevent recursive calls
    _inside_hook = threading.local() # track when we're inside a hook wrapper

    @classmethod
    def dispatch_with_profiler(cls, func, types, args=(), kwargs=None):
        import os
        _debug = os.environ.get("DEBUG_DISPATCH", "0") == "1"

        if getattr(cls._reentrancy_guard, 'active', False):
            # If we're in a reentrant call, check if we're inside a hook
            # If so, unwrap and execute without propagate_taints
            inside_hook = getattr(cls._inside_hook, 'active', False)
            if _debug:
                print(f"[dispatch_with_profiler] SKIPPED (reentrancy guard): {func}, inside_hook={inside_hook}")

            if inside_hook:
                # Unwrap and execute without calling propagate_taints
                def unwrap(x):
                    if isinstance(x, TaintedTensor):
                        with torch._C._DisableTorchDispatch():
                            return x.detach()
                    elif isinstance(x, (list, tuple)):
                        return type(x)(unwrap(i) for i in x)
                    elif isinstance(x, dict):
                        return {k: unwrap(v) for k, v in x.items()}
                    return x

                plain_args = unwrap(args)
                plain_kwargs = unwrap(kwargs or {})

                with torch._C._DisableTorchDispatch():
                    return func(*plain_args, **plain_kwargs)
            else:
                return cls._original_dispatch(func, types, args, kwargs)

        if _debug:
            inside_hook = getattr(cls._inside_hook, 'active', False)
            print(f"[dispatch_with_profiler] {func}, inside_hook={inside_hook}")

        cls._reentrancy_guard.active = True
        try:
            kwargs = kwargs or {}

            input_shapes = []
            input_ints = []

            def format_tensor_shape(tensor):
                if isinstance(tensor, TaintedTensor):
                    return tensor.taint_str_with_history()
                elif isinstance(tensor, torch.Tensor):
                    _dt = getattr(tensor, "_dim_taints", None)
                    sizes = torch.Tensor.size(tensor)
                    dims = [_format_dim(s, (_dt[i] if _dt is not None and i < len(_dt) else None))
                            for i, s in enumerate(sizes)]
                    return "[" + ", ".join(dims) + "]"
                return None

            def collect_from(obj):
                if isinstance(obj, TaintedInt):
                    if obj.taint is not None:
                        input_ints.append(f"{obj.taint}={obj.value}")
                elif isinstance(obj, torch.Tensor):
                    shape_str = format_tensor_shape(obj)
                    if shape_str:
                        input_shapes.append(shape_str)
                elif isinstance(obj, (list, tuple)) and not isinstance(obj, TaintedShape):
                    for item in obj:
                        collect_from(item)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        collect_from(v)

            collect_from(args)
            collect_from(kwargs)

            op_name = str(func.name()) if hasattr(func, 'name') else str(func)
            taint_str = " | ".join(input_shapes) if input_shapes else "no_inputs"
            annotated_name = f"{op_name} IN:[{taint_str}]"
            if input_ints:
                # # Deduplicate while preserving order
                # seen = set()
                # unique_ints = []
                # for item in input_ints:
                #     if item not in seen:
                #         seen.add(item)
                #         unique_ints.append(item)
                annotated_name += " ARGS:[" + ", ".join(unique_ints) + "]"

            with record_function(annotated_name):
                # If we're inside a hook, only add the annotation but don't call propagate_taints
                # The hook will handle taint propagation itself
                if getattr(cls._inside_hook, 'active', False):
                    # Execute the operation but return plain result (no taint propagation)
                    # Unwrap to plain tensors
                    def unwrap(x):
                        if isinstance(x, TaintedTensor):
                            with torch._C._DisableTorchDispatch():
                                return x.detach()
                        elif isinstance(x, (list, tuple)):
                            return type(x)(unwrap(i) for i in x)
                        elif isinstance(x, dict):
                            return {k: unwrap(v) for k, v in x.items()}
                        return x

                    plain_args = unwrap(args)
                    plain_kwargs = unwrap(kwargs)

                    # Call the actual operation with plain tensors
                    with torch._C._DisableTorchDispatch():
                        return func(*plain_args, **plain_kwargs)
                else:
                    # Normal path: call original dispatch which does taint propagation
                    if _debug:
                        print(f"  -> Calling _original_dispatch (will call propagate_taints)")
                    return cls._original_dispatch(func, types, args, kwargs)
        finally:
            cls._reentrancy_guard.active = False

    # Store original dispatch
    _original_dispatch = __torch_dispatch__


# ---------------------------------------------------------------------------
# TaintedParameter: preserves taint when wrapping in nn.Parameter
# ---------------------------------------------------------------------------

class TaintedParameter(torch.nn.Parameter, TaintedTensor):
    """Preserves TaintedTensor behavior when wrapped in nn.Parameter."""

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.tensor([])
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(self, data, requires_grad=True):
        # If taints were already set (e.g., by patched nn.Parameter.__new__), preserve them
        if hasattr(self, '_dim_taints') and self._dim_taints is not None:
            if any(t is not None for t in self._dim_taints):
                return  # Already has taints, don't overwrite

        if isinstance(data, TaintedTensor):
            self._dim_taints = data._dim_taints
        else:
            self._dim_taints = tuple([None] * data.ndim)

