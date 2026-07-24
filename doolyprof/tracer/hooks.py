"""Monkey-patching hooks for torch factory functions, custom ops, and dispatch."""

import copyreg
import functools
import importlib
import threading
import dataclasses
from typing import Optional, Tuple
import torch
from torch.profiler import record_function

from .types import Taint, DimTaint, TaintedInt, TaintedFloat, TaintedShape, unwrap_taint
from .tensor import TaintedTensor

__all__ = [
    'install_hooks', 'uninstall_hooks',
    'patch_int_in_modules', 'unpatch_int_in_modules',
]

# ---------------------------------------------------------------------------
# Captured originals (before any hooks replace them)
# ---------------------------------------------------------------------------

# torch.compile (captured before any patching). Patched in install_hooks()
# with a wrapper that handles TaintedTensor inputs without going through
# dynamo (which can't trace TaintedTensor's __torch_dispatch__).
_original_torch_compile = torch.compile

# Tensor creation hooks
_original_zeros = torch.zeros
_original_ones = torch.ones
_original_empty = torch.empty
_original_randn = torch.randn
_original_rand = torch.rand
_original_arange = torch.arange
_original_torch_size = torch.Size
_original_copy = torch.Tensor.copy_

# Tensor shape manipulation hooks
# View/Reshape operations
_original_tensor_view = torch.Tensor.view
_original_torch_reshape = torch.reshape  # torch.reshape EXISTS!
_original_tensor_reshape = torch.Tensor.reshape

# Transpose/Permute operations
_original_tensor_permute = torch.Tensor.permute
_original_torch_transpose = torch.transpose  # torch.transpose EXISTS!
_original_tensor_transpose = torch.Tensor.transpose

# Concatenation/Stacking operations (only torch. versions exist)
_original_torch_cat = torch.cat
_original_torch_stack = torch.stack

# Splitting operations
_original_torch_split = torch.split
_original_tensor_split = torch.Tensor.split  # Tensor.split EXISTS!
_original_torch_chunk = torch.chunk
_original_tensor_chunk = torch.Tensor.chunk  # Tensor.chunk EXISTS!

# Squeeze/Unsqueeze operations
_original_torch_unsqueeze = torch.unsqueeze
_original_torch_squeeze = torch.squeeze
_original_tensor_unsqueeze = torch.Tensor.unsqueeze
_original_tensor_squeeze = torch.Tensor.squeeze

# Unbind operation
_original_torch_unbind = torch.unbind
_original_tensor_unbind = torch.Tensor.unbind  # Tensor.unbind EXISTS!

# Flatten operation
_original_torch_flatten = torch.flatten
_original_tensor_flatten = torch.Tensor.flatten

# Expand operation (only Tensor.expand exists)
_original_tensor_expand = torch.Tensor.expand

# Known operations
_original_torch_linear = torch.nn.functional.linear
_original_torch_pad = torch.nn.functional.pad
_original_torch_matmul = torch.matmul
_original_tensor_matmul = torch.Tensor.matmul
_original_torch_einsum = torch.einsum

# Einops operations (imported lazily to avoid import errors if einops not installed)
_original_einops_rearrange = None
_original_einops_reduce = None
_original_einops_repeat = None

# Register pickle handler for original torch.Size to avoid identity mismatch
# when torch.Size is patched with PatchedSize. This handles instances created
# by PyTorch C++ code that bypass our PatchedSize.__new__.
def _pickle_original_torch_size(size_instance):
    return (tuple, (tuple(size_instance),))

copyreg.pickle(_original_torch_size, _pickle_original_torch_size)

_original_module_call = None


def _strip_tainted_scalars(obj):
    """Recursively replace TaintedInt/TaintedFloat with raw Python scalars.

    Tensors (including TaintedTensor) are left untouched so dispatch-level
    taint propagation still works. Only non-tensor scalar wrappers are
    stripped. Triton kernel launches reject TaintedInt as a constexpr
    because its AST parser only accepts {int, bool, NoneType}; this helper
    is called at the boundary where arguments are forwarded to `func`,
    ensuring any TaintedInt flowing through a torch op into a downstream
    Triton kernel is unwrapped.
    """
    if isinstance(obj, (TaintedInt, TaintedFloat)):
        return obj.value
    if isinstance(obj, (TaintedTensor, torch.Tensor)):
        return obj
    if isinstance(obj, TaintedShape):
        return obj
    if isinstance(obj, tuple):
        out = tuple(_strip_tainted_scalars(x) for x in obj)
        return out if type(obj) is tuple else type(obj)(*out)
    if isinstance(obj, list):
        return [_strip_tainted_scalars(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _strip_tainted_scalars(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Triton JIT patch
# ---------------------------------------------------------------------------
# vLLM kernels (e.g. unified_attention_with_output for gpt-oss) stash
# TaintedInt values as attributes on backend impls during construction, then
# forward them into Triton kernel launches as constexpr args. Triton's AST
# parser rejects TaintedInt because it only allows {int, bool, NoneType} in
# compile-time branches like `if SLIDING_WINDOW:`. Intercept at the
# JITFunction.run level to unwrap tainted scalars before compilation.

def _wrap_triton_run(original_run):
    @functools.wraps(original_run)
    def _run(self, *args, **kwargs):
        return original_run(
            self,
            *_strip_tainted_scalars(args),
            **_strip_tainted_scalars(kwargs),
        )
    return _run


def _patch_triton_jit():
    try:
        import triton.runtime.jit as _triton_jit
    except ImportError:
        return  # Triton not installed; nothing to patch.
    jit_cls = getattr(_triton_jit, "JITFunction", None)
    if jit_cls is None or hasattr(jit_cls, "_dooly_original_run"):
        return  # already patched, or API drifted
    jit_cls._dooly_original_run = jit_cls.run
    jit_cls.run = _wrap_triton_run(jit_cls.run)


def _unpatch_triton_jit():
    try:
        import triton.runtime.jit as _triton_jit
    except ImportError:
        return
    jit_cls = getattr(_triton_jit, "JITFunction", None)
    if jit_cls is None:
        return
    orig = getattr(jit_cls, "_dooly_original_run", None)
    if orig is not None:
        jit_cls.run = orig
        del jit_cls._dooly_original_run


# ---------------------------------------------------------------------------
# TorchFunction Hook
# ---------------------------------------------------------------------------

def _torch_function_implementation(cls, func, types, args=(), kwargs=None):
    """
    Implementation of __torch_function__ for TaintedTensor.
    This allows capturing high-level Python API calls (like F.linear)
    before they decompose into lower-level dispatched operations.
    """
    if kwargs is None:
        kwargs = {}
    if not all(issubclass(t, (torch.Tensor, TaintedTensor)) for t in types):
        return NotImplemented

    # Log the function call - handle wrapped functions
    if hasattr(func, '__name__'):
        op_name = func.__name__
        # If this is a wrapper function, try to get the original function name
        if op_name == 'wrapper' and hasattr(func, '__wrapped__'):
            op_name = func.__wrapped__.__name__ if hasattr(func.__wrapped__, '__name__') else op_name
    else:
        op_name = str(func)

    taint_str = "no_inputs"
    annotated_name = f"{op_name}"

    # Use DisableTorchFunction to avoid recursion when inspecting tensor properties
    with torch._C.DisableTorchFunction():
        try:
            input_shapes = []
            input_ints = []

            def format_tensor_shape(tensor):
                if isinstance(tensor, TaintedTensor):
                    return tensor.taint_str_with_history()
                elif isinstance(tensor, torch.Tensor):
                    dims = [f"?({d})" for d in tensor.shape]
                    return "[" + ", ".join(dims) + "]"
                return None

            def collect_from(obj):
                if isinstance(obj, TaintedInt):
                    if obj.taint is not None:
                        # Handle nested DimTaint - unwrap to get the actual taint
                        taint_repr = obj.taint
                        if isinstance(taint_repr, DimTaint):
                            # Unwrap nested DimTaints
                            while isinstance(taint_repr.taint, DimTaint):
                                taint_repr = taint_repr.taint
                            # Use just the taint name, not the full DimTaint repr
                            taint_repr = taint_repr.taint if taint_repr.taint else taint_repr
                        input_ints.append(f"{taint_repr}={obj.value}")
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

            if input_shapes:
                taint_str = " | ".join(input_shapes)

            annotated_name = f"API: {op_name} IN:[{taint_str}]"
            if input_ints:
                # For now, don't deduplicate - just show all
                annotated_name += " ARGS:[" + ", ".join(input_ints) + "]"
        except Exception:
            # Fallback if logging fails
            annotated_name = f"API: {op_name} (Log Error)"

    # Strip TaintedInt/TaintedFloat wrappers before handing off to `func`.
    # Required for downstream Triton kernel launches (e.g. gpt-oss's
    # unified_attention_with_output -> kernel_unified_attention_2d), where
    # TaintedInt-as-constexpr is rejected by Triton's AST parser.
    stripped_args = _strip_tainted_scalars(args)
    stripped_kwargs = _strip_tainted_scalars(kwargs)

    with record_function(annotated_name):
        with torch._C.DisableTorchFunction():
            return func(*stripped_args, **stripped_kwargs)


# ---------------------------------------------------------------------------
# nn.Module Hook
# ---------------------------------------------------------------------------

def _extract_module_state_hash(module: "torch.nn.Module") -> str:
    """Short deterministic hash of a module instance's primitive attrs.

    Captures behavior-affecting config that differs across instances of the
    same class (e.g. an Attention layer's ``sliding_window`` being ``None``
    vs an int) without bloating the trace annotation.

    For backends that do NOT template-specialize on these runtime args
    (FLASH_ATTN, TRITON_ATTN), the kernel symbol is identical across SWA and
    full-attention layers — so without this, both variants collapse to one
    signature. Including a hash of the module's primitive attrs in the
    annotation forces distinct signatures whenever the module's config
    differs, letting the resolver split them correctly.

    Filter policy — include only primitive/compositional values:
      - ``None``, ``bool``, ``int``, ``float``
      - ``tuple``/``list`` whose elements are the above
    Everything else is skipped:
      - tensors, nested modules, callables, and other objects (can't be
        hashed stably)
      - strings (almost always identifier-like: ``layer_name``, ``prefix``;
        including them would fragment signatures uselessly per layer)
      - private attrs (keys starting with ``_``)

    Returns a 12-char hex digest, or ``""`` if no capturable attrs exist
    (keeps old signature behavior for modules that have nothing relevant).
    """
    import hashlib

    def is_primitive(v) -> bool:
        if v is None or isinstance(v, (bool, int, float)):
            return True
        if isinstance(v, (tuple, list)):
            return all(x is None or isinstance(x, (bool, int, float)) for x in v)
        return False

    def canonicalize(v):
        """Strip taint-wrapper classes (TaintedInt subclasses int but
        overrides __repr__ to add the taint label — that makes the same
        numeric value hash differently at trace time vs resolve time). Force
        every value down to its plain-primitive representation so repr() is
        stable across processes."""
        if v is None:
            return None
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            return float(v)
        if isinstance(v, (tuple, list)):
            return tuple(canonicalize(x) for x in v)
        return v

    # Attrs whose value depends on framework/runtime context rather than the
    # module's configuration — excluded so the hash stays stable across
    # trace-time and resolve-time process boundaries.
    #   training   : nn.Module default state flag, can differ if .eval()
    #                  was called in one process but not the other
    #   quant_config, query_quant : late-bound infrastructure objects that
    #                  may be None in one context and non-None in another
    _NON_CONFIG_ATTRS = frozenset({"training", "quant_config", "query_quant"})

    try:
        d = vars(module)
    except TypeError:
        return ""

    canonical = tuple(
        (k, canonicalize(v))
        for k, v in sorted(d.items())
        if not k.startswith("_")
        and k not in _NON_CONFIG_ATTRS
        and is_primitive(v)
    )
    if not canonical:
        return ""
    return hashlib.blake2b(repr(canonical).encode(), digest_size=6).hexdigest()


def _module_call_wrapper(self, *args, **kwargs):
    """
    Wrapper for nn.Module.__call__ that logs taint information.

    This allows us to capture taint info at the Module level, so if leaf operations
    don't have taints, we can walk up the hierarchy to find a tainted Module.
    """
    module_name = self.__class__.__name__

    # Collect taint info from input tensors
    taint_str = "no_inputs"

    try:
        input_shapes = []
        input_ints = []
        input_dtypes = []  # Collect dtypes for each tensor input
        kwarg_tensor_names = []  # Names of tensor kwargs (in collection order)

        def format_tensor_shape(tensor):
            if isinstance(tensor, TaintedTensor):
                return tensor.taint_str_with_history()
            elif isinstance(tensor, torch.Tensor):
                dims = [f"?({d})" for d in tensor.shape]
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
                    input_dtypes.append(str(obj.dtype))
            elif isinstance(obj, (list, tuple)) and not isinstance(obj, TaintedShape):
                for item in obj:
                    collect_from(item)
            elif isinstance(obj, dict):
                for v in obj.values():
                    collect_from(v)

        # Collect positional args first
        collect_from(args)
        num_positional_tensors = len(input_shapes)

        # Collect kwargs explicitly so we can record their parameter names
        for name, val in kwargs.items():
            if isinstance(val, torch.Tensor):
                shape_str = format_tensor_shape(val)
                if shape_str:
                    input_shapes.append(shape_str)
                    input_dtypes.append(str(val.dtype))
                    kwarg_tensor_names.append(name)
            elif isinstance(val, TaintedInt):
                if val.taint is not None:
                    input_ints.append(f"{val.taint}={val.value}")

        if input_shapes:
            taint_str = " | ".join(input_shapes)

        annotated_name = f"MODULE: {module_name} IN:[{taint_str}]"
        if input_dtypes:
            annotated_name += f" DTYPES:[{', '.join(input_dtypes)}]"
        if kwarg_tensor_names:
            # Record how many positional tensors there are, then kwarg names
            # Format: KWARGS:[name1, name2, ...] for the kwargs portion only
            annotated_name += f" KWARGS:[{', '.join(kwarg_tensor_names)}]"
        if input_ints:
            # For now, don't deduplicate - just show all
            annotated_name += " ARGS:[" + ", ".join(input_ints) + "]"
        # STATE:<hash> distinguishes module instances that differ only in
        # runtime-branching attrs (e.g. sliding_window). See
        # _extract_module_state_hash for the filter policy.
        _state_hash = _extract_module_state_hash(self)
        if _state_hash:
            annotated_name += f" STATE:{_state_hash}"
    except Exception:
        annotated_name = f"MODULE: {module_name} (Log Error)"

    with record_function(annotated_name):
        return _original_module_call(self, *args, **kwargs)

def patch_module_call():
    """Patch nn.Module.__call__ to log taint information."""
    global _original_module_call

    if _original_module_call is not None:
        return  # Already patched

    _original_module_call = torch.nn.Module.__call__
    torch.nn.Module.__call__ = _module_call_wrapper

def unpatch_module_call():
    """Restore original nn.Module.__call__."""
    global _original_module_call

    if _original_module_call is not None:
        torch.nn.Module.__call__ = _original_module_call
        _original_module_call = None


# ---------------------------------------------------------------------------
# Module-level int() patching
# ---------------------------------------------------------------------------

_real_int = int  # captured once at import time
_patched_int_modules = {}

def _preserving_int(x=0, /, *args, **kwargs):
    """Drop-in replacement for int() that preserves TaintedInt."""
    if isinstance(x, TaintedInt) and not args and not kwargs:
        # print(f"[DEBUG @ _preserving_int] (TaintedInt) x: {x}, args: {args}, kwargs: {kwargs}")
        return x
    if isinstance(x, TaintedFloat) and not args and not kwargs:
        # print(f"[DEBUG @ _preserving_int] (TaintedFloat) x: {x}, args: {args}, kwargs: {kwargs}")
        output = TaintedInt(_real_int(x), x.taint)
        # print(f"[DEBUG @ _preserving_int] (TaintedFloat) output: {output}, {output.taint}")
        return output

    # print(f"[DEBUG @ _preserving_int] (else) x: {x}, args: {args}, kwargs: {kwargs}")
    return _real_int(x, *args, **kwargs)


def patch_int_in_modules(module_paths):
    """Replace the ``int`` name in each listed module so int(TaintedInt)
    is a no-op.  Call from ``install_hooks``; reversed by ``uninstall_hooks``.
    """

    for path in module_paths:
        # print(f"[DEBUG @ patch_int_in_modules] module_paths: {module_paths}")
        # print(f"[DEBUG @ patch_int_in_modules] _patched_int_modules: {_patched_int_modules}")
        if path in _patched_int_modules:
            continue
        try:
            mod = importlib.import_module(path)
        except ModuleNotFoundError:
            continue

        _patched_int_modules[path] = getattr(mod, 'int', _real_int)
        mod.int = _preserving_int


def unpatch_int_in_modules():
    """Restore original ``int`` in every module we touched."""
    for path, orig in _patched_int_modules.items():
        try:
            mod = importlib.import_module(path)
            mod.int = orig
        except ModuleNotFoundError:
            pass
    _patched_int_modules.clear()


# ---------------------------------------------------------------------------
# Factory hook helpers
# ---------------------------------------------------------------------------

def _extract_taints_from_args(args, kwargs) -> Tuple[bool, Tuple[Optional[Taint], ...]]:
    """
    Extract taints from factory function arguments.
    Returns (has_taint, taints) where taints correspond to dimensions.
    """
    size = None
    if args and isinstance(args[0], (list, tuple, TaintedShape)):
        size = args[0]
    elif 'size' in kwargs:
        size = kwargs['size']
    elif args:
        if all(isinstance(a, (int, TaintedInt)) for a in args):
            size = args

    if size is None:
        return False, ()

    taints = []
    has_taint = False

    # Debug: track if we're iterating over a TaintedShape with 269
    # if isinstance(size, TaintedShape):
    #     try:
    #         if 269 in size:
    #             print(f"[EXTRACT_TAINTS DEBUG] Processing TaintedShape with 269: {size}, _taints={size._taints}", flush=True)
    #     except:
    #         pass

    for s in size:
        # Debug: track when we see 269
        # try:
        #     s_int = int(s)
        #     if s_int == 269:
        #         print(f"[EXTRACT_TAINTS DEBUG] Found 269: s={s}, type={type(s)}, is_TaintedInt={isinstance(s, TaintedInt)}", flush=True)
        #         if hasattr(s, 'taint'):
        #             print(f"[EXTRACT_TAINTS DEBUG]   taint attr exists: {s.taint}", flush=True)
        #         if hasattr(s, '_taint'):
        #             print(f"[EXTRACT_TAINTS DEBUG]   _taint attr exists: {s._taint}", flush=True)
        # except:
        #     pass

        if isinstance(s, TaintedInt) and s.taint:
            taints.append(s.taint)
            has_taint = True
        else:
            taints.append(None)

    return has_taint, tuple(taints)


def _unwrap_for_factory(obj):
    """Recursively unwrap TaintedInt to int for factory function arguments."""
    if isinstance(obj, TaintedInt):
        return _real_int(obj)
    if isinstance(obj, TaintedFloat):
        return float(obj)
    if isinstance(obj, TaintedShape):
        return tuple(_unwrap_for_factory(x) for x in obj)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_unwrap_for_factory(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _unwrap_for_factory(v) for k, v in obj.items()}
    if isinstance(obj, TaintedTensor):
        with torch._C._DisableTorchDispatch():
            return obj.detach()
    return obj

def _wrap_factory_output(original_fn):
    """Wrap a tensor creation function to propagate taints from TaintedInt sizes."""
    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        has_taint, taints = _extract_taints_from_args(args, kwargs)

        plain_args = _unwrap_for_factory(args)
        plain_kwargs = _unwrap_for_factory(kwargs)

        result = original_fn(*plain_args, **plain_kwargs)
        if has_taint:
            return TaintedTensor(result, taints)
        return result

    return wrapper


def _wrap_arange(original_fn):
    """Wrapper specifically for torch.arange to correctly propagate taints to the output dimension."""
    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        # Identify if any range defining argument is tainted.
        # arange args: (end,), (start, end), (start, end, step)
        # plus keywords.
        
        has_taint = False
        taint_label = None
        
        # Helper to check a value
        def check_taint(val):
            nonlocal has_taint, taint_label
            if isinstance(val, TaintedInt) and val.taint:
                has_taint = True
                # Simple policy: keep the first taint found.
                # Ideally we might merge them if multiple exist, but single source is common.
                if taint_label is None:
                    taint_label = val.taint
        
        for arg in args:
            check_taint(arg)
            
        for k, v in kwargs.items():
            if k in ('start', 'end', 'step'):
                check_taint(v)
        
        # Unwrap and call
        plain_args = _unwrap_for_factory(args)
        plain_kwargs = _unwrap_for_factory(kwargs)
        
        result = original_fn(*plain_args, **plain_kwargs)
        
        if has_taint:
            # arange always returns a 1D tensor
            # The size of this 1D tensor depends on start/end/step.
            # So the dimension 0 is tainted.
            return TaintedTensor(result, (taint_label,))
            
        return result
    return wrapper


def _wrap_copy(original_fn):
    """
    Wrap torch.Tensor.copy_() - CURRENTLY DISABLED for testing.

    Testing hypothesis: If most tensors are TaintedTensor (from config TaintedInt),
    then __torch_dispatch__ handles copy_() automatically, and we don't need
    special wrapper logic at all.
    """
    @functools.wraps(original_fn)
    def wrapper(self, src, non_blocking=False):
        # Just pass through to original function
        result = original_fn(self, src, non_blocking)
        return result

    return wrapper

## wrappers for shape manipulation
def _wrap_torch_reshape(original_fn):
    """
    Wrap torch.reshape (module-level function) to respect shape argument taints.

    This is separate from _wrap_view_reshape because torch.reshape is a module function,
    not a method, so the first argument is the tensor, not self.

    Uses simplified TaintedInt arithmetic approach.
    """
    from .types import DimTaint, TaintedInt, TaintedFloat

    @functools.wraps(original_fn)
    def wrapper(tensor, shape):
        # Flatten shape args (handle both reshape(t, (a,b,c)) and reshape(t, [a,b,c]))
        if isinstance(shape, (tuple, list, TaintedShape)):
            shape_args = shape
        else:
            # Single argument case (shouldn't happen for reshape but handle it)
            shape_args = (shape,)

        # Check if we're already inside a hook
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        TaintedTensor._inside_hook.active = True

        try:
            result = original_fn(tensor, shape)

            if was_already_in_hook:
                return result

            # Only propagate taints if input is TaintedTensor
            if not isinstance(tensor, TaintedTensor):
                return result

            # Build output taints using TaintedInt arithmetic
            output_taints = []
            minus_one_idx = None

            for i, s in enumerate(shape_args):
                if s == -1:
                    minus_one_idx = i
                    output_taints.append(None)  # Will compute later
                elif isinstance(s, (TaintedInt, TaintedFloat)):
                    # Explicit TaintedInt/Float: use its taint
                    output_taints.append(DimTaint.from_taint(s.taint))
                else:
                    # Plain int: no taint
                    output_taints.append(None)

            # Compute -1 dimension taint using TaintedInt arithmetic
            taint_dict={}
            if minus_one_idx is not None:
                # Compute total elements as product of all input dimensions
                total = TaintedInt(1, None)  # Start with neutral element
                for i, (size, taint) in enumerate(zip(tensor.shape, tensor._dim_taints)):
                    dim_val = tensor.shape[i]
                    if isinstance(dim_val, TaintedInt):
                        # Extract plain Taint from DimTaint if needed (for dictionary key)
                        plain_taint = unwrap_taint(dim_val.taint)
                        taint_dict[plain_taint] = dim_val.value * taint_dict.get(plain_taint, 1)
                    elif taint is not None:
                        # Create TaintedInt from size and taint
                        taint_obj = taint.taint if isinstance(taint, DimTaint) else taint
                        total = total * TaintedInt(int(size), taint_obj)
                    else:
                        # Plain dimension
                        total = total * int(size) if hasattr(total, '__mul__') else TaintedInt(int(total) * int(size), total.taint if isinstance(total, TaintedInt) else None)

                

                # Compute product of explicit dimensions (not -1)
                explicit_product = TaintedInt(1, None)
                for i, s in enumerate(shape_args):
                    if i != minus_one_idx:
                        if isinstance(s, (TaintedInt, TaintedFloat)):
                            explicit_product = explicit_product * s
                        else:
                            val = int(s)
                            if isinstance(explicit_product, TaintedInt):
                                explicit_product = TaintedInt(explicit_product.value * val, explicit_product.taint)
                            else:
                                explicit_product = explicit_product * val

                # Inferred dimension = total / explicit_product
                inferred = total / explicit_product

                # Extract taint from the result
                if isinstance(inferred, (TaintedInt, TaintedFloat)):
                    if inferred.taint is not None:
                        output_taints[minus_one_idx] = DimTaint.from_taint(inferred.taint)
                # else: remains None

            # Wrap result if we have any taints
            if any(t is not None for t in output_taints):
                return TaintedTensor(result, tuple(output_taints))

            return result

        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False

    return wrapper

def _wrap_view_reshape(original_fn, fn_name):
    """
    Registry-based reshape wrapper with product-aggregation taint propagation.

    Algorithm:
    1. Aggregate input taints from self._dim_taints using merge_history
    2. Extract consumed taints from TaintedInt reshape args
    3. Validate divisions >= 1 (error on expansion)
    4. Compute remaining taints
    5. For -1 dimension: lookup registry first, then compute if not found
    6. Register new values in global registry
    """
    from .types import DimTaint, TaintedInt, TaintedFloat, Taint
    from .registry import register_taint, lookup_taint
    from . import is_taint_tracking_enabled

    @functools.wraps(original_fn)
    def wrapper(self, *shape):
        # Handle special case: view(dtype)
        if fn_name == 'view' and len(shape) == 1 and isinstance(shape[0], torch.dtype):
            TaintedTensor._inside_hook.active = True
            try:
                return original_fn(self, shape[0])
            finally:
                TaintedTensor._inside_hook.active = False

        # Flatten shape args
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, TaintedShape)):
            shape_args = shape[0]
        else:
            shape_args = shape

        # Check if we're already inside a hook
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        TaintedTensor._inside_hook.active = True

        try:
            result = original_fn(self, *shape)

            # If taint tracking is disabled (during init / warmup / dummy_run),
            # skip the shape-analysis path entirely. This prevents warmup-only
            # tensor shapes (e.g. vLLM's _dummy_run trial num_tokens) from
            # leaving phantom entries in the global registry that later collide
            # with real workload values (Qwen3-30B-A3B + FLASHINFER + B=16
            # was the original symptom: 16 → None registered during warmup).
            if not is_taint_tracking_enabled():
                return result

            # print(f"[DEBUG @ _wrap_view_reshape] {fn_name} ( {self.shape}, {shape} ) -> {result.shape}")

            # if result.shape[1] == 9216 or result.shape[1] == 18432:
            #     print(f"[DEBUG @ _wrap_view_reshape] Result shape has dimension 9216 or 18432: {result.shape}", flush=True)

            if was_already_in_hook:
                return result

            # Only propagate taints if input is TaintedTensor
            if not isinstance(self, TaintedTensor):
                return result

            minus_one_idx = None
            for i, arg in enumerate(shape_args):
                if arg == -1:
                    minus_one_idx = i
                    break
            
            # If no -1, we can skip the complex logic and just assign taints based on explicit TaintedInt args
            if minus_one_idx is None:
                # No -1, just use arg taints directly 
                output_taints = []
                for arg in shape_args:
                    if isinstance(arg, (TaintedInt, TaintedFloat)) and arg.taint:
                        # Handle nested DimTaints properly
                        taint_to_use = arg.taint
                        while isinstance(taint_to_use, DimTaint) and isinstance(taint_to_use.taint, DimTaint):
                            taint_to_use = taint_to_use.taint

                        # Extract the plain taint
                        if isinstance(taint_to_use, DimTaint):
                            plain_taint = taint_to_use.taint
                            # If it already has merge history, preserve it
                            if taint_to_use.has_history:
                                output_taints.append(taint_to_use)
                            else:
                                arg_value = int(arg)
                                output_taints.append(DimTaint(plain_taint, merge_history={plain_taint: arg_value}))
                        else:
                            # Plain Taint
                            arg_value = int(arg)
                            output_taints.append(DimTaint(taint_to_use, merge_history={taint_to_use: arg_value}))
                    else:
                        # Plain int (untainted)
                        output_taints.append(DimTaint.from_taint(None))

                # Wrap result if we have any taints
                if any(t is not None and t.taint is not None for t in output_taints):
                    return TaintedTensor(result, tuple(output_taints))
                return result
            else:
                # STEP 1: Aggregate input taints from self._dim_taints
                aggregated = {}

                for dim_idx, dim_taint in enumerate(self._dim_taints):
                    if dim_taint is None:
                        continue

                    # Get the actual dimension size
                    dim_size = int(self.shape[dim_idx])

                    # Check if dimension has merge_history (from previous reshape)
                    if dim_taint.merge_history:
                        # Use components from merge_history
                        for taint, qty in dim_taint.merge_history.items():
                            aggregated[taint] = aggregated.get(taint, 1) * qty
                    else:
                        # Simple dimension: use actual size as quantity
                        if dim_taint.taint is not None:
                            plain_taint = unwrap_taint(dim_taint.taint)
                            aggregated[plain_taint] = aggregated.get(plain_taint, 1) * dim_size
                            
                # print(f"[DEBUG @ _wrap_view_reshape] Step 1 - aggregated input taints: {aggregated}")

                # STEP 2: Extract consumed taints from reshape args (TaintedInt objects)
                consumed = {}

                for i, arg in enumerate(shape_args):
                    if arg == -1:
                        minus_one_idx = i
                        continue

                    # Extract taint directly from TaintedInt/TaintedFloat
                    if isinstance(arg, (TaintedInt, TaintedFloat)) and arg.taint:
                        # Extract plain Taint from DimTaint if needed (for dictionary key)
                        taint = unwrap_taint(arg.taint)
                        value = int(arg)
                        consumed[taint] = consumed.get(taint, 1) * value
                        
                # print(f"[DEBUG @ _wrap_view_reshape] Step 2 - consumed taints from reshape args: {consumed}, minus_one_idx: {minus_one_idx}")

                # STEP 3: Validate and compute remaining taints
                remaining = {}

                for taint, agg_qty in aggregated.items():
                    if taint in consumed:
                        cons_qty = consumed[taint]

                        # Validate: no expansion allowed
                        # if cons_qty > agg_qty:
                        #     raise ValueError(
                        #         f"Reshape error: Cannot expand taint {taint} from {agg_qty} to {cons_qty}. "
                        #         f"Expansion (division < 1) is not supported."
                        #     )

                        division = agg_qty // cons_qty
                        
                        if division < 1:
                            remaining[taint] = None                            
                            continue

                        # Only keep if not fully consumed (division > 1)
                        if division > 1:
                            remaining[taint] = division
                        
                    else:
                        # Taint not consumed at all
                        remaining[taint] = agg_qty

                # print(f"[DEBUG @ _wrap_view_reshape] aggregated: {aggregated}, consumed: {consumed}, remaining: {remaining}, minus_one_idx: {minus_one_idx}")

                # STEP 4: Build output taints
                output_taints = []

                for i, arg in enumerate(shape_args):
                    if i == minus_one_idx:
                        # -1 dimension - THIS IS WHERE WE USE REGISTRY
                        resolved_value = int(result.shape[i])

                        # Look up in registry FIRST
                        existing_entry = lookup_taint(resolved_value)
                        # print(f"[DEBUG @ _wrap_view_reshape] minus_one resolved_value: {resolved_value}, existing_entry: {existing_entry}")

                        if existing_entry is not None:
                            # Found in registry - use it
                            # print(f"[DEBUG @ _wrap_view_reshape] Found existing entry in registry for value {resolved_value}: {existing_entry}")
                            if isinstance(existing_entry, dict):
                                # MIX entry
                                output_taints.append(DimTaint(existing_entry['taint'], merge_history=existing_entry['components']))
                                # print(f"[DEBUG @ _wrap_view_reshape] Found existing entry in registry for value {resolved_value}: {existing_entry}, {existing_entry['taint']}")
                            else:
                                if len(remaining) == 1 or len(remaining) == 0:
                                    taint = list(remaining.keys())[0] if remaining else None

                                    # Ensure existing_entry is a Taint, not a DimTaint
                                    if isinstance(existing_entry, DimTaint):
                                        actual_taint = existing_entry.taint
                                        print(f"[WARNING] Registry returned DimTaint for {resolved_value}, extracting inner taint: {actual_taint}")
                                    else:
                                        actual_taint = existing_entry

                                    output_taints.append(DimTaint(actual_taint, merge_history={actual_taint: resolved_value}))
                                    # print(f"[DEBUG @ _wrap_view_reshape] Found existing entry in registry for value {resolved_value}: {existing_entry}, pure taint")
                                else:
                                    raise ValueError("Registry entry is pure taint but we have multiple remaining components. Change number of tokens to prevent overlap.")
                        else:
                            # Not in registry - compute from remaining and register
                            # print(f"[DEBUG @ _wrap_view_reshape] No existing entry in registry for value {resolved_value}. Computing from remaining: {remaining}")
                            if len(remaining) == 0:
                                # All consumed
                                output_taints.append(DimTaint.from_taint(None))
                            elif len(remaining) == 1:
                                # Pure taint
                                taint = list(remaining.keys())[0]
                                qty = remaining[taint]

                                # Register as pure
                                # print(f"[DEBUG @ _wrap_view_reshape] Registering {resolved_value} as pure taint {taint} with quantity {qty}")
 
                                # Ensure taint is not already a DimTaint before registering
                                if isinstance(taint, DimTaint):
                                    # Extract the actual Taint object
                                    actual_taint = taint.taint
                                    print(f"[WARNING] taint was a DimTaint, extracting inner taint: {actual_taint}")
                                else:
                                    actual_taint = taint

                                register_taint(resolved_value, actual_taint)

                                # Create DimTaint with history for logging
                                # Make sure we use the actual Taint, not a DimTaint
                                output_taints.append(DimTaint(actual_taint, merge_history=remaining))
                            else:
                                # MIX taint
                                # DEBUG
                                # if resolved_value == 960:
                                    # print(f"\n[REGISTER MIX DEBUG] Registering 960 as MIX")
                                    # print(f"  components (remaining): {remaining}")
                                    # print(f"  type of components: {type(remaining)}")
                                    # print(f"  components keys: {list(remaining.keys()) if remaining else 'N/A'}")

                                # Register with components
                                # print(f"[DEBUG @ _wrap_view_reshape] Registering {resolved_value} as MIX with components: {remaining}")
                                register_taint(resolved_value, Taint('MIX'), components=remaining)

                                # Create DimTaint with history for logging
                                output_taints.append(DimTaint(Taint('MIX'), merge_history=remaining))

                    elif isinstance(arg, (TaintedInt, TaintedFloat)) and arg.taint:
                        # Explicit tainted dimension - preserve existing history if available
                        arg_value = int(arg)

                        if isinstance(arg.taint, DimTaint):
                            # Already a DimTaint - preserve it as-is (including merge_history)
                            inner_taint = unwrap_taint(arg.taint)
                            if arg.taint.merge_history:
                                # Has existing history - use it
                                output_taints.append(DimTaint(inner_taint, merge_history=arg.taint.merge_history))
                            else:
                                output_taints.append(DimTaint(inner_taint, merge_history={inner_taint: arg_value}))
                        else:
                            # Plain Taint - create new history
                            # Safety: ensure arg.taint is a base Taint, not DimTaint
                            base_taint = unwrap_taint(arg.taint)
                            output_taints.append(DimTaint(base_taint, merge_history={base_taint: arg_value}))
                    else:
                        # Plain int (untainted)
                        output_taints.append(DimTaint.from_taint(None))

            # Wrap result if we have any taints
            if any(t is not None and t.taint is not None for t in output_taints):
                return TaintedTensor(result, tuple(output_taints))

            return result

        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False
    
    return wrapper


def _wrap_permute(original_fn, fn_name):
    """
    Example:
        t = TaintedTensor((8, 32, 128), [BATCH, SEQ, HIDDEN])
        t.permute(2, 0, 1)  # dims=(2, 0, 1)
        # Output: (128, 8, 32) with taints [HIDDEN, BATCH, SEQ]
    """

    @functools.wraps(original_fn)
    def wrapper(self, *dims):
        # Handle both permute(1, 2, 3) and permute([1, 2, 3]) calling conventions
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = dims[0]
       
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False) 
        TaintedTensor._inside_hook.active = True
        try:
            result = original_fn(self, *dims)
            
            if was_already_in_hook:
                return result

            # If input has taints, reorder them according to dims
            if isinstance(self, TaintedTensor):
                new_taints = []
                for d in dims:
                    if d < len(self._dim_taints):
                        new_taints.append(self._dim_taints[d])
                    else:
                        new_taints.append(None)

                return TaintedTensor(result, tuple(new_taints))
                    
            return result
        
        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False


    return wrapper


def _wrap_transpose(original_fn, fn_name):
    """
    Wrap torch.Tensor.transpose to use explicit dimension indices.

    transpose(dim0, dim1) swaps two dimensions - this is simpler than permute
    since we only swap two positions.

    Example:
        t = TaintedTensor((8, 32, 128), [BATCH, SEQ, HIDDEN])
        t.transpose(0, 1)  # Swap BATCH and SEQ
        # Output: (32, 8, 128) with taints [SEQ, BATCH, HIDDEN]
    """
    from .types import DimTaint

    @functools.wraps(original_fn)
    def wrapper(self, dim0, dim1):
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        TaintedTensor._inside_hook.active = True
        try:
            result = original_fn(self, dim0, dim1)
            
            if was_already_in_hook:
                return result

            # If input has taints, swap the two dimensions
            if isinstance(self, TaintedTensor):
                # transpose just swaps two dimensions
                new_taints = list(self._dim_taints)

                # Handle negative indices
                ndim = len(new_taints)
                if dim0 < 0:
                    dim0 = ndim + dim0
                if dim1 < 0:
                    dim1 = ndim + dim1

                # Swap
                if dim0 < len(new_taints) and dim1 < len(new_taints):
                    new_taints[dim0], new_taints[dim1] = new_taints[dim1], new_taints[dim0]

                return TaintedTensor(result, tuple(new_taints))

            return result
    
        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False

    return wrapper


def _wrap_cat(original_fn):
    """
    Wrap torch.cat to explicitly handle concatenation along a dimension.

    torch.cat concatenates tensors along an existing dimension:
    - Tensors must have same shape except along the cat dimension
    - The cat dimension size is the sum of input sizes
    - Other dimensions are preserved

    Taint propagation:
    - Non-cat dimensions: preserve taint from first tensor
    - Cat dimension: preserve taint only if ALL inputs have the same taint,
      otherwise mark as None (mixed/independent)

    Example:
        a = TaintedTensor((4, 128), [BATCH_A, HIDDEN])
        b = TaintedTensor((8, 128), [BATCH_A, HIDDEN])  # Same BATCH_A taint
        torch.cat([a, b], dim=0)
        # Output: (12, 128) with taints [BATCH_A, HIDDEN]  # Preserved since same taint

        a = TaintedTensor((4, 128), [BATCH_A, HIDDEN])
        b = TaintedTensor((8, 128), [BATCH_B, HIDDEN])  # Different BATCH_B taint
        torch.cat([a, b], dim=0)
        # Output: (12, 128) with taints [None, HIDDEN]  # None since mixed taints

    Note: We currently don't track cat history (which tensors were concatenated).
    This could be added in the future if needed for split operations.
    """
    from .types import DimTaint

    @functools.wraps(original_fn)
    def wrapper(tensors, dim=0, *, out=None):
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        
        TaintedTensor._inside_hook.active = True
        try:
            result = original_fn(tensors, dim=dim, out=out)
            
            if was_already_in_hook:
                return result

            # Check if any input is tainted
            tainted_inputs = [t for t in tensors if isinstance(t, TaintedTensor)]
            if not tainted_inputs:
                return result

            # Get reference shape and taints from first tainted input
            ref_tensor = tainted_inputs[0]
            ndim = len(ref_tensor.shape)

            # Normalize negative dim
            if dim < 0:
                dim = ndim + dim

            # Build output taints
            output_taints = []
            for d in range(ndim):
                if d == dim:
                    # For the concatenated dimension:
                    # Preserve taint only if all inputs have the same taint
                    cat_dim_taints = []
                    for t in tainted_inputs:
                        if d < len(t._dim_taints):
                            taint_obj = t._dim_taints[d]
                            # Extract the plain Taint for comparison
                            if isinstance(taint_obj, DimTaint):
                                # DimTaint object - extract the taint attribute
                                cat_dim_taints.append(taint_obj.taint)
                            elif taint_obj is not None:
                                # Plain Taint object - use directly
                                cat_dim_taints.append(taint_obj)
                            else:
                                # None
                                cat_dim_taints.append(None)
                        else:
                            cat_dim_taints.append(None)

                    # Check if all taints are the same (ignoring None values for comparison)
                    non_none_taints = [t for t in cat_dim_taints if t is not None]
                    if non_none_taints and all(t == non_none_taints[0] for t in non_none_taints):
                        # All non-None taints are the same - preserve it
                        output_taints.append(DimTaint.from_taint(non_none_taints[0]))
                    else:
                        # Mixed taints or all None - mark as independent
                        output_taints.append(None)
                else:
                    # For non-cat dimensions: take taint from reference tensor
                    if d < len(ref_tensor._dim_taints):
                        output_taints.append(ref_tensor._dim_taints[d])
                    else:
                        output_taints.append(None)

            return TaintedTensor(result, tuple(output_taints))
    
        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False 

    return wrapper


def _wrap_stack(original_fn):
    """
    Wrap torch.stack to explicitly handle stacking along a new dimension.

    torch.stack creates a NEW dimension by stacking tensors:
    - All input tensors must have the same shape
    - A new dimension is inserted at position 'dim'
    - The new dimension size equals the number of tensors
    - All other dimensions are preserved but shifted

    Taint propagation:
    - New stacked dimension: marked as None (independent - represents "which tensor")
    - Other dimensions: preserved from input tensors, shifted by the stack position

    Example:
        a = TaintedTensor((8, 128), [BATCH, HIDDEN])
        b = TaintedTensor((8, 128), [BATCH, HIDDEN])
        torch.stack([a, b], dim=0)
        # Output: (2, 8, 128) with taints [None, BATCH, HIDDEN]
        #         New dim[0] is independent (which of 2 tensors)

        torch.stack([a, b], dim=1)
        # Output: (8, 2, 128) with taints [BATCH, None, HIDDEN]
        #         New dim[1] is independent, others shifted
    """
    from .types import DimTaint

    @functools.wraps(original_fn)
    def wrapper(tensors, dim=0, *, out=None):
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        TaintedTensor._inside_hook.active = True
        try:
            result = original_fn(tensors, dim=dim, out=out)
            if was_already_in_hook:
                    return result

            # Check if any input is tainted
            tainted_inputs = [t for t in tensors if isinstance(t, TaintedTensor)]
            if not tainted_inputs:
                return result

            # Get reference taints from first tainted input
            ref_tensor = tainted_inputs[0]
            ref_taints = ref_tensor._dim_taints

            # Normalize negative dim
            result_ndim = len(result.shape)
            if dim < 0:
                dim = result_ndim + dim

            # Build output taints by inserting None at the stack dimension
            output_taints = []
            input_idx = 0

            for out_idx in range(result_ndim):
                if out_idx == dim:
                    # New dimension created by stacking - mark as independent
                    output_taints.append(None)
                else:
                    # Existing dimension from input tensors (shifted if after stack dim)
                    if input_idx < len(ref_taints):
                        output_taints.append(ref_taints[input_idx])
                    else:
                        output_taints.append(None)
                    input_idx += 1

            return TaintedTensor(result, tuple(output_taints))
    
        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False

    return wrapper


def _wrap_split(original_fn):
    """
    Wrap torch.split to handle TaintedInt in split_size_or_sections argument.

    Two modes with different taint propagation rules:

    Mode 1: split_size_or_sections is a LIST
    - Each chunk's split dimension inherits taint from the corresponding list element
    - If list element is TaintedInt → use that taint
    - If list element is plain int → split dimension loses taint (None)
    - Non-split dimensions always preserve original taints

    Mode 2: split_size_or_sections is a single int
    - All chunks preserve ALL original taints (split dimension keeps its taint)

    Examples:
        # Mode 1: List with TaintedInt
        x = TaintedTensor((12, 4), [BATCH, HIDDEN])
        sizes = [TaintedInt(4, MC), TaintedInt(5, MC), TaintedInt(3, MC)]
        torch.split(x, sizes, dim=0)
        # → chunks get [MODEL_CONFIG, HIDDEN] (from TaintedInt)

        # Mode 1: List with plain int
        x = TaintedTensor((12, 4), [BATCH, HIDDEN])
        torch.split(x, [4, 5, 3], dim=0)
        # → chunks get [None, HIDDEN] (lost BATCH taint!)

        # Mode 2: Single int
        x = TaintedTensor((12, 4), [BATCH, HIDDEN])
        torch.split(x, 3, dim=0)
        # → chunks get [BATCH, HIDDEN] (preserved)
    """
    from .types import DimTaint

    @functools.wraps(original_fn)
    def wrapper(tensor, split_size_or_sections, dim=0):
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        TaintedTensor._inside_hook.active = True
        try:
            results = original_fn(tensor, split_size_or_sections, dim=dim)
            
            if was_already_in_hook:
                return results

            # If input is not tainted, return plain results
            if not isinstance(tensor, TaintedTensor):
                return results

            # Normalize negative dim
            ndim = len(tensor.shape)
            if dim < 0:
                dim = ndim + dim

            is_list = isinstance(split_size_or_sections, (list, tuple))

            if is_list:
                # Mode 1: List of sizes - use taint from each list element
                new_results = []
                for i, result in enumerate(results):
                    new_taints = list(tensor._dim_taints)

                    # Get taint from corresponding list element
                    if i < len(split_size_or_sections):
                        size_val = split_size_or_sections[i]
                        if isinstance(size_val, TaintedInt):
                            # TaintedInt → use its taint
                            new_taints[dim] = DimTaint.from_taint(size_val.taint)
                        else:
                            # Plain int → dimension loses taint
                            new_taints[dim] = None

                    new_results.append(TaintedTensor(result, tuple(new_taints)))
                return tuple(new_results)
            else:
                # Mode 2: Single int - preserve all original taints
                new_results = []
                for result in results:
                    new_results.append(TaintedTensor(result, tensor._dim_taints))
                return tuple(new_results)

        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False
                
    return wrapper


def _wrap_chunk(original_fn):
    """
    Wrap torch.chunk - preserves all original taints.

    torch.chunk(tensor, chunks, dim=0) divides into N chunks.
    Since it only takes a count (not sizes), we always preserve original taints.

    Example:
        x = TaintedTensor((12, 4), [BATCH, HIDDEN])
        torch.chunk(x, 4, dim=0)
        # → All chunks get [BATCH, HIDDEN]
    """
    @functools.wraps(original_fn)
    def wrapper(tensor, chunks, dim=0):
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        
        TaintedTensor._inside_hook.active = True
        try:
            results = original_fn(tensor, chunks, dim=dim)
            
            if was_already_in_hook:
                return results

            # If input is not tainted, return plain results
            if not isinstance(tensor, TaintedTensor):
                return results

            # Preserve all taints
            new_results = []
            for result in results:
                new_results.append(TaintedTensor(result, tensor._dim_taints))
                
            return tuple(new_results)

        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False
    return wrapper


def _wrap_unsqueeze(original_fn):
    """
    Wrap torch.unsqueeze to insert a new dimension with None taint.

    torch.unsqueeze(tensor, dim) inserts a new dimension of size 1 at position dim.
    The new dimension is always size 1, which is independent (workload-independent).

    Taint propagation:
    - New dimension at position 'dim': marked as None (independent, size 1)
    - All other dimensions: preserve original taints, shifted if after insertion point

    Example:
        x = TaintedTensor((8, 128), [BATCH, HIDDEN])
        torch.unsqueeze(x, dim=1)
        # Output: (8, 1, 128) with taints [BATCH, None, HIDDEN]
        #         New dim[1] is independent (size 1)

        torch.unsqueeze(x, dim=0)
        # Output: (1, 8, 128) with taints [None, BATCH, HIDDEN]
        #         New dim[0] is independent
    """
    from .types import DimTaint

    @functools.wraps(original_fn)
    def wrapper(input, dim):
        # Execute the operation
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        TaintedTensor._inside_hook.active = True
        try:
            result = original_fn(input, dim)
            if was_already_in_hook:
                return result

            # If input is not tainted, return plain result
            if not isinstance(input, TaintedTensor):
                return result

            # Normalize negative dim to positive index in the OUTPUT tensor
            result_ndim = len(result.shape)
            if dim < 0:
                dim = result_ndim + dim
                
            # Build output taints by inserting None at the unsqueezed dimension
            output_taints = []
            input_idx = 0

            for out_idx in range(result_ndim):
                if out_idx == dim:
                    # New dimension created by unsqueeze - always size 1, mark as independent
                    output_taints.append(None)
                else:
                    # Existing dimension from input tensor (shifted if after unsqueeze dim)
                    if input_idx < len(input._dim_taints):
                        output_taints.append(input._dim_taints[input_idx])
                    else:
                        output_taints.append(None)
                    input_idx += 1

            new_tensor = TaintedTensor(result, tuple(output_taints))

            return new_tensor

        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False
    return wrapper


def _wrap_squeeze(original_fn):
    """
    Wrap torch.squeeze to remove dimensions of size 1.

    torch.squeeze has two modes:
    1. squeeze(tensor) - removes ALL dimensions of size 1
    2. squeeze(tensor, dim) - removes dimension 'dim' ONLY if it has size 1

    Taint propagation:
    - Dimensions of size 1 are removed (their taints are dropped)
    - All other dimensions preserve their taints

    Examples:
        # Mode 1: Remove all size-1 dimensions
        x = TaintedTensor((1, 8, 1, 128), [None, BATCH, None, HIDDEN])
        torch.squeeze(x)
        # Output: (8, 128) with taints [BATCH, HIDDEN]

        # Mode 2: Remove specific dimension if size 1
        x = TaintedTensor((1, 8, 128), [None, BATCH, HIDDEN])
        torch.squeeze(x, dim=0)
        # Output: (8, 128) with taints [BATCH, HIDDEN]

        torch.squeeze(x, dim=1)  # dim 1 has size 8, not 1
        # Output: (1, 8, 128) with taints [None, BATCH, HIDDEN] (unchanged)
    """
    from .types import DimTaint

    @functools.wraps(original_fn)
    def wrapper(input, dim=None):
        # Execute the operation
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        
        TaintedTensor._inside_hook.active = True
        
        try:
            if dim is None:
                result = original_fn(input)
            else:
                result = original_fn(input, dim)
                
            if was_already_in_hook:
                return result
        
            # If input is not tainted, return plain result
            if not isinstance(input, TaintedTensor):
                return result

            # Determine which dimensions were actually removed
            input_shape = tuple(input.shape)
            result_shape = tuple(result.shape)

            # If shapes are the same, no dimensions were removed
            if input_shape == result_shape:
                return TaintedTensor(result, input._dim_taints)

            if dim is None:
                # Mode 1: Remove all size-1 dimensions
                # Build output taints by skipping dimensions with size 1
                output_taints = []
                for i, size in enumerate(input_shape):
                    if size != 1:
                        # Keep this dimension's taint
                        if i < len(input._dim_taints):
                            output_taints.append(input._dim_taints[i])
                        else:
                            output_taints.append(None)
                    # else: skip size-1 dimensions
            else:
                # Mode 2: Remove specific dimension if it has size 1
                # Normalize negative dim
                input_ndim = len(input_shape)
                if dim < 0:
                    dim = input_ndim + dim

                # Check if the dimension was actually removed (had size 1)
                if dim < len(input_shape) and input_shape[dim] == 1:
                    # Build output taints by skipping the specified dimension
                    output_taints = []
                    for i in range(input_ndim):
                        if i != dim:
                            if i < len(input._dim_taints):
                                output_taints.append(input._dim_taints[i])
                            else:
                                output_taints.append(None)
                else:
                    # Dimension was not removed (didn't have size 1)
                    # Preserve all taints unchanged
                    output_taints = list(input._dim_taints)

            return TaintedTensor(result, tuple(output_taints))

        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False
                
    return wrapper


def _wrap_unbind(original_fn):
    """
    Wrap torch.unbind - removes a dimension and returns a tuple of tensors.

    torch.unbind(tensor, dim=0) removes dimension 'dim' and returns a tuple
    of slices along that dimension.

    Taint propagation (Tier 1 - Pure Argument-Based):
    - The dimension at 'dim' is removed (taint dropped)
    - All other dimensions preserve their taints

    Example:
        x = TaintedTensor((4, 8, 128), [BATCH, SEQ, HIDDEN])
        torch.unbind(x, dim=1)
        # Returns tuple of 8 tensors, each with shape (4, 128)
        # Each tensor has taints [BATCH, HIDDEN] (SEQ dimension removed)
    """
    from .types import DimTaint

    @functools.wraps(original_fn)
    def wrapper(input, dim=0):
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        TaintedTensor._inside_hook.active = True
        try:
            results = original_fn(input, dim)
            if was_already_in_hook:
                return results

            # If input is not tainted, return plain results
            if not isinstance(input, TaintedTensor):
                return results

            # Normalize negative dim
            input_ndim = len(input.shape)
            if dim < 0:
                dim = input_ndim + dim

            # Build output taints by removing the unbind dimension
            output_taints = []
            for i in range(input_ndim):
                if i != dim:
                    if i < len(input._dim_taints):
                        output_taints.append(input._dim_taints[i])
                    else:
                        output_taints.append(None)

            # Apply the same taints to all result tensors
            output_taints_tuple = tuple(output_taints)
            new_results = []
            for result in results:
                new_results.append(TaintedTensor(result, output_taints_tuple))

            return tuple(new_results)
        
        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False

    return wrapper


def _wrap_matmul(original_fn):
    """
    Wrap torch.matmul - matrix multiplication with dimension contraction.

    Taint propagation (Tier 2 - Semantic Rules + Validation):
    - Matmul contracts the last dim of A with the second-to-last dim of B
    - Batch dimensions are broadcast-merged
    - Output gets: [...batch..., A[-2], B[-1]]

    Cases:
    1. (n,) @ (n,) → scalar (no taints)
    2. (n,) @ (n, k) → (k,)
    3. (m, n) @ (n,) → (m,)
    4. (m, n) @ (n, k) → (m, k)
    5. (..., m, n) @ (..., n, k) → (..., m, k) - batched matmul

    Examples:
        A: (8, 1024) with (BATCH, HIDDEN_IN)
        B: (1024, 512) with (HIDDEN_IN, HIDDEN_OUT)
        Result: (8, 512) with (BATCH, HIDDEN_OUT)
    """
    from .types import DimTaint

    @functools.wraps(original_fn)
    def wrapper(input, other):
        # Set flag to indicate we're inside a hook
        # This tells dispatch_with_profiler to skip propagate_taints but still add annotation
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)

        TaintedTensor._inside_hook.active = True
        try:
            # Call original function with TaintedTensors (no unwrapping)
            # dispatch_with_profiler will add annotation with input taints, then return plain result
            result = original_fn(input, other)

            # If nested hook call, return immediately without wrapping
            if was_already_in_hook:
                return result

            # Both must be TaintedTensor for custom taint propagation
            if not (isinstance(input, TaintedTensor) and isinstance(other, TaintedTensor)):
                return result

            # Custom taint propagation logic
            input_shape = input.shape
            other_shape = other.shape
            input_ndim = len(input_shape)
            other_ndim = len(other_shape)

            # Case 1: (n,) @ (n,) → scalar
            if input_ndim == 1 and other_ndim == 1:
                return result  # Scalar has no dimensions to taint

            # Case 2: (n,) @ (n, k) → (k,)
            elif input_ndim == 1 and other_ndim == 2:
                # Output gets taint from second dim of other
                new_taints = (other._dim_taints[1],)
                return TaintedTensor(result, new_taints)

            # Case 3: (m, n) @ (n,) → (m,)
            elif input_ndim == 2 and other_ndim == 1:
                # Output gets taint from first dim of input
                new_taints = (input._dim_taints[0],)
                return TaintedTensor(result, new_taints)

            # Case 4: (m, n) @ (n, k) → (m, k)
            elif input_ndim == 2 and other_ndim == 2:
                # Output: [input_dim0, other_dim1]
                new_taints = (input._dim_taints[0], other._dim_taints[1])
                return TaintedTensor(result, new_taints)

            # Case 5: Batched matmul (..., m, n) @ (..., n, k) → (..., m, k)
            else:
                # Merge batch dimensions using broadcast rules
                # For simplicity, take batch dims from input (left operand)
                # In a full implementation, we'd handle broadcasting properly

                input_batch_taints = input._dim_taints[:-2] if input_ndim > 2 else ()
                other_batch_taints = other._dim_taints[:-2] if other_ndim > 2 else ()

                # Simple merge: if both have batch dims and they match in length, use input's
                # Otherwise, use the longer one (handles broadcasting)
                if len(input_batch_taints) >= len(other_batch_taints):
                    batch_taints = input_batch_taints
                else:
                    batch_taints = other_batch_taints

                # Matmul on last two dims: output gets [input[-2], other[-1]]
                m_taint = input._dim_taints[-2] if input_ndim >= 2 else None
                k_taint = other._dim_taints[-1] if other_ndim >= 1 else None

                new_taints = batch_taints + (m_taint, k_taint)
                return TaintedTensor(result, new_taints)
        finally:
            # Only turn off flag if we were the ones who turned it on
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False

    return wrapper


def _wrap_flatten(original_fn):
    """
    Wrap torch.flatten - converts multi-dimensional tensor to 1D or 2D.

    flatten(input, start_dim=0, end_dim=-1) flattens dimensions [start_dim, end_dim].

    Taint propagation using input arguments:
    - Dimensions before start_dim: preserved as-is
    - Dimensions [start_dim, end_dim]: merged into single dimension with:
      * If all merged dims have the SAME taint → use that taint
      * If merged dims have DIFFERENT taints → merge using TaintedInt multiplication logic
        - Both contain 'CONFIG' → 'CONFIG'
        - One is NUM_XX and other contains 'CONFIG' → 'MIX'
        - Otherwise → None
    - Dimensions after end_dim: preserved as-is

    Examples:
        x = TaintedTensor((2, 3, 4, 5), [BATCH, NUM_REQS, MODEL_CONFIG, C])
        torch.flatten(x, start_dim=1, end_dim=2)
        # Output: (2, 12, 5) with taints [BATCH, MIX, C]
        # NUM_REQS * MODEL_CONFIG = MIX

        x = TaintedTensor((2, 3, 4, 5), [BATCH, SEQ, SEQ, C])
        torch.flatten(x, start_dim=1, end_dim=2)
        # Output: (2, 12, 5) with taints [BATCH, SEQ, C]
        # Both merged dims are SEQ, so merged dim keeps SEQ
    """
    from .types import DimTaint

    @functools.wraps(original_fn)
    def wrapper(input, start_dim=0, end_dim=-1):
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        TaintedTensor._inside_hook.active = True
        try:
            result = original_fn(input, start_dim, end_dim)
            if was_already_in_hook:
                return result

            # If input is not tainted, return plain result
            if not isinstance(input, TaintedTensor):
                return result

            # Normalize negative indices
            input_ndim = len(input.shape)
            if start_dim < 0:
                start_dim = input_ndim + start_dim
            if end_dim < 0:
                end_dim = input_ndim + end_dim

            # Clamp to valid range
            start_dim = max(0, min(start_dim, input_ndim - 1))
            end_dim = max(0, min(end_dim, input_ndim - 1))

            # Ensure start <= end
            if start_dim > end_dim:
                start_dim, end_dim = end_dim, start_dim

            # Build output taints:
            # [preserved dims before start] + [merged dim] + [preserved dims after end]
            new_taints = []

            # 1. Preserve dimensions before start_dim
            for i in range(start_dim):
                new_taints.append(input._dim_taints[i])

            # 2. Merged dimension: merge taints using TaintedInt multiplication logic
            merged_taints = [input._dim_taints[i] for i in range(start_dim, end_dim + 1)]

            # Check if all merged taints are the same
            first_taint = merged_taints[0]
            if all(t == first_taint for t in merged_taints):
                # All merged dims have same taint - preserve it
                new_taints.append(first_taint)
            else:
                # Different taints - merge them using TaintedInt multiplication logic
                from .types import TaintedInt
                merged_taint = None
                for taint in merged_taints:
                    if taint is not None:
                        # Extract raw taint from DimTaint if needed
                        raw_taint = taint.taint if isinstance(taint, DimTaint) else taint

                        if merged_taint is None:
                            merged_taint = raw_taint
                        else:
                            # Merge taints using TaintedInt multiplication logic
                            dummy1 = TaintedInt(1, merged_taint)
                            dummy2 = TaintedInt(1, raw_taint)
                            result_taint = (dummy1 * dummy2).taint
                            merged_taint = result_taint
                new_taints.append(DimTaint.from_taint(merged_taint) if merged_taint is not None else None)

            # 3. Preserve dimensions after end_dim
            for i in range(end_dim + 1, input_ndim):
                new_taints.append(input._dim_taints[i])

            return TaintedTensor(result, tuple(new_taints))

        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False

    return wrapper


def _wrap_pad(original_fn):
    """
    Wrap torch.nn.functional.pad - pads selected trailing dimensions.

    pad is a flat tuple (pad_last_begin, pad_last_end, pad_second_last_begin,
    pad_second_last_end, ...), grouped in pairs from the LAST input dim
    backward. Output dim size for each padded dim is:
        output_dim = input_dim + pad_begin + pad_end

    Taint propagation:
    - Non-padded leading dims: preserve the input's taint unchanged.
    - Padded dims: preserve the input's taint on that dim. Padding enlarges
      the physical size but does not change the dim's semantic role. For
      models like gpt-oss, FusedMoE pads hidden from MODEL_CONFIG(2880) to
      MODEL_CONFIG(3072); without this hook the padded dim falls back to an
      untainted ?(3072) via the generic shape-matching path.

    Example:
        x = TaintedTensor((60, 2880), [NUM_TOKS, MODEL_CONFIG(2880)])
        F.pad(x, (0, 192), mode='constant', value=0.0)
        # result shape (60, 3072), taints [NUM_TOKS, MODEL_CONFIG]
    """
    @functools.wraps(original_fn)
    def wrapper(input, pad, *args, **kwargs):
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        TaintedTensor._inside_hook.active = True
        try:
            result = original_fn(input, pad, *args, **kwargs)
            if was_already_in_hook:
                return result
            if not isinstance(input, TaintedTensor):
                return result

            # Copy input dim taints. Pad preserves the semantic role of each dim.
            new_taints = list(input._dim_taints)
            # Sanity: result should have same ndim as input.
            if len(result.shape) != len(new_taints):
                return result
            return TaintedTensor(result, tuple(new_taints))
        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False

    return wrapper


def _wrap_expand(original_fn):
    """
    Wrap torch.expand - broadcasts tensor to a new shape.

    expand(*sizes) broadcasts dimensions of size 1 to larger sizes.
    Can also add new leading dimensions.

    Taint propagation:
    - New leading dimensions: None (independent)
    - Broadcasted dimensions (1 -> n): None (workload-dependent)
    - Preserved dimensions: keep original taint

    Examples:
        x = TaintedTensor((1, 5), [BATCH, HIDDEN])
        x.expand(3, 5)
        # Output: (3, 5) with taints [None, HIDDEN]
        # BATCH dimension was size 1, expanded to 3, so loses taint

        x = TaintedTensor((5, 1), [HIDDEN, None])
        x.expand(5, 10)
        # Output: (5, 10) with taints [HIDDEN, None]
        # First dim preserved, second was already None

        x = TaintedTensor((3, 5), [BATCH, HIDDEN])
        x.expand(2, 3, 5)
        # Output: (2, 3, 5) with taints [None, BATCH, HIDDEN]
        # New leading dimension added
    """
    from .types import DimTaint

    @functools.wraps(original_fn)
    def wrapper(input, *sizes):
        # Handle both expand(2, 3, 4) and expand([2, 3, 4]) calling styles
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list, torch.Size)):
            sizes = sizes[0]
        
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        TaintedTensor._inside_hook.active = True
        try:
            result = original_fn(input, *sizes)
            if was_already_in_hook:
                return result

            # If input is not tainted, return plain result
            if not isinstance(input, TaintedTensor):
                return result

            input_shape = tuple(input.shape)
            output_shape = tuple(result.shape)
            input_ndim = len(input_shape)
            output_ndim = len(output_shape)

            # Calculate offset for new leading dimensions
            # expand can add dimensions at the front
            dim_offset = output_ndim - input_ndim

            out_taints = []

            for out_idx in range(output_ndim):
                if out_idx < dim_offset:
                    # New leading dimension - always independent
                    out_taints.append(None)
                else:
                    # Map to input dimension
                    in_idx = out_idx - dim_offset
                    in_size = input_shape[in_idx]
                    out_size = output_shape[out_idx]

                    if in_size == 1 and out_size > 1:
                        # Broadcasted dimension (1 -> n) - becomes independent
                        out_taints.append(None)
                    elif in_size == out_size:
                        # Preserved dimension - keep taint
                        if in_idx < len(input._dim_taints):
                            out_taints.append(input._dim_taints[in_idx])
                        else:
                            out_taints.append(None)
                    else:
                        # This shouldn't happen in valid expand operation
                        # (in_size must be 1 or equal to out_size)
                        out_taints.append(None)

            return TaintedTensor(result, tuple(out_taints))

        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False
    return wrapper


def _wrap_linear(original_fn):
    """
    Wrap torch.nn.functional.linear - applies a linear transformation.

    linear(input, weight, bias=None) computes: input @ weight.T + bias

    Taint propagation (Tier 2 - Semantic Rules + Validation):
    - Input: (..., in_features)
    - Weight: (out_features, in_features)
    - Output: (..., out_features)

    Rule: Preserve all batch dims from input, last dim from weight's first dim

    Example:
        input: (8, 128, 1024) with (BATCH, SEQ, HIDDEN_IN)
        weight: (512, 1024) with (HIDDEN_OUT, HIDDEN_IN)
        output: (8, 128, 512) with (BATCH, SEQ, HIDDEN_OUT)
    """

    @functools.wraps(original_fn)
    def wrapper(input, weight, bias=None):
        # Set flag to indicate we're inside a hook
        # This tells dispatch_with_profiler to skip propagate_taints but still add annotation
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)

        TaintedTensor._inside_hook.active = True
        try:
            result = original_fn(input, weight, bias)

            # If nested hook call, return immediately without wrapping
            if was_already_in_hook:
                return result

            # Input must be TaintedTensor
            if not isinstance(input, TaintedTensor):
                return result

            # If weight is not tainted, preserve batch dims and mark last dim as None
            if not isinstance(weight, TaintedTensor):
                new_taints = input._dim_taints[:-1] + (None,)
                return TaintedTensor(result, new_taints)

            # VALIDATION: Check shapes match expectations
            # input: (..., in_features), weight: (out_features, in_features)
            if len(weight.shape) != 2:
                # Unexpected weight shape, fall back to no taint
                return result

            if input.shape[-1] != weight.shape[1]:
                # Shape mismatch - shouldn't happen in valid linear operation
                return result

            # SEMANTIC RULE: batch dims from input + output dim from weight[0]
            batch_taints = input._dim_taints[:-1]
            output_dim_taint = weight._dim_taints[0]

            new_taints = batch_taints + (output_dim_taint,)
            return TaintedTensor(result, new_taints)
        finally:
            # Only turn off flag if we were the ones who turned it on
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False

    return wrapper


class PatchedSize(tuple):
    """
    Class acting as a proxy for torch.Size.
    Subclassing tuple ensures it's treated as a type (compatible with | operator).

    For clean (non-tainted) shapes, we create actual PatchedSize instances
    instead of returning torch._original_size, to avoid pickle identity issues
    when torch.Size is patched.
    """
    def __new__(cls, sizes):
        # Scan for taints
        has_taint = False
        taints = []
        plain_sizes = []

        # Handle single integer argument case: torch.Size(5) -> (5,)
        if isinstance(sizes, int):
            sizes = (sizes,)

        for s in sizes:
            if isinstance(s, TaintedInt):
                has_taint = True
                taints.append(s.taint)
                plain_sizes.append(int(s))
            else:
                taints.append(None)
                plain_sizes.append(s)

        # If tainted, return TaintedShape (which is a tuple subclass)
        if has_taint:
            return TaintedShape(tuple(plain_sizes), tuple(taints))

        # Create actual PatchedSize instance for clean shapes
        # This avoids pickle identity mismatch (original torch.Size vs patched torch.Size)
        return tuple.__new__(cls, plain_sizes)

    def __reduce__(self):
        # Pickle as a plain tuple to avoid any torch.Size identity issues
        return (tuple, (tuple(self),))

    def __reduce_ex__(self, protocol):
        return self.__reduce__()


# ---------------------------------------------------------------------------
# Einops and Einsum Wrappers
# ---------------------------------------------------------------------------

def _parse_einops_pattern(pattern, input_shape, axes_lengths):
    """
    Parse einops pattern to create complete mapping from input to output dimensions.

    Returns: list where each element corresponds to an output dimension and contains:
        - Single int: copy taint from that input dimension
        - List of ints: merge taints from those input dimensions
        - None: new dimension (untainted)
    """
    # Split the pattern into input and output parts
    parts = pattern.split('->')
    if len(parts) != 2:
        return None  # Can't parse

    input_pattern, output_pattern = parts[0].strip(), parts[1].strip()

    def tokenize_pattern(pat):
        """Tokenize a pattern into dimension names/groups."""
        tokens = []
        i = 0
        while i < len(pat):
            c = pat[i]
            if c == '(':
                # Find matching closing paren
                depth = 1
                j = i + 1
                while j < len(pat) and depth > 0:
                    if pat[j] == '(':
                        depth += 1
                    elif pat[j] == ')':
                        depth -= 1
                    j += 1
                # Extract content inside parens - these are grouped dims
                group_content = pat[i+1:j-1].strip()
                sub_dims = [d.strip() for d in group_content.split() if d.strip()]
                tokens.append(('group', sub_dims))
                i = j
            elif c == '.' and i + 2 < len(pat) and pat[i:i+3] == '...':
                tokens.append(('ellipsis', '...'))
                i += 3
            elif c.isalpha() or c.isdigit() or c == '_':
                # Dimension name
                j = i
                while j < len(pat) and (pat[j].isalnum() or pat[j] == '_'):
                    j += 1
                tokens.append(('dim', pat[i:j]))
                i = j
            else:
                i += 1  # Skip whitespace
        return tokens

    input_tokens = tokenize_pattern(input_pattern)
    output_tokens = tokenize_pattern(output_pattern)

    # Count non-ellipsis dimensions in input and output
    input_named_dims = sum(1 for tok_type, tok_val in input_tokens
                           if tok_type != 'ellipsis')
    output_named_dims = sum(1 for tok_type, tok_val in output_tokens
                            if tok_type != 'ellipsis')

    # Calculate how many dimensions the ellipsis covers
    ellipsis_count_input = len(input_shape) - input_named_dims
    ellipsis_count_output = len(input_shape) - input_named_dims  # Should be the same

    # Build a mapping: dimension name -> input position(s)
    # For grouped input dims (e.g., "(h d)"), we need to know which input dim they came from
    dim_to_input_pos = {}
    input_pos = 0
    for tok_type, tok_val in input_tokens:
        if tok_type == 'group':
            # Grouped dimensions in input (e.g., "(h d)") -> this is ONE dimension that will split
            # Each sub-dimension in the group should map to the SAME input position
            for sub_dim in tok_val:
                dim_to_input_pos[sub_dim] = [input_pos]
            input_pos += 1
        elif tok_type == 'dim':
            dim_to_input_pos[tok_val] = [input_pos]
            input_pos += 1
        elif tok_type == 'ellipsis':
            # Ellipsis represents multiple dimensions - store the range
            # The ellipsis covers dimensions [0 ... ellipsis_count_input - 1]
            dim_to_input_pos['...'] = list(range(ellipsis_count_input))
            input_pos += ellipsis_count_input

    # Build output mapping
    output_mapping = []
    for tok_type, tok_val in output_tokens:
        if tok_type == 'group':
            # Grouped dimensions in output (e.g., "(h d)") -> merge multiple input dims
            input_positions = []
            for sub_dim in tok_val:
                if sub_dim in dim_to_input_pos and dim_to_input_pos[sub_dim]:
                    # Only add if not already in list (avoid duplicates for splits)
                    for pos in dim_to_input_pos[sub_dim]:
                        if pos not in input_positions:
                            input_positions.append(pos)
            if len(input_positions) > 1:
                output_mapping.append(input_positions)  # MERGE these dimensions
            elif len(input_positions) == 1:
                output_mapping.append(input_positions[0])  # Single dimension
            else:
                output_mapping.append(None)  # New dimension
        elif tok_type == 'dim':
            # Check if this dimension exists in input
            if tok_val in dim_to_input_pos and dim_to_input_pos[tok_val]:
                positions = dim_to_input_pos[tok_val]
                if len(positions) == 1:
                    # Could be COPY or SPLIT
                    # If multiple output dims map to same input, it's a SPLIT
                    output_mapping.append(positions[0])
                else:
                    output_mapping.append(positions)  # Multiple inputs - merge
            elif tok_val in axes_lengths:
                # New dimension specified in axes_lengths
                output_mapping.append(None)
            else:
                # Unknown dimension
                output_mapping.append(None)
        elif tok_type == 'ellipsis':
            # Ellipsis in output - expand to preserve corresponding dims from input
            ellipsis_dims = dim_to_input_pos.get('...', [])
            for dim_idx in ellipsis_dims:
                output_mapping.append(dim_idx)

    return output_mapping


def _wrap_einops_rearrange(original_fn):
    """
    Wrap einops.rearrange - flexible tensor reshaping with pattern notation.

    einops.rearrange(tensor, pattern, **axes_lengths)

    Patterns:
    - MERGE: "... h d -> ... (h d)" - merge dimensions
    - SPLIT: "... (h d) -> ... h d" with h=8 - split dimension
    - PERMUTE: "b h w c -> b c h w" - reorder dimensions

    Taint propagation:
    - Parse pattern to get direct mapping from input to output dimensions
    - COPY: Single input dim -> output dim (preserve taint)
    - SPLIT: Input dim -> multiple output dims
      * If axes_lengths provides TaintedInt for split dim, use that taint
      * Otherwise, broadcast taint from source dimension
    - MERGE: Multiple input dims -> output dim (multiply taints using TaintedInt logic)
    """
    from .types import TaintedInt, DimTaint

    @functools.wraps(original_fn)
    def wrapper(tensor, pattern, **axes_lengths):
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        TaintedTensor._inside_hook.active = True

        try:
            # If not a TaintedTensor, just call original
            if not isinstance(tensor, TaintedTensor):
                return original_fn(tensor, pattern, **axes_lengths)

            # Get input taints
            input_taints = tensor._dim_taints

            # Call original operation
            result = original_fn(tensor, pattern, **axes_lengths)

            if was_already_in_hook:
                return result

            # Parse pattern to get output mapping and dimension names
            parts = pattern.split('->')
            if len(parts) != 2:
                return result

            input_pattern, output_pattern = parts[0].strip(), parts[1].strip()

            # Tokenize output pattern to get dimension names
            output_tokens = []
            i = 0
            while i < len(output_pattern):
                c = output_pattern[i]
                if c == '(':
                    # Find matching closing paren
                    depth = 1
                    j = i + 1
                    while j < len(output_pattern) and depth > 0:
                        if output_pattern[j] == '(':
                            depth += 1
                        elif output_pattern[j] == ')':
                            depth -= 1
                        j += 1
                    group_content = output_pattern[i+1:j-1].strip()
                    sub_dims = [d.strip() for d in group_content.split() if d.strip()]
                    output_tokens.extend(sub_dims)  # Flatten grouped dims
                    i = j
                elif c == '.' and i + 2 < len(output_pattern) and output_pattern[i:i+3] == '...':
                    output_tokens.append('...')
                    i += 3
                elif c.isalpha() or c.isdigit() or c == '_':
                    j = i
                    while j < len(output_pattern) and (output_pattern[j].isalnum() or output_pattern[j] == '_'):
                        j += 1
                    output_tokens.append(output_pattern[i:j])
                    i = j
                else:
                    i += 1  # Skip whitespace

            # Parse pattern to get output mapping
            output_mapping = _parse_einops_pattern(pattern, tuple(tensor.shape), axes_lengths)

            if output_mapping is None:
                # Couldn't parse pattern - return untainted result
                return result

            # Build output taints based on mapping
            out_taints = []
            for idx, mapping in enumerate(output_mapping):
                # Get the dimension name for this output position
                dim_name = output_tokens[idx] if idx < len(output_tokens) else None

                if mapping is None:
                    # New dimension - check if it has a taint from axes_lengths
                    if dim_name and dim_name in axes_lengths:
                        axis_value = axes_lengths[dim_name]
                        if isinstance(axis_value, TaintedInt):
                            out_taints.append(axis_value.taint)
                        else:
                            out_taints.append(None)
                    else:
                        out_taints.append(None)
                elif isinstance(mapping, int):
                    # COPY or SPLIT: Single input dimension
                    # Check if this is a split dimension with tainted size
                    if dim_name and dim_name in axes_lengths:
                        axis_value = axes_lengths[dim_name]
                        if isinstance(axis_value, TaintedInt):
                            # Use the taint from the axes_lengths parameter
                            out_taints.append(axis_value._taint)
                        else:
                            # No taint in axes_lengths, use source dimension taint
                            if mapping < len(input_taints):
                                out_taints.append(input_taints[mapping])
                            else:
                                out_taints.append(None)
                    else:
                        # Regular copy - preserve taint from source
                        if mapping < len(input_taints):
                            out_taints.append(input_taints[mapping])
                        else:
                            out_taints.append(None)
                elif isinstance(mapping, list):
                    # MERGE or SPLIT: Multiple input dimensions
                    # If all point to same input dim, it's a SPLIT - check for tainted size
                    # If different input dims, it's a MERGE - combine taints
                    unique_inputs = list(set(mapping))
                    if len(unique_inputs) == 1:
                        # SPLIT: All from same input dim
                        # Check if this split dimension has a tainted size
                        if dim_name and dim_name in axes_lengths:
                            axis_value = axes_lengths[dim_name]
                            if isinstance(axis_value, TaintedInt):
                                out_taints.append(axis_value._taint)
                            else:
                                # No taint, use source
                                in_idx = unique_inputs[0]
                                if in_idx < len(input_taints):
                                    out_taints.append(input_taints[in_idx])
                                else:
                                    out_taints.append(None)
                        else:
                            # No axes_lengths for this dim, use source taint
                            in_idx = unique_inputs[0]
                            if in_idx < len(input_taints):
                                out_taints.append(input_taints[in_idx])
                            else:
                                out_taints.append(None)
                    else:
                        # MERGE: Different input dims - combine taints
                        merged_taints = [input_taints[i] for i in unique_inputs if i < len(input_taints)]

                        # Check if all merged taints are the same
                        first_taint = merged_taints[0] if merged_taints else None
                        if all(t == first_taint for t in merged_taints):
                            out_taints.append(first_taint)
                        else:
                            # Different taints - merge using TaintedInt multiplication logic
                            merged_taint = None
                            for taint in merged_taints:
                                if taint is not None:
                                    # Extract raw taint from DimTaint if needed
                                    raw_taint = taint.taint if isinstance(taint, DimTaint) else taint

                                    if merged_taint is None:
                                        merged_taint = raw_taint
                                    else:
                                        dummy1 = TaintedInt(1, merged_taint)
                                        dummy2 = TaintedInt(1, raw_taint)
                                        result_taint = (dummy1 * dummy2).taint
                                        merged_taint = result_taint
                            out_taints.append(DimTaint.from_taint(merged_taint) if merged_taint is not None else None)
                else:
                    # Unknown mapping type
                    out_taints.append(None)

            return TaintedTensor(result, tuple(out_taints))

        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False

    return wrapper


def _wrap_einops_reduce(original_fn):
    """
    Wrap einops.reduce - reduction operations with pattern notation.

    einops.reduce(tensor, pattern, reduction, **axes_lengths)

    Pattern example: "b h w c -> b c" with reduction='mean'

    Taint propagation:
    - Use pattern parsing to determine which dimensions are preserved
    - Reduced dimensions are removed, preserved dimensions keep their taints
    """
    from .types import DimTaint

    @functools.wraps(original_fn)
    def wrapper(tensor, pattern, reduction, **axes_lengths):
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        TaintedTensor._inside_hook.active = True

        try:
            # If not a TaintedTensor, just call original
            if not isinstance(tensor, TaintedTensor):
                return original_fn(tensor, pattern, reduction, **axes_lengths)

            # Get input taints
            input_taints = tensor._dim_taints

            # Call original operation
            result = original_fn(tensor, pattern, reduction, **axes_lengths)

            if was_already_in_hook:
                return result

            # Parse pattern to get output mapping (same as rearrange)
            output_mapping = _parse_einops_pattern(pattern, tuple(tensor.shape), axes_lengths)

            if output_mapping is None:
                # Couldn't parse - return untainted
                return result

            # Build output taints (preserved dimensions keep taints)
            out_taints = []
            for mapping in output_mapping:
                if mapping is None or mapping == '...':
                    out_taints.append(None)
                elif isinstance(mapping, int):
                    if mapping < len(input_taints):
                        out_taints.append(input_taints[mapping])
                    else:
                        out_taints.append(None)
                elif isinstance(mapping, list):
                    # For reduce, grouped dims should not be merged - just take first
                    in_idx = mapping[0]
                    if in_idx < len(input_taints):
                        out_taints.append(input_taints[in_idx])
                    else:
                        out_taints.append(None)
                else:
                    out_taints.append(None)

            return TaintedTensor(result, tuple(out_taints))

        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False

    return wrapper


def _wrap_einops_repeat(original_fn):
    """
    Wrap einops.repeat - repeat tensor elements with pattern notation.

    einops.repeat(tensor, pattern, **axes_lengths)

    Pattern example: "h w -> h w c" with c=3

    Taint propagation:
    - New repeated dimensions get None (workload-dependent)
    - Preserved dimensions keep their taints
    """
    from .types import DimTaint

    @functools.wraps(original_fn)
    def wrapper(tensor, pattern, **axes_lengths):
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        TaintedTensor._inside_hook.active = False

        try:
            # If not a TaintedTensor, just call original
            if not isinstance(tensor, TaintedTensor):
                return original_fn(tensor, pattern, **axes_lengths)

            # Get input taints
            input_taints = tensor._dim_taints

            # Call original operation
            result = original_fn(tensor, pattern, **axes_lengths)

            if was_already_in_hook:
                return result

            # Parse pattern to get output mapping
            output_mapping = _parse_einops_pattern(pattern, tuple(tensor.shape), axes_lengths)

            if output_mapping is None:
                # Couldn't parse - return untainted
                return result

            # Build output taints (new dimensions are None, preserved keep taints)
            out_taints = []
            for mapping in output_mapping:
                if mapping is None or mapping == '...':
                    out_taints.append(None)
                elif isinstance(mapping, int):
                    if mapping < len(input_taints):
                        out_taints.append(input_taints[mapping])
                    else:
                        out_taints.append(None)
                elif isinstance(mapping, list):
                    # For repeat, new dimensions from repetition are None
                    out_taints.append(None)
                else:
                    out_taints.append(None)

            return TaintedTensor(result, tuple(out_taints))

        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False

    return wrapper


def _wrap_einsum(original_fn):
    """
    Wrap torch.einsum - Einstein summation convention.

    torch.einsum(equation, *operands)

    Examples:
    - "mhd,nd->hmn" - complex matrix multiplication with broadcasting
    - "bd,dn->bn" - standard matrix multiplication
    - "bhwc,hkc->bhwk" - custom einsum operation

    Taint propagation:
    - Parse einsum equation to understand dimension mappings
    - Match output dimensions to input dimensions by letter
    - Contracted dimensions (not in output) are removed
    """
    from .types import DimTaint

    @functools.wraps(original_fn)
    def wrapper(equation, *operands):
        was_already_in_hook = getattr(TaintedTensor._inside_hook, 'active', False)
        TaintedTensor._inside_hook.active = True

        try:
            # Check if any operand is TaintedTensor
            tainted_operands = [op for op in operands if isinstance(op, TaintedTensor)]
            if not tainted_operands:
                return original_fn(equation, *operands)

            # Call original operation
            result = original_fn(equation, *operands)

            if was_already_in_hook:
                return result

            # Parse einsum equation
            # Format: "input1,input2,...->output" or "input1,input2,..." (implicit output)
            parts = equation.split('->')
            if len(parts) == 2:
                inputs_str, output_str = parts[0], parts[1]
            else:
                # Implicit output - not supported for now
                return result

            input_strs = inputs_str.split(',')
            output_str = output_str.strip()

            # Build a mapping: output dimension letter -> (operand_idx, dim_idx)
            dim_to_source = {}
            for op_idx, (input_str, operand) in enumerate(zip(input_strs, operands)):
                if not isinstance(operand, TaintedTensor):
                    continue
                input_str = input_str.strip()
                for dim_idx, dim_letter in enumerate(input_str):
                    if dim_letter not in dim_to_source:
                        dim_to_source[dim_letter] = []
                    if dim_idx < len(operand._dim_taints):
                        dim_to_source[dim_letter].append((op_idx, dim_idx, operand._dim_taints[dim_idx]))

            # Build output taints
            out_taints = []
            for output_letter in output_str:
                if output_letter in dim_to_source:
                    # Get all taints for this dimension letter
                    sources = dim_to_source[output_letter]
                    if len(sources) == 1:
                        # Single source - use that taint
                        _, _, taint = sources[0]
                        out_taints.append(taint)
                    else:
                        # Multiple sources - this shouldn't happen in valid einsum
                        # Use first source
                        _, _, taint = sources[0]
                        out_taints.append(taint)
                else:
                    # New dimension or broadcast - untainted
                    out_taints.append(None)

            return TaintedTensor(result, tuple(out_taints))

        finally:
            if not was_already_in_hook:
                TaintedTensor._inside_hook.active = False

    return wrapper


# ---------------------------------------------------------------------------
# torch.compile wrapper — taint-aware
# ---------------------------------------------------------------------------
#
# Problem: TaintedTensor's custom __torch_dispatch__ is incompatible with
# torch.dynamo tracing, so @torch.compile-decorated functions (e.g.
# Cohere's layer_norm_func, GemmaRMSNorm._forward_static_*) crash when called
# with TaintedTensor inputs. The previous mitigation was to disable dynamo
# entirely (`torch._dynamo.config.disable = True`), which forced compiled
# functions to run in eager mode — but each constituent ATen op still went
# through TaintedTensor.__torch_dispatch__, decomposing the function into
# many primitive ops that the tracer captured separately.
#
# Fix: wrap torch.compile so that when its produced callable is invoked with
# TaintedTensor inputs, we (1) collect input shapes + taints, (2) unwrap to
# plain tensors and call the original function with custom dispatch disabled
# (treating the function body as a single black box — no per-op decomposition),
# (3) apply propagate_taints (shape-matching rules) to compute output taints,
# (4) re-wrap outputs as TaintedTensors. When the wrapped callable is invoked
# with non-tainted inputs, we delegate to the real torch.compile path so
# dynamo can do its normal optimization work.

def _make_taint_aware_compiled(original_fn, original_compiled_fn):
    """Return a callable that handles TaintedTensor inputs without dynamo.

    For TaintedTensor inputs: bypass compilation, run original_fn as a black
    box on plain tensors, apply shape-based taint propagation to outputs.
    For non-tainted inputs: delegate to torch.compile's compiled wrapper.
    """
    @functools.wraps(original_fn)
    def wrapped(*call_args, **call_kwargs):
        from .propagation import propagate_taints

        input_infos = []
        shape_strs = []
        dtype_strs = []
        has_tainted = False

        def collect(obj):
            nonlocal has_tainted
            if isinstance(obj, TaintedTensor):
                has_tainted = True
                shape = tuple(torch.Tensor.size(obj))
                input_infos.append((shape, obj._dim_taints))
                shape_strs.append(obj.taint_str_with_history())
                dtype_strs.append(str(obj.dtype))
            elif isinstance(obj, torch.Tensor):
                shape = tuple(torch.Tensor.size(obj))
                input_infos.append((shape, tuple(None for _ in range(len(shape)))))
                shape_strs.append("[" + ", ".join(f"?({d})" for d in shape) + "]")
                dtype_strs.append(str(obj.dtype))
            elif isinstance(obj, (list, tuple)):
                for x in obj:
                    collect(x)
            elif isinstance(obj, dict):
                for x in obj.values():
                    collect(x)

        collect(call_args)
        collect(call_kwargs)

        if not has_tainted:
            # Pure plain-tensor path — let torch.compile do its thing.
            return original_compiled_fn(*call_args, **call_kwargs)

        # TaintedTensor inputs present. Unwrap, run original (uncompiled) fn
        # with our custom dispatch disabled so its body executes as a single
        # black box (no per-op TaintedTensor decomposition).
        def unwrap(x):
            if isinstance(x, TaintedTensor):
                with torch._C._DisableTorchDispatch():
                    return x.detach()
            elif isinstance(x, (list, tuple)):
                return type(x)(unwrap(i) for i in x)
            elif isinstance(x, dict):
                return {k: unwrap(v) for k, v in x.items()}
            return x

        plain_args = unwrap(call_args)
        plain_kwargs = unwrap(call_kwargs)

        # Emit a COMPILE: marker. The trace parser stacks 'compile' events into
        # the kernel parent chain and treats them as self-tainted anchors, so
        # kernels inside this scope land on the function as their profiling
        # target — matching what the resolver needs to call back into the real
        # torch.compile-decorated module (e.g. Cohere's layer_norm_func via
        # its containing LayerNorm).
        fn_name = getattr(original_fn, "__name__", "compiled_fn")
        in_str = " | ".join(shape_strs) if shape_strs else "no_inputs"
        annotated_name = f"COMPILE: {fn_name} IN:[{in_str}]"
        if dtype_strs:
            annotated_name += f" DTYPES:[{', '.join(dtype_strs)}]"

        with record_function(annotated_name):
            with torch._C._DisableTorchDispatch():
                out = original_fn(*plain_args, **plain_kwargs)

        # Apply shape-based taint propagation to each output tensor.
        def wrap_output(x):
            if isinstance(x, torch.Tensor) and not isinstance(x, TaintedTensor):
                out_shape = tuple(x.shape)
                try:
                    out_taints = propagate_taints(input_infos, out_shape)
                except Exception:
                    # Conservative fallback: leave outputs un-tainted if
                    # shape propagation hits an unexpected case (e.g., a
                    # dimension-collision raise inside propagate_taints).
                    out_taints = tuple(None for _ in range(len(out_shape)))
                return TaintedTensor(x, out_taints)
            elif isinstance(x, (list, tuple)):
                return type(x)(wrap_output(i) for i in x)
            elif isinstance(x, dict):
                return {k: wrap_output(v) for k, v in x.items()}
            return x

        return wrap_output(out)

    # Stash references for debugging / introspection.
    wrapped._taint_aware_original_fn = original_fn
    wrapped._taint_aware_compiled_fn = original_compiled_fn
    return wrapped


def _patched_torch_compile(model=None, *args, **kwargs):
    """Drop-in replacement for torch.compile that produces taint-aware
    callables. Supports decorator (with or without args) and direct usage."""
    if model is None:
        # Decorator-with-args form: @torch.compile(mode=..., backend=...)
        def decorator(fn):
            compiled = _original_torch_compile(fn, *args, **kwargs)
            return _make_taint_aware_compiled(fn, compiled)
        return decorator

    # Direct call: torch.compile(fn, ...)  OR  @torch.compile (no args)
    compiled = _original_torch_compile(model, *args, **kwargs)
    return _make_taint_aware_compiled(model, compiled)


# ---------------------------------------------------------------------------
# Install / Uninstall
# ---------------------------------------------------------------------------

def install_hooks(int_patch_modules=()):
    """Install hooks on tensor creation functions and enable profiler integration."""
    import os

    # Replace torch.compile with a taint-aware wrapper. TaintedTensor's
    # __torch_dispatch__ is incompatible with dynamo's tracing, so when a
    # compiled callable is invoked with TaintedTensor inputs we bypass
    # compilation, run the original function as a single black box, and
    # apply shape-based taint propagation on outputs (see propagation.py).
    # Plain-tensor calls still go through the normal torch.compile path.
    _disable_compile_wrapper = os.environ.get("DISABLE_TAINT_COMPILE_WRAPPER", "0") == "1"
    if _disable_compile_wrapper:
        # Legacy behaviour: globally disable dynamo. Compiled functions run
        # in eager mode and decompose into per-op TaintedTensor dispatch.
        importlib.import_module("torch._dynamo").config.disable = True
    else:
        torch.compile = _patched_torch_compile

    _disable_factory_wrappers = os.environ.get("DISABLE_FACTORY_WRAPPERS", "0") == "1"

    if _disable_factory_wrappers:
        print("[DEBUG] Factory wrappers DISABLED - no TaintedTensors will be created")
    else:
        torch.zeros = _wrap_factory_output(_original_zeros)
        torch.ones = _wrap_factory_output(_original_ones)
        torch.empty = _wrap_factory_output(_original_empty)
        torch.randn = _wrap_factory_output(_original_randn)
        torch.rand = _wrap_factory_output(_original_rand)
        torch.arange = _wrap_arange(_original_arange)
        torch.Tensor.copy_ = _wrap_copy(_original_copy)

        # Wrap view/reshape to detect TaintedInt in shape args
        torch.Tensor.view = _wrap_view_reshape(_original_tensor_view, 'view')
        torch.reshape = _wrap_torch_reshape(_original_torch_reshape)  # Use module-level wrapper
        torch.Tensor.reshape = _wrap_view_reshape(_original_tensor_reshape, 'reshape')

        # Wrap permute and transpose to use explicit dimension indices
        torch.Tensor.permute = _wrap_permute(_original_tensor_permute, 'permute')
        torch.transpose = _wrap_transpose(_original_torch_transpose, 'transpose')
        torch.Tensor.transpose = _wrap_transpose(_original_tensor_transpose, 'transpose')

        # Wrap cat and stack to use explicit dimension and tensor list arguments
        torch.cat = _wrap_cat(_original_torch_cat)
        torch.stack = _wrap_stack(_original_torch_stack)

        # Wrap split and chunk to handle TaintedInt arguments
        torch.split = _wrap_split(_original_torch_split)
        torch.Tensor.split = _wrap_split(_original_tensor_split)
        torch.chunk = _wrap_chunk(_original_torch_chunk)
        torch.Tensor.chunk = _wrap_chunk(_original_tensor_chunk)

        # Wrap unsqueeze and squeeze to handle dimension insertion/removal
        torch.unsqueeze = _wrap_unsqueeze(_original_torch_unsqueeze)
        torch.squeeze = _wrap_squeeze(_original_torch_squeeze)
        torch.Tensor.unsqueeze = _wrap_unsqueeze(_original_tensor_unsqueeze)
        torch.Tensor.squeeze = _wrap_squeeze(_original_tensor_squeeze)

        # Wrap unbind to handle dimension removal
        torch.unbind = _wrap_unbind(_original_torch_unbind)
        torch.Tensor.unbind = _wrap_unbind(_original_tensor_unbind)

        # Wrap flatten to handle dimension merging
        torch.flatten = _wrap_flatten(_original_torch_flatten)
        torch.Tensor.flatten = _wrap_flatten(_original_tensor_flatten)

        # Wrap expand to handle broadcasting (only Tensor.expand exists, not torch.expand)
        torch.Tensor.expand = _wrap_expand(_original_tensor_expand)

        # Wrap matmul and linear for semantic taint propagation (Tier 2)
        torch.matmul = _wrap_matmul(_original_torch_matmul)
        torch.Tensor.matmul = _wrap_matmul(_original_tensor_matmul)
        torch.nn.functional.linear = _wrap_linear(_original_torch_linear)
        torch.nn.functional.pad = _wrap_pad(_original_torch_pad)

        # Wrap einsum for taint propagation
        torch.einsum = _wrap_einsum(_original_torch_einsum)

        # Wrap einops operations (lazy import to avoid errors if not installed)
        try:
            import einops
            global _original_einops_rearrange, _original_einops_reduce, _original_einops_repeat
            if _original_einops_rearrange is None:
                _original_einops_rearrange = einops.rearrange
                _original_einops_reduce = einops.reduce
                _original_einops_repeat = einops.repeat
            einops.rearrange = _wrap_einops_rearrange(_original_einops_rearrange)
            einops.reduce = _wrap_einops_reduce(_original_einops_reduce)
            einops.repeat = _wrap_einops_repeat(_original_einops_repeat)
        except ImportError:
            pass  # einops not installed, skip

        # 1. Capture original (ensure this is done only once)
        if not hasattr(torch, "_original_size"):
            torch._original_size = torch.Size

        torch.Size = PatchedSize

        if not hasattr(torch, "_original_tensor"):
            torch._original_tensor = torch.tensor
            torch._original_full = torch.full
            torch._original_as_tensor = torch.as_tensor
            torch._original_from_numpy = torch.from_numpy

        torch.tensor = _wrap_factory_output(torch._original_tensor)
        torch.full = _wrap_factory_output(torch._original_full)
        torch.as_tensor = _wrap_factory_output(torch._original_as_tensor)
        torch.from_numpy = _wrap_factory_output(torch._original_from_numpy)
        
                
        # to speed up init process
        torch.Tensor.uniform_ = lambda self, *args, **kwargs: self

    # Swap dispatch to profiler-annotated version
    # DEBUG: Set DISABLE_PROFILER_DISPATCH=1 to use original dispatch
    import os
    _disable_profiler_dispatch = os.environ.get("DISABLE_PROFILER_DISPATCH", "0") == "1"
    if _disable_profiler_dispatch:
        print("[DEBUG] Using original __torch_dispatch__ (profiler annotations disabled)")
    else:
        TaintedTensor.__torch_dispatch__ = TaintedTensor.dispatch_with_profiler

    # Enable __torch_function__ hook to capture parent hierarchy
    # DEBUG: Set DISABLE_TAINTED_TORCH_FUNCTION=1 to disable
    _disable_torch_function = os.environ.get("DISABLE_TAINTED_TORCH_FUNCTION", "0") == "1"
    if _disable_torch_function:
        print("[DEBUG] TaintedTensor.__torch_function__ DISABLED")
    else:
        TaintedTensor.__torch_function__ = classmethod(_torch_function_implementation)

    # Patch nn.Module.__call__ to log taint info at module level
    _disable_module_wrapper = os.environ.get("DISABLE_MODULE_WRAPPER", "0") == "1"
    if _disable_module_wrapper:
        print("[DEBUG] Module call wrapper DISABLED")
    else:
        patch_module_call()

    if int_patch_modules:
        patch_int_in_modules(int_patch_modules)

    # Strip TaintedInt/TaintedFloat at Triton kernel launches (needed for
    # gpt-oss and other models that stash tainted scalars on backend impls).
    _patch_triton_jit()


def uninstall_hooks():
    """Restore original tensor creation functions and dispatch."""
    torch.compile = _original_torch_compile
    torch.zeros = _original_zeros
    torch.ones = _original_ones
    torch.empty = _original_empty
    torch.randn = _original_randn
    torch.rand = _original_rand
    torch.arange = _original_arange
    torch.Tensor.copy_ = _original_copy
    torch.Tensor.view = _original_tensor_view
    torch.reshape = _original_torch_reshape
    torch.Tensor.reshape = _original_tensor_reshape
    torch.Tensor.permute = _original_tensor_permute
    torch.transpose = _original_torch_transpose
    torch.Tensor.transpose = _original_tensor_transpose
    torch.cat = _original_torch_cat
    torch.stack = _original_torch_stack
    torch.split = _original_torch_split
    torch.Tensor.split = _original_tensor_split
    torch.chunk = _original_torch_chunk
    torch.Tensor.chunk = _original_tensor_chunk
    torch.unsqueeze = _original_torch_unsqueeze
    torch.squeeze = _original_torch_squeeze
    torch.Tensor.unsqueeze = _original_tensor_unsqueeze
    torch.Tensor.squeeze = _original_tensor_squeeze
    torch.unbind = _original_torch_unbind
    torch.Tensor.unbind = _original_tensor_unbind
    torch.flatten = _original_torch_flatten
    torch.Tensor.flatten = _original_tensor_flatten
    torch.Tensor.expand = _original_tensor_expand
    torch.matmul = _original_torch_matmul
    torch.Tensor.matmul = _original_tensor_matmul
    torch.nn.functional.linear = _original_torch_linear
    torch.nn.functional.pad = _original_torch_pad
    torch.einsum = _original_torch_einsum

    # Restore einops operations
    if _original_einops_rearrange is not None:
        try:
            import einops
            einops.rearrange = _original_einops_rearrange
            einops.reduce = _original_einops_reduce
            einops.repeat = _original_einops_repeat
        except ImportError:
            pass

    if hasattr(torch, "_original_tensor"):
        torch.tensor = torch._original_tensor
        torch.full = torch._original_full
        torch.as_tensor = torch._original_as_tensor
        torch.from_numpy = torch._original_from_numpy

    TaintedTensor.__torch_dispatch__ = TaintedTensor._original_dispatch
    
    # Remove __torch_function__ hook (restore default behavior which might be None or strict)
    # TaintedTensor doesn't define it by default, so we can delete it or set to None?
    # Actually, Tensor subclasses don't start with __torch_function__ usually,
    # but we should check if TaintedTensor had one. It likely didn't.
    if hasattr(TaintedTensor, '__torch_function__'):
        del TaintedTensor.__torch_function__

    _unpatch_triton_jit()

    unpatch_int_in_modules()
    unpatch_module_call()