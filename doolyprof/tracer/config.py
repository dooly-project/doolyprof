"""Config tainting: wraps HuggingFace config integer attributes as TaintedInt."""

import functools

from .types import TaintedInt, TaintedFloat, Taint
from .registry import register_taint

__all__ = ['create_tainted_config', 'patch_autoconfig', 'unpatch_autoconfig', 'patch_input_buffers', 'unpatch_input_buffers' ]


# Cache for dynamically created tainted config subclasses
_tainted_config_cache: dict[type, type] = {}


def _reconstruct_config(cls, state):
    """Pickle helper: reconstruct and re-wrap the config."""
    obj = object.__new__(cls)
    obj.__dict__.update(state)
    return create_tainted_config(obj)

# create tainted config object that wraps integer attributes as TaintedInt
def create_tainted_config(config):
    # get the original class's config object (e.g. LlamaConfig)
    config_cls = type(config)

    # if the class is already in the cache, return it
    if config_cls not in _tainted_config_cache:
        _tainted_config_cache[config_cls] = _create_tainted_config_class(config_cls)

    # create tainted config object
    TaintedConfigClass = _tainted_config_cache[config_cls]
    return TaintedConfigClass(config)

SKIP_ATTRS = {
    'bos_token_id', 'eos_token_id', 'pad_token_id', 'sep_token_id',
    'decoder_start_token_id', 'forced_eos_token_id',
    'return_dict', 'use_cache', 'tie_word_embeddings',
    'num_beams', 'num_return_sequences', 'num_beam_groups',
    'max_length', 'min_length'
}

def _taint_value(value, skip_attrs=None):
    """Recursively taint integers in a value (int, dict, list, or nested config)."""
    if skip_attrs is None:
        skip_attrs = set()

    # Already tainted
    if isinstance(value, TaintedInt):
        return value

    # Plain integer -> taint it (exclude bools which are int subclass)
    if isinstance(value, int) and not isinstance(value, bool):
        tainted = TaintedInt(value, 'MODEL_CONFIG')
        # Register in global registry
        register_taint(value, Taint('MODEL_CONFIG'))
        return tainted

    # Plain float -> taint it
    if isinstance(value, float):
        tainted_float = TaintedFloat(value, 'MODEL_CONFIG')
        # Note: Registry only stores int values
        return tainted_float

    # Nested config object (PretrainedConfig subclass) -> wrap recursively
    try:
        from transformers import PretrainedConfig
        if isinstance(value, PretrainedConfig):
            return create_tainted_config(value)
    except ImportError:
        pass

    # Dictionary -> wrap with _TaintedDict
    if isinstance(value, dict):
        return _TaintedDict(value, skip_attrs)

    # List/tuple -> recursively taint elements
    if isinstance(value, (list, tuple)):
        tainted_items = [_taint_value(item, skip_attrs) for item in value]
        return type(value)(tainted_items)

    return value

class _TaintedDict(dict):
    """Dict wrapper that taints integer values on access."""

    def __init__(self, original_dict, skip_attrs=None):
        super().__init__(original_dict)
        self._skip_attrs = skip_attrs or set()
        self._taint_cache = {}

    def __getitem__(self, key):
        if key in self._taint_cache:
            return self._taint_cache[key]

        value = super().__getitem__(key)

        # Skip certain keys
        if key in self._skip_attrs:
            return value

        tainted = _taint_value(value, self._skip_attrs)
        if tainted is not value:
            self._taint_cache[key] = tainted
            return tainted
        return value

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]


# helper function that creates a tainted config class that inherits from the original config class
def _create_tainted_config_class(config_cls):

    # create a tainted config class that inherits from the original config class
    class TaintedConfigSubclass(config_cls):
        """Dynamic subclass that taints integer attributes with MODEL_CONFIG."""

        # copy all data from original config into new tainted object
        # don't call super().__init__() because it calls from_pretrained() which causes infinite recursion
        def __init__(self, original_config=None, **kwargs):
            if original_config is None:
                super().__init__(**kwargs)
                self._taint_cache = {}
            else:
                self.__dict__.update(original_config.__dict__)
                self._taint_cache = {}

        # override __getattribute__ to wrap integer attributes as TaintedInt
        def __getattribute__(self, name):
            # return the original attribute if private
            if name.startswith('_'):
                return super().__getattribute__(name)

            # retrieve actual value from parent class
            value = super().__getattribute__(name)
            # if this is a function or method, we return as is
            if callable(value):
                return value

            # check if we have already tainted this attribute
            try:
                cache = super().__getattribute__('_taint_cache')
                if name in cache:
                    return cache[name]
            except AttributeError:
                return value

            # skip certain attributes
            if name in SKIP_ATTRS:
                return value

            # recursively taint the value
            tainted = _taint_value(value, SKIP_ATTRS)
            if tainted is not value:
                cache[name] = tainted
                return tainted

            return value

        def __reduce__(self):
            """Pickle as the original config class so spawn-based subprocesses work."""
            state = {k: v for k, v in self.__dict__.items() if k != '_taint_cache'}
            return (_reconstruct_config, (config_cls, state))

        # def __deepcopy__(self, memo):
        #     """Preserve taint through copy.deepcopy() operations.

        #     This is critical for vLLM which calls deepcopy on configs in with_hf_config().
        #     Without this, deepcopy creates a plain config object and taints are lost.
        #     """
        #     import copy
        #     # Create a new instance of the original config class
        #     new_config = object.__new__(config_cls)
        #     # Deepcopy the __dict__ (excluding our taint cache)
        #     new_dict = {}
        #     for k, v in self.__dict__.items():
        #         if k == '_taint_cache':
        #             continue
        #         new_dict[k] = copy.deepcopy(v, memo)
        #     new_config.__dict__.update(new_dict)
        #     # Re-wrap as a tainted config
        #     return create_tainted_config(new_config)

        def to_dict(self):
            """Override to_dict to return tainted values instead of raw dict values."""
            import copy
            # Get the raw dict
            output = copy.deepcopy(self.__dict__)

            # Remove internal attributes (HuggingFace and our cache)
            for key in list(output.keys()):
                if key.startswith('_'):
                    del output[key]

            # Recursively taint all values
            def taint_dict_values(d):
                for key, value in list(d.items()):
                    if key in SKIP_ATTRS:
                        continue
                    if isinstance(value, dict):
                        taint_dict_values(value)
                    elif hasattr(value, 'to_dict'):
                        # Nested config - call its to_dict to get dict
                        # This fixes "TypeError: Object of type XConfig is not JSON serializable"
                        nested_dict = value.to_dict()
                        # Taint values inside the resulting dictionary if it wasn't already tainted
                        if not hasattr(value, '_taint_cache'):
                            for nk, nv in list(nested_dict.items()):
                                if nk not in SKIP_ATTRS:
                                    nested_dict[nk] = _taint_value(nv, SKIP_ATTRS)
                        d[key] = nested_dict
                    else:
                        d[key] = _taint_value(value, SKIP_ATTRS)

            taint_dict_values(output)

            # Add model_type if present
            if hasattr(self.__class__, "model_type"):
                output["model_type"] = self.__class__.model_type

            return output

    # Keep the same class name for type() lookups
    TaintedConfigSubclass.__name__ = config_cls.__name__
    TaintedConfigSubclass.__qualname__ = config_cls.__qualname__
    TaintedConfigSubclass.__module__ = config_cls.__module__

    return TaintedConfigSubclass


# ---------------------------------------------------------------------------
# AutoConfig patching
# ---------------------------------------------------------------------------

_original_autoconfig_from_pretrained = None
_original_pretrainedconfig_from_pretrained = None
_original_pretrainedconfig_from_dict = None
_original_vllm_get_config = None

# patch transformers.AutoConfig.from_pretrained to return TaintedConfig
# This allows taints to propagate into vLLM model code automatically,
# since vLLM loads config via AutoConfig.from_pretrained internally.
def patch_autoconfig(verbose=False):
    global _original_autoconfig_from_pretrained
    global _original_pretrainedconfig_from_pretrained
    global _original_pretrainedconfig_from_dict
    global _original_vllm_get_config

    try:
        from transformers import AutoConfig, PretrainedConfig
    except ImportError:
        return

    if _original_autoconfig_from_pretrained is not None:
        if verbose:
            print("[tracer] patch_autoconfig: Already patched, skipping")
        return  # Already patched

    if verbose:
        print("[tracer] patch_autoconfig: Applying patches...")

    _original_autoconfig_from_pretrained = AutoConfig.from_pretrained
    _original_pretrainedconfig_from_pretrained = PretrainedConfig.from_pretrained
    _original_pretrainedconfig_from_dict = PretrainedConfig.from_dict

    @functools.wraps(_original_autoconfig_from_pretrained)
    def patched_autoconfig_from_pretrained(*args, **kwargs):
        raw_config = _original_autoconfig_from_pretrained(*args, **kwargs)
        if verbose:
            print(f"[tracer] AutoConfig.from_pretrained called: {type(raw_config).__name__}")
        return create_tainted_config(raw_config)

    @classmethod
    def patched_pretrainedconfig_from_pretrained(cls, *args, **kwargs):
        func = _original_pretrainedconfig_from_pretrained.__func__
        raw_config = func(cls, *args, **kwargs)
        if verbose:
            print(f"[tracer] PretrainedConfig.from_pretrained called: {type(raw_config).__name__}")
        return create_tainted_config(raw_config)

    # Patch from_dict to catch Mistral configs and other direct dict-based loading
    @classmethod
    def patched_pretrainedconfig_from_dict(cls, config_dict, **kwargs):
        func = _original_pretrainedconfig_from_dict.__func__
        raw_config = func(cls, config_dict, **kwargs)
        if verbose:
            print(f"[tracer] PretrainedConfig.from_dict called: {type(raw_config).__name__}")
        return create_tainted_config(raw_config)

    AutoConfig.from_pretrained = patched_autoconfig_from_pretrained
    PretrainedConfig.from_pretrained = patched_pretrainedconfig_from_pretrained
    PretrainedConfig.from_dict = patched_pretrainedconfig_from_dict
    if verbose:
        print("[tracer] Patched AutoConfig.from_pretrained, PretrainedConfig.from_pretrained, PretrainedConfig.from_dict")

    # Also patch vLLM's get_config as a safety net for any edge cases
    try:
        import vllm.transformers_utils.config as vllm_config_mod
        _original_vllm_get_config = vllm_config_mod.get_config

        @functools.wraps(_original_vllm_get_config)
        def patched_vllm_get_config(*args, **kwargs):
            raw_config = _original_vllm_get_config(*args, **kwargs)
            # Only wrap if not already tainted (avoid double-wrapping)
            if hasattr(raw_config, '_taint_cache'):
                if verbose:
                    print(f"[tracer] vllm.get_config: Already tainted, skipping")
                return raw_config
            if verbose:
                print(f"[tracer] vllm.get_config called: {type(raw_config).__name__}")
            return create_tainted_config(raw_config)

        vllm_config_mod.get_config = patched_vllm_get_config
        if verbose:
            print("[tracer] Patched vllm.transformers_utils.config.get_config")
    except ImportError:
        if verbose:
            print("[tracer] vLLM not installed, skipping vllm.get_config patch")



def unpatch_autoconfig():
    """Restore original AutoConfig.from_pretrained and related patches."""
    global _original_autoconfig_from_pretrained
    global _original_pretrainedconfig_from_pretrained
    global _original_pretrainedconfig_from_dict
    global _original_vllm_get_config

    if _original_autoconfig_from_pretrained is not None:
        try:
            from transformers import AutoConfig, PretrainedConfig
            AutoConfig.from_pretrained = _original_autoconfig_from_pretrained
            _original_autoconfig_from_pretrained = None

            if _original_pretrainedconfig_from_pretrained is not None:
                PretrainedConfig.from_pretrained = _original_pretrainedconfig_from_pretrained
                _original_pretrainedconfig_from_pretrained = None

            if _original_pretrainedconfig_from_dict is not None:
                PretrainedConfig.from_dict = _original_pretrainedconfig_from_dict
                _original_pretrainedconfig_from_dict = None
        except ImportError:
            pass

    if _original_vllm_get_config is not None:
        try:
            import vllm.transformers_utils.config as vllm_config_mod
            vllm_config_mod.get_config = _original_vllm_get_config
            _original_vllm_get_config = None
        except ImportError:
            pass

_original_scheduler_config_getattribute = None
_original_input_batch_num_reqs = None

# Attributes to taint in SchedulerConfig
SCHEDULER_CONFIG_TAINT_MAP = {
    # Note: Commenting out max_num_batched_tokens to prevent buffer tainting
    # Only runtime values (total_num_scheduled_tokens) should be tainted as NUM_TOKS
    'max_num_batched_tokens': 'MAX_NUM_TOKS',
    'max_num_seqs': 'MAX_NUM_REQS',

}

def _recursively_taint_config(obj, skip_attrs=None, taint_str='MODEL_CONFIG'):
    """Recursively taint all integer attributes in a config object.

    This function is idempotent - safe to call multiple times on the same object.
    It will skip values that are already TaintedInt/TaintedFloat to preserve taints.

    NOTE: We don't use a _taint_cache flag because adding it to __dict__ breaks
    vLLM's replace() function (vllm/config/utils.py:107-115), which iterates over
    __dict__ and validates all keys as dataclass fields. The isinstance() checks
    below make this function naturally idempotent without needing explicit caching.
    """
    if skip_attrs is None:
        skip_attrs = SKIP_ATTRS

    print(f"[TAINT] Processing {type(obj).__name__} with taint_str={taint_str}", flush=True)

    # Get all attributes
    if hasattr(obj, '__dict__'):
        for attr_name in list(obj.__dict__.keys()):
            # Skip private attributes and those in skip list
            if attr_name.startswith('_') or attr_name in skip_attrs:
                continue

            try:
                value = getattr(obj, attr_name)

                # Taint integers (but not bools which are int subclass)
                # Skip if already tainted to preserve more specific taints
                if isinstance(value, int) and not isinstance(value, (bool, TaintedInt)):
                    print(f"[TAINT]   {attr_name}={value} -> TaintedInt({value}, {taint_str})", flush=True)
                    tainted_int = TaintedInt(value, taint_str)
                    object.__setattr__(obj, attr_name, tainted_int)
                    # Register in global registry
                    register_taint(value, Taint(taint_str))
                # Taint floats
                # Skip if already tainted to preserve more specific taints
                elif isinstance(value, float) and not isinstance(value, TaintedFloat):
                    print(f"[TAINT]   {attr_name}={value} -> TaintedFloat({value}, {taint_str})", flush=True)
                    object.__setattr__(obj, attr_name, TaintedFloat(value, taint_str))
                    # Note: Registry only stores int values
                elif isinstance(value, (TaintedInt, TaintedFloat)):
                    # Already tainted, skip (this makes the function idempotent)
                    pass
                # Recursively process nested config objects
                elif hasattr(value, '__dict__') and not isinstance(value, type):
                    _recursively_taint_config(value, skip_attrs, taint_str)
            except (AttributeError, TypeError):
                # Some attributes might not be accessible
                continue


def patch_input_buffers():
    """Patch SchedulerConfig and InputBatch for workload tainting.

    1. SchedulerConfig: max_num_batched_tokens -> NUM_TOKS, max_num_seqs -> NUM_REQS
       These taint the buffer sizes used in tensor creation.

    2. InputBatch.num_reqs: Returns TaintedInt for the ACTUAL runtime batch size.
       This is needed because num_reqs = len(req_id_to_index) is a plain int,
       and we need the runtime batch dimension to be tainted for LogitsProcessor etc.

    """
    import os
    global _original_scheduler_config_getattribute
    global _original_input_batch_num_reqs

    # print(f"[DEBUG @ patch_input_buffers] Called! PID={os.getpid()}", flush=True)

    # Patch SchedulerConfig
    try:
        from vllm.config.scheduler import SchedulerConfig
        # print("[DEBUG @ patch_input_buffers] Patching SchedulerConfig", flush=True)

        if _original_scheduler_config_getattribute is None:
            _original_scheduler_config_getattribute = SchedulerConfig.__getattribute__

            def patched_getattribute(self, name):
                value = _original_scheduler_config_getattribute(self, name)
                if name in SCHEDULER_CONFIG_TAINT_MAP:
                    if isinstance(value, int) and not isinstance(value, TaintedInt):
                        register_taint(value, Taint(SCHEDULER_CONFIG_TAINT_MAP[name]))
                        return TaintedInt(value, SCHEDULER_CONFIG_TAINT_MAP[name])
                return value

            SchedulerConfig.__getattribute__ = patched_getattribute
            # print("[DEBUG @ patch_input_buffers] SchedulerConfig patch applied", flush=True)
    except ImportError as e:
        print(f"[DEBUG @ patch_input_buffers] SchedulerConfig ImportError - {e}", flush=True)

    # Patch SchedulerOutput to taint num_scheduled_tokens and total_num_scheduled_tokens
    try:
        from vllm.v1.core.sched.output import SchedulerOutput
        # print("[DEBUG @ patch_input_buffers] Patching SchedulerOutput", flush=True)

        # Custom dict that returns TaintedInt for len()
        class TaintedLenDict(dict):
            def __len__(self):
                return TaintedInt(super().__len__(), 'NUM_REQS')

        _original_scheduler_output_getattribute = SchedulerOutput.__getattribute__

        def patched_scheduler_output_getattribute(self, name):
            value = _original_scheduler_output_getattribute(self, name)
            if name == 'num_scheduled_tokens':
                if isinstance(value, dict) and not isinstance(value, TaintedLenDict):
                    register_taint(len(value), Taint('NUM_REQS'))
                    return TaintedLenDict(value)
            elif name == 'total_num_scheduled_tokens':
                if isinstance(value, int) and not isinstance(value, TaintedInt):
                    register_taint(value, Taint('NUM_TOKS'))
                    return TaintedInt(value, 'NUM_TOKS')
            return value

        SchedulerOutput.__getattribute__ = patched_scheduler_output_getattribute
        # print("[DEBUG @ patch_input_buffers] SchedulerOutput patch applied", flush=True)
    except ImportError as e:
        print(f"[DEBUG @ patch_input_buffers] SchedulerOutput ImportError - {e}", flush=True)

    # Patch model_runner's len() to prevent it from stripping int subclasses
    try:
        import vllm.v1.worker.gpu.model_runner as model_runner_mod
        # print("[DEBUG @ patch_input_buffers] Patching model_runner.len (v2)", flush=True)
        import builtins
        _original_len = builtins.len
        
        def tainted_len(obj):
            if hasattr(obj, '__class__') and obj.__class__.__name__ == 'TaintedLenDict':
                return obj.__len__()
            return _original_len(obj)
            
        model_runner_mod.len = tainted_len
        # print("[DEBUG @ patch_input_buffers] model_runner.len (v2) patch applied", flush=True)
    except ImportError as e:
        print(f"[DEBUG @ patch_input_buffers] v2 model_runner ImportError - {e}", flush=True)

    # Patch ModelArchitectureConfig to prevent Pydantic from stripping taints during coercion
    try:
        from vllm.config.model_arch import ModelArchitectureConfig
        if not hasattr(ModelArchitectureConfig, '_is_taint_patched'):
            _original_model_arch_config_init = ModelArchitectureConfig.__init__

            def patched_model_arch_config_init(self, *args, **kwargs):
                # Save a copy of the kwargs to preserve taints before Pydantic coerces them
                _kwargs_copy = kwargs.copy()

                # Run original pydantic init which coerces types
                _original_model_arch_config_init(self, *args, **kwargs)

                # Restore tainted ints
                for k, v in _kwargs_copy.items():
                    if isinstance(v, TaintedInt):
                        object.__setattr__(self, k, v)

            ModelArchitectureConfig.__init__ = patched_model_arch_config_init
            ModelArchitectureConfig._is_taint_patched = True
    except ImportError as e:
        print(f"[DEBUG @ patch_input_buffers] ModelArchitectureConfig ImportError - {e}", flush=True)

    # Patch ModelConfig to recursively taint all integer attributes after __post_init__
    try:
        from vllm.config.model import ModelConfig
        if not hasattr(ModelConfig, '_is_recursive_taint_patched'):
            _original_model_config_post_init = ModelConfig.__post_init__

            def patched_model_config_post_init(self, *args, **kwargs):
                # Run original __post_init__ with all arguments
                _original_model_config_post_init(self, *args, **kwargs)
                # Recursively taint all integer attributes
                print(f"[TAINT] ModelConfig.__post_init__ called, tainting as MODEL_CONFIG", flush=True)
                _recursively_taint_config(self, taint_str='MODEL_CONFIG')
        
            def patched_model_config_setattr(self, name, value):
                # Intercept changes to num_attention_heads and other integer attributes that might be set after __post_init__
                if isinstance(value, int) and not isinstance(value, (bool, TaintedInt)):
                    register_taint(value, Taint('MODEL_CONFIG'))
                    value = TaintedInt(value, 'MODEL_CONFIG')
                elif isinstance(value, float) and not isinstance(value, TaintedFloat):
                    register_taint(value, Taint('MODEL_CONFIG'))
                    value = TaintedFloat(value, 'MODEL_CONFIG')
                object.__setattr__(self, name, value)

            ModelConfig.__post_init__ = patched_model_config_post_init
            ModelConfig.__setattr__ = patched_model_config_setattr
            
            ModelConfig._is_recursive_taint_patched = True
            
    except ImportError as e:
        print(f"[DEBUG @ patch_input_buffers] ModelConfig ImportError - {e}", flush=True)

    # Patch CacheConfig via __init__ and __setattr__ since block_size is set after init
    # try:
    #     from vllm.config.cache import CacheConfig
    #     if not hasattr(CacheConfig, '_is_recursive_taint_patched'):
    #         _original_cache_config_init = CacheConfig.__init__
    #         _original_cache_config_setattr = CacheConfig.__setattr__

    #         def patched_cache_config_init(self, *args, **kwargs):
    #             _original_cache_config_init(self, *args, **kwargs)
    #             # Taint all integer/float attributes after initialization
    #             print(f"[TAINT] CacheConfig.__init__ called, tainting as CACHE_CONFIG", flush=True)
    #             _recursively_taint_config(self, taint_str='CACHE_CONFIG')

    #         def patched_cache_config_setattr(self, name, value):
    #             # Taint integers/floats when they're set (e.g., block_size set by platform)
    #             if isinstance(value, int) and not isinstance(value, (bool, TaintedInt)):
    #                 value = TaintedInt(value, 'CACHE_CONFIG')
    #             elif isinstance(value, float) and not isinstance(value, TaintedFloat):
    #                 value = TaintedFloat(value, 'CACHE_CONFIG')
    #             _original_cache_config_setattr(self, name, value)

    #         CacheConfig.__init__ = patched_cache_config_init
    #         CacheConfig.__setattr__ = patched_cache_config_setattr
    #         CacheConfig._is_recursive_taint_patched = True
    # except (ImportError, AttributeError) as e:
    #     print(f"[DEBUG @ patch_input_buffers] CacheConfig error - {e}", flush=True)

    # Patch ParallelConfig for tensor/pipeline/data parallel settings
    # try:
    #     from vllm.config.parallel import ParallelConfig
    #     if not hasattr(ParallelConfig, '_is_recursive_taint_patched'):
    #         if hasattr(ParallelConfig, '__post_init__'):
    #             _original_parallel_config_post_init = ParallelConfig.__post_init__

    #             def patched_parallel_config_post_init(self):
    #                 _original_parallel_config_post_init(self)
    #                 _recursively_taint_config(self, taint_str='PARALLEL_CONFIG')

    #             ParallelConfig.__post_init__ = patched_parallel_config_post_init
    #             ParallelConfig._is_recursive_taint_patched = True
    # except (ImportError, AttributeError) as e:
    #     print(f"[DEBUG @ patch_input_buffers] ParallelConfig error - {e}", flush=True)

    # # Patch VllmConfig which contains all sub-configs
    # try:
    #     from vllm.config import VllmConfig
    #     if not hasattr(VllmConfig, '_is_recursive_taint_patched'):
    #         if hasattr(VllmConfig, '__post_init__'):
    #             _original_vllm_config_post_init = VllmConfig.__post_init__

    #             def patched_vllm_config_post_init(self):
    #                 _original_vllm_config_post_init(self)
    #                 # Recursively taint all nested configs (including CacheConfig, etc.)
    #                 _recursively_taint_config(self, taint_str='VLLM_CONFIG')

    #             VllmConfig.__post_init__ = patched_vllm_config_post_init
    #             VllmConfig._is_recursive_taint_patched = True
    # except (ImportError, AttributeError) as e:
    #     print(f"[DEBUG @ patch_input_buffers] VllmConfig error - {e}", flush=True)

    # try:
    #     import vllm.v1.worker.gpu_model_runner as model_runner_mod_v1
    #     print("[DEBUG @ patch_input_buffers] Patching model_runner.len (v1)", flush=True)
    #     import builtins
    #     if '_original_len' not in locals():
    #         _original_len = builtins.len
        
    #     def tainted_len_v1(obj):
    #         if hasattr(obj, '__class__') and obj.__class__.__name__ == 'TaintedLenDict':
    #             return obj.__len__()
    #         return _original_len(obj)
            
    #     model_runner_mod_v1.len = tainted_len_v1
    #     print("[DEBUG @ patch_input_buffers] model_runner.len (v1) patch applied", flush=True)
    # except ImportError as e:
    #     print(f"[DEBUG @ patch_input_buffers] v1 model_runner ImportError - {e}", flush=True)

def unpatch_input_buffers():
    global _original_scheduler_config_getattribute

    if _original_scheduler_config_getattribute is not None:
        from vllm.config import SchedulerConfig
        SchedulerConfig.__getattribute__ = _original_scheduler_config_getattribute
        _original_scheduler_config_getattribute = None
