from typing import Dict, List, Any, Optional, Tuple, Callable
import torch
import json

from doolyprof.profiler.module import ModuleInfo, ModuleComparator
from doolyprof.profiler.trace import TraceParser
from vllm.transformers_utils.config import get_config

class ModelInfo:
    def __init__(
        self,
        name: str,
        trace_dir: str,
        dtype: torch.dtype
    ):
        # load model configuration
        self.config = get_config(name, trust_remote_code=True)
        self.name = name
        self.dtype = dtype
        self.model_family = self.get_model_family()

        # model config values
        self.vocab_size = self.get_vocab_size()
        self.hidden_size = self.get_hidden_size()
        self.num_layers = self.get_num_layers()
        self.num_heads = self.get_num_heads()
        self.max_model_len = self.get_max_model_len()
        self.num_kv_heads, self.head_dim = self.get_kv_params()

        # Only validate for transformer models with attention
        if self.head_dim is not None and self.num_heads is not None:
            assert self.head_dim * self.num_heads == self.hidden_size, f"head_dim * num_heads != hidden_size for {self.name}"
        
        parser = TraceParser()
        self.all_ops: List[ModuleInfo] = parser.parse_trace(
            trace_path=trace_dir,
            verbose=False,
            metadata_path=None
        )
        # Preserve the tracer-time taint registry so the resolver can do
        # inverse lookups (e.g. which value carries NUM_REQS?) without having
        # to rescan the trace file.
        self.taint_registry = getattr(parser, 'taint_registry', {}) or {}

        self.used_ops: List[ModuleInfo] = [op for op in self.all_ops if not op.is_collective_op]
        self.collective_ops: List[ModuleInfo] = [op for op in self.all_ops if op.is_collective_op]
        self.to_profile: List[ModuleInfo] = self.used_ops
        
        self.all_resolved_ops: List[Tuple[ModuleInfo, Callable, str, str]] = []  # All ops with signatures
        self.to_profile_resolved: List[Tuple[ModuleInfo, Callable]] = []
        
    def compare_with_many(self, models: List['ModelInfo'], db_path: Optional[str] = None, overwrite=False) -> None:
        all_models = list(models)
        if self not in all_models:
            all_models.insert(0, self)

        comparator = ModuleComparator()
        # list of model names
        model_names = [model.name for model in all_models]
        # list of [list of (ModuleInfo, Callable)]
        model_modules = [model.to_profile_resolved for model in all_models]
        pruned, all_ops = comparator.prune_overlaps(model_modules, model_names, db_path=db_path, overwrite=overwrite)

        # update all models
        for model in all_models:
            # to_profile_resolved: operations that need profiling
            model.to_profile_resolved = pruned.get(model.name, [])
            model.to_profile = [module for module, _, _, _ in model.to_profile_resolved]
            # all_resolved_ops: ALL operations with signatures (for model_operations table)
            model.all_resolved_ops = all_ops.get(model.name, [])


    # common values across model configs
    def get_model_family(self) -> str:
        """Get model family from config's model_type attribute."""
        return getattr(self.config, "model_type", "unknown")

    def get_hidden_size(self) -> int:
        return getattr(self.config, "hidden_size", None)
    def get_num_heads(self) -> int:
        return getattr(self.config, "num_attention_heads", None)
    def get_vocab_size(self) -> int:
        return getattr(self.config, "vocab_size", None)
    def get_num_layers(self) -> int:
        return getattr(self.config, "num_hidden_layers", None)

    def get_max_model_len(self) -> Optional[int]:
        candidates = [
            "max_position_embeddings",
            "max_sequence_length",
            "seq_length",
            "n_positions",
        ]
        for attr in candidates:
            value = getattr(self.config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    def get_kv_params(self) -> Tuple[Optional[int], Optional[int]]:
        # SSM models (Mamba) don't have attention heads
        if self.num_heads is None:
            return None, None
        return self._extract_kv_params(self.config)

    def _extract_kv_params(self, config) -> Tuple[Optional[int], Optional[int]]:
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = getattr(config, "num_kv_heads", None)
        if num_kv_heads is None:
            num_kv_heads = self.num_heads
        head_dim = self.hidden_size // self.num_heads

        return num_kv_heads, head_dim

    def get_max_num_blocks(
        self,
        world_config: Dict[str, List[int]],
        block_size: int = 16,
        dtype: Optional[torch.dtype] = None,
        gpu_memory_utilization: float = 0.9,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> int:
        dtype = dtype or self.dtype
        if num_kv_heads is None or head_dim is None:
            config_num_kv_heads, config_head_dim = self._extract_kv_params(self.config)
            num_kv_heads = num_kv_heads or config_num_kv_heads
            head_dim = head_dim or config_head_dim
        num_layers = num_layers or self.num_layers

        if not isinstance(num_kv_heads, int) or not isinstance(head_dim, int):
            raise ValueError(f"Missing KV params for model {self.name}")
        if not isinstance(num_layers, int):
            raise ValueError(f"Missing num_layers for model {self.name}")

        element_size = torch.empty(1, dtype=dtype).element_size()
        block_memory_size = (
            2
            * block_size
            * element_size
            * num_kv_heads // min(world_config["tp"])
            * head_dim
        )
        assert num_layers % world_config["pp"][0] == 0
        block_memory_total = block_memory_size * (
            num_layers // world_config["pp"][0]
        )
        return int((torch.cuda.mem_get_info()[1] * gpu_memory_utilization) / block_memory_total)
    
    # save a json file with Dict[(model_name), (List[Dict[(module_name), (operation_name), (params)]])]
    def to_json(self, models: List['ModelInfo'], json_output: str) -> Dict[str, List[Dict[str, Any]]]: 
        print(f"Outputting profiling results to {json_output}")
        all_models = list(models)
        if self not in all_models:
            all_models.insert(0, self)

        output_data: Dict[str, Dict[str, List[str]]] = {}
        for model in all_models:
            output_data[model.name] = {}
            for module in model.to_profile:
                if module.module_name not in output_data[model.name]:
                    output_data[model.name][module.module_name] = []
                output_data[model.name][module.module_name].append(module.operation_name)

        with open(json_output, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        return output_data
