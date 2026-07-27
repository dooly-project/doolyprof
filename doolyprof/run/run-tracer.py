"""
Single-process TP=N profiling: Load sharded weights on 1 GPU, profile compute.

This patches vLLM's parallel state so linear layers see tp_size=N while
running in a single process. Weights are sharded but no actual NCCL is used.

Usage:
    python test.py --model meta-llama/Llama-3.1-8B --tp 2 --trace_dir ./traces
"""
import os
import sys
import socket
import argparse


def find_free_port() -> int:
    """Find a free port by binding to port 0 and letting the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


# Add project root to path for doolyprof imports
# run-tracer.py -> run/ -> doolyprof/ -> repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Parse arguments BEFORE any heavy imports
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Model name or path")
parser.add_argument("--tp", "--tensor_parallel_size", type=int, default=1, dest="tensor_parallel_size", help="Fake tensor parallel size (weights sharded but run on 1 GPU)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype")
parser.add_argument("--prompts", type=str, default="Hello, how are you? I am fine, and ", help="Prompt to run")
parser.add_argument("--batch-size", type=int, default=5, help="Number of identical prompts to send in a single generate() call")
parser.add_argument("--max-tokens", type=int, default=1, help="Max tokens to generate")
parser.add_argument("--trace_dir", type=str, default="./vllm_traces_tp", help="Directory to save trace")
parser.add_argument("--gpu", type=int, default=0, help="GPU device index to use")
parser.add_argument("--attention-backend", type=str, default="FLASH_ATTN", help="Attention backend (FLASHINFER, FLASH_ATTN, XFORMERS)")
parser.add_argument("--flash-attn-version", type=int, default=2, choices=[2, 3, 4], help="Pin flash-attention version when --attention-backend=FLASH_ATTN. Without this, vLLM auto-picks FA3 on Hopper / FA4 on Blackwell.")
parser.add_argument("--quantization", type=str, default=None, choices=["fp8", "awq", "gptq"], help="Optional quantization scheme. 'fp8' requires Hopper+ (H100). With load_format=dummy, vLLM initializes random quantized weights — no quantized checkpoint needed.")
parser.add_argument("--tokenizer", type=str, default=None, help="Override tokenizer (useful for models with broken tokenizer configs)")
parser.add_argument("--kv-cache-dtype", type=str, default="auto", help="KV cache dtype passed to vLLM (e.g. 'fp8'). Match your serving config so the trace captures the same attention kernels.")
parser.add_argument("--block-size", type=int, default=None, help="KV cache block size passed to vLLM. Match your serving config (e.g. 256).")
args = parser.parse_args()

import torch
import torch.profiler
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

print(f"[Tracer] Set VLLM_ATTENTION_BACKEND={args.attention_backend}")

FAKE_TP = args.tensor_parallel_size
os.environ["VLLM_FAKE_TP"] = str(FAKE_TP)

# Install taint hooks BEFORE any vLLM imports
# This patches AutoConfig.from_pretrained to return TaintedConfig objects
# and installs hooks on tensor factory functions for taint propagation
from doolyprof.tracer.hooks import install_hooks
from doolyprof.tracer.config import patch_autoconfig, patch_input_buffers
from doolyprof.tracer.registry import TAINT_REGISTRY
from doolyprof.tracer import disable_taint_tracking, enable_taint_tracking

# Disable taint tracking initially - we'll enable it only during the actual generate() call
# This prevents semantic confusion during warm-up/compilation where values are reused incorrectly
disable_taint_tracking()
print("[Tracer] Taint tracking DISABLED during initialization/warm-up")

patch_autoconfig()
# patch_common_attention_metadata()
patch_input_buffers()
install_hooks(
    int_patch_modules=[
        "vllm.model_executor.layers.rotary_embedding",
    ]
)

# Patch torch.nn.Parameter.__new__ so that when constructed from a
# TaintedTensor, the data is unwrapped to a plain torch.Tensor before
# delegating. This preserves the Parameter subclass identity (e.g.
# vLLM's BasevLLMParameter / ModelWeightParameter), which carries
# attributes like `weight_loader` via @property — those are unreachable
# if the resulting object is downgraded to a plain TaintedTensor by
# Parameter.__new__'s alt-path (data.detach().requires_grad_(rg)).
_original_parameter_new = torch.nn.Parameter.__new__

def _patched_parameter_new(cls, data=None, requires_grad=True):
    from doolyprof.tracer.tensor import TaintedTensor
    if isinstance(data, TaintedTensor):
        # Unwrap: get the plain Tensor view of the same storage, so that
        # type(data) is torch.Tensor and Parameter.__new__ takes the
        # _make_subclass(cls, ...) path that actually returns `cls`.
        with torch._C._DisableTorchDispatch():
            data = torch.Tensor._make_subclass(torch.Tensor, data, False)
    return _original_parameter_new(cls, data=data, requires_grad=requires_grad)

torch.nn.Parameter.__new__ = _patched_parameter_new

# Patch DummyModelLoader to force CUDA move
# (dummy models are loaded into CPU for some reason when tainting is enabled)
try:
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    original_dummy_load_weights = DummyModelLoader.load_weights

    def patched_dummy_load_weights(self, model, model_config):
        if torch.cuda.is_available():
            # vLLM's FP8 / quantization paths capture `layer._load_device =
            # torch.get_default_device()` at create_weights time, then later
            # use it inside process_weights_after_loading to allocate the
            # real (non-meta) weight. With our taint hooks active, that
            # captured device can resolve to CPU even though vLLM wraps
            # initialize_model in `with target_device:` — and a CPU weight
            # then fails scaled_fp8_quant (CUDA-only). Pin it to cuda:0
            # before dummy load so vLLM's per-layer materialize+quantize
            # loop runs on CUDA (it frees the BF16 staging tensor after
            # each layer, so this doesn't double-allocate).
            cuda_dev = torch.device("cuda:0")
            for module in model.modules():
                if hasattr(module, "_load_device"):
                    module._load_device = cuda_dev

        original_dummy_load_weights(self, model, model_config)

        if torch.cuda.is_available():
            # Move CPU-placed dummy weights to CUDA (tainting sometimes
            # leaves them on CPU). Leave meta tensors alone — vLLM's
            # quantization init handles them in process_weights_after_loading.
            def _convert(t):
                if t.is_meta or t.device.type == "cuda":
                    return t
                return t.to("cuda")
            model._apply(_convert)

    DummyModelLoader.load_weights = patched_dummy_load_weights
except ImportError:
    print("[DebugPatch] Could not import DummyModelLoader")

# Debug patch: Track num_tokens in _dummy_run
try:
    import vllm.v1.worker.gpu_model_runner as gmr_module
    from doolyprof.tracer.types import TaintedInt

    _orig_dummy_run = None
    _orig_get_slot_mappings = None

    def patch_for_debugging():
        global _orig_dummy_run, _orig_get_slot_mappings
        if hasattr(gmr_module, 'GPUModelRunner'):
            GPUModelRunner = gmr_module.GPUModelRunner

            if _orig_dummy_run is None:
                _orig_dummy_run = GPUModelRunner._dummy_run
                def debug_dummy_run(self, num_tokens, *args, **kwargs):
                    return _orig_dummy_run(self, num_tokens, *args, **kwargs)
                GPUModelRunner._dummy_run = debug_dummy_run

            if _orig_get_slot_mappings is None:
                _orig_get_slot_mappings = GPUModelRunner._get_slot_mappings
                def debug_get_slot_mappings(self, num_tokens_padded, *args, **kwargs):
                    return _orig_get_slot_mappings(self, num_tokens_padded, *args, **kwargs)
                GPUModelRunner._get_slot_mappings = debug_get_slot_mappings

            print("[DebugPatch] Installed debug wrappers for _dummy_run and _get_slot_mappings", flush=True)

    # Will patch after GPUModelRunner is defined
except Exception as e:
    print(f"[DebugPatch] Could not setup debug patches - {e}", flush=True)

# Apply fake TP patches using shared utility, then import vLLM
from doolyprof.profiler.utils.fake_tp import apply_fake_tp_patches
if FAKE_TP > 1:
    apply_fake_tp_patches(FAKE_TP)

from vllm import LLM, SamplingParams
from transformers import AutoConfig

# Debug: verify multiprocessing mode
import vllm.envs as envs
print(f"[DEBUG] VLLM_ENABLE_V1_MULTIPROCESSING = {envs.VLLM_ENABLE_V1_MULTIPROCESSING}", flush=True)

def get_max_model_len(model_name: str, default: int = 4096) -> int:
    """Auto-detect max_model_len from model config."""
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Try common attribute names for max position embeddings
        for attr in ["max_position_embeddings", "n_positions", "max_seq_len"]:
            if hasattr(config, attr):
                val = getattr(config, attr)
                if val is not None:
                    print(f"[Tracer] Auto-detected max_model_len={val} from {attr}")
                    return int(val)
        print(f"[Tracer] Could not find max position embeddings, using default={default}")
        return default
    except Exception as e:
        print(f"[Tracer] Error loading config: {e}, using default={default}")
        return default

if __name__ == "__main__":
    print(f"[Tracer] Tracing {args.model} with fake TP={FAKE_TP}")

    # Auto-detect max_model_len from model config, capped at 32768 to avoid OOM
    max_model_len = min(get_max_model_len(args.model), 32768)

    try:
        # Apply debug patches before LLM init
        try:
            patch_for_debugging()
        except:
            pass

        llm_kwargs = dict(
            model=args.model,
            tokenizer=args.tokenizer if args.tokenizer else args.model,
            dtype=args.dtype,
            tensor_parallel_size=1,  # Tell vLLM executor we're single process
            load_format="dummy",
            enforce_eager=True,
            gpu_memory_utilization=0.9,
            attention_backend=args.attention_backend,
            max_model_len=max_model_len,
            trust_remote_code=True,
            max_num_batched_tokens=521,
            max_num_seqs=263,
        )
        if args.attention_backend == "FLASH_ATTN":
            llm_kwargs["attention_config"] = {"flash_attn_version": args.flash_attn_version}
        if args.quantization:
            llm_kwargs["quantization"] = args.quantization
        if args.kv_cache_dtype != "auto":
            llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
        if args.block_size is not None:
            llm_kwargs["block_size"] = args.block_size
        llm = LLM(**llm_kwargs)

        # Check weight shapes to verify sharding
        print(f"\n=== Checking weight shapes (should be divided by TP={FAKE_TP}) ===")
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        for name, param in model.named_parameters():
            if "layers.0" in name and ("q_proj" in name or "down_proj" in name):
                print(f"{name}: {param.shape}")

        sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

        # Enable taint tracking now - warm-up/compilation is done
        enable_taint_tracking()
        print("\n[Tracer] Taint tracking ENABLED for actual generation")

        def dooly_trace_handler(dir_name: str, model_name: str, tp: int):
            """torch.profiler handler that writes to a predictable filename
            ``{model_name}_{tp}_dooly_trace.json`` and appends the
            TAINT_REGISTRY snapshot as a top-level key. Mirrors the pattern
            of torch.profiler.tensorboard_trace_handler.

            The handler runs after the profile context exits, at which point
            TAINT_REGISTRY is fully populated by the preceding generate()
            call, so the snapshot is complete.
            """
            import json as _json

            safe_model = model_name.replace("/", "_")
            file_name = f"{safe_model}_{tp}_dooly_trace.json"

            def handler_fn(prof) -> None:
                if not os.path.isdir(dir_name):
                    try:
                        os.makedirs(dir_name, exist_ok=True)
                    except Exception as e:
                        raise RuntimeError(f"Can't create directory: {dir_name}") from e
                trace_path = os.path.join(dir_name, file_name)
                prof.export_chrome_trace(trace_path)

                # Inject the taint registry as a top-level key.
                with open(trace_path, "r") as f:
                    data = _json.load(f)
                data["dooly_taint_registry"] = {
                    str(v): str(t) for v, t in TAINT_REGISTRY.items()
                    if not isinstance(t, dict)
                }
                with open(trace_path, "w") as f:
                    _json.dump(data, f)
                print(f"[Tracer] Wrote trace + taint registry "
                      f"({len(data['dooly_taint_registry'])} entries) to {trace_path}")

            return handler_fn

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # with_stack=True,
            record_shapes=True,
            on_trace_ready=dooly_trace_handler(args.trace_dir, args.model, args.tensor_parallel_size),
        ) as prof:
            llm.generate([args.prompts] * args.batch_size, sampling_params=sampling_params)

        print(f"[Tracer] Done tracing {args.model}, output saved to {args.trace_dir}")

        # Print registry contents
        print("\n" + "="*80)
        print("TAINT_REGISTRY CONTENTS")
        print("="*80)
        print(f"Total entries: {len(TAINT_REGISTRY)}")

        if len(TAINT_REGISTRY) > 0:
            # Separate pure and MIX entries
            pure_entries = {k: v for k, v in TAINT_REGISTRY.items() if not isinstance(v, dict)}
            mix_entries = {k: v for k, v in TAINT_REGISTRY.items() if isinstance(v, dict)}

            if pure_entries:
                print(f"\nPure Taints ({len(pure_entries)} entries):")
                # Sort by value (taint name) then by key
                for key in sorted(pure_entries.keys()):
                    value = pure_entries[key]
                    print(f"  {key:6d} → {value}")

            if mix_entries:
                print(f"\nMIX Taints ({len(mix_entries)} entries):")
                # Sort by key
                for key in sorted(mix_entries.keys()):
                    value = mix_entries[key]
                    components_str = ', '.join(f"{qty}×{taint}" for taint, qty in value['components'].items())
                    print(f"  {key:6d} → MIX [{components_str}]")
        else:
            print("  (Registry is empty)")

        print("="*80 + "\n")
            
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
