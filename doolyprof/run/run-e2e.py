import os
import sys
import glob
import argparse
import subprocess


# Tracing prompt. Long enough that avg_packed_qo_len (tokens-per-request x GQA
# group size) clears FlashInfer's cta_tile_q threshold of 64, so the dispatched
# kernel symbol -- and therefore the operation signature -- does not depend on
# how a model's tokenizer happens to render the prompt. With the previous short
# default, Qwen2.5-32B and DeepSeek-R1-Distill-Qwen-32B (both 40q/8kv, group 5)
# tokenized to 12 vs 13 tokens, landing at 60 vs 65 and selecting tile 64 vs
# 128, which split the attention signature of two architecturally identical
# models. Override with DOOLY_TRACE_PROMPT if needed.
DEFAULT_TRACE_PROMPT = (
    "Hello, how are you? I am fine, and I hope that you are also doing well "
    "today. The weather here is quite pleasant, so I decided to take a long "
    "walk through the park this morning before starting my work. "
)
TRACE_PROMPT = os.environ.get("DOOLY_TRACE_PROMPT", DEFAULT_TRACE_PROMPT)


def model_to_dirname(model: str) -> str:
    """Convert model name to a safe directory name."""
    return model.replace("/", "_").replace(".", "_")


def find_trace_file(trace_dir: str) -> str | None:
    """Find the .json trace file in trace_dir. Returns None if not found."""
    pattern = os.path.join(trace_dir, "*.json")
    files = glob.glob(pattern)
    return files[0] if files else None


def run_tracer(model: str, tp: int, trace_dir: str, dtype: str, gpu: int, attention_backend: str, tokenizer: str | None = None, quantization: str | None = None):
    """Run the tracer as a subprocess (required due to module-level patching)."""
    tracer_script = os.path.join(os.path.dirname(__file__), "run-tracer.py")
    cmd = [
        sys.executable, tracer_script,
        "--model", model,
        "--tp", str(tp),
        "--trace_dir", trace_dir,
        "--dtype", dtype,
        "--gpu", str(gpu),
        "--attention-backend", attention_backend,
        "--prompts", TRACE_PROMPT,
        # Trace at batch 7: batch 5 gives cu_seqlens=6 which collides with
        # top_k=6 in DeepSeek-V2-Lite's MLA taint propagation. 7 -> cu_seqlens=8,
        # avoiding the collision so MLA models trace via the scenario driver.
        "--batch-size", "7",
    ]
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    if quantization:
        cmd.extend(["--quantization", quantization])
    print(f"[E2E] Running tracer: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=True)

    return result.returncode == 0


def run_profiler(models: list[str], traces: list[str], tp: int, profile_output: str, dtype: str, attention_backend: str, max_batch_size: int, max_seq_len: int, gpu: int, db_path: str | None = None, profile_comm: bool = False, quantization: str | None = None, overwrite: bool = False):
    """Run the profiler as a subprocess."""
    profiler_script = os.path.join(os.path.dirname(__file__), "run-profiler.py")
    cmd = [
        sys.executable, profiler_script,
        "--models", *models,
        "--traces", *traces,
        "--tp", str(tp),
        "--profile-output", profile_output,
        "--dtype", dtype,
        "--attention-backend", attention_backend,
        "--max-batch-size", str(max_batch_size),
        "--max-seq-len", str(max_seq_len),
        "--gpu", str(gpu),
    ]
    if db_path:
        cmd.extend(["--db-path", db_path])
    if profile_comm:
        cmd.append("--profile-comm")
    if quantization:
        cmd.extend(["--quantization", quantization])
    if overwrite:
        cmd.append("--overwrite")
    print(f"[E2E] Running profiler: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="UTNS Sim Profiler Workflow")

    parser.add_argument("--models", nargs='+', type=str, default=["meta-llama/Llama-3.1-8B"], help="Model name or path")
    parser.add_argument("--attention-backend", type=str, default="FLASHINFER", help="Attention backend (FLASHINFER, FLASH_ATTN, XFORMERS)")
    parser.add_argument("--max-batch-size", type=int, default=5, help="Max batch size for profiling")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Max sequence length for profiling")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype")
    parser.add_argument("--tp", "--tensor_parallel_size", nargs='+', type=int, default=[1])
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index to use")
    parser.add_argument("--profile-only", action="store_true", help="Skip tracing, only run profiler")
    parser.add_argument("--workspace-dir", type=str, default="vllm_traces_tp", help="Directory to save trace and profile")
    parser.add_argument("--tokenizer", type=str, default=None, help="Override tokenizer (useful for models with broken tokenizer configs)")
    parser.add_argument("--db-path", type=str, default=None, help="Path to SQLite database for persistent storage and deduplication")
    parser.add_argument("--profile-comm", action="store_true",
                        help="Also profile collective operations (only meaningful when TP>1).")
    parser.add_argument("--quantization", type=str, default=None, choices=["fp8", "awq", "gptq"],
                        help="Optional quantization scheme. 'fp8' requires Hopper+ (H100). "
                             "Combined with load_format=dummy, vLLM initializes random "
                             "quantized weights — no quantized checkpoint needed.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Pass --overwrite to run-profiler to force re-profile of existing sigs.")
    args = parser.parse_args()

    workspace_path = args.workspace_dir
    trace_base = os.path.join(workspace_path, f"traces-{args.attention_backend}-{args.dtype}")
    profile_base = os.path.join(workspace_path, f"profiles-{args.attention_backend}-{args.dtype}")

    os.makedirs(workspace_path, exist_ok=True)
    os.makedirs(trace_base, exist_ok=True)
    os.makedirs(profile_base, exist_ok=True)

    for tp in args.tp:
        tp_trace_dir = os.path.join(trace_base, f"tp_{tp}")
        profile_dir = os.path.join(profile_base, f"tp_{tp}")
        os.makedirs(tp_trace_dir, exist_ok=True)
        os.makedirs(profile_dir, exist_ok=True)

        # Step 1: Run tracer for each model (each gets its own subdirectory)
        if not args.profile_only:
            for model in args.models:
                model_trace_dir = os.path.join(tp_trace_dir, model_to_dirname(model))
                os.makedirs(model_trace_dir, exist_ok=True)
                
                print(f"\n[E2E] Tracing {model} with TP={tp}")

                run_tracer(
                    model=model,
                    tp=tp,
                    trace_dir=model_trace_dir,
                    dtype=args.dtype,
                    gpu=args.gpu,
                    attention_backend=args.attention_backend,
                    tokenizer=args.tokenizer,
                    quantization=args.quantization,
                )

        # Step 2: Collect trace files in order matching models list
        trace_files = []
        for model in args.models:
            model_trace_dir = os.path.join(tp_trace_dir, model_to_dirname(model))
            trace_file = find_trace_file(model_trace_dir)
            if trace_file:
                trace_files.append(trace_file)
            else:
                print(f"[E2E] Warning: No trace file found for {model} in {model_trace_dir}")

        if len(trace_files) != len(args.models):
            print(f"[E2E] Error: Found {len(trace_files)} traces but have {len(args.models)} models, skipping profiler")
            continue

        print(f"\n[E2E] Found {len(trace_files)} trace files for TP={tp}")
        profile_output = os.path.join(profile_dir, "profile.csv")

        run_profiler(
            models=args.models,
            traces=trace_files,
            tp=tp,
            profile_output=profile_output,
            dtype=args.dtype,
            attention_backend=args.attention_backend,
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
            db_path=args.db_path,
            gpu=args.gpu,
            profile_comm=args.profile_comm,
            quantization=args.quantization,
            overwrite=args.overwrite,
        )

    print("\n[E2E] Workflow complete!")


if __name__ == "__main__":
    main()


