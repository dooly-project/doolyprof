"""Unit tests for taint-string dimension parsing and trace batch selection.

* ``parse_taint_string`` / ``extract_inputs_from_taint`` must keep every traced
  dimension, including ``SINGLETON(1)`` broadcast dims, so dimension types stay
  aligned with the traced shape (a dropped dim shifted MODEL_CONFIG padding onto
  the token axis and leaked the trace batch size into signatures).
* ``choose_trace_batch_size`` must avoid every registry collision the scheduler
  would otherwise raise as a TAINT CONFLICT.

Runs under pytest or directly: ``python doolyprof/tests/taint_parse_test.py``. CPU only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from doolyprof.profiler.taint import DimensionType, parse_taint_string  # noqa: E402
from doolyprof.profiler.finput import extract_inputs_from_taint  # noqa: E402
from doolyprof.profiler.parser import Event  # noqa: E402
from doolyprof.tracer.registry import choose_trace_batch_size, choose_trace_batch  # noqa: E402

MC, NT, NR, MIX = (DimensionType.MODEL_CONFIG, DimensionType.NUM_TOKENS,
                   DimensionType.NUM_REQUESTS, DimensionType.MIX)


def _types(taint_input):
    return [d.dim_type for d in taint_input.dimensions]


def test_singleton_dim_is_kept():
    s = "aten::masked_fill_.Scalar IN:[[NUM_TOKS(504), MODEL_CONFIG(2304)] | [NUM_TOKS(504), SINGLETON(1)] | []]"
    inputs = parse_taint_string(s)
    assert len(inputs) == 3
    assert _types(inputs[0]) == [NT, MC]
    assert _types(inputs[1]) == [NT, MC], _types(inputs[1])  # SINGLETON kept, as a fixed dim
    assert [d.value for d in inputs[1].dimensions] == [504, 1]
    assert inputs[2].dimensions == []


def test_singleton_keeps_token_axis_aligned_with_trace_shape():
    s = "aten::masked_fill_.Scalar IN:[[NUM_TOKS(504), MODEL_CONFIG(2304)] | [NUM_TOKS(504), SINGLETON(1)] | []]"
    profile_inputs = extract_inputs_from_taint(
        s, trace_input_dims=[[504, 2304], [504, 1], []],
        trace_input_dtypes=["c10::BFloat16", "bool", "Scalar"], concrete_inputs=["", "", "0.0"])
    mask = profile_inputs[1]
    assert mask.type == "Tensor" and mask.actual_shape == [504, 1]
    assert mask.dimensions == [NT, MC], mask.dimensions  # before: [MC, NT] -> 504 became a "model constant"


def test_mla_kv_tensor_with_singleton_middle_dim():
    s = "vllm::unified_mla_kv_cache_update IN:[[NUM_TOKS(36)[hist:NUM_TOKS:36], MODEL_CONFIG(512)] | [NUM_TOKS(36)[hist:NUM_TOKS:36], SINGLETON(1), MODEL_CONFIG(64)] | []]"
    inputs = parse_taint_string(s)
    assert _types(inputs[0]) == [NT, MC]
    assert _types(inputs[1]) == [NT, MC, MC]
    assert [d.value for d in inputs[1].dimensions] == [36, 1, 64]


def test_known_labels_and_max_prefix():
    inputs = parse_taint_string("op IN:[[MAX_NUM_TOKS(521), MAX_NUM_REQS(263), NUM_REQS(7), ?(9), MODEL_CONFIG(128)]]")
    assert _types(inputs[0]) == [NT, NR, NR, MC, MC]


def test_mix_history_tracer_order_and_legacy_order():
    # tracer writes taint:qty (tensor._format_dim); older strings had qty:taint
    d = parse_taint_string("op IN:[[MIX(8064)[hist:NUM_TOKS:504,MODEL_CONFIG:16], MODEL_CONFIG(128)]]")[0].dimensions[0]
    assert d.dim_type == MIX and d.value == 8064 and d.components == {"NUM_TOKS": 504, "MODEL_CONFIG": 16}, d
    d = parse_taint_string("op IN:[[MIX(240)[hist:6:NUM_TOKS,40:MODEL_CONFIG], MODEL_CONFIG(128)]]")[0].dimensions[0]
    assert d.dim_type == MIX and d.value == 240 and d.components == {"NUM_TOKS": 6, "MODEL_CONFIG": 40}
    d = parse_taint_string("op IN:[[MIX(1077248)[hist:MAX_NUM_REQS:1,MODEL_CONFIG:1]]]")[0].dimensions[0]
    assert d.components == {"MAX_NUM_REQS": 1, "MODEL_CONFIG": 1}


def test_mix_component_axis_is_exact_label():
    from doolyprof.profiler.input_generator import _component_axis
    assert _component_axis("NUM_TOKS") == "num_tokens"
    assert _component_axis("NUM_REQS") == "num_requests"
    assert _component_axis("MAX_NUM_REQS") is None     # buffer bound: fixed, not swept
    assert _component_axis("MAX_NUM_TOKS") is None
    assert _component_axis("MODEL_CONFIG") is None and _component_axis("?") is None


def _mix_module(taint, dims):
    from doolyprof.profiler.module import ModuleInfo
    inputs = extract_inputs_from_taint(taint, trace_input_dims=dims, trace_input_dtypes=["c10::BFloat16"] * len(dims))
    return ModuleInfo(module_name="synthetic", operation_name="aten::mul.Tensor", inputs=inputs, dtype="bfloat16")


def test_generator_sweeps_mix_dimension_with_tokens():
    """End-to-end through the real generator on GPU: a flattened tokens x heads tensor
    (MIX = NUM_TOKS * 16) must follow the token sweep, and the op must run on the result."""
    import torch
    if not torch.cuda.is_available():
        print("SKIP (no CUDA)"); return
    from doolyprof.profiler.input_generator import InputGenerator
    taint = ("aten::mul.Tensor IN:[[MIX(8064)[hist:NUM_TOKS:504,MODEL_CONFIG:16], MODEL_CONFIG(128)] | "
             "[MIX(8064)[hist:NUM_TOKS:504,MODEL_CONFIG:16], MODEL_CONFIG(128)]]")
    module = _mix_module(taint, [[8064, 128], [8064, 128]])
    assert module.inputs[0].dimensions == [MIX, MC] and module.inputs[0].dimension_components[0] == {"NUM_TOKS": 504, "MODEL_CONFIG": 16}
    gen = InputGenerator(module, max_num_request=4, max_num_token=256, test_counts=3)
    configs = gen.prepare_inputs(mode="vidur", use_batch_config=False)
    assert configs and len({c["num_tokens"] for c in configs}) > 1, configs  # the token axis is swept
    for cfg in configs:
        tensors, params = gen._create_tensor_inputs_from_params(cfg)
        expect = (params["num_tokens"] * 16, 128)
        assert tuple(tensors[0].shape) == expect and tuple(tensors[1].shape) == expect, (cfg, tensors[0].shape)
        out = torch.ops.aten.mul.Tensor(tensors[0], tensors[1])
        assert tuple(out.shape) == expect
    torch.cuda.synchronize()


def test_generator_keeps_max_bound_mix_fixed():
    import torch
    if not torch.cuda.is_available():
        print("SKIP (no CUDA)"); return
    from doolyprof.profiler.input_generator import InputGenerator
    taint = "aten::mul.Tensor IN:[[MIX(538624)[hist:MAX_NUM_REQS:263,MODEL_CONFIG:2048]] | [MIX(538624)[hist:MAX_NUM_REQS:263,MODEL_CONFIG:2048]]]"
    module = _mix_module(taint, [[538624], [538624]])
    gen = InputGenerator(module, max_num_request=4, max_num_token=256, test_counts=3)
    for cfg in ({"num_tokens": 8, "num_requests": 1}, {"num_tokens": 256, "num_requests": 4}):
        tensors, _ = gen._create_tensor_inputs_from_params(cfg)
        assert tuple(tensors[0].shape) == (538624,), tensors[0].shape  # MAX_* bound: never swept


def test_event_dims_from_taint_string_keeps_singleton():
    ev = Event(cat="cpu_op", name="x", ts=0, dur=1, ev_idx=None, external_id=None, python_id=None, python_parent_id=None)
    dims = ev.get_dims_from_taint_string("[[NUM_TOKS(504), MODEL_CONFIG(2304)] | [NUM_TOKS(504), SINGLETON(1)]]")
    assert dims == [[504, 2304], [504, 1]], dims


def test_choose_trace_batch_size_kimi_linear():
    # Kimi-Linear registers 2..27 (kda_layers + full_attn_layers) as MODEL_CONFIG;
    # 18-token prompt, max_num_seqs=263, max_num_batched_tokens=521.
    reg = {v: "MODEL_CONFIG" for v in range(2, 28)}
    reg.update({32: "MODEL_CONFIG", 263: "MAX_NUM_REQS", 521: "MAX_NUM_TOKS", 512: "MODEL_CONFIG"})
    assert choose_trace_batch_size(7, 18, 263, 521, reg) == 28   # 2..27 taken; 28*18=504 fits one step
    assert choose_trace_batch_size(2, 18, 263, 521, reg) == 28
    assert choose_trace_batch_size(28, 18, 263, 521, reg) == 28
    assert choose_trace_batch_size(40, 18, 263, 521, reg) == 28  # 29*18 > 521: fall back downward


def test_choose_trace_batch_truncates_prompt_when_needed():
    reg = {v: "MODEL_CONFIG" for v in range(2, 28)}
    reg.update({32: "MODEL_CONFIG", 263: "MAX_NUM_REQS", 521: "MAX_NUM_TOKS", 512: "MODEL_CONFIG"})
    # default 47-token prompt: 521 // 47 = 11 and 2..27 are taken -> shorten to 18 tokens, batch 28
    assert choose_trace_batch(7, 47, 263, 521, reg) == (28, 18)
    assert choose_trace_batch(7, 18, 263, 521, reg) == (28, 18)   # nothing to truncate
    assert choose_trace_batch(7, 19, 263, 521, {}) == (7, 19)     # empty registry: as requested
    big = {v: "MODEL_CONFIG" for v in range(2, 300)}
    try:
        choose_trace_batch(7, 47, 263, 521, big)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_choose_trace_batch_size_derived_values():
    assert choose_trace_batch_size(5, 10, 263, 521, {6: "MODEL_CONFIG"}) == 7    # 5 -> cu_seqlens 6 collides; 6 collides itself
    assert choose_trace_batch_size(5, 10, 263, 521, {50: "MODEL_CONFIG"}) == 6   # 5*10 tokens collides
    assert choose_trace_batch_size(5, 10, 263, 521, {}) == 5                     # nothing to avoid
    try:
        choose_trace_batch_size(2, 300, 263, 521, {})                             # no batch fits one step
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                failures += 1; print(f"FAIL {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failures += 1; print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
