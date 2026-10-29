"""Unit tests for the two signature fields that identify an op's numeric configuration:

* the module state hash (``doolyprof.tracer.hooks._extract_module_state_hash``), which must
  separate attention layers that differ in KV-cache dtype, logit soft-capping, ALiBi or
  Mamba state dtype while staying identical for ordinary modules, and
* the per-op dtype (``doolyprof.profiler.trace._derive_op_dtype``), derived from the op's own
  traced tensor inputs rather than a run-level constant.

Runs under pytest or directly: ``python doolyprof/tests/signature_fields_test.py``. CPU only.
"""
import enum
import hashlib
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch  # noqa: E402

from doolyprof.tracer.hooks import _extract_module_state_hash  # noqa: E402
from doolyprof.profiler.trace import _derive_op_dtype  # noqa: E402
from doolyprof.profiler.finput import ProfileInput  # noqa: E402
from doolyprof.profiler.taint import DimensionType  # noqa: E402

try:
    from vllm.v1.attention.backend import AttentionType
    from vllm.config.cache import CacheConfig
    HAVE_VLLM = True
except Exception:  # pragma: no cover - vLLM optional for the pure-torch tests
    HAVE_VLLM = False


class FakeImpl:
    """Stands in for a vLLM AttentionImpl: the constructor contract is read from __init__."""

    def __init__(self, num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window,
                 kv_cache_dtype, logits_soft_cap, attn_type, kv_sharing_target_layer_name, sinks=None):
        self.num_heads, self.head_size, self.scale = num_heads, head_size, scale
        self.num_kv_heads, self.alibi_slopes, self.sliding_window = num_kv_heads, alibi_slopes, sliding_window
        self.kv_cache_dtype, self.logits_soft_cap, self.attn_type = kv_cache_dtype, logits_soft_cap, attn_type
        self.kv_sharing_target_layer_name, self.sinks = kv_sharing_target_layer_name, sinks
        self.support_trtllm_attn = False  # backend/hardware flag: not a constructor arg, must be ignored


def attention_like(kv_cache_dtype="auto", kv_cache_torch_dtype=torch.bfloat16, sliding_window=None,
                   logits_soft_cap=None, alibi_slopes=None, sinks=None, attn_type="decoder",
                   layer_name="model.layers.0.self_attn.attn", kv_sharing_target_layer_name=None):
    at = AttentionType(attn_type) if HAVE_VLLM else attn_type
    m = types.SimpleNamespace(
        num_heads=32, head_size=128, num_kv_heads=8, sliding_window=sliding_window, has_sink=sinks is not None,
        use_output=True, kv_cache_dtype=kv_cache_dtype, kv_cache_torch_dtype=kv_cache_torch_dtype,
        layer_name=layer_name, kv_sharing_target_layer_name=kv_sharing_target_layer_name, attn_type=at,
        dtype=torch.bfloat16, training=False, quant_config=None, _private=object(),
    )
    m.impl = FakeImpl(32, 128, 0.088, 8, alibi_slopes, sliding_window, kv_cache_dtype, logits_soft_cap, at,
                      kv_sharing_target_layer_name, sinks)
    return m


def test_plain_module_hash_unchanged():
    """Modules without engine config hash exactly as before (primitive attrs only)."""
    m = types.SimpleNamespace(hidden_size=4096, eps=1e-5, training=False, weight_name="w", _p=1)
    canonical = (("eps", 1e-5), ("hidden_size", 4096))
    assert _extract_module_state_hash(m) == hashlib.blake2b(repr(canonical).encode(), digest_size=6).hexdigest()
    assert _extract_module_state_hash(types.SimpleNamespace(name="x")) == ""


def test_kv_cache_dtype_splits_and_auto_canonicalises():
    bf16_auto = _extract_module_state_hash(attention_like("auto", torch.bfloat16))
    bf16_explicit = _extract_module_state_hash(attention_like("bfloat16", torch.bfloat16))
    fp8 = _extract_module_state_hash(attention_like("fp8", torch.uint8))
    fp8_e4m3 = _extract_module_state_hash(attention_like("fp8_e4m3", torch.uint8))
    fp8_e5m2 = _extract_module_state_hash(attention_like("fp8_e5m2", torch.uint8))
    assert bf16_auto != fp8 != fp8_e5m2 and bf16_auto != fp8_e5m2
    if HAVE_VLLM:
        assert bf16_auto == bf16_explicit, "auto and an explicit bfloat16 cache are the same configuration"
        assert fp8 == fp8_e4m3, "bare fp8 is the platform e4m3 flavour"
    # auto resolves through the model dtype, so a float16 model gives a different cache
    assert bf16_auto != _extract_module_state_hash(attention_like("auto", torch.float16))


def test_identifier_strings_do_not_fragment():
    a = _extract_module_state_hash(attention_like(layer_name="model.layers.0.self_attn.attn"))
    b = _extract_module_state_hash(attention_like(layer_name="model.layers.17.self_attn.attn"))
    assert a == b


def test_impl_contract_soft_cap_alibi_sinks():
    base = _extract_module_state_hash(attention_like())
    assert _extract_module_state_hash(attention_like(logits_soft_cap=50.0)) != base
    assert _extract_module_state_hash(attention_like(alibi_slopes=torch.tensor([0.7, 0.5]))) != base
    # tensors contribute shape only: different learned sink values, same hash
    s1 = _extract_module_state_hash(attention_like(sinks=torch.zeros(32)))
    s2 = _extract_module_state_hash(attention_like(sinks=torch.ones(32)))
    assert s1 == s2 and s1 != base
    # a backend/hardware flag on impl that is not a constructor argument is ignored
    m = attention_like(); m.impl.support_trtllm_attn = True
    assert _extract_module_state_hash(m) == base


def test_sliding_window_still_splits():
    assert _extract_module_state_hash(attention_like(sliding_window=None)) != \
        _extract_module_state_hash(attention_like(sliding_window=4096))


def test_enum_attr_by_value():
    if not HAVE_VLLM:
        return
    assert _extract_module_state_hash(attention_like(attn_type="decoder")) != \
        _extract_module_state_hash(attention_like(attn_type="encoder_only"))


def test_mamba_cache_config_dtype():
    if not HAVE_VLLM:
        return
    model_cfg = types.SimpleNamespace(dtype=torch.bfloat16)
    def mixer(cache_dtype):
        return types.SimpleNamespace(num_heads=128, cache_config=CacheConfig(mamba_cache_dtype=cache_dtype,
                                     mamba_ssm_cache_dtype=cache_dtype), model_config=model_cfg, training=False)
    assert _extract_module_state_hash(mixer("auto")) != _extract_module_state_hash(mixer("float32"))
    # the KV cache_dtype field is not a Mamba field: changing it must not split the mixer
    a = mixer("auto"); b = mixer("auto"); b.cache_config.cache_dtype = "fp8"
    assert _extract_module_state_hash(a) == _extract_module_state_hash(b)


def _inp(dtype, shape=(65, 4096), typ="Tensor"):
    return ProfileInput(type=typ, dtype=dtype, dimensions=[], actual_shape=list(shape))


def test_layer_index_attrs_excluded():
    # vLLM stores the layer position on KDA / linear-attention / MoE layers as
    # layer_idx (or layer_id, block_idx); it names the instance, not its config.
    a, b = attention_like(), attention_like()
    a.layer_idx, b.layer_idx = 0, 19
    assert _extract_module_state_hash(a) == _extract_module_state_hash(b)
    a.layer_id, b.layer_id = 3, 4
    a.block_idx, b.block_idx = 1, 2
    assert _extract_module_state_hash(a) == _extract_module_state_hash(b)
    b.sliding_window = 4096  # a real configuration difference must still split
    assert _extract_module_state_hash(a) != _extract_module_state_hash(b)


def test_signature_hash_ignores_descendants():
    """Two anchors that agree on op name, own dims, state hash and dtype hash equal even if the
    ops executed under them differ (one-time autotuning, cached metadata, tracer records)."""
    from doolyprof.profiler.module import ModuleComparator, ModuleInfo
    from doolyprof.profiler.parser import Event, OperationHierarchy

    def ev(name, cat="cpu_op", **args):
        return Event(cat=cat, name=name, ts=0, dur=1, ev_idx=None, external_id=None, python_id=None,
                     python_parent_id=None, args=args)

    def info(children, state="2dcd83b04195", dtype="bfloat16", hidden=2304):
        anchor = ev("KimiDeltaAttention", cat="module")
        for i, c in enumerate(children):
            anchor.children[i] = ev(c)
        h = OperationHierarchy(anchor=anchor, root_module=anchor, nearest_module=anchor, kernels=[], taints=[])
        inp = ProfileInput(type="Tensor", dtype="c10::BFloat16", dimensions=[DimensionType.NUM_TOKENS, DimensionType.MODEL_CONFIG],
                           actual_shape=[504, hidden])
        return ModuleInfo(module_name="KimiDeltaAttention", operation_name="KimiDeltaAttention", inputs=[inp],
                          is_module=True, dtype=dtype, hierarchy=h, state_hash=state)

    cmp = ModuleComparator()
    steady = info(["aten::linear"] * 18 + ["vllm::kda_attention"])
    first = info(["aten::linear"] * 18 + ["vllm::kda_attention", "aten::cumsum", "aten::cat", "PythonSubclass"])
    assert cmp._hash_signature(cmp._module_signature(steady)) == cmp._hash_signature(cmp._module_signature(first))
    assert len(cmp._descendant_signature(first)) == 22 and len(cmp._descendant_signature(steady)) == 19  # still recorded
    for other in (info(["aten::linear"], state="ffffffffffff"), info(["aten::linear"], dtype="float16"),
                  info(["aten::linear"], hidden=4096)):
        assert cmp._hash_signature(cmp._module_signature(steady)) != cmp._hash_signature(cmp._module_signature(other))


def test_impl_class_in_state_hash():
    """Backends sharing one constructor contract must still hash apart."""
    a, b = attention_like(), attention_like()
    class OtherImpl(FakeImpl):
        pass
    b.impl = OtherImpl(32, 128, 0.088, 8, None, None, "auto", None, b.attn_type, None, None)
    assert _extract_module_state_hash(a) != _extract_module_state_hash(b)


def test_derive_op_dtype():
    assert _derive_op_dtype([_inp("long int", (65,)), _inp("c10::BFloat16")]) == "bfloat16"  # rotary: positions first
    assert _derive_op_dtype([_inp("c10::Half")]) == "float16"
    assert _derive_op_dtype([_inp("torch.float32")]) == "float32"
    assert _derive_op_dtype([_inp("float")]) == "float32"
    assert _derive_op_dtype([_inp("int", (7,)), _inp("long int", (7,))]) == "int32"
    assert _derive_op_dtype([_inp("c10::Float8_e4m3fn")]) == "float8_e4m3fn"
    assert _derive_op_dtype([_inp("Scalar", (), "Scalar")], fallback="bfloat16") == "bfloat16"
    assert _derive_op_dtype([], fallback="") == ""


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
