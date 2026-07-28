"""Model-agnostic MoE forced routing +
the activated_experts grid, for the 2-D (num_tokens, activated_experts) MoE profile.

Every vLLM MoE model routes its expert computation through the common `FusedMoE`
layer (or its subclass `SharedFusedMoE`, which folds in the shared expert), bound as
`self.experts` on the model-specific wrapper. So hooking the base `FusedMoE` covers
Mixtral, Qwen2/3-MoE, Phi-MoE, DeepSeek-V2, GPT-OSS, Llama-4, ... with no per-model
code. We force the router *instance's* `_compute_routing` (reached from every router
subclass via `select_experts`) so exactly `activated_experts` distinct experts fire,
regardless of the (dummy-weight) gate -- LLMServingSim's methodology
(profiler/core/hooks/moe_hook.py), but keyed on type rather than a per-model class name.
"""
from contextlib import contextmanager

import torch


def find_fused_moe(module):
    """Return the FusedMoE / SharedFusedMoE instance for ``module`` -- whether the
    module IS a FusedMoE or a wrapper (e.g. Qwen2MoeSparseMoeBlock) that contains one.
    Returns None if there is no FusedMoE (i.e. not an MoE module)."""
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    except Exception:
        return None
    if isinstance(module, FusedMoE):
        return module
    if hasattr(module, "modules"):
        for m in module.modules():
            if isinstance(m, FusedMoE):
                return m
    return None


def moe_activated_grid(top_k, num_experts, num_tokens):
    """Power-of-two ``activated_experts`` grid in [top_k, min(num_experts,
    num_tokens*top_k)] -- LLMServingSim ExpertCategory.compose_shots. Always includes
    the top_k and the upper-bound endpoints so the interpolation table is well-bounded."""
    hi = min(num_experts, num_tokens * top_k)
    if hi < top_k:
        return []
    grid, v = set(), 1
    while v <= num_experts:
        if top_k <= v <= hi:
            grid.add(v)
        v *= 2
    grid.add(top_k)
    grid.add(hi)
    return sorted(grid)


def _cycle_ids(num_tokens, top_k, activated):
    # ids[t, off] = (t*top_k + off) % activated  -> exactly `activated` distinct experts
    return [[(t * top_k + off) % activated for off in range(top_k)]
            for t in range(num_tokens)]


@contextmanager
def force_moe_routing(fused, num_tokens, activated_experts):
    """Patch ``fused.router._compute_routing`` so exactly ``activated_experts`` distinct
    experts fire for a batch of ``num_tokens`` tokens. Router-subclass-agnostic
    (FusedTopKRouter, GroupedTopKRouter, ...). Restores on exit."""
    top_k = fused.top_k
    if not (top_k <= activated_experts <= num_tokens * top_k):
        raise ValueError(
            f"need top_k({top_k}) <= activated({activated_experts}) "
            f"<= num_tokens*top_k({num_tokens * top_k})")
    dev = next(fused.parameters()).device
    idt = fused.router._get_indices_type()
    ids = torch.tensor(
        _cycle_ids(num_tokens, top_k, activated_experts),
        device=dev, dtype=torch.int32 if idt is None else idt)
    weights = torch.full((num_tokens, top_k), 1.0 / top_k, device=dev, dtype=torch.float32)

    router = fused.router
    original = router._compute_routing

    def forced(_hidden_states, _router_logits, _indices_type):
        return weights, ids

    router._compute_routing = forced
    try:
        yield
    finally:
        router._compute_routing = original
