#!/usr/bin/env python3
"""
Taint string parser for extracting dimension categorization.

This module parses taint annotations from PyTorch traces to identify
which dimensions are MODEL_CONFIG (fixed model parameters) vs WORKLOAD
(runtime parameters that should be swept during profiling).
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class DimensionType(Enum):
    """Simple dimension categorization for profiling"""
    MODEL_CONFIG = "MODEL_CONFIG"  # Model parameter (fixed) - DON'T sweep, use for comparison
    # WORKLOAD = "WORKLOAD"          # Runtime parameter (?, batch, seq_len) - SWEEP during profiling
    NUM_TOKENS = "NUM_TOKS"
    NUM_REQUESTS = "NUM_REQS"
    MIX = "MIX"  # Mixed dimension (e.g., NUM_TOKS × MODEL_CONFIG)
    # UNKOWN = "UNKNOWN"

@dataclass
class TaintDimension:
    """Represents a single dimension from taint"""
    dim_type: DimensionType
    value: int  # Actual value from trace
    components: Optional[Dict[str, int]] = None  # For MIX: {taint_name: quantity}


@dataclass
class TaintInput:
    """Represents a single input with dimension types"""
    dimensions: List[TaintDimension]


_DIM_RE = re.compile(r'([A-Za-z_?]+)\((\d+)\)(?:\[hist:([^\]]*)\])?')
_LABEL_TO_TYPE = {t.value: t for t in DimensionType}


def _make_dimension(label: str, value: int, hist: str) -> TaintDimension:
    """Map one traced label to a TaintDimension.

    ``MAX_<label>`` folds onto its base type (MAX_NUM_TOKS -> NUM_TOKENS).
    A label that is not a sweepable workload axis ("?", "SINGLETON", any
    label this profiler does not know) is MODEL_CONFIG: fixed, never swept.
    MIX history pairs are "taint:qty" as the tracer writes them
    (tensor._format_dim: ``f"{taint}:{qty}"``); the legacy "qty:taint" order
    is accepted too. A pair with no integer side is skipped.
    """
    base = label[4:] if label.startswith("MAX_") else label
    dim_type = _LABEL_TO_TYPE.get(base, DimensionType.MODEL_CONFIG)
    components = None
    if dim_type == DimensionType.MIX:
        components = {}
        for component in (hist or "").split(","):
            parts = [x.strip() for x in component.split(":")]
            if len(parts) != 2:
                continue
            if parts[1].isdigit():          # taint:qty (tracer)
                components[parts[0]] = int(parts[1])
            elif parts[0].isdigit():        # qty:taint (legacy)
                components[parts[1]] = int(parts[0])
    return TaintDimension(dim_type, value, components)


def parse_taint_string(taint_name: str) -> List[TaintInput]:
    # Use greedy match (.*) to capture all inputs until the final ]]
    # e.g., [[?(7), MODEL_CONFIG(4096)] | [?(7), MODEL_CONFIG(1024)] | [?(7), MODEL_CONFIG(1024)]]
    match = re.search(r'\[\[(.*)\]\]', taint_name)
    if not match:
        return []


    content = match.group(1)

    # Split by "|" to get individual inputs
    input_strs = content.split('|')
    # print(f"[DEBUG @ taint.py] match : {match}")
    # print(f"[DEBUG @ taint.py] content : {content}")
    # print(f"[DEBUG @ taint.py] input_strs : {input_strs}")
     

    taint_inputs = []

    for input_str in input_strs:
        input_str = input_str.strip()
        if not input_str or input_str == '[]':
            # Empty input (scalar with no shape)
            taint_inputs.append(TaintInput(dimensions=[]))
            # print(f"[DEBUG @ taint.py] taint_inputs: {taint_inputs}")
            continue

        # Do not strip brackets: the outer [[...]] match already consumed the
        # list's own brackets, and a trailing "]" here belongs to a MIX
        # history ("MIX(n)[hist:...]" as the last dim) that the regex below
        # needs intact. _DIM_RE ignores stray brackets anyway.

        # print(f"[DEBUG @ taint.py] input_str: {input_str}")

        dimensions = []

        # Every dimension label the tracer emits has the shape LABEL(N),
        # optionally followed by [hist:...] (DimTaint.display in
        # doolyprof.tracer.types, "?(N)" for untracked dims and "SINGLETON(1)"
        # for broadcast dims in tensor._format_dim). Match that shape rather
        # than a fixed list of labels, so no dimension is dropped and the
        # result stays aligned with the traced shape; the label is mapped to a
        # DimensionType afterwards.
        for label, value, hist in _DIM_RE.findall(input_str):
            dimensions.append(_make_dimension(label, int(value), hist))

        taint_inputs.append(TaintInput(dimensions=dimensions))

    return taint_inputs


# Test function
if __name__ == "__main__":
    # Test parse_taint_string
    test_taint = "_C::rms_norm IN:[[NUM_TOKS(15), MODEL_CONFIG(4096)] | [NUM_TOKS(15), MODEL_CONFIG(4096)] | [MODEL_CONFIG(4096)]]"

    print(f"Parsing: {test_taint}")
    print()

    result = parse_taint_string(test_taint)

    print(f"Found {len(result)} inputs:")
    for i, taint_input in enumerate(result):
        print(f"  Input {i}:")
        for j, dim in enumerate(taint_input.dimensions):
            print(f"    Dim {j}: {dim.dim_type.name}({dim.value})")
    print()

    # Test with scalar at end
    test_taint2 = "_C::fused_add_rms_norm IN:[[?(15), MODEL_CONFIG(4096)] | [?(15), MODEL_CONFIG(4096)] | [MODEL_CONFIG(4096)] | []]"
    print(f"Parsing with scalar: {test_taint2}")
    print()

    result2 = parse_taint_string(test_taint2)
    print(f"Found {len(result2)} inputs:")
    for i, taint_input in enumerate(result2):
        print(f"  Input {i}: {len(taint_input.dimensions)} dimensions")
        for j, dim in enumerate(taint_input.dimensions):
            print(f"    Dim {j}: {dim.dim_type.name}({dim.value})")
    print()

    # Test with MIX dimension
    test_taint3 = "test_op IN:[[MIX(10760)[hist:269:NUM_TOKS,40:MODEL_CONFIG], MODEL_CONFIG(128)]]"
    print(f"Parsing with MIX: {test_taint3}")
    print()

    result3 = parse_taint_string(test_taint3)
    print(f"Found {len(result3)} inputs:")
    for i, taint_input in enumerate(result3):
        print(f"  Input {i}:")
        for j, dim in enumerate(taint_input.dimensions):
            if dim.components:
                print(f"    Dim {j}: {dim.dim_type.name}({dim.value}) with components: {dim.components}")
            else:
                print(f"    Dim {j}: {dim.dim_type.name}({dim.value})")
