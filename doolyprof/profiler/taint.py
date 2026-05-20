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

        # Remove outer brackets if present
        input_str = input_str.strip('[]')

        # print(f"[DEBUG @ taint.py] input_str: {input_str}")

        dimensions = []

        # Find all dimension patterns:
        # - MODEL_CONFIG(value)
        # - ?(value)

        # Patterns for different taint types
        # MIX(value)[hist:qty:taint,qty:taint,...] - with component history
        mix_pattern = r'MIX\((\d+)\)\[hist:([^\]]+)\]'
        # MODEL_CONFIG(value)
        model_config_pattern = r'MODEL_CONFIG\((\d+)\)'
        # NUM_TOKS(value) or MAX_NUM_TOKS(value)
        num_toks_pattern = r'(?:MAX_)?NUM_TOKS\((\d+)\)'
        # NUM_REQS(value) or MAX_NUM_REQS(value)
        num_reqs_pattern = r'(?:MAX_)?NUM_REQS\((\d+)\)'
        # ?(value) - unknown/untracked dimensions
        unknown_pattern = r'\?\((\d+)\)'

        # Find all dimension patterns (including MIX with history)
        # Note: Order matters - check MIX first since it has [hist:...] suffix
        dim_pattern = r'MIX\(\d+\)\[hist:[^\]]+\]|MODEL_CONFIG\(\d+\)|(?:MAX_)?NUM_TOKS\(\d+\)|(?:MAX_)?NUM_REQS\(\d+\)|\?\(\d+\)'
        dim_matches = re.findall(dim_pattern, input_str)

        # print(f"[DEBUG @ taint.py] dim_matches: {dim_matches}")

        for dim_match in dim_matches:
            # Check if it's MIX with history
            mix_match = re.match(mix_pattern, dim_match)
            if mix_match:
                value = int(mix_match.group(1))
                history_str = mix_match.group(2)
                # Parse history: "qty:taint,qty:taint,..." -> {taint: qty}
                components = {}
                for component in history_str.split(','):
                    parts = component.strip().split(':')
                    if len(parts) == 2:
                        qty_str, taint_name = parts
                        try:
                            qty = int(qty_str)
                            components[taint_name] = qty
                        except ValueError:
                            pass
                dimensions.append(TaintDimension(DimensionType.MIX, value, components))
                continue
            # Check if it's MODEL_CONFIG
            mc_match = re.match(model_config_pattern, dim_match)
            if mc_match:
                value = int(mc_match.group(1))
                dimensions.append(TaintDimension(DimensionType.MODEL_CONFIG, value))
                # print(f"[DEBUG @ taint.py] dimensions (mc_match): {dimensions}")
                continue

            # Check if it's NUM_TOKS
            num_toks_match = re.match(num_toks_pattern, dim_match)
            if num_toks_match:
                value = int(num_toks_match.group(1))
                dimensions.append(TaintDimension(DimensionType.NUM_TOKENS, value))
                # print(f"[DEBUG @ taint.py] dimensions (num_toks_match): {dimensions}")
                continue

            # Check if it's NUM_REQS
            num_reqs_match = re.match(num_reqs_pattern, dim_match)
            if num_reqs_match:
                value = int(num_reqs_match.group(1))
                dimensions.append(TaintDimension(DimensionType.NUM_REQUESTS, value))
                # print(f"[DEBUG @ taint.py] dimensions (num_reqs_match): {dimensions}")
                continue

            # Check if it's ?(value) - treat as MODEL_CONFIG (conservative, won't sweep)
            unknown_match = re.match(unknown_pattern, dim_match)
            if unknown_match:
                value = int(unknown_match.group(1))
                dimensions.append(TaintDimension(DimensionType.MODEL_CONFIG, value))
                # print(f"[DEBUG @ taint.py] dimensions (unknown_match): {dimensions}")
                continue


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
