#!/usr/bin/env python3
"""Time the trace parser on one exported trace, CPU only.

    python doolyprof/tests/parse_benchmark.py <trace.json> [--filter] [--out result.json]

Reports JSON load, HierarchicalTraceParser.parse_from_events (tree, pruning,
taint association) and extract_module_to_operation_mapping separately, plus
event / taint counts and a summary of the hierarchies found, so two versions of
the parser can be compared on the same input. ``--filter`` keeps only the
cpu_op / kernel / user_annotation events before parsing, which is what the
tracer now exports.

Reference (Kimi-Linear-48B, fake TP 2, 270 MB unfiltered trace, 1,329,293
events, 5,489 taint annotations, 2026-10-29): before the indexed taint lookup
parse 178.7 s + mapping 330.9 s; after 1.8 s + 0.8 s, identical output.
"""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from doolyprof.profiler.parser import HierarchicalTraceParser  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--filter", action="store_true", help="drop events the parser never reads first")
    ap.add_argument("--out", default=None, help="write the result as JSON here")
    a = ap.parse_args()

    t = {}
    t0 = time.perf_counter()
    data = json.load(open(a.trace))
    t["json_load_s"] = time.perf_counter() - t0
    events = data["traceEvents"]
    n_all = len(events)
    if a.filter:
        keep = {"cpu_op", "kernel", "user_annotation"}
        events = [e for e in events if e.get("cat") in keep]

    parser = HierarchicalTraceParser(events=events)
    t0 = time.perf_counter()
    parser.parse_from_events()
    t["parse_from_events_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    hierarchies = parser.extract_module_to_operation_mapping()
    t["extract_mapping_s"] = time.perf_counter() - t0

    summary = collections.Counter()
    for h in hierarchies:
        summary[(h.nearest_module_name, h.operation_name, h.get_taint_string() or "")] += 1
    result = {
        "trace": a.trace, "filtered": a.filter, "events_in_file": n_all, "events_parsed": len(events),
        "taints": len(parser.taints), "cpu_ops": len(parser.cpu_ops), "kernels": len(parser.kernels),
        "modules": len(parser.modules), "hierarchies": len(hierarchies), "distinct_hierarchies": len(summary),
        "timings": {k: round(v, 2) for k, v in t.items()},
    }
    print(json.dumps(result, indent=1))
    if a.out:
        result["hierarchy_summary"] = sorted([[list(k), v] for k, v in summary.items()])
        json.dump(result, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
