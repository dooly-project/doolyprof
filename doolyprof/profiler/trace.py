"""
Trace loading and parsing utilities for vLLM profiler traces.
"""

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

from doolyprof.profiler.module import ModuleInfo
from doolyprof.profiler.parser import HierarchicalTraceParser, OperationHierarchy
from doolyprof.profiler.finput import ProfileInput, extract_inputs_from_taint
from doolyprof.profiler.taint import parse_taint_string

class TraceLoader:
    @staticmethod
    def load_trace(json_path: str) -> Dict[str, Any]:
        json_path = str(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    @staticmethod
    def get_trace_events(trace_data: Dict[str, Any]) -> List[Dict]:
        return trace_data.get("traceEvents", [])

    @staticmethod
    def get_trace_metadata(trace_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "cuda_runtime_version": trace_data.get("cuda_runtime_version"),
            "cuda_driver_version": trace_data.get("cuda_driver_version"),
            "record_shapes": trace_data.get("record_shapes", False),
            "with_flops": trace_data.get("with_flops", False),
            "distributedInfo": trace_data.get("distributedInfo", {}),
            "schema_version": trace_data.get("schemaVersion"),
        }

    @staticmethod
    def get_taint_registry(trace_data: Dict[str, Any]) -> Dict[int, str]:
        """Extract the TAINT_REGISTRY snapshot that run-tracer.py stored at
        the trace's top level. Maps dimension value -> taint name
        (e.g. 5 -> "NUM_REQS"). Returns {} for traces captured before this
        instrumentation (resolver will fall back to its legacy defaults)."""
        raw = trace_data.get("dooly_taint_registry", {}) or {}
        out: Dict[int, str] = {}
        for v, t in raw.items():
            try:
                out[int(v)] = str(t)
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def save_to_json(data: Any, json_path: str) -> None:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def _derive_op_dtype(inputs, fallback: str = "") -> str:
    """Compute dtype of an op from its traced tensor inputs.

    Uses the dtype of the first floating-point tensor input (the activation
    dtype the kernel computes in), else the first tensor input's dtype, else
    ``fallback``. Trace dtype strings (``c10::BFloat16``, ``torch.float16``,
    ``long int`` ...) are resolved through the same mapping the input
    generator uses to materialise tensors, and named by torch
    (``str(torch.dtype)`` without the ``torch.`` prefix), so the value is
    identical whether the op was traced as a raw op or inside a MODULE
    annotation.
    """
    from doolyprof.profiler.input_generator import trace_dtype_to_torch, canonical_dtype_name

    first_tensor = None
    for inp in inputs or []:
        if getattr(inp, "type", None) != "Tensor" or not getattr(inp, "dtype", None):
            continue
        td = trace_dtype_to_torch(inp.dtype)
        if td is None:
            continue
        if td.is_floating_point:
            return canonical_dtype_name(td)
        if first_tensor is None:
            first_tensor = td
    if first_tensor is not None:
        return canonical_dtype_name(first_tensor)
    return fallback


class TraceParser:
    def __init__(self, loader: Optional[TraceLoader] = None) -> None:
        self.loader = loader or TraceLoader()
        # Taint registry captured at trace time (value -> taint name).
        # Populated by parse_trace(); read by callers (e.g. run-profiler.py)
        # that need to pass it down to the resolver.
        self.taint_registry: Dict[int, str] = {}

    def parse_trace(
        self,
        trace_path: str,
        output_path: str = "vllm_operations.json",
        verbose: bool = False,
        metadata_path: str | None = None,
    ) -> List[ModuleInfo]:

        try:
            trace_data = self.loader.load_trace(trace_path)
        except FileNotFoundError:
            print(f"Error: Trace file not found: {trace_path}")
            return []

        trace_events = self.loader.get_trace_events(trace_data)
        self.taint_registry = self.loader.get_taint_registry(trace_data)
        if self.taint_registry:
            print(f"[TraceParser] Loaded taint_registry ({len(self.taint_registry)} entries) from trace")
        else:
            print(f"[TraceParser] No dooly_taint_registry key in trace (older format); "
                  f"resolver will fall back to tensor-axis heuristics")

        profile_metadata = {}
        module_infos = self.extract_module_infos(
            trace_events,
            profile_metadata=profile_metadata,
        )

        # output_data = {
        #     "summary": {
        #         "unique_module_profiles": len(module_infos),
        #     },
        #     "module_infos": [profile.to_dict() for profile in module_infos],
        # }

        # self.loader.save_to_json(output_data, output_path)

        return module_infos

    # build a hierarchical tree of events in the trace
    # convert into ModuleInfo
    def extract_module_infos(
        self,
        trace_events: List[Dict],
        profile_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ModuleInfo]:

        # return list of ModuleInfo
        module_infos: List[ModuleInfo] = []
        collective_infos: List[Any] = []

        # extract module and operation relationships from trace file using HierarchicalTraceParser

        print(f"Extracting module and operation relationships from trace file using HierarchicalTraceParser...")
        hierarchical_parser = HierarchicalTraceParser(events=trace_events)

        print(f"Parsing trace events...")
        hierarchical_parser.parse_from_events()
        
        # collective_infos = hierarchical_parser.comm_ops
        # print(collective_infos)

        # Returns List[OperationHierarchy] with full ancestry info:
        # - hierarchy.anchor: the profiling target (cpu_op or module)
        # - hierarchy.root_module: outermost nn.Module
        # - hierarchy.nearest_module: nearest nn.Module parent
        # - hierarchy.kernels: GPU kernels using this anchor
        # - hierarchy.taints: taint events for the anchor (remove?)

        print("Extracting module to operation mapping...")
        operation_hierarchies: List[OperationHierarchy] = hierarchical_parser.extract_module_to_operation_mapping()

        print("Making module infos from operation hierarchies...")
        profile_metadata = profile_metadata or {}
        backend = profile_metadata.get("backend", "")
        # Per-op dtype is derived from the op's own traced inputs (see
        # _derive_op_dtype); the metadata value is only a fallback for ops
        # whose inputs carry no tensor dtype.
        fallback_dtype = profile_metadata.get("dtype", "bfloat16")

        # --- DEBUG-TRACE: per-reason drop counters + per-op tallies ---
        from collections import Counter as _TCounter
        _drops = _TCounter()
        _drop_by_op = {
            "no_taint": _TCounter(),
            "no_inputs": _TCounter(),
            "taint_extract_failed": _TCounter(),
            "comm_no_params": _TCounter(),
            "comm_no_dims": _TCounter(),
            "comm_empty_inputs": _TCounter(),
        }
        _accepted_by_op = _TCounter()
        # --- END DEBUG-TRACE ---

        for hierarchy in operation_hierarchies:

            module_name, op_name, inputs, is_module, is_collective = "", "", [], False, False
            
            if hierarchy.anchor.cat == 'comm':
                module_name = hierarchy.nearest_module_name
                comm_event = hierarchy.anchor
                op_name = comm_event.name

                if not comm_event.params:
                    _drops["comm_no_params"] += 1
                    _drop_by_op["comm_no_params"][op_name] += 1
                    continue

                # Parse dims from comm event's taint string
                dims_lists = comm_event.get_dims_from_taint_string(comm_event.params)
                if not dims_lists:
                    _drops["comm_no_dims"] += 1
                    _drop_by_op["comm_no_dims"][op_name] += 1
                    continue

                # Extract inputs
                try:
                    inputs = extract_inputs_from_taint(
                        taint_string=comm_event.params,
                        trace_input_dims=dims_lists,
                        trace_input_dtypes=[],
                        concrete_inputs=[]
                    )
                except Exception as e:
                    print(f"⚠ Input extraction failed for {op_name}: {e}")
                    _drops["comm_extract_failed"] += 1
                    continue

                if not inputs:
                    _drops["comm_empty_inputs"] += 1
                    _drop_by_op["comm_empty_inputs"][op_name] += 1
                    continue

                is_collective = True

            else:

                module_name = hierarchy.nearest_module_name   # root_module.name or "None"
                op_name = hierarchy.operation_name        # anchor.name
                anchor_args = hierarchy.operation_args        # anchor.args
                is_module = hierarchy.is_module

                # Compile anchors: the anchor's name is the wrapped function
                # (e.g. "layer_norm_func"), but the resolver imports modules
                # by class name AND the profiler will call the wrapping
                # nn.Module's __call__ — whose signature differs from the
                # compiled fn's (e.g. LayerNorm(hidden_states, residual) vs
                # layer_norm_func(hidden_states, weight, eps)). Switch both
                # the lookup name and the input-extraction source to the
                # nearest MODULE event so the profiler gets the right shapes
                # for the right call. The compile fn name remains accessible
                # via hierarchy.anchor.name for diagnostics.
                if hierarchy.is_compile and hierarchy.nearest_module is not None:
                    op_name = module_name
                    anchor_args = hierarchy.nearest_module.args
                    # Use the module's params as the taint string source
                    # (overrides hierarchy.get_taint_string() below).
                    _compile_taint_override = hierarchy.nearest_module.params
                else:
                    _compile_taint_override = None
                
                if len(hierarchy.comm_ops) > 0:
                    print(f"[DEBUG] Module {op_name} has {len(hierarchy.comm_ops)}, nearest_modul_name: {module_name}, comm ops: {[c.name for c in hierarchy.comm_ops]}, module_taint={hierarchy.get_taint_string()[:50] if hierarchy.get_taint_string() else None}, anchor_args={anchor_args if anchor_args else None}")


                # Non-module (cpu_op) anchored hierarchy: use trace args for input extraction
                shapes = anchor_args.get("Input Dims", [])
                dtypes = anchor_args.get("Input type", [])
                concrete_inputs = anchor_args.get("Concrete Inputs", [])

                # Get taint string from hierarchy (or from nearest MODULE event
                # for compile anchors — see comment above)
                matching_taint = _compile_taint_override or hierarchy.get_taint_string()
                
                

                # Debug module anchors with comm ops (to understand why input extraction fails)
                if hierarchy.comm_ops:
                    first_comm = hierarchy.comm_ops[0]
                    print(f"[DEBUG] Module {op_name} has comm: {first_comm.name}, module_taint={matching_taint[:50] if matching_taint else None}, shapes={shapes[:2] if shapes else []}")

                if matching_taint is None:
                    _drops["no_taint"] += 1
                    _drop_by_op["no_taint"][op_name] += 1
                    continue

                # Try taint-based extraction
                inputs = []
                if matching_taint:
                    try:
                        inputs = extract_inputs_from_taint(
                            taint_string=matching_taint,
                            trace_input_dims=shapes,
                            trace_input_dtypes=dtypes,
                            concrete_inputs=concrete_inputs
                        )
                    except Exception as e:
                        print(f"⚠ Taint-based extraction failed for {op_name}: {e}")
                        _drops["taint_extract_failed"] += 1
                        _drop_by_op["taint_extract_failed"][op_name] += 1
                        inputs = []

                # Skip if no inputs extracted
                if not inputs:
                    if not hierarchy.comm_ops:
                        _drops["no_inputs"] += 1
                        _drop_by_op["no_inputs"][op_name] += 1
                    continue

            # Create ModuleInfo with hierarchy reference for full ancestry navigation.
            # state_hash comes from the MODULE annotation's STATE: trailer
            # (captured by _module_call_wrapper in hooks.py, stashed on the
            # event's args by parser.py). Only populated for modules — raw
            # ops don't carry it.
            state_hash = ""
            if is_module:
                # hierarchy.operation_args is the anchor event's args dict;
                # the parser put 'state_hash' there when it stripped STATE:
                # off the MODULE annotation name.
                state_hash = (hierarchy.operation_args or {}).get("state_hash", "") or ""
            info = ModuleInfo(
                module_name=module_name,
                operation_name=op_name,
                inputs=inputs,
                backend=backend,
                dtype=_derive_op_dtype(inputs, fallback_dtype),
                is_module=is_module,
                is_collective_op=is_collective,
                hierarchy=hierarchy,
                state_hash=state_hash,
            )

            # Add module_info to anchor event
            hierarchy.anchor.module_info = info
            _accepted_by_op[op_name] += 1

            module_infos.append(info)

        # --- DEBUG-TRACE summary ---
        print(f"[DEBUG-TRACE] total_hierarchies={len(operation_hierarchies)}  "
              f"accepted={sum(_accepted_by_op.values())}  "
              f"drops_total={sum(_drops.values())}")
        print(f"[DEBUG-TRACE] drops by reason: {dict(_drops)}")
        print(f"[DEBUG-TRACE] accepted by op (top 12):")
        for n, c in _accepted_by_op.most_common(12):
            print(f"    {c:5d}  {n}")
        for reason, tally in _drop_by_op.items():
            if not tally:
                continue
            print(f"[DEBUG-TRACE] dropped ({reason}) top 12:")
            for n, c in tally.most_common(12):
                print(f"    {c:5d}  {n}")
        # --- END DEBUG-TRACE ---

        # print(f"Processed {comm_count} communication operations")
        print(f"Deduplicating module infos...")

        # for module in module_infos:
        #     if module.is_collective_op:
        #         print(f"Module: {module.module_name}, Operation: {module.operation_name}, Inputs: {len(module.inputs)}")
        #         print(f"  - Count: {module.count}")

        dedup_module_infos = self._deduplicate_module_infos(module_infos)

        # A MIX dimension without components cannot follow the sweep: the
        # generator would hold it at its traced size while the NUM_TOKS dims
        # move, and the op would run on inconsistent shapes. Say so loudly.
        for info in dedup_module_infos:
            for i, inp in enumerate(info.inputs):
                if inp.type != "Tensor":
                    continue
                for j in inp.get_mix_dims():
                    comps = inp.dimension_components[j] if inp.dimension_components else None
                    if not comps:
                        print(f"[TraceParser] WARNING: {info.module_name}/{info.operation_name} "
                              f"input{i} dim{j} is MIX({inp.actual_shape[j]}) with no components; "
                              f"it will stay fixed during sweeps", flush=True)

        # for module in dedup_module_infos:
        #     if module.is_collective_op:
        #         print(f"Module: {module.module_name}, Operation: {module.operation_name}, Inputs: {len(module.inputs)}")
        #         print(f"  - Count: {module.count}")


        print(f"\nDeduplicated module infos:")
        for module in dedup_module_infos:
            # Create a more descriptive signature showing key differentiators
            signature_parts = [f"Module: {module.module_name}", f"Operation: {module.operation_name}"]

            # Add input count
            signature_parts.append(f"Inputs: {len(module.inputs)}")

            # For operations with multiple variants, show distinguishing features
            if len(module.inputs) > 0:
                # Show first input dtype and shape (most distinctive)
                first_input = module.inputs[0]
                if first_input.type == "Tensor" and first_input.actual_shape:
                    # Show shape with dimension types
                    shape_str = "x".join(str(s) for s in first_input.actual_shape)
                    signature_parts.append(f"shape=[{shape_str}]")
                    signature_parts.append(f"dtype={first_input.dtype}")

                # Show scalar parameters if present (e.g., for aten::to dtype conversion)
                scalar_params = [inp.scalar_value for inp in module.inputs[1:]
                                if inp.type == "Scalar" and inp.scalar_value and inp.scalar_value not in ('', 'False', 'True')]
                if scalar_params:
                    signature_parts.append(f"params={scalar_params[0]}")

            print(", ".join(signature_parts))

        return dedup_module_infos

    @staticmethod
    def _index_python_events(trace_events: List[Dict]) -> Dict[tuple, List[Dict[str, Any]]]:
        python_events_by_thread: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for event in trace_events:
            if event.get("cat") != "python_function":
                continue
            ts = event.get("ts")
            dur = event.get("dur")
            if ts is None or dur is None:
                continue
            pid = event.get("pid")
            tid = event.get("tid")
            python_events_by_thread[(pid, tid)].append(
                {
                    "ts": ts,
                    "dur": dur,
                    "name": event.get("name", ""),
                }
            )

        for key in python_events_by_thread:
            python_events_by_thread[key].sort(key=lambda e: (e["ts"], -e["dur"]))

        return python_events_by_thread

    @staticmethod
    def _call_chain_for_op(
        op_event: Dict[str, Any],
        python_events_by_thread: Dict[tuple, List[Dict[str, Any]]],
    ) -> List[str]:
        ts = op_event.get("ts")
        dur = op_event.get("dur")
        if ts is None or dur is None:
            return []
        te = ts + dur
        key = (op_event.get("pid"), op_event.get("tid"))
        chain = []

        for event in python_events_by_thread.get(key, []):
            ev_start = event["ts"]
            ev_end = ev_start + event["dur"]
            if ev_start <= ts and ev_end >= te:
                name = event["name"]
                if not name.startswith("nn.Module:"):
                    chain.append(name)

        return chain

    @staticmethod
    def _deduplicate_module_infos(module_infos: List[ModuleInfo]) -> List[ModuleInfo]:
        """
        Deduplicate module infos by grouping modules with same base name and configuration.

        For example:
        - QKVParallelLinear_0..31 with same operation and model params → QKVParallelLinear
        - SiluAndMul_0..31 with same config → SiluAndMul
        - aten::linear and aten::matmul with same params → single entry (prefer linear)
        - But QKVParallelLinear with different params are kept separate
        """
        groups: Dict[Any, List[ModuleInfo]] = defaultdict(list)

        for info in module_infos:
            module_name = info.module_name
            if "_" in module_name and module_name.rsplit("_", 1)[1].isdigit():
                base_name = module_name.rsplit("_", 1)[0]
            else:
                base_name = module_name

            # Use existing method instead of re-extracting MODEL_CONFIG params
            model_params_list = info._semantic_params_list()
            model_params_tuple = tuple(sorted(model_params_list[0].items())) if model_params_list else ()

            nearest_module = info.hierarchy.nearest_module if info.hierarchy else None

            # Module-state discriminator: instances of one class that differ in
            # configuration (sliding window vs full attention, KV-cache dtype,
            # attention backend ...) land in separate groups. GPU kernel names
            # are deliberately NOT part of the key: they vary with the traced
            # token count, and the state hash already separates every variant
            # they used to (audited 2026-10-29). Must match the signature hash
            # fields in ModuleComparator._module_signature.
            #
            # For MODULE anchors (is_module=True), use the module's own hash.
            # For raw-op anchors (e.g. vllm::unified_kv_cache_update), inherit
            # from the nearest containing module so two raw ops living under
            # differently-configured parents land in separate groups.
            own_state_hash = info.state_hash or ""
            parent_state_hash = ""
            if nearest_module is not None:
                parent_args = getattr(nearest_module, 'args', {}) or {}
                parent_state_hash = parent_args.get('state_hash', '') or ''
            state_hash = own_state_hash or parent_state_hash

            signature = (
                base_name,
                info.operation_name,
                model_params_tuple,
                state_hash,
                info.dtype,
            )

            # dictionary of lists that group module infos by signature
            groups[signature].append(info)

        deduplicated: List[ModuleInfo] = []
        for (
            base_name,
            _op_name,
            _model_params_tuple,
            _state_hash,
            _dtype,
        ), infos in groups.items():
            # use the first info as template since all have same signature
            template = infos[0]
            count = len(infos)

            print(
                f"[DEBUG] Deduplicating {count} modules with signature: {base_name}, "
                f"operation: {template.operation_name}, model_params: {_model_params_tuple}, "
                f"state: {_state_hash or '-'}"
            )

            dedup_info = ModuleInfo(
                module_name=base_name,
                operation_name=template.operation_name,
                inputs=template.inputs,
                is_module=template.is_module,
                backend=template.backend,
                dtype=template.dtype,
                call_chain=template.call_chain,
                hierarchy=template.hierarchy,
                is_collective_op=template.is_collective_op,
                count = count,  # track how many were deduplicated into this one
                # State hash carries over from template; members of a dedup
                # group share the same module-instance state by construction
                # since state_hash participates in the grouping key.
                state_hash=template.state_hash,
            )

            # Add module_info to anchor event
            dedup_info.hierarchy.anchor.module_info = dedup_info
            deduplicated.append(dedup_info)
        return deduplicated

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--trace-dir", type=str, help="Directory of trace")
    args = parser.parse_args()
    
    trace_parser = TraceParser()

    print(f"Loading trace from {args.trace_dir}...") 
    trace_data = trace_parser.loader.load_trace(args.trace_dir)
    
    print(f"Getting trace events...")
    trace_events = trace_parser.loader.get_trace_events(trace_data) 
    print(f"Number of trace events: {len(trace_events)}") 

    print(f"Extracting module infos from trace...")
    used_ops = trace_parser.extract_module_infos(trace_events)
    
    for op in used_ops:
        print(op)
