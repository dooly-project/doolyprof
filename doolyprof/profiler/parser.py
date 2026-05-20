"""
Hierarchical trace parser that extracts Module→CPU_OP→Kernel hierarchies.

Builds a complete hierarchy of nn.Module events, CPU operations,
and GPU kernels using temporal containment and External ID linkage.

Strategy:
1. Parse all events: modules (nn.Module), cpu_ops, kernels, taints (user_annotation)
2. For each kernel, find the leaf CPU_OP that launches it (via External ID)
3. Find the highest-level CPU_OP that contains the leaf CPU_OP
4. Find the innermost (last) nn.Module that contains the highest CPU_OP
5. Find associated taint events that overlap with the operation
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# choose which operation to import / profile
# use_highest = True: use highest_cpu_op
# use_highest = False: use leaf_cpu_op that calls Kernel directly
use_highest = False

def prune_non_kernel_paths(event):
    """
    Recursively check if event has kernel descendants.
    Returns True if this branch should be kept (contains kernel).
    Prunes children that don't lead to a kernel.
    """
    # 1. Base case: Is this a kernel?
    if event.cat == 'kernel':
        return True
    
    # 2. Recursive step: Check children
    # keys_to_remove = []
    # relevant_children = {}
    has_relevant_child = False
    
    # We need to iterate over a copy of keys or items to modify dict
    # But wait, logic is: Only keep children that return True
    
    relevant_child_keys = []
    
    for key, child in event.children.items():
        if prune_non_kernel_paths(child):
            relevant_child_keys.append(key)
            has_relevant_child = True
            
    # Remove irrelevant children
    # We can rebuild the dict to be safe
    new_children = {k: event.children[k] for k in relevant_child_keys}
    event.children = new_children
    
    # 3. Return True if we have relevant children OR if we are a Module
    # (Sometimes users want to see Modules even if empty? 
    #  user said "those that call kernels; not all side operations")
    #  Strict interpretation: Only ancestors of kernels.
    
    return has_relevant_child


@dataclass(eq=False)
class Event:
    """Represents a trace event with full metadata"""
    cat: str                            # category: 'module', 'cpu_op', 'kernel', 'taint'
    name: str
    ts: float                           # timestamp
    dur: float                          # duration
    ev_idx: Optional[int]               # Event index
    external_id: Optional[int]          # External ID
    python_id: Optional[int]            # Python ID (for python_function)
    python_parent_id: Optional[int]     # Python parent ID
    args: Dict[str, Any] = field(default_factory=dict)  # Full args dict
    params: Optional[str] = None         # parameter for the event if it is a taint (None if not a taint)
    pid: Optional[int] = None           # Process ID
    tid: Optional[int] = None           # Thread ID
    module_info = None                  # Module info retrieval handle

    # Hierarchy
    children: Dict[int, 'Event'] = field(default_factory=dict, repr=False)
    parent: Optional['Event'] = field(default=None, repr=False)

    @property
    def end_ts(self):

        """End timestamp"""
        return self.ts + self.dur

    def contains(self, other: 'Event') -> bool:
        """Check if this event temporally contains another event"""
        return self.ts <= other.ts and self.end_ts >= other.end_ts

    def __repr__(self):
        return f"Event(cat={self.cat}, name={self.name[:50]}, params={self.params}, args={self.args})"
    
    def print_tree(self, indent=0):
        prefix = "  " * indent
        print(f"{prefix}{self}")
        for child in self.children.values():
            child.print_tree(indent + 2)

    def traverse_up(self):
        curr = self
        orig = self.name
        while curr.parent:
            print(f"{orig} -> {curr.name}")
            curr = curr.parent

    def get_dims_from_taint_string(self, taint_string: str) -> Optional[List[List[int]]]:
        """Get the dimensions from the taint string.

        Parses strings like:
            [[?(7), MODEL_CONFIG(32), MODEL_CONFIG(128)] | [?(7), MODEL_CONFIG(8), MODEL_CONFIG(128)]]

        Returns:
            List of dimension lists, e.g. [[7, 32, 128], [7, 8, 128]]
        """
        import re
        if not taint_string:
            return None
        s = taint_string.strip()
       
        # Strip outer [[ and ]]
        if s.startswith('[[') and s.endswith(']]'):
            s = s[1:-1]  # Remove outer [ and ], leaving [dim1, ...] | [dim2, ...]

        # Split by ' | ' to get individual tensor dimension strings
        tensor_strs = s.split(' | ')

        result = []
        for tensor_str in tensor_strs:
            # Extract integers from ?(N) or MODEL_CONFIG(N) patterns
            dims = re.findall(r'(?:\?|NUM_TOKS|NUM_REQS|MODEL_CONFIG)\((\d+)\)', tensor_str)
            result.append([int(d) for d in dims])

        return result

@dataclass
class OperationHierarchy:
    """
    Captures the full hierarchy for a profiling operation (anchor).

    This dataclass makes the kernel→operation→module relationships explicit:
    - anchor: The profiling anchor (nearest tainted ancestor of kernels)
    - root_module: Top-level (outermost) nn.Module containing the anchor
    - nearest_module: Nearest nn.Module parent of the anchor
    - kernels: GPU kernels that use this anchor
    - taints: Taint events associated with the anchor

    Navigation:
    - From kernel to anchor: kernel in self.kernels → self.anchor
    - From anchor to module: self.anchor.parent (traverse until cat='module')
    - Full traversal: use self.anchor.parent chain
    """
    anchor: Event                              # Nearest tainted ancestor (profiling target)
    root_module: Optional[Event]               # Top-level (outermost) nn.Module
    nearest_module: Optional[Event]            # Nearest nn.Module parent of anchor
    kernels: List[Event] = field(default_factory=list)  # Kernels using this anchor
    taints: List[Event] = field(default_factory=list)   # Taint events for the anchor
    comm_ops: List[Event] = field(default_factory=list)  # Communication ops associated with this anchor

    @property
    def module_name(self) -> str:
        """Get the root module name (for compatibility with existing code)."""
        if self.root_module:
            return self.root_module.name
        return "None"

    @property
    def operation_name(self) -> str:
        """Get the anchor operation name."""
        return self.anchor.name

    @property
    def operation_args(self) -> Dict[str, Any]:
        """Get the anchor operation args."""
        return self.anchor.args

    @property
    def nearest_module_name(self) -> str:
        """Get the nearest module name."""
        if self.nearest_module:
            return self.nearest_module.name
        return "None"

    @property
    def is_module(self) -> bool:
        """Check if this anchor should be resolved as an nn.Module callable.

        Includes 'compile' anchors — they don't *are* modules, but the
        resolver needs to look up their containing nn.Module (the class that
        holds the @torch.compile-decorated function) and call its forward,
        which internally invokes the real compiled fn.
        """
        return self.anchor.cat in ('module', 'compile')

    @property
    def is_compile(self) -> bool:
        """True if the anchor is a torch.compile scope (cat='compile')."""
        return self.anchor.cat == 'compile'

    def get_taint_string(self) -> Optional[str]:
        """Get the first taint's full name with params (for input extraction)."""
        return self.anchor.params

    def kernel_for_module(self, module: Event) -> List[Event]:
        """Get kernels whose ancestry includes the given module."""
        result = []
        for kernel in self.kernels:
            current = kernel.parent
            while current:
                if current is module:
                    result.append(kernel)
                    break
                current = current.parent
        return result
class HierarchicalTraceParser:
    """Extended parser that builds Module→CPU_OP→Kernel hierarchy"""

    def __init__(self, trace_path: Optional[str] = None, events: Optional[List[Dict]] = None):
        """
        Initialize parser from either a file path or events list.

        Args:
            trace_path: Path to trace JSON file
            events: List of trace events (alternative to trace_path)
        """
        self.trace_path = trace_path
        self.raw_events = events
        self.events: List[Event] = []
        self.modules: List[Event] = []      # nn.Module events
        self.cpu_ops: List[Event] = []      # CPU operation events
        self.kernels: List[Event] = []      # GPU kernel events
        self.taints: List[Event] = []       # user_annotation events
        self.comm_ops: List[Event] = []     # Communication operations (separate from hierarchy)
        self.external_id_to_events: Dict[int, List[Event]] = {}

    def parse(self):
        """Parse from file"""
        if self.trace_path is None:
            raise ValueError("No trace_path provided")

        with open(self.trace_path, 'r') as f:
            data = json.load(f)

        self._parse_events(data['traceEvents'])

    def parse_from_events(self):
        """Parse from events list"""
        if self.raw_events is None:
            raise ValueError("No events provided")

        self._parse_events(self.raw_events)

    def build_tree(self, enable_pruning: bool = True):
        """
        Build the parent-child relationships between events.
        1. Thread-local hierarchy based on timestamps (Stack).
        2. CPU->GPU linkage based on External ID.

        Args:
            enable_pruning: If True, prune non-kernel paths (expensive).
                           Set to False for 50-80% speedup if pruning not needed.
        """
        print("Building event tree...")

        # 1. Group by Thread
        thread_events = defaultdict(list)
        for event in self.events:
            # We only stack CPU-side events (modules, compile-scopes, ops, comm)
            # Taints are excluded as per user request
            # Kernels are on GPU/different thread, handled separately
            if event.cat in ('module', 'compile', 'cpu_op', 'comm'):
            # if event.cat in ('module', 'cpu_op'):
                # Use a combined key for pid/tid
                key = (event.pid, event.tid)
                thread_events[key].append(event)

        # Track roots during building
        roots = []

        # 2. Build hierarchy per thread
        for key, events in thread_events.items():
            # Sort by start time. If tied, assume longer duration (parent) comes first
            events.sort(key=lambda x: (x.ts, -x.dur))

            stack: List[Event] = []

            for event in events:
                # Pop finished events
                while stack and event.ts >= stack[-1].end_ts:
                    stack.pop()

                if stack:
                    parent = stack[-1]
                    # Link
                    event.parent = parent
                    # Use ev_idx as key if available, else standard hash/id
                    key = event.ev_idx if event.ev_idx is not None else id(event)
                    parent.children[key] = event
                else:
                    # No parent = root
                    roots.append(event)

                # Push this event as potential parent
                stack.append(event)

        # 3. Pre-build cpu_op index by external_id for faster kernel linking
        cpu_op_by_external_id = defaultdict(list)
        for cpu_op in self.cpu_ops:
            if cpu_op.external_id is not None:
                cpu_op_by_external_id[cpu_op.external_id].append(cpu_op)

        # 4. Link GPU Kernels to CPU Ops using index
        for kernel in self.kernels:
            if kernel.external_id is None:
                continue

            cpu_ops = cpu_op_by_external_id.get(kernel.external_id)
            if not cpu_ops:
                continue

            # Pick parent op
            if len(cpu_ops) == 1:
                parent_op = cpu_ops[0]
            else:
                # Tie-breaking: prefer one with ev_idx
                parent_op = next((c for c in cpu_ops if c.ev_idx is not None), cpu_ops[0])

            # Link
            kernel.parent = parent_op
            key = kernel.ev_idx if kernel.ev_idx is not None else id(kernel)
            parent_op.children[key] = kernel

        print("Building event tree complete.")

        # 5. Prune the tree to only keep paths leading to kernels (OPTIONAL)
        if enable_pruning:
            print("Pruning non-kernel paths...")
            self._prune_non_kernel_paths_optimized()
            print("Pruning complete.")
        else:
            print("Skipping pruning (disabled for performance).")

    def _prune_non_kernel_paths_optimized(self):
        """Optimized pruning using two-pass marking instead of recursive rebuild."""
        # Pass 1: Mark all ancestors of kernels AND comm ops (bottom-up)
        has_kernel_descendant = set()

        # Mark kernel ancestry
        for kernel in self.kernels:
            current = kernel
            while current:
                event_id = id(current)
                if event_id in has_kernel_descendant:
                    break  # Already marked, ancestors are marked too
                has_kernel_descendant.add(event_id)
                current = current.parent

        # Mark comm op ancestry (so comm ops survive pruning)
        for comm in self.comm_ops:
            current = comm
            while current:
                event_id = id(current)
                if event_id in has_kernel_descendant:
                    break  # Already marked, ancestors are marked too
                has_kernel_descendant.add(event_id)
                current = current.parent

        # Pass 2: Prune children that aren't marked (single pass, in-place)
        for event in self.events:
            if event.cat in ('module', 'compile', 'cpu_op', 'comm'):
                # Filter children in-place (only one dict rebuild per node)
                event.children = {
                    k: v for k, v in event.children.items()
                    if id(v) in has_kernel_descendant
                }

    def _parse_events(self, trace_events: List[Dict]):
        """Common parsing logic for both file and events input"""

        print(f"Building event tree...")

        for event_data in trace_events:
            # Skip events without timestamp or duration
            if 'ts' not in event_data or 'dur' not in event_data:
                continue

            event_param = None
            cat = event_data.get('cat', '')
            name = event_data.get('name', '')
            args = event_data.get('args', {})
            # Determine event category
            if cat == 'user_annotation' and name.startswith('MODULE:'):
                # Module scope annotation emitted by _module_call_wrapper in hooks.py.
                # Replaces python_function nn.Module: events when with_stack=False.
                # Name format:
                #   "MODULE: ClassName IN:[...] DTYPES:[...] KWARGS:[...] STATE:<hex>"
                # STATE:<hex> is an optional trailer that hashes the module
                # instance's primitive attrs (e.g. sliding_window). We pull it
                # off first so it doesn't leak into downstream IN/DTYPES/KWARGS
                # slices, and stash it on the event's args for the signature
                # builder to consume.
                if ' STATE:' in name:
                    name, _, _state_part = name.rpartition(' STATE:')
                    args['state_hash'] = _state_part.strip()
                clean_name = name.split(' IN:')[0].replace('MODULE: ', '').strip()
                event_cat = 'module'
                event_name = clean_name
                if ' IN:' in name:
                    # Extract everything between IN: and DTYPES: (or end if no DTYPES)
                    # Preserve KWARGS section which comes after DTYPES
                    in_section = name.split(' IN:')[1]
                    # Stop at DTYPES if present, but preserve KWARGS
                    if ' DTYPES:' in in_section:
                        # Get IN section (before DTYPES)
                        in_part = in_section.split(' DTYPES:')[0]
                        # Get KWARGS section if present (after DTYPES)
                        dtypes_and_after = in_section.split(' DTYPES:')[1]
                        if ' KWARGS:' in dtypes_and_after:
                            kwargs_part = ' KWARGS:' + dtypes_and_after.split(' KWARGS:')[1]
                            event_param = in_part + kwargs_part
                        else:
                            event_param = in_part
                    else:
                        event_param = in_section
                if ' DTYPES:' in name:
                    dtypes_section = name.split(' DTYPES:')[1]
                    # Extract just the DTYPES part (before KWARGS if present)
                    if ' KWARGS:' in dtypes_section:
                        dtypes_section = dtypes_section.split(' KWARGS:')[0]
                    args['Input type'] = list(
                        dtypes_section.replace('[', '').replace(']', '').split(', ')
                    )
            elif cat == 'user_annotation' and name.startswith('COMPILE:'):
                # torch.compile-decorated function scope. Emitted by
                # _make_taint_aware_compiled in hooks.py wrapping the call to
                # the original Python fn under _DisableTorchDispatch — so its
                # internal aten ops have no per-op taint and would otherwise
                # leave their kernels orphaned. Treat the COMPILE event itself
                # as a self-tainted anchor (cat='compile', stacked into the
                # tree) so kernel walks land here.
                # Name format: "COMPILE: <fn_name> IN:[...] DTYPES:[...]"
                clean_name = name.split(' IN:')[0].replace('COMPILE: ', '').strip()
                event_cat = 'compile'
                event_name = clean_name
                if ' IN:' in name:
                    in_section = name.split(' IN:')[1]
                    if ' DTYPES:' in in_section:
                        event_param = in_section.split(' DTYPES:')[0]
                    else:
                        event_param = in_section
                if ' DTYPES:' in name:
                    dtypes_section = name.split(' DTYPES:')[1]
                    args['Input type'] = list(
                        dtypes_section.replace('[', '').replace(']', '').split(', ')
                    )
            elif cat == 'cpu_op':
                # CPU operation
                event_cat = 'cpu_op'
                event_name = name
            elif cat == 'user_annotation' and name.startswith('COMM:'):
                # Communication operation annotation - handle separately
                # Name format: "COMM: vllm::all_reduce IN:[...] ARGS:[...]"
                event_cat = 'comm'
                event_name = name.split(' IN:')[0].replace('COMM: ', '').strip()
                if ' IN:' in name:
                    # Extract IN section (tensor shapes)
                    in_section = name.split(' IN:')[1]
                    if ' ARGS:' in in_section:
                        event_param = in_section.split(' ARGS:')[0]
                        # Extract ARGS section and store in args dict
                        args_section = in_section.split(' ARGS:')[1]
                        args['comm_args'] = args_section.strip('[]')
                    else:
                        event_param = in_section

            elif cat == 'user_annotation':
                # Taint/user annotation: API:, aten::, etc. (not MODULE: or COMM:)
                event_cat = 'taint'
                event_name = name
                if " IN" in name:
                    event_name = str(name.split(" IN:")[0])
                    event_param = str(name.split(" IN:")[1])
                # this makes the structure of Module and Op Events the same
                if " DTYPES:" in name:
                    args['Input type'] = list(name.split(" DTYPES:")[1].replace("[","").replace("]","").split(", "))
            elif cat == 'kernel':
                # GPU kernel
                event_cat = 'kernel'
                event_name = name
            else:
                # Skip other event types (python_function, gpu_user_annotation, etc.)
                continue

            event = Event(
                cat=event_cat,
                name=event_name,
                params=event_param,
                ts=event_data['ts'],
                dur=event_data['dur'],
                ev_idx=args.get('Ev Idx'),
                external_id=args.get('External id'),
                python_id=args.get('Python id'),
                python_parent_id=args.get('Python parent id'),
                args=args,
                pid=event_data.get('pid'),
                tid=event_data.get('tid')
            )

            self.events.append(event)

            # Categorize events
            if event_cat == 'module':
                # Populate Input Dims from the MODULE annotation's own IN
                # section. Without this, modules with no separately-emitted
                # matching taint event (e.g. modules above a torch.compile
                # scope where __torch_dispatch__ is bypassed) have no shape
                # info — the trace.py compile-anchor handler reads from this.
                if event_param:
                    event.args['Input Dims'] = event.get_dims_from_taint_string(event_param)
                self.modules.append(event)
            elif event_cat == 'compile':
                # Compile events are self-tainted: populate Input Dims from
                # their own params now so the trace.py anchor handler can read
                # shapes the same way it does for cpu_op anchors.
                if event_param:
                    event.args['Input Dims'] = event.get_dims_from_taint_string(event_param)
            elif event_cat == 'comm':
                # print(f"[DEBUG] Adding comm event: {event}")
                self.comm_ops.append(event)
            elif event_cat == 'cpu_op':
                self.cpu_ops.append(event)
            elif event_cat == 'kernel':
                self.kernels.append(event)
            elif event_cat == 'taint':
                self.taints.append(event)

            # Build external ID mapping
            if event.external_id is not None:
                if event.external_id not in self.external_id_to_events:
                    self.external_id_to_events[event.external_id] = []
                self.external_id_to_events[event.external_id].append(event)
        
        # After parsing all, build the tree by filling in children and parent dict
        self.build_tree()

        print("Finding taint association...")

        # IMPORTANT: find associated taints and add to events
        # this links taints to events and can be looked up later by resolver
        tainted_categories = set()
        for event in self.events:
            if event.cat != 'taint': # only link taints to computation events
                associated_taints = self.find_associated_taints(event)
                if associated_taints:
                    tainted_categories.add(event.cat)
                assert(len(associated_taints) <= 1)
                if len(associated_taints) == 1:
                    taint = associated_taints[0]
                    if 'DTYPES:' in taint.params:
                        event.params = taint.params.split(' DTYPES:')[0]
                        event.args['Input Dims'] = event.get_dims_from_taint_string(taint.params)
                    else:
                        event.params = taint.params

                    if 'Input type' in taint.args and event.cat == 'module':
                        event.args['Input type'] = taint.args['Input type']

                    # For cpu_ops: if the best taint is a more specific aten overload
                    # (e.g. "aten::mean.dim" vs the cpu_op's "aten::mean"), promote
                    # that name so ModuleInfo.operation_name carries the exact overload.
                    if event.cat == 'cpu_op' and '::' in taint.name and '.' in taint.name.split('::')[-1]:
                        event.name = taint.name
        print(f"Taint association done. Tainted categories: {tainted_categories}")

        print("Finding taints done.")

    def find_associated_taints(self, event: Event) -> List[Event]:
        """
        Find taint events that match this CPU_OP or Module, preferring most specific.

        When both API: XXX (from __torch_function__) and aten::XXX (from __torch_dispatch__)
        taints exist, prefer the aten:: one (smaller duration, more precise timing).

        For modules, matches MODULE: <ClassName> taints.

        Returns:
            List containing the single best matching taint Event (shortest duration)
        """
        # Extract base name from event
        # For cpu_ops: "linear" from "aten::linear" or "_C::rms_norm"
        # For modules: "QKVParallelLinear_0" stays as is
        if event.cat == 'module':
            # Module names may have _N suffix (e.g., "QKVParallelLinear_0")
            # Strip the suffix to get the class name for matching
            event_base = event.name.rsplit('_', 1)[0] if '_' in event.name and event.name.rsplit('_', 1)[-1].isdigit() else event.name
        else:
            event_base = event.name.split("::")[-1]
            
        # if (event.cat == 'comm'):
        #     print(f"event_base: {event_base}, event.name: {event.name}, event.params: {event.params}")

        # Find the single best taint (shortest duration) that matches by name
        best_taint = None

        for taint in self.taints:
            # Check temporal overlap
            if not (taint.ts <= event.end_ts and taint.end_ts >= event.ts):
                continue

            # Extract base name from taint
            # Handles: "API: linear", "aten::linear", "MODULE: QKVParallelLinear", "aten::index.Tensor"
            taint_name = taint.name
            if taint_name.startswith("MODULE: "):
                taint_base = taint_name.replace("MODULE: ", "")
            elif taint_name.startswith("API: "):
                taint_base = taint_name.replace("API: ", "")
            elif taint_name.startswith("COMM: "):
                taint_base = taint_name.replace("COMM: ", "")
            else:
                taint_base = taint_name.split("::")[-1]

            # 1. Exact base name match
            if taint_base == event_base:
                pass
            # 2. Check for overload suffix (e.g. "index.Tensor" matches "index")
            elif taint_base.startswith(event_base + "."):
                pass
            else:
                continue

            # Keep the shortest duration taint
            if best_taint is None or taint.dur < best_taint.dur:
                best_taint = taint

        return [best_taint] if best_taint else []

    def find_nearest_tainted_ancestor(self, event: Event) -> Optional[Event]:
        """
        Walk up the parent hierarchy to find the nearest ancestor that has associated taints.

        Args:
            event: The starting event (usually a kernel or leaf CPU op)

        Returns:
            The nearest tainted ancestor Event, or None if no tainted ancestor is found.
        """
        current = event
        # print(f"DEBUG: Searching ancestor for {event.name}")
        while current is not None:
            # 'compile' scopes (from torch.compile-decorated fns wrapped in
            # hooks.py) are self-tainted: their own params hold the taint
            # signature, no separate taint event is emitted by design (the
            # whole point is to skip per-op dispatch inside the compiled fn).
            if current.cat == 'compile' and current.params:
                return current
            # We treat 'cpu_op' and 'module' as valid candidates for having taints.
            if current.cat in ('cpu_op', 'module'):
                taints = self.find_associated_taints(current)
                if taints:
                    # print(f"DEBUG: Found tainted ancestor {current.name} for {event.name}")
                    return current

            # Move up
            current = current.parent

        return None

    def get_profiling_ops(self) -> List[Dict]:
        """
        Identify the unique set of operations that should be profiled based on kernel ancestry.
        
        Strategy:
        1. For each GPU kernel, walk up the hierarchy to find the nearest tainted ancestor.
        2. Collect all such unique tainted ancestors ("anchors").
        3. Deduplicate: If Anchor A is an ancestor of Anchor B, we discard B.
           (Profiling A implicitly covers B, and we want the highest-level/most-inclusive coverage).
        4. Format the remaining anchors as operation dicts for the profiler.
        
        Returns:
            List of operation dicts suitable for 'operations' field in the output.
        """
        # 1. Collect all unique anchor events
        # Map (name, ts) -> Event object to deduplicate identical events found via different kernels
        unique_anchors = {} 

        for kernel in self.kernels:
            start_node = kernel.parent
            if start_node is None:
                continue
                
            ancestor = self.find_nearest_tainted_ancestor(start_node)
            
            if ancestor:
                op_key = (ancestor.name, ancestor.ts)
                if op_key not in unique_anchors:
                    unique_anchors[op_key] = ancestor

        # 2. Deduplicate: Remove anchors that are descendants of other anchors
        # If A contains B (A is ancestor of B), and both are anchors, we keep A.
        anchor_objects = set(unique_anchors.values())
        final_anchors_list = []

        print(f"[DEBUG @ extract_module_infos] Unique anchors before deduplication: {unique_anchors}")
        print(f"[DEBUG @ extract_module_infos] Anchor objects: {anchor_objects}")

        for anchor in unique_anchors.values():
            is_redundant = False
            curr = anchor.parent
            while curr:
                # If any ancestor is also an anchor, this node is redundant
                if curr in anchor_objects:
                    is_redundant = True
                    break
                curr = curr.parent
            
            if not is_redundant:
                final_anchors_list.append(anchor)

        print(f"[DEBUG @ extract_module_infos] Found {len(final_anchors_list)} unique anchors after deduplication")
        print(f"{final_anchors_list}")

        # 3. Format valid anchors
        profiling_ops = []
        for anchor in final_anchors_list:
            # Find taints for this ancestor to include in the output
            taints = self.find_associated_taints(anchor)
            formatted_taints = []
            for t in taints:
                full_name = t.name
                if t.params:
                    full_name = f"{t.name} IN:{t.params}"
                formatted_taints.append({
                    'cat': 'taint',
                    'name': full_name,
                    'ts': t.ts,
                    'dur': t.dur
                })

            op_dict = {
                'cat': 'op', # or 'module' if it is one? Trace.py expects 'operations' list.
                'name': anchor.name,
                'ts': anchor.ts,
                'dur': anchor.dur,
                'args': anchor.args,
                'pid': anchor.pid,
                'tid': anchor.tid,
                'taints': formatted_taints
            }
            profiling_ops.append(op_dict)
        
        return profiling_ops

    def build_full_hierarchy(self) -> List[Event]:
        """
        Return top-level module Events that have children (contain operations).

        The Event tree is already built by build_tree() during parsing.
        This just filters to modules that have cpu_op children after pruning.
        """
        return [m for m in self.modules if m.children]

    def _annotate_module_descendant_kernels(self) -> None:
        """For each module event, record the set of GPU kernel names launched
        anywhere inside its forward, stored on the event as
        ``event.descendant_kernels`` (frozenset[str]).

        Implementation: walk each kernel's ``.parent`` chain once and add its
        name to every module ancestor encountered. Total work is
        O(num_kernels × average_depth); negligible next to tree build.
        """
        per_module: Dict[int, set] = defaultdict(set)
        module_refs: Dict[int, Event] = {}
        for kernel in self.kernels:
            name = kernel.name
            if not name:
                continue
            cur = kernel.parent
            while cur is not None:
                if cur.cat == 'module':
                    per_module[id(cur)].add(name)
                    module_refs[id(cur)] = cur
                cur = cur.parent
        for mid, names in per_module.items():
            module_refs[mid].descendant_kernels = frozenset(names)

    def extract_module_to_operation_mapping(self) -> List[OperationHierarchy]:
        """
        Extract module→operation mappings as OperationHierarchy dataclasses.

        Returns a list of OperationHierarchy objects, each representing a unique
        profiling anchor with its full ancestry information:
        - anchor: The nearest tainted ancestor (profiling target)
        - root_module: Top-level (outermost) nn.Module
        - nearest_module: Nearest nn.Module parent of anchor
        - kernels: GPU kernels that use this anchor
        - taints: Taint events for the anchor

        This allows traversing:
        - kernel → anchor → nearest_module → root_module
        - anchor.parent chain for full path
        """
        # Annotate every module event with the set of GPU kernel names
        # reachable from inside its subtree. Used by trace.py's dedup key so
        # CPU ops at different sibling-kernel contexts (e.g. Gemma 2 layers
        # with alternating SWA attention kernels) split correctly instead of
        # collapsing into a single group by their own direct kernel alone.
        self._annotate_module_descendant_kernels()

        # 1. Collect kernel → anchor mappings
        anchor_to_kernels: Dict[int, List[Event]] = defaultdict(list)  # id(anchor) → kernels
        anchor_events: Dict[int, Event] = {}  # id(anchor) → anchor Event
        anchor_to_collectives: Dict[int, List[Event]] = defaultdict(list)  # id(anchor) → comm ops
        
        # Note: We don't create OperationHierarchy objects for comm ops here.
        # Comm ops are processed separately in trace.py to create ModuleInfo objects.
        # This avoids creating module anchors that fail input extraction.

        # Link communication ops to anchors
        comm_module_anchors = set()
        comm_cpu_op_anchors = set()
        for comm in self.comm_ops:
            # if comm.parent is None:
            #     continue
            # if comm.parent.cat == 'module':
            #     ancestor = comm.parent
            # else:
            #     ancestor = self.find_nearest_tainted_ancestor(comm.parent)
            # for comm ops, anchor becomes itself 
            ancestor = comm 
            anchor_id = id(ancestor)
            anchor_events[anchor_id] = ancestor
            anchor_to_collectives[anchor_id].append(comm)

            # Track anchor type for comm ops
            if ancestor.cat == 'module':
                comm_module_anchors.add(anchor_id)
            elif ancestor.cat == 'cpu_op':
                comm_cpu_op_anchors.add(anchor_id)

        # Track which anchors are modules vs cpu_ops
        module_anchors = set()
        cpu_op_anchors = set()

        for kernel in self.kernels:
            start_node = kernel.parent
            if start_node is None:
                continue
            ancestor = self.find_nearest_tainted_ancestor(start_node)
            if ancestor:
                anchor_id = id(ancestor)
                anchor_events[anchor_id] = ancestor
                anchor_to_kernels[anchor_id].append(kernel)

                # Track anchor type
                if ancestor.cat == 'module':
                    module_anchors.add(anchor_id)
                elif ancestor.cat == 'cpu_op':
                    cpu_op_anchors.add(anchor_id)

        # 2. Deduplicate: Remove anchors that are descendants of other anchors
        # If A contains B (A is ancestor of B), keep only A and remap B's kernels to A
        anchor_objects = set(anchor_events.values())
        canonical_anchors: Dict[int, Event] = {}

        for anchor_id, anchor in anchor_events.items():
            is_redundant = False
            curr = anchor.parent
            while curr:
                if curr in anchor_objects:
                    is_redundant = True
                    # Remap kernels to the canonical parent anchor
                    parent_id = id(curr)
                    anchor_to_kernels[parent_id].extend(anchor_to_kernels[anchor_id])
                    break
                curr = curr.parent

            if not is_redundant:
                canonical_anchors[anchor_id] = anchor

        # --- DEBUG: expose anchor landscape ---
        from collections import Counter as _DbgCounter
        _dbg_raw = _DbgCounter(
            f"{ev.cat}:{ev.name}" for ev in anchor_events.values()
        )
        _dbg_can = _DbgCounter(
            f"{ev.cat}:{ev.name}" for ev in canonical_anchors.values()
        )
        print(f"[DEBUG-ANCHORS] raw_anchors={len(anchor_events)}, "
              f"canonical={len(canonical_anchors)}")
        print("[DEBUG-ANCHORS] raw anchor name counts (top 30):")
        for k, v in _dbg_raw.most_common(30):
            print(f"    {v:5d}  {k}")
        print("[DEBUG-ANCHORS] canonical anchor name counts:")
        for k, v in sorted(_dbg_can.items(), key=lambda kv: -kv[1]):
            print(f"    {v:5d}  {k}")

        # Report kernels whose walk never landed on a tainted ancestor
        _orphans = _DbgCounter()
        for kernel in self.kernels:
            start_node = kernel.parent
            if start_node is None:
                _orphans["<no_parent>"] += 1
                continue
            anc = self.find_nearest_tainted_ancestor(start_node)
            if anc is None:
                # walk up to find what non-tainted parent chain looks like
                chain_names = []
                cur = start_node
                while cur is not None and len(chain_names) < 6:
                    chain_names.append(f"{cur.cat}:{cur.name}")
                    cur = cur.parent
                _orphans[" -> ".join(chain_names[:3])] += 1
        print(f"[DEBUG-ORPHANS] kernels without tainted ancestor (top 15):")
        for k, v in _orphans.most_common(15):
            print(f"    {v:5d}  {k}")
        # --- END DEBUG ---

        # 3. Build OperationHierarchy for each canonical anchor
        results: List[OperationHierarchy] = []

        # print(f"[DEBUG] Building OperationHierarchy for {len(canonical_anchors)} canonical anchors")
        for anchor_id, anchor in canonical_anchors.items():
            # Find nearest module (first module in parent chain)
            nearest_module = self._find_nearest_module(anchor)

            # Find root module (outermost module in parent chain)
            root_module = self._find_root_module(anchor)

            # Get taints for this anchor
            taints = self.find_associated_taints(anchor)

            # Get kernels and comm ops using this anchor
            kernels = anchor_to_kernels[anchor_id]
            comm_ops = anchor_to_collectives.get(anchor_id, [])
 
            hierarchy = OperationHierarchy(
                anchor=anchor,
                root_module=root_module,
                nearest_module=nearest_module,
                kernels=kernels,
                taints=taints,
                comm_ops=comm_ops,
            )
            results.append(hierarchy)

        # Sort by anchor timestamp
        results.sort(key=lambda h: h.anchor.ts)

        return results

    def _find_nearest_module(self, event: Event) -> Optional[Event]:
        """Find the nearest (innermost) nn.Module parent of an event."""
        current = event
        while current:
            if current.cat == 'module':
                return current
            current = current.parent
        return None

    def _find_root_module(self, event: Event) -> Optional[Event]:
        """Find the root (outermost) nn.Module containing an event."""
        current = event
        root_module = None
        while current:
            if current.cat == 'module':
                root_module = current
            current = current.parent
        return root_module


def main():
    """Command-line interface for testing"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hierarchical_parser.py <trace.json> [output.json]")
        sys.exit(1)

    trace_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "hierarchy_output.json"

    # Parse trace
    parser = HierarchicalTraceParser(trace_path=trace_path)
    parser.parse()

    # print(f"Found {len(parser.modules)} modules, {len(parser.cpu_ops)} CPU ops, "
        #   f"{len(parser.kernels)} kernels, {len(parser.taints)} taints")

    def print_tree(event: Event, depth: int = 0):
        indent = "  " * depth
        print(f"{indent}{event.name} ({event.cat})")
        if event.cat == 'cpu_op':
            taints = parser.find_associated_taints(event)
            print(f"{indent}Taints: {[t for t in taints]}")
        # specific handling to show kernels clearly
        for child_idx, child in event.children.items():
            print_tree(child, depth + 1)

    # print("\n--- Event Hierarchy ---")
    # for event in parser.events:
    #     # Only print top-level modules (those with no parent)
    #     if event.cat == 'module' and event.parent is None:
    #         print_tree(event)


    # for event in parser.events:
    #     if event.cat == 'cpu_op':
    #         taints = parser.find_associated_taints(event)
    #         print(event.name, [t.param for t in taints])

    # Extract hierarchies
    hierarchies = parser.build_full_hierarchy()
    print(f"Built {len(hierarchies)} hierarchies")

    # --- Show Profiling Anchors (Hierarchical Taint Propagation) ---
    print("\n" + "="*80)
    print("PROFILING ANCHORS (Hierarchical Taint Propagation)")
    print("="*80)
    print("For each kernel, shows the path to its nearest tainted ancestor (the 'anchor').")
    print("Anchors are deduplicated - same anchor for multiple kernels = profiled once.\n")

    # 1. Get Canonical Anchors from deduplication logic
    profiling_ops = parser.get_profiling_ops()
    canonical_anchors = set()
    for op in profiling_ops:
        # Match by name and ts, assuming unique enough
        canonical_anchors.add((op['name'], op['ts']))

    anchor_to_kernels = {}  # Map anchor_key -> info
    kernels_without_anchor = []

    for kernel in parser.kernels:
        start_node = kernel.parent
        if start_node is None:
            kernels_without_anchor.append(kernel.name[:60])
            continue

        # Build path from kernel's parent up to the anchor
        path = []
        current = start_node
        anchor = None
        raw_anchor = None

        # Trace up to find ANY tainted ancestor first
        temp_curr = current
        while temp_curr is not None:
             if temp_curr.cat in ('cpu_op', 'module'):
                 taints = parser.find_associated_taints(temp_curr)
                 has_taint = len(taints) > 0
                 taint_str = "✓ TAINTED" if has_taint else "✗ no taint"
                 path.append(f"{temp_curr.name[:50]} ({temp_curr.cat}) [{taint_str}]")
                 
                 if has_taint and raw_anchor is None:
                     raw_anchor = temp_curr
             temp_curr = temp_curr.parent

        # Now resolve raw_anchor to a canonical anchor
        if raw_anchor:
            # Check if raw_anchor is canonical
            key = (raw_anchor.name, raw_anchor.ts)
            if key in canonical_anchors:
                anchor = raw_anchor
            else:
                # Check ancestors of raw_anchor
                curr_ancestor = raw_anchor.parent
                while curr_ancestor:
                    k = (curr_ancestor.name, curr_ancestor.ts)
                    if k in canonical_anchors:
                        anchor = curr_ancestor
                        break
                    curr_ancestor = curr_ancestor.parent
        
        if anchor:
            anchor_key = (anchor.name, anchor.ts)
            if anchor_key not in anchor_to_kernels:
                # Re-find taints for display
                taints = parser.find_associated_taints(anchor)
                taint_obj = taints[0] if taints else None
                anchor_to_kernels[anchor_key] = {
                    'anchor_name': anchor.name,
                    'anchor_cat': anchor.cat,
                    'kernels': [],
                    'path_example': path,
                    'taint': taint_obj
                }
            anchor_to_kernels[anchor_key]['kernels'].append(kernel.name[:40])
        else:
            kernels_without_anchor.append(kernel.name[:60])

    # Print summary by anchor
    print(f"Found {len(anchor_to_kernels)} unique profiling anchors for {len(parser.kernels)} kernels\n")

    for i, (anchor_key, info) in enumerate(anchor_to_kernels.items()):
        anchor_name = info['anchor_name']
        anchor_cat = info['anchor_cat']
        kernels = info['kernels']
        path = info['path_example']
        taint = info['taint']

        print(f"[Anchor {i+1}] {anchor_name} ({anchor_cat})")
        print(f"  Kernels using this anchor: {len(kernels)}")
        if len(kernels) <= 3:
            for k in kernels:
                print(f"    - {k}")
        else:
            for k in kernels[:2]:
                print(f"    - {k}")
            print(f"    ... and {len(kernels)-2} more")

        print(f"  Taint: {taint.name if taint else 'NONE'}")
        if taint and taint.params:
            print(f"  Params: {taint.params[:80]}...")
        print(f"  Path (kernel→anchor): {' → '.join(reversed(path[:3]))}")
        print()

    if kernels_without_anchor:
        print(f"\n⚠ WARNING: {len(kernels_without_anchor)} kernels have NO tainted ancestor!")
        for k in kernels_without_anchor[:5]:
            print(f"  - {k}")
        if len(kernels_without_anchor) > 5:
            print(f"  ... and {len(kernels_without_anchor)-5} more")

    print("\n" + "="*80)

    # Extract module mappings (now returns List[OperationHierarchy])
    op_hierarchies = parser.extract_module_to_operation_mapping()
    print(f"Extracted {len(op_hierarchies)} operation hierarchies")

    # Convert OperationHierarchy objects to JSON-serializable dicts
    def hierarchy_to_dict(h: OperationHierarchy) -> dict:
        return {
            'module_name': h.module_name,
            'operation_name': h.operation_name,
            'anchor': h.anchor.name,
            'root_module': h.root_module.name if h.root_module else None,
            'nearest_module': h.nearest_module.name if h.nearest_module else None,
            'kernels': [k.name for k in h.kernels],
            'taints': [t.name for t in h.taints],
        }

    # Export to JSON
    output = {
        'summary': {
            'total_events': len(parser.events),
            'modules': len(parser.modules),
            'cpu_ops': len(parser.cpu_ops),
            'kernels': len(parser.kernels),
            'taints': len(parser.taints),
            'hierarchies': len(hierarchies),
            'operation_hierarchies': len(op_hierarchies)
        },
        'operation_hierarchies': [hierarchy_to_dict(h) for h in op_hierarchies]
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Exported to {output_path}")


    for kernel in parser.kernels:
        ancestor = parser.find_nearest_tainted_ancestor(kernel.parent)
        if ancestor and 'split_with_sizes' in ancestor.name:
            print(f"Kernel: {kernel.name}")
            print(f"  Ancestor chain: ", end="")
            curr = kernel.parent
            while curr:
                print(f"{curr.name[:30]} <- ", end="")
                curr = curr.parent
            print()


if __name__ == "__main__":
    main()
