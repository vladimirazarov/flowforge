"""
FragmentForest manages a hierarchical (tree-like) collection of fragments and
generates structured code. It maintains fragment relationships (parent–child or
sequential siblings) and provides a single entry point for final code generation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterator, TYPE_CHECKING, Literal, Set, cast, Any
import random
from loguru import logger
from rich.console import Console 
from src.utils.logging_helpers import (
    print_fragment_gen_start, print_fragment_gen_details, print_fragment_gen_end,
    log_fragment_gen_if_else_map, log_fragment_gen_if_else_targets,
    log_fragment_gen_if_else_lookup, log_fragment_gen_ops
)

from src.core.fragments.fragment_types import FragmentRole
from src.core.cfg.cfg_context import CFGContext
from src.core.cfg_content.expression import Expression
from src.core.nodes.cfg_basic_block_node import CFGBasicBlockNode
from src.config.config import config
if TYPE_CHECKING:
    from src.core.fragments.cfg_fragment import Fragment
    from src.core.cfg.cfg import CFG 


__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

@dataclass
class _FragmentTreeNode:
    """Internal representation of a node in the FragmentForest tree"""
    fragment_id: int
    parent_id: Optional[int] = None
    child_ids: List[int] = field(default_factory=list) 

@dataclass
class FragmentForest:
    """
    Manages a hierarchical (multi-child tree) collection of Fragments
    Maintains ordered relationships between nested children and sequential roots/siblings

    Attributes:
        _nodes: Dictionary mapping fragment IDs to their _FragmentTreeNode
        _roots: List of root fragment IDs, maintaining insertion order
        _fragments: Dictionary mapping fragment IDs to Fragment objects
        _cfg: Optional[CFG] = None 
    """
    _nodes: Dict[int, _FragmentTreeNode] = field(default_factory=dict)
    _roots: List[int] = field(default_factory=list)
    _fragments: Dict[int, Fragment] = field(default_factory=dict)
    _cfg: Optional[CFG] = None 

    def set_cfg_reference(self, cfg: CFG) -> None:
        """Sets the reference to the main CFG object"""
        self._cfg = cfg

    def _ensure_node_exists(self, frag_id: int):
        """Creates a _FragmentTreeNode if it doesn't exist"""
        if frag_id not in self._nodes:
            self._nodes[frag_id] = _FragmentTreeNode(fragment_id=frag_id)

    def add_fragment(self, fragment: Fragment, nest: bool, origin_fragment_id: Optional[int],
                     position: Literal['before', 'after'] = 'after') -> None:
        """
        Adds a fragment relative to another, preserving order

        Args:
            fragment: The Fragment object to add
            nest: True to add as a nested child, False for sibling/root insertion
            origin_fragment_id: ID of the fragment to add relative to. If None, add as a root
            position: If nest=False or origin_fragment_id=None, specifies order ('before'/'after')
        """
        frag_id = fragment.fragment_id
        self._fragments[frag_id] = fragment
        self._ensure_node_exists(frag_id)
        new_node = self._nodes[frag_id]

        if origin_fragment_id is None:
            # Adding as a root
            if frag_id in self._roots:
                if config.developer_options.log_fragment_forest:
                    logger.warning(f"Fragment {frag_id} is already a root. Ignoring add root request.")
                return
            insert_idx = 0 if position == 'before' and self._roots else len(self._roots)
            if position == 'before' and self._roots:
                insert_idx = 0
            elif position == 'after':
                insert_idx = len(self._roots)
            else: 
                insert_idx = 0
            self._roots.insert(insert_idx, frag_id)
            new_node.parent_id = None
            if config.developer_options.log_fragment_forest:
                logger.debug(f"Added fragment {frag_id} as a root at index {insert_idx}.")
            return

        # Adding relative to an existing fragment
        self._ensure_node_exists(origin_fragment_id)
        origin_node = self._nodes[origin_fragment_id]

        if nest:
            # Nested insertion: New fragment B becomes a child of origin A.
            # Origin A is the parent
            parent_id = origin_fragment_id 
            if config.developer_options.log_fragment_forest:
                logger.debug(f"Attempting nested insertion of {frag_id}(B) as child of {origin_fragment_id}(A).")

            # Ensure parent node (origin) exists
            self._ensure_node_exists(parent_id)
            parent_node = self._nodes[parent_id]

            new_node.parent_id = parent_id

            # Add the new fragment ID to the parent's ordered child list based on position
            if position == 'before':
                parent_node.child_ids.insert(0, frag_id)
                if config.developer_options.log_fragment_forest:
                    logger.debug(f"Added {frag_id}(B) as the *first* child of {parent_id}(A). Children: {parent_node.child_ids}")
            else: # Default to 'after' or append if position is invalid
                parent_node.child_ids.append(frag_id)
                if config.developer_options.log_fragment_forest:
                    logger.debug(f"Added {frag_id}(B) as the *last* child of {parent_id}(A). Children: {parent_node.child_ids}")

        else:
            # Sequential insertion: Add as sibling before/after origin, mirroring previous logic
            target_sibling_id = origin_fragment_id
            parent_id = origin_node.parent_id

            if parent_id is not None:
                # Origin has a parent: insert into parent's ordered child list
                if parent_id not in self._nodes:
                    if config.developer_options.log_fragment_forest:
                        logger.error(f"Parent node {parent_id} not found for origin {origin_fragment_id}. Cannot insert sequentially.")
                    return 
                parent_node = self._nodes[parent_id]
                siblings = parent_node.child_ids
                try:
                    idx = siblings.index(target_sibling_id)
                    insert_idx = idx if position == 'before' else idx + 1
                    siblings.insert(insert_idx, frag_id)
                    # Explicitly set parent for the new node
                    new_node.parent_id = parent_id
                    if config.developer_options.log_fragment_forest:
                        logger.debug(f"Fragment {frag_id} added as sibling {position} {target_sibling_id} under {parent_id} at index {insert_idx}. Children: {siblings}")
                except ValueError:
                    # Fallback if target sibling isn't found
                    if config.developer_options.log_fragment_forest:
                        logger.error(f"Target sibling {target_sibling_id} not found in parent {parent_id}'s children list {siblings}. Appending {frag_id} at end.")
                    siblings.append(frag_id)
                    new_node.parent_id = parent_id 
            else:
                # Origin is a root: insert into the root list
                try:
                    root_idx = self._roots.index(target_sibling_id)
                    insert_idx = root_idx if position == 'before' else root_idx + 1
                    self._roots.insert(insert_idx, frag_id)
                    # Explicitly set parent to None for the new root node
                    new_node.parent_id = None
                    if config.developer_options.log_fragment_forest:
                        logger.debug(f"Fragment {frag_id} added as root sibling {position} {target_sibling_id} at index {insert_idx}. Roots: {self._roots}")
                except ValueError:
                    # target root isn't found
                    if config.developer_options.log_fragment_forest:
                        logger.error(f"Root sibling {target_sibling_id} not found in roots list {self._roots}. Appending {frag_id} at end.")
                    self._roots.append(frag_id)
                    new_node.parent_id = None 

    def get_fragment(self, frag_id: int) -> Optional[Fragment]:
        return self._fragments.get(frag_id)

    def get_children(self, frag_id: int) -> List[Fragment]:
        """
        Returns an ordered list of child fragments for a given fragment ID
        """
        children: List[Fragment] = []
        if frag_id in self._nodes:
            node = self._nodes[frag_id]
            for child_id in node.child_ids:
                child_frag = self.get_fragment(child_id)
                if child_frag:
                    children.append(child_frag)
        return children

    def is_root(self, frag_id: int) -> bool:
        return frag_id in self._roots

    def traverse(self) -> Iterator[Fragment]:
        """
        Performs a depth-first traversal (pre-order) of the forest
        Visits node, then traverses children in order
        """
        visited: Set[int] = set()
        for root_id in self._roots:
            if root_id not in visited:
                stack: List[int] = [root_id]
                while stack:
                    current_frag_id = stack.pop() 
                    if current_frag_id not in visited and current_frag_id in self._nodes:
                        visited.add(current_frag_id)
                        fragment = self.get_fragment(current_frag_id)
                        if fragment:
                            yield fragment 

                        # Push children onto stack in reverse order for correct DFS visit order
                        node = self._nodes[current_frag_id]
                        for child_id in reversed(node.child_ids):
                            if child_id not in visited:
                                stack.append(child_id)

    def is_descendant(self, potential_descendant_id: int, ancestor_id: int) -> bool:
        """
        Checks if potential_descendant_id is a descendant of ancestor_id by traversing up the parent links
        """
        if potential_descendant_id == ancestor_id:
            return False
        if potential_descendant_id not in self._nodes or ancestor_id not in self._nodes:
            return False

        current_id: Optional[int] = potential_descendant_id
        # Prevent cycles in parent chain
        visited_asc = {current_id} 
        while current_id is not None:
            node = self._nodes[current_id]
            parent_id = node.parent_id
            if parent_id == ancestor_id:
                return True
            # Reached a root
            if parent_id is None:
                break 
            if parent_id in visited_asc:
                if config.developer_options.log_fragment_forest:
                    logger.warning(f"Cycle detected in parent chain for fragment {potential_descendant_id} at node {parent_id}. Stopping descendant check.")
                break
            visited_asc.add(parent_id)
            current_id = parent_id
        return False

    def generate_code(self, context: Optional[CFGContext] = None) -> str:
        """
        Generates C code for all root fragments in a single function
        Dynamically determines branch successors from the CFG graph
        """
        cfg_graph = self._cfg.graph 
        if config.developer_options.log_fragment_forest:
            logger.warning("Reviewing FragmentForest.generate_code for multi-child structure.")
        # 1) one shared set for the whole codegen run
        processed_nodes: Set[CFGBasicBlockNode] = set()

        def _generate_operations_code(node: CFGBasicBlockNode, indent: str, role_label: Optional[str] = None) -> str:
            if config.developer_options.log_fragment_forest:
                log_fragment_gen_ops(getattr(node, 'node_id', None), "entry")
            # 2) skip re‑emitting ops for a node we've already handled
            if node in processed_nodes:
                if config.developer_options.log_fragment_forest:
                    logger.debug(f"Skipping ops for node {node.node_id}")
                return ""
            processed_nodes.add(node)

            ops_code = ""
            if config.developer_options.log_branch_labels_in_code and role_label:
                 ops_code += f"{indent}/* {role_label} */\n"
            if config.developer_options.log_node_ids_in_code:
                 ops_code += f"{indent}/* Node ID: {node.node_id} */\n"
            if node and node.instructions:
                for op in node.instructions.operations:
                    op_c_code = op.to_c()
                    ops_code += f"{indent}{op_c_code}\n"
            return ops_code

        def _generate_fragment_code(fragment_id: int, indent_level: int = 1) -> str:
            fragment = self._fragments[fragment_id]
            node = self._nodes[fragment_id]
            role = fragment.fragment_role
            role_name = role.name if role else "Unknown"

            console = Console()
            if config.developer_options.log_fragment_forest:
                print_fragment_gen_start(console, fragment_id, role_name, indent_level)
                print_fragment_gen_details(console, fragment)

            entry_node: Optional[CFGBasicBlockNode] = fragment.entry_node

            indent = "    " * indent_level
            code = ""

            assert entry_node is not None
            code += _generate_operations_code(entry_node, indent)

            original_expression: Optional[Expression] = None
            if entry_node.jump:
                original_expression = entry_node.jump.expression
            if config.developer_options.log_fragment_forest:
                logger.debug(f"[FragForest] Fragment {fragment_id}: original_expression = {original_expression.to_c() if original_expression else 'None'}")

            successors = list(cfg_graph.successors(entry_node)) if entry_node in cfg_graph else []
            if config.developer_options.log_fragment_forest:
                logger.debug(f"[GenCode] Fragment {fragment_id} ({role_name}) Entry {entry_node.node_id} has successors in main CFG: {[s.node_id for s in successors]}")
            
            true_successor: Optional[CFGBasicBlockNode] = None
            false_successor: Optional[CFGBasicBlockNode] = None
            body_node: Optional[CFGBasicBlockNode] = None 
            exit_node: Optional[CFGBasicBlockNode] = None 

            # merge_node is the node AFTER the branches rejoin or loop exits
            # latch node is inside the loop body, pointing back to the header
            # Identify successors based on edge labels
            if self._cfg is not None:
                for succ in successors:
                    label = self._cfg.get_edge_label(entry_node, succ)
                    if label == 'true':
                        true_successor = succ
                        if config.developer_options.log_fragment_forest:
                            logger.debug(f"  -> Identified True successor: {succ.node_id} via label '{label}'")
                    elif label == 'false':
                        false_successor = succ
                        if config.developer_options.log_fragment_forest:
                            logger.debug(f"  -> Identified False successor: {succ.node_id} via label '{label}'")
                    elif label == 'loop':
                        body_node = succ
                        if config.developer_options.log_fragment_forest:
                            logger.debug(f"  -> Identified Loop Body successor: {succ.node_id} via label '{label}'")
                    elif label == 'exit':
                        exit_node = succ
                        if config.developer_options.log_fragment_forest:
                            logger.debug(f"  -> Identified Loop Exit successor: {succ.node_id} via label '{label}'")
                    else:
                        if config.developer_options.log_fragment_forest:
                            logger.warning(f"  -> Successor {succ.node_id} has unrecognized or missing label: '{label}'")
            else:
                if config.developer_options.log_fragment_forest:
                    logger.error("FragmentForest._cfg is None, cannot get edge labels for code generation.")

            # if exactly two successors and only one was labeled, the other is false
            if true_successor and not false_successor and len(successors) == 2:
                false_successor = next(s for s in successors if s is not true_successor)
                if config.developer_options.log_fragment_forest:
                    logger.debug(f"  -> Falling back: treating {false_successor.node_id} as false‐branch")

            # Generate child fragment code first (recursive call)
            children_code_map: Dict[int, str] = {} 
            child_entry_node_map: Dict[int, Optional[CFGBasicBlockNode]] = {} 
            for child_id in node.child_ids: 
                child_frag = self._fragments.get(child_id)
                if child_frag:
                    if config.developer_options.log_fragment_forest:
                        logger.debug(f"[GenCode] Generating code for child fragment {child_id} of parent {fragment_id}")
                    children_code_map[child_id] = _generate_fragment_code(child_id, indent_level + 1)
                    child_entry_node_map[child_id] = child_frag.entry_node
                else:
                    if config.developer_options.log_fragment_forest:
                        logger.warning(f"[GenCode] Child fragment ID {child_id} not found in forest for parent {fragment_id}")

            # Match CFG successors to child fragments based on entry nodes
            successor_to_child_map: Dict[CFGBasicBlockNode, int] = {}
            child_id_to_successor_map: Dict[int, CFGBasicBlockNode] = {}
            unmatched_successors: List[CFGBasicBlockNode] = list(successors) 

            if config.developer_options.log_fragment_forest:
                logger.debug(f"[GenCode] Matching {len(successors)} successors {[s.node_id for s in successors]} to {len(child_entry_node_map)} children {[n.node_id if n else 'None' for n in child_entry_node_map.values()]}")
            for child_id, child_entry_node in child_entry_node_map.items():
                if child_entry_node is not None:
                    matched_succ = next((succ for succ in successors if succ == child_entry_node), None)
                    if matched_succ:
                        if config.developer_options.log_fragment_forest:
                            logger.debug(f"  - Matched Child {child_id} (Entry: {child_entry_node.node_id}) to Successor {matched_succ.node_id}")
                        successor_to_child_map[matched_succ] = child_id
                        child_id_to_successor_map[child_id] = matched_succ
                        if matched_succ in unmatched_successors:
                            unmatched_successors.remove(matched_succ)
                    else:
                        if config.developer_options.log_fragment_forest:
                            logger.warning(f"  - Child {child_id} (Entry: {child_entry_node.node_id}) did not match any successor of parent {fragment_id} (Entry: {entry_node.node_id})")
                else:
                    if config.developer_options.log_fragment_forest:
                        logger.warning(f"  - Child {child_id} has no entry node.")
            if config.developer_options.log_fragment_forest:
                logger.debug(f"[GenCode] Unmatched successors: {[s.node_id for s in unmatched_successors]}")

            if role == FragmentRole.IF:
                condition_str = original_expression.to_c() if original_expression else "/* condition? */"
                if config.developer_options.log_fragment_forest:
                    logger.debug(f"[FragForest] IF Fragment {fragment_id}: condition_str = {condition_str}")
                code += f"{indent}if ({condition_str}) {{\n"
                inner_indent = indent + "    "
                #code += f"{inner_indent} /*TRUE BRANCH (Label-Identified)*/\n"
                
                identified_true_child_id = successor_to_child_map.get(true_successor) if true_successor else None

                # Generate True branch code (using true_successor)
                if true_successor:
                    code += _generate_operations_code(true_successor, inner_indent, role_label="TRUE BRANCH")
                    # Append code for the matched child fragment, if any
                    if identified_true_child_id is not None and identified_true_child_id in children_code_map:
                        code += children_code_map[identified_true_child_id]
                else:
                    if config.developer_options.log_fragment_forest:
                        logger.warning(f"[GenCode] IF {fragment_id}: No 'true' labeled successor found.")
                
                code += f"{indent}}}\n"
                # Emit any nested fragments in the false path after the IF
                for child_id in node.child_ids:
                    # Skip the true-branch child 
                    if true_successor and successor_to_child_map.get(true_successor) == child_id:
                        continue
                    # Only emit if code was generated
                    if child_id in children_code_map:
                        code += children_code_map[child_id]

            elif role == FragmentRole.IF_ELSE:
                condition_str = original_expression.to_c() if original_expression else "/* condition? */"
                if config.developer_options.log_fragment_forest:
                    logger.debug(f"[FragForest] IF_ELSE Fragment {fragment_id}: condition_str = {condition_str}")
                code += f"{indent}if ({condition_str}) {{\n"
                true_indent = indent + "    "
                #code += f"{true_indent}/* TRUE BRANCH (Label-Identified) */\n"
                
                identified_true_child_id = successor_to_child_map.get(true_successor) if true_successor else None
                identified_false_child_id = successor_to_child_map.get(false_successor) if false_successor else None

                # Generate True branch code
                if true_successor:
                    code += _generate_operations_code(true_successor, true_indent, role_label="TRUE BRANCH")
                    if identified_true_child_id is not None and identified_true_child_id in children_code_map:
                        code += children_code_map[identified_true_child_id]
                else:
                    if config.developer_options.log_fragment_forest:
                        logger.warning(f"[GenCode] IF_ELSE {fragment_id}: No 'true' labeled successor found.")
                
                code += f"{indent}}} else {{\n"
                false_indent = indent + "    "
                #code += f"{false_indent}/* FALSE BRANCH (Label-Identified) */\n"

                # Generate False branch code
                if false_successor:
                    code += _generate_operations_code(false_successor, false_indent, role_label="FALSE BRANCH")
                    if identified_false_child_id is not None and identified_false_child_id in children_code_map:
                        code += children_code_map[identified_false_child_id]
                else:
                    if config.developer_options.log_fragment_forest:
                        logger.warning(f"[GenCode] IF_ELSE {fragment_id}: No 'false' labeled successor found.")

                for child_id in node.child_ids:
                    if child_id in (identified_true_child_id, identified_false_child_id):
                        continue
                    code += children_code_map.get(child_id, "")
                code += f"{indent}}}\n"

            elif role in (FragmentRole.WHILE, FragmentRole.FOR, FragmentRole.FOR_EARLY_EXIT):
                # Loop construct, print header
                condition = original_expression.to_c() if original_expression else "/* condition? */"
                if config.developer_options.log_fragment_forest:
                    logger.debug(f"[FragForest] LOOP Fragment {fragment_id}: condition = {condition}")
                if role == FragmentRole.WHILE:
                    header = f"{indent}while ({condition}) {{\n"
                else:
                    # For loops
                    loop_var_name = None
                    context_for_vars = self._cfg.context if self._cfg else None
                    existing = {v.name for v in context_for_vars.used_variables} if context_for_vars else set()
                    choices = [n for n in config.complexity_tuner.for_loop_variable_names if n not in existing]
                    loop_var_name = (random.choice(choices) if choices else f"loop_var_{fragment_id}")
                    init_part = f"int {loop_var_name} = 0"
                    incr = random.choice(["++", "--"]);
                    header = f"{indent}for ({init_part}; {condition}; {loop_var_name}{incr}) {{\n"
                code += header
                inner = indent + "    "
                # Body operations 
                if body_node: 
                    code += _generate_operations_code(body_node, inner, role_label="LOOP BODY")
                # Always include child fragment code inside loop
                for child_id in node.child_ids:
                    if child_id in children_code_map:
                        code += children_code_map[child_id]
                code += f"{indent}}}\n"

            # Basic block or other simple fragment types
            else: 
                # Basic block logic (emit children sequentially)
                if config.developer_options.log_fragment_forest:
                    logger.debug(f"[GenCode] BASIC {fragment_id}: Generating {len(children_code_map)} children sequentially.")
                for child_id in node.child_ids:
                     if child_id in children_code_map:
                         code += children_code_map[child_id]

            node_to_check: Optional[CFGBasicBlockNode] = None
            node_to_check_role: Optional[str] = None  
            if role in (FragmentRole.WHILE, FragmentRole.FOR, FragmentRole.FOR_EARLY_EXIT):
                node_to_check = exit_node # Use the loop's exit node
                if node_to_check: 
                    node_to_check_role = "LOOP EXIT" 
            elif role in (FragmentRole.IF, FragmentRole.IF_ELSE):
                # Calculate the merge node for conditionals
                unmatched = [s for s in successors if s not in (true_successor, false_successor)]
                if len(unmatched) == 1:
                    node_to_check = unmatched[0]

            if node_to_check: 
                merge_exit_id = node_to_check.node_id
                if config.developer_options.log_fragment_forest:
                    logger.debug(f"[GenCode] Checking unified merge/exit node {merge_exit_id} for return (Role: {role_name}).")
                # Emit all of the node's ops (including return)
                code += _generate_operations_code(node_to_check, indent, role_label=node_to_check_role)
            if config.developer_options.log_fragment_forest:
                print_fragment_gen_end(console, fragment_id)
            return code

        parameters_str = ""
        indent = "    "
        if context and context.used_parameters:
            param_parts = []
            sorted_params = sorted(list(context.used_parameters), key=lambda v: v.name)
            for param in sorted_params:
                param_type_val = param.type.value
                c_type = 'int' if param_type_val == 'bool' else param_type_val
                param_name = param.name
                param_parts.append(f"{c_type} {param_name}")
            parameters_str = ", ".join(param_parts)

        func_name = "foo" if parameters_str else "main"
        result = f"int {func_name}({parameters_str}) {{\n"

        # Generate code by traversing roots
        processed_root_fragments: Set[int] = set()
        for root_id in self._roots:
            if root_id not in processed_root_fragments:
                result += _generate_fragment_code(root_id, indent_level=1)
                # Mark root as processed
                processed_root_fragments.add(root_id)

        exit_node = self._cfg.exit_node
        if exit_node:
            if config.developer_options.log_fragment_forest:
                logger.debug(f"[GenCode] Appending global exit node {exit_node.node_id} operations.")
            result += "\n" + _generate_operations_code(exit_node, indent, role_label="GLOBAL EXIT") 

        result += "}\n"
        return result

    def update_fragment_related_nodes(self, deleted_node: CFGBasicBlockNode) -> None:
        """Removes references to a deleted CFG node from all fragments"""
        for fragment in self._fragments.values():
            if fragment.entry_node == deleted_node:
                fragment.entry_node = None
            if fragment.exit_node == deleted_node:
                fragment.exit_node = None
            if getattr(fragment, 'body_node', None) == deleted_node:
                fragment.body_node = None # type: ignore[assignment]
            if getattr(fragment, 'true_branch_node', None) == deleted_node:
                fragment.true_branch_node = None # type: ignore[assignment]
            if getattr(fragment, 'false_branch_node', None) == deleted_node:
                fragment.false_branch_node = None # type: ignore[assignment]
            if getattr(fragment, 'break_node', None) == deleted_node:
                fragment.break_node = None # type: ignore[assignment]