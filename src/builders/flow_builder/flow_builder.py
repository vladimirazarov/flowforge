"""
Orchestrates CFG construction using fragments and manages FragmentForest

The `FlowBuilder` class manages the construction of a Control Flow Graph (CFG)
by incrementally adding structural `Fragment`s (like if-else, loops). It uses
configuration parameters (e.g., target complexity, desired structures) to
select fragments

Workflow:
1.  Starts with an initial CFG (often a single entry/exit node)
2.  Selects and creates a `Fragment` instance
3.  Merges the fragment's sub-CFG into the main CFG at a target location
    This can be sequential (appending after a node) or nested (inserting
    within an existing fragment's placeholder, like a branch or loop body)
4.  Updates the associated `FragmentForest` by registering the new fragment
    and linking it to its semantic parent based on the insertion type

`Fragment`s serve as blueprints; their internal sub-CFGs are used for merging
and then effectively discarded, making them ephemeral construction helpers
"""

from __future__ import annotations
from dataclasses import dataclass, field
import random
from typing import TYPE_CHECKING, Optional, Literal, Dict

from rich.console import Console
from loguru import logger

from src.config.config import AppConfig, config as global_config
from src.core.nodes.cellings_assigner import assign_shtv_cellings
from src.core.fragments import (
    FragmentRole,
    Fragment
)
from src.core.nodes.cfg_basic_block_node import CFGBasicBlockNode
from src.core.nodes.cfg_instructions_node import (
    CFGTrueBranch, CFGFalseBranch, CFGWhileLoopBody, CFGForLoopBody
)
from src.core.nodes.cfg_jump_node import CFGWhileLoopJump, CFGForLoopJump
from src.utils.logging_helpers import (
    print_cfg_rich,
    print_code,
    print_flow_builder_start_info,
    print_flow_builder_target_cc_one,
    print_flow_builder_finish_info,
    print_flow_builder_shtv_assigned,
    print_fragment_forest_rich,
    _NullConsole
)
if TYPE_CHECKING:
    from src.core.cfg.cfg import CFG
    from src.core.nodes.cfg_basic_block_node import CFGBasicBlockNode

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"


@dataclass
class FlowBuilder:
    """
    CFG Builder is responsible for creating a Control Flow Graph with the specified
    cyclomatic oper_builder and structure based on configuration parameters
    
    Input state:
        - CFG (passed as an argument and modified in-place)
        - Configuration parameters (provided via app_config)
        
    Output state:
        - CFG with cyclomatic oper_builder equal to target oper_builder and all fragments
          added according to the configuration parameters
    """
    
    app_config: AppConfig = field(default_factory=lambda: global_config)
    
    target_cc:           int = field(init=False)
    max_depth:           int = field(init=False)
    nesting_probability: float = field(init=False)
    seed:                Optional[int] = field(init=False)
    console: Optional[Console|_NullConsole] = field(init=False)
    
    def __post_init__(self):
        """Initialize fields from the provided or global config object"""

        cfg_builder_conf = self.app_config.cfg_builder

        assert cfg_builder_conf.total_cyclomatic_complexity is not None
        self.target_cc = cfg_builder_conf.total_cyclomatic_complexity
        
        assert cfg_builder_conf.max_nesting_depth is not None
        self.max_depth = cfg_builder_conf.max_nesting_depth
        
        assert cfg_builder_conf.nesting_probability is not None
        self.nesting_probability = cfg_builder_conf.nesting_probability
        
        self.seed = self.app_config.seed
    
    def build_random_cfg(self, cfg: CFG):
        if self.app_config.developer_options.show_rich:
            self.console = Console()
        else:
            self.console = _NullConsole()

        print_flow_builder_start_info(self.console, self.target_cc, self.max_depth, self.nesting_probability, self.seed) #type: ignore

        if self.target_cc == 1:
            print_flow_builder_target_cc_one(self.console) #type: ignore
            return

        iteration = 0
        while cfg.cc < self.target_cc:
            iteration += 1
            # Select random fragment role
            all_roles_set = set(FragmentRole) - {FragmentRole.FOR_EARLY_EXIT, FragmentRole.SWITCH}
            # Convert to list AND sort deterministically
            all_roles = sorted(list(all_roles_set), key=lambda role: role.name)
            chosen_role = random.choice(all_roles)
            fragment = Fragment(fragment_role=chosen_role)  # type: ignore
            # Don't choose too expensive fragment
            if (cfg.cc + fragment.cc - 1) > self.target_cc:
                continue
            # First ever fragment treatment
            if cfg.cc == 1:
                assert cfg.entry_node is not None
                # Initial fragment is always sequential relative to entry
                self._connect_fragment(fragment, cfg, cfg.entry_node, nest=True) 
                continue
            # Bernoulli trial for nesting decision
            # X ~ Bernoulli(p) where p = nesting_probability
            # PMF: P(X = x) = p if x = 1 (nest), 1-p if x = 0 (no nest)
            # Implementation via inverse transform sampling:
            # 1) Generate U ~ Uniform(0,1)
            # 2) X = 1 if U < p, X = 0 if U >= p
            # E[X] = p, Var(X] = p(1-p)
            nest = 1 if random.random() < self.nesting_probability else 0

            # Nest
            if nest:
                # Nest only inside true/false branches or loop bodies
                candidates = [
                    node for node in cfg
                    if 0 < node.depth <= self.max_depth
                    and isinstance(node.instructions, (CFGTrueBranch, CFGFalseBranch, CFGWhileLoopBody, CFGForLoopBody))
                ]
                if not candidates:
                    if self.app_config.developer_options.flow_builder_log:
                        logger.warning("No valid branch/body nodes found for nesting. Skipping fragment addition this iteration.")
                    continue
                candidates.sort(key=lambda node: node.node_id)
                target_node = random.choice(candidates)
                if self.app_config.developer_options.flow_builder_log:
                    logger.info(f"Chosen node for nesting: {target_node.node_id}")
                self._connect_fragment(fragment, cfg, target_node, nest=True)

            # Don't nest
            else:
                # Sequential insertion only on decision nodes (>=2 outgoing edges)
                candidates = [
                    node for node in cfg
                    if len(cfg.get_node_children(node)) >= 2
                ]
                if not candidates:
                    if self.app_config.developer_options.flow_builder_log:
                        logger.warning("No decision nodes found for sequential addition. Skipping fragment addition this iteration.")
                    continue
                candidates.sort(key=lambda node: node.node_id)
                target_node = random.choice(candidates)
                if self.app_config.developer_options.flow_builder_log:
                    logger.info(f"Chosen node for sequential insertion: {target_node.node_id}")
                self._connect_fragment(fragment, cfg, target_node, nest=False)

        print_flow_builder_finish_info(self.console, iteration, cfg.cc) #type: ignore

        if self.app_config.developer_options.flow_builder_log:
            logger.info("Triggering CFG analysis after flow building.")
        cfg.run_analysis()

        assign_shtv_cellings(cfg)
        print_flow_builder_shtv_assigned(self.console) #type:ignore
 
        print_cfg_rich(cfg)

        print_code(cfg, self.console) #type:ignore

    def _connect_fragment(self, fragment: Fragment, cfg: CFG, node: CFGBasicBlockNode, nest: bool):
        """
        Connect fragment to the selected node, respecting the nest parameter.
        - nest=True: Fragment becomes a nested child in the forest, CFG wired accordingly (_nest_inside).
        - nest=False: Fragment becomes a sibling/root in the forest, CFG wired sequentially (_connect_sequential).
        """
        assert fragment.entry_node is not None 

        # 1. Merge sub-CFG into the main CFG
        fragment.introduce_into(cfg)

        origin_frag_id = node.origin_fragment_id # Get the ID of the fragment we are connecting relative to

        # Determine connection position for sequential additions
        position: Literal['before', 'after'] = 'after' # Default
        if not nest: # Only determine position if adding sequentially
            parents_count = len(cfg.get_node_parents(node))
            if node.is_entry:
                position = 'before'
            elif node.is_exit:
                position = 'after'
            elif node.jump is not None: # branch/decision node -> adding before split/loop check
                position = 'before'
            elif parents_count > 1: # join/merge node (no jump) -> adding after merge
                position = 'after'

        if self.app_config.developer_options.flow_builder_log:
            logger.info(f"Updating forest: Adding fragment {fragment.fragment_id} relative to origin {origin_frag_id}. Nest={nest}, Position='{position if not nest else 'N/A'}'")
        cfg.fragment_forest.add_fragment(fragment, 
                                         nest=nest, 
                                         origin_fragment_id=origin_frag_id, 
                                         position=position)
        assert cfg.fragment_forest is not None
        print_fragment_forest_rich(cfg.fragment_forest, self.console) #type: ignore


        if nest:
            if self.app_config.developer_options.flow_builder_log:
                logger.info(f"Connecting nested: Wiring fragment {fragment.fragment_id} inside node {node.node_id}'s scope.")
            self._nest_inside(fragment, cfg, node)
        else:
            if self.app_config.developer_options.flow_builder_log:
                logger.info(f"Connecting sequentially: Wiring fragment {fragment.fragment_id} relative to node {node.node_id} (position: {position}).")
            self._connect_sequential(fragment, cfg, node) 

        if self.app_config.developer_options.flow_builder_log:
            logger.info(f"Successfully connected fragment {fragment.fragment_id}")

        self._wire_fragment_entry_in_main_cfg(fragment, cfg)

    def _connect_sequential(self, fragment: Fragment, cfg: CFG, target: CFGBasicBlockNode):
        assert fragment.entry_node is not None
        assert fragment.exit_node is not None

        # 1) stash the labels on all parent -> target
        incoming_labels = {
            parent: cfg.get_edge_label(parent, target)
            for parent in cfg.get_node_parents(target)
            if cfg.get_edge_label(parent, target) is not None
        }
        # 2) stash the labels on all target -> child
        outgoing_labels = {
            child: cfg.get_edge_label(target, child)
            for child in cfg.get_node_children(target)
            if cfg.get_edge_label(target, child) is not None
        }

        # Entry -> put fragment above
        if target.is_entry:
            if self.app_config.developer_options.flow_builder_log:
                logger.info(f"Connecting sequentially to ENTRY node: {target.node_id}")
            # 1) Mark the fragment's entry node as the new entry
            fragment.entry_node.is_entry = True
            cfg.entry_node = fragment.entry_node
            
            # 2) The original entry node is no longer the entry
            target.is_entry = False
            
            parents_backup = fragment.sub_cfg.get_node_parents(fragment.exit_node)
            # 3) Disconnect all parents from fragment exit node
            for parent in fragment.sub_cfg.get_node_parents(fragment.exit_node):
                cfg.remove_edge(parent, fragment.exit_node)
            # 4) Reconnect all parents (originally pointing to fragment exit) to the main cfg entry node (original target)
            for parent in parents_backup: 
                cfg.add_edge(parent, target)
                # Reapply label if this parent edge originally led into the target
                if label := incoming_labels.get(parent):
                    cfg.set_edge_label(parent, target, label)
                    if self.app_config.developer_options.flow_builder_log:
                        logger.debug(f"Re-applied incoming label '{label}' to ENTRY edge {parent.node_id} -> {target.node_id}")

            # Erase fragment's original exit node 
            if fragment.exit_node and fragment.exit_node in cfg.graph:
                exit_node_to_erase = fragment.exit_node
                fragment.exit_node = None # Avoid dangling reference
                cfg.erase_node(exit_node_to_erase)
            elif fragment.exit_node:
                if self.app_config.developer_options.flow_builder_log:
                    logger.error(f"Fragment exit node {fragment.exit_node.node_id} not found in graph before erase in Seq+Entry case.")
            else:
                if self.app_config.developer_options.flow_builder_log:
                    logger.error("Fragment exit node was None before erase in Seq+Entry case.")


        # Not entry and not exit -> put fragment into the middle 
        elif not target.is_entry and not target.is_exit:
            if self.app_config.developer_options.flow_builder_log:
                logger.info(f"Connecting sequentially to MIDDLE node: {target.node_id}")
            # 1) Back up the parents of the target node
            parents_backup = cfg.get_node_parents(target)
            
            # 2) Disconnect external parents from target and connect them to fragment entry
            for parent in parents_backup:
                # Check if parent -> target is an internal loop edge that should NOT be redirected
                is_internal_edge = False
                if isinstance(target.jump, (CFGWhileLoopJump, CFGForLoopJump)) and \
                   parent.origin_fragment_id is not None and \
                   target.origin_fragment_id is not None:
                    
                    is_same_fragment = (parent.origin_fragment_id == target.origin_fragment_id)
                    is_nested_loop_back = cfg.fragment_forest.is_descendant(
                        parent.origin_fragment_id,
                        target.origin_fragment_id
                    )
                    is_internal_edge = is_same_fragment or is_nested_loop_back

                    if is_internal_edge:
                        reason = "same fragment" if is_same_fragment else "nested fragment"
                        if self.app_config.developer_options.flow_builder_log:
                            logger.debug(f"Identified internal loop edge {parent.node_id}(frag {parent.origin_fragment_id}) -> {target.node_id}(frag {target.origin_fragment_id}) via {reason}.")
      
                if is_internal_edge:
                    if self.app_config.developer_options.flow_builder_log:
                        logger.debug(f"Skipping redirection of internal loop edge: {parent.node_id} -> {target.node_id}")
                    continue # Don't redirect the loop-back edge itself
                
                # If not an internal loop edge, proceed with redirection
                if self.app_config.developer_options.flow_builder_log:
                    logger.debug(f"Redirecting external edge: removing {parent.node_id} -> {target.node_id}, adding {parent.node_id} -> {fragment.entry_node.node_id}")
                # Remove edge from external parent to target
                cfg.remove_edge(parent, target)

                # Add edge from external parent to fragment entry
                cfg.add_edge(parent, fragment.entry_node)
                # Reapply label (original parent -> target label)
                if label := incoming_labels.get(parent):
                    cfg.set_edge_label(parent, fragment.entry_node, label)
                    if self.app_config.developer_options.flow_builder_log:
                        logger.debug(f"Re-applied incoming label '{label}' to new edge {parent.node_id} -> {fragment.entry_node.node_id}")
            
            # 4) Backup parents of the fragment's exit node (get from main CFG)
            if fragment.exit_node is None or fragment.exit_node not in cfg.graph:
                 raise RuntimeError(f"Fragment {fragment.fragment_id} exit node is invalid or not in CFG before getting parents.")
            exit_parents = cfg.get_node_parents(fragment.exit_node)
            
            # 5) Disconnect fragment exit node from its parents and erase it
            exit_node_to_erase = fragment.exit_node 
            fragment.exit_node = None 
            for parent in exit_parents:
                 cfg.remove_edge(parent, exit_node_to_erase)
                
            # 6) Connect all parents of the (now erased) fragment exit node to the target node
            for parent in exit_parents:
                cfg.add_edge(parent, target)
                # Apply the original target -> child label, assuming 'parent' corresponds to 'child'
                if label := outgoing_labels.get(parent): 
                    cfg.set_edge_label(parent, target, label)
                    if self.app_config.developer_options.flow_builder_log:
                        logger.debug(
                          f"Re-applied outgoing label '{label}' from original child {parent.node_id} to new edge {parent.node_id} -> {target.node_id}"
                        )

            cfg.erase_node(exit_node_to_erase)

        else:  # target.is_exit - the "Seq + last" case
            if self.app_config.developer_options.flow_builder_log:
                logger.info(f"Connecting sequentially to EXIT node: {target.node_id}")
            target.jump = fragment.entry_node.jump
            
            # 1) Identify children of the fragment's ENTRY node
            entry_children = fragment.sub_cfg.get_node_children(fragment.entry_node)
            
            # Find nodes within the fragment that point back to the fragment's entry node
            entry_parents_in_subcfg = fragment.sub_cfg.get_node_parents(fragment.entry_node)
            for parent_in_frag in entry_parents_in_subcfg:
                if cfg.graph.has_edge(parent_in_frag, fragment.entry_node):
                    if self.app_config.developer_options.flow_builder_log:
                        logger.debug(f"Redirecting loop-back edge from {parent_in_frag.node_id} -> {fragment.entry_node.node_id} to {parent_in_frag.node_id} -> {target.node_id}")
                    cfg.remove_edge(parent_in_frag, fragment.entry_node)
                    cfg.add_edge(parent_in_frag, target) # Point back to the target (new loop condition node)
                    # Reapply label if this loop-back edge was originally an incoming edge to target
                    if label := incoming_labels.get(parent_in_frag):
                        cfg.set_edge_label(parent_in_frag, target, label)
                        if self.app_config.developer_options.flow_builder_log:
                            logger.debug(f"Re-applied loop-back label '{label}' to new edge {parent_in_frag.node_id} -> {target.node_id}")

                else:
                    if self.app_config.developer_options.flow_builder_log:
                        logger.warning(f"Expected loop-back edge {parent_in_frag.node_id} -> {fragment.entry_node.node_id} not found in main CFG during redirection.")

            # 2) Connect the exit node (target) DIRECTLY to the children of fragment's entry node
            for child in entry_children:
                # Remove the original edge from the fragment's entry to its child
                if cfg.graph.has_edge(fragment.entry_node, child):
                    cfg.remove_edge(fragment.entry_node, child)
                # Add the edge from the target (new loop condition) to the child
                cfg.add_edge(target, child)
                # Reapply label (original target -> child label)
                if label := outgoing_labels.get(child):
                    cfg.set_edge_label(target, child, label)
                    if self.app_config.developer_options.flow_builder_log:
                        logger.debug(f"Re-applied outgoing label '{label}' to new edge {target.node_id} -> {child.node_id}")
            
            # Store original entry node before modifying fragment reference
            original_entry_node = fragment.entry_node 

            # Transfer fragment identity to the reused node (target now acts as entry)
            target.origin_fragment_id = fragment.fragment_id
            # Update the fragment's reference to its effective entry node
            fragment.entry_node = target 

            # 3) erase the original entry node of the fragment
            if original_entry_node in cfg.graph:
                cfg.erase_node(original_entry_node) 
            else:
                if self.app_config.developer_options.flow_builder_log:
                    logger.warning(f"Fragment original entry node {original_entry_node.node_id} not found in graph before erase in Seq+Exit case.")

            # Original target is no longer exit, fragment's exit node becomes new exit
            if fragment.exit_node and fragment.exit_node in cfg.graph: 
                fragment.exit_node.is_exit = True
                target.is_exit = False
                cfg.exit_node = fragment.exit_node
            else:
                # This might happen if the fragment was very simple (e.g., single node)
                if self.app_config.developer_options.flow_builder_log:
                    logger.warning(f"Fragment exit node was likely erased or reused. Setting target {target.node_id} as CFG exit.")
                target.is_exit = True # 
                cfg.exit_node = target
                # Ensure fragment reference is updated if its original exit was different
                if fragment.exit_node is not None and fragment.exit_node != target:
                    if self.app_config.developer_options.flow_builder_log:
                        logger.debug(f"Updating fragment {fragment.fragment_id} exit node reference to {target.node_id}")
                    fragment.exit_node = target 

    def _nest_inside(self, fragment: Fragment, cfg: CFG, target: CFGBasicBlockNode):
        assert fragment.entry_node is not None
        assert fragment.exit_node is not None
        assert fragment.sub_cfg is not None
        # 1) Initial state
        old_parents = cfg.get_node_parents(target)
        old_children = cfg.get_node_children(target)
        
        # 2) Increment fragment depths
        for node in fragment.sub_cfg:
            node.depth += target.depth
        
        # 3) Handle special case: target is an entry node
        if target.is_entry:
            # Mark fragment entry as new entry node
            fragment.entry_node.is_entry = True
            fragment.exit_node.is_exit = True
            cfg.entry_node = fragment.entry_node
            cfg.exit_node = fragment.exit_node
            # Erase original entry node
            cfg.erase_node(target)
        
        # 4) Other cases 
        else:
            # Standard nesting case (non-entry node)
            
            incoming_labels = {}
            for parent in old_parents:
                if cfg.graph.has_edge(parent, target):
                    label = cfg.get_edge_label(parent, target)
                    if label:
                        incoming_labels[parent] = label
                        if self.app_config.developer_options.flow_builder_log:
                            logger.debug(f"Stored label '{label}' for incoming edge {parent.node_id} -> {target.node_id}")
                else:
                     if self.app_config.developer_options.flow_builder_log:
                         logger.warning(f"Edge {parent.node_id} -> {target.node_id} not found when storing incoming labels in _nest_inside.")

            outgoing_labels = {}
            for child in old_children:
                if cfg.graph.has_edge(target, child):
                    label = cfg.get_edge_label(target, child)
                    if label:
                        outgoing_labels[child] = label
                        if self.app_config.developer_options.flow_builder_log:
                            logger.debug(f"Stored label '{label}' for outgoing edge {target.node_id} -> {child.node_id}")
                else:
                     if self.app_config.developer_options.flow_builder_log:
                         logger.warning(f"Edge {target.node_id} -> {child.node_id} not found when storing outgoing labels in _nest_inside.")

            # Disconnect target from its parents and children
            for parent in old_parents:
                if cfg.graph.has_edge(parent, target):
                    cfg.remove_edge(parent, target)
            
            for child in old_children:
                if cfg.graph.has_edge(target, child):
                    cfg.remove_edge(target, child)
            
            # Connect parents to fragment entry and fragment exit to children
            for parent in old_parents:
                cfg.add_edge(parent, fragment.entry_node)
                if label := incoming_labels.get(parent):
                    cfg.set_edge_label(parent, fragment.entry_node, label)
                    if self.app_config.developer_options.flow_builder_log:
                        logger.debug(f"Re-applied label '{label}' to new incoming edge {parent.node_id} -> {fragment.entry_node.node_id}")
            
            for child in old_children:
                cfg.add_edge(fragment.exit_node, child)
                if label := outgoing_labels.get(child):
                    cfg.set_edge_label(fragment.exit_node, child, label)
                    if self.app_config.developer_options.flow_builder_log:
                        logger.debug(f"Re-applied label '{label}' to new outgoing edge {fragment.exit_node.node_id} -> {child.node_id}")
            
            # Transfer jump behavior 
            if target.jump is not None:
                fragment.exit_node.jump = target.jump
            
            cfg.erase_node(target)

    def _wire_fragment_entry_in_main_cfg(self, fragment: Fragment, cfg: CFG):
        """
        Finds the effective entry node of the fragment within the main CFG
        (after splicing) and calls its wire_edges method
        Handles fragments with and without jump instructions on their entry node
        """
        original_entry_ref = fragment.entry_node 
        if not original_entry_ref:
            if self.app_config.developer_options.flow_builder_log:
                logger.error(f"Fragment {fragment.fragment_id} has invalid original entry node reference. Cannot wire edges.")
            return

        # Find all nodes in the main CFG belonging to this fragment
        candidates = [n for n in cfg.graph if getattr(n, 'origin_fragment_id', None) == fragment.fragment_id]
        if not candidates:
             if self.app_config.developer_options.flow_builder_log:
                 logger.error(f"Could not locate any nodes for Fragment {fragment.fragment_id} in main CFG. Wiring skipped.")
             return

        decision_node = None

        # Case 1: Entry node has a jump (standard control flow) 
        if original_entry_ref.jump:
            expected_jump_type = type(original_entry_ref.jump)
            if self.app_config.developer_options.flow_builder_log:
                logger.debug(f"Fragment {fragment.fragment_id} entry has jump type: {expected_jump_type.__name__}. Searching candidates...")

            # Filter candidates to find the one that now functions as the entry (matching jump type)
            entry_candidates = [c for c in candidates if isinstance(c.jump, expected_jump_type)]

            if not entry_candidates:
                if self.app_config.developer_options.flow_builder_log:
                    logger.error(f"Could not locate effective entry node for Fragment {fragment.fragment_id} with jump type {expected_jump_type.__name__} in main CFG among {len(candidates)} candidates. Wiring skipped.")
                return
            
            if len(entry_candidates) > 1:
                if self.app_config.developer_options.flow_builder_log:
                    logger.warning(f"Found multiple ({len(entry_candidates)}) potential entry nodes for Fragment {fragment.fragment_id} with jump type {expected_jump_type.__name__}. Using the first found: {entry_candidates[0].node_id}. Candidates: {[c.node_id for c in entry_candidates]}")

            decision_node = entry_candidates[0]
        
        # Case 2: Entry node has NO jump (simple instruction block)
        else:
            if self.app_config.developer_options.flow_builder_log:
                logger.debug(f"Fragment {fragment.fragment_id} entry has no jump. Assuming simple block.")
            # If it was a simple block, there should ideally only be one node left from this fragment after splicing
            if len(candidates) == 1:
                decision_node = candidates[0]
                if self.app_config.developer_options.flow_builder_log:
                    logger.debug(f"Found single candidate node {decision_node.node_id} for simple block Fragment {fragment.fragment_id}.")
            elif len(candidates) > 1:
                 # If multiple nodes remain, it's ambiguous which one acts as the entry point now.
                 # This might indicate an issue with how simple fragments are structured or spliced
                 if self.app_config.developer_options.flow_builder_log:
                     logger.warning(f"Found multiple ({len(candidates)}) nodes for jump-less Fragment {fragment.fragment_id}. Cannot reliably determine entry point for wiring. Using first candidate: {candidates[0].node_id}. Candidates: {[c.node_id for c in candidates]}")
                 decision_node = candidates[0] # Fallback, might be incorrect
            else:
                # This case is already handled by the initial check for candidates
                if self.app_config.developer_options.flow_builder_log:
                    logger.error(f"Logic error: No candidates found for jump-less Fragment {fragment.fragment_id}, but initial check passed.")
                return

        # Perform wiring 
        if decision_node:
            if self.app_config.developer_options.flow_builder_log:
                logger.info(f"Wiring edges for Fragment {fragment.fragment_id} using identified main CFG node {decision_node.node_id} (Jump: {type(decision_node.jump).__name__ if decision_node.jump else 'None'}).")
            # Update the fragment's reference to point to the live node before wiring
            if self.app_config.developer_options.flow_builder_log:
                logger.debug(f"Updating Fragment {fragment.fragment_id}'s entry_node reference from {getattr(fragment.entry_node, 'node_id', 'N/A')} to {decision_node.node_id}.")
            fragment.entry_node = decision_node
            
            fragment.wire_edges(cfg, decision_node)
        else:
            # Should not be reachable if candidate checks work, but added for safety
            if self.app_config.developer_options.flow_builder_log:
                logger.error(f"Failed to identify a decision node for Fragment {fragment.fragment_id}. Wiring skipped.")

 

      