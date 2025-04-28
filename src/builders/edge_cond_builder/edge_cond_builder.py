"""
Assigns conditions to CFG edges to make predefined test paths feasible

Uses Z3 to check feasibility of paths in `cfg.test_paths`, leveraging
`cfg.analysis` results to handle loop edge conditions correctly during checks

Process:
1.  Assigns initial random conditions to branches
2.  Iteratively checks path feasibility (`TestPath._check_feasibility_for_builder`)
3.  If infeasible, identifies a branch condition to change (using Z3 unsat core
    or path structure, prioritizing non-loop edges)
4.  Regenerates the condition *randomly*
5.  Repeats until paths are feasible or retries expire
6.  Calls `LoopTerminator` afterwards to ensure loops terminate

Note:
-   Feasibility guarantee is ONLY for `cfg.test_paths`
-   Relies on `LoopTerminator` for actual loop termination logic
-   Fixing is heuristic and based on random regeneration; may fail

Depends on: `CFG`, `TestPath`, `LoopAnalysisInfo`, `ExpressionRandomFactory`,
            `LoopTerminator`, `z3`
"""

from __future__ import annotations
from typing import Any, List, Set, Dict, Optional, Tuple, TYPE_CHECKING, cast
from dataclasses import dataclass, field

import networkx as nx
from rich.console import Console
from loguru import logger

from src.config.config import AppConfig
from src.core.nodes.cfg_basic_block_node import CFGBasicBlockNode
from src.core.paths.cfg_test_path import TestPath
from src.builders.edge_cond_builder.expression_random_factory import ExpressionRandomFactory
from src.core.cfg_content import Expression as EdgeCondition
from src.builders.edge_cond_builder.loop_terminator import LoopTerminator

from src.utils.logging_helpers import (
    log_phase, 
    print_edge_cond_start_info, 
    print_loop_analysis_summary, 
    print_edge_cond_finish_info,
    print_all_test_path_details,
    print_code,
    log_builder_event, 
    log_feasibility_check_attempt,
    log_all_paths_feasible,
    log_infeasible_paths_found,
    log_max_retries_reached,
    log_structural_infeasibility_error,
    log_branch_pair_found,
    log_condition_assignment,
    log_initial_conditions_assigned,
    print_initial_jump_conditions,
    log_jump_condition_set,
    log_path_infeasible,
    log_fix_attempt_start,
    log_fix_strategy_info,
    log_fix_condition_regenerate,
    log_fix_condition_update,
    log_fix_no_suitable_edge,
    print_loop_condition_assignment_details,
)

if TYPE_CHECKING:
    from src.core.cfg.cfg import CFG
    from src.core.cfg.cfg_analysis import LoopAnalysisInfo
    from src.core.cfg.cfg_context import CFGContext

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

@dataclass
class EdgeCondBuilderError(Exception):
    """Base exception for errors within EdgeCondBuilder"""
    pass

@dataclass
class StructurallyInfeasiblePathError(EdgeCondBuilderError):
    """Raised when structurally infeasible test paths are detected and cannot be fixed"""
    infeasible_paths: Set[TestPath]
    message: str = "Detected structurally infeasible test paths that could not be fixed"

    def __post_init__(self):
        log_structural_infeasibility_error(len(self.infeasible_paths), self.infeasible_paths)

    def __str__(self):
        path_ids = [p.path_id for p in self.infeasible_paths]
        return f"{self.message}: {path_ids}"

@dataclass
class EdgeCondBuilder:
    """Generates edge conditions to ensure test path feasibility using Z3

    Assigns conditions to CFG edges, checks feasibility of paths in
    `cfg.test_paths`, and attempts to fix infeasible paths by randomly
    regenerating conditions. Uses `LoopTerminator` to ensure loop termination

    Attributes:
        cfg: The Control Flow Graph to operate on
        expression_factory: Factory for creating random condition expressions
        app_config: Application configuration settings
        max_retries: Maximum attempts to fix infeasible paths
        context: Shared context from the CFG
        _branch_edge_pairs: Internal mapping of branch nodes to their successor edges
        structurally_infeasible_paths: Set of paths deemed impossible to fix
        console: Rich console instance for output
    """
    cfg: CFG
    expression_factory: ExpressionRandomFactory
    app_config: AppConfig
    max_retries: int = 10

    context: CFGContext = field(init=False)
    _branch_edge_pairs: Dict[CFGBasicBlockNode, Dict[str, Optional[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]]]] = field(init=False)
    structurally_infeasible_paths: Set[TestPath] = field(init=False)
    console: Console = field(init=False)

    def __post_init__(self):
        """Initialize remaining fields after dataclass initialization"""
        self.context = self.cfg.context
        self._branch_edge_pairs = {}
        self.structurally_infeasible_paths = set()
        self.console = Console()

    @log_phase("Edge Condition Building")
    def build_conditions(self, max_retries: int = 30) -> bool:
        """
        Build and validate edge conditions for pre-defined test paths

        Args:
            max_retries (int): Overrides the instance's max_retries for this run

        Returns:
            bool: True if the process completes (regardless of final feasibility)
        """
        num_paths: Optional[int] = None
        if self.cfg.test_paths is not None:
            num_paths = len(self.cfg.test_paths)
        else:
            log_builder_event('warning', "cfg.test_paths is None. Cannot determine number of paths.")

        print_edge_cond_start_info(self.console, max_retries, num_paths)

        # Ensure analysis is run if not already done
        if self.cfg.analysis is None:
             log_builder_event('info', "CFG analysis not yet run. Running now.")
             self.cfg.run_analysis() 
             if self.cfg.analysis is None:
                 log_builder_event('error', "Failed to run or initialize CFG analysis. Proceeding without loop-aware logic.")

        # Log analysis summary if analysis results exist
        if self.cfg.analysis:
            loop_info = self.cfg.analysis.get_loop_info()
            if loop_info:
                self._log_loop_analysis_summary(loop_info)
                log_builder_event('info', "Using pre-computed loop analysis results from CFG.")
            else:
                log_builder_event('info', "CFG analysis available but found no loops.")
        else:
             log_builder_event('warning', "CFG analysis unavailable. Proceeding without loop-aware logic.")


        self._initialize_build_step() 
        self._check_and_fix_feasibility_step(max_retries) 
        self._finalize_and_terminate_loops_step() 
        print_code(self.cfg, self.console) 

        final_infeasible_paths = self.cfg.statically_infeasible_test_paths
        if final_infeasible_paths:
             log_builder_event('success', "Edge condition building deemed structurally feasible, proceeding despite remaining static infeasibilities.")
        else:
             log_builder_event('success', "Edge condition building completed successfully. All paths statically feasible.")

        assert final_infeasible_paths is not None
        print_edge_cond_finish_info(self.console, final_infeasible_paths)
        print_all_test_path_details(self.console, list(self.cfg.test_paths) if self.cfg.test_paths else None, self.cfg)
             
        return True

    @log_phase("Build Step Initialization")
    def _initialize_build_step(self) -> None: 
        """Handles the initial setup."""
        self._initialize_parameters()
        self._identify_branch_pairs()
        self._assign_initial_conditions() 

    def _log_loop_analysis_summary(self, loop_analysis_results: Optional['LoopAnalysisInfo']) -> None:
        """Logs the detailed summary of the loop analysis results using the helper"""
        print_loop_analysis_summary(self.console, loop_analysis_results, self.cfg.graph)

    @log_phase("Feasibility Check and Fix")
    def _check_and_fix_feasibility_step(self, max_retries: int) -> None: 
        """
        Iteratively checks test path feasibility and attempts to fix infeasible paths
        """
        for attempt in range(max_retries):
            log_feasibility_check_attempt(attempt + 1, max_retries)
            infeasible_paths_data = self._get_infeasible_paths() 

            if not infeasible_paths_data:
                log_all_paths_feasible()
                self.cfg.statically_infeasible_test_paths = [] 
                return

            log_infeasible_paths_found(len(infeasible_paths_data))
            self._fix_infeasible_paths(infeasible_paths_data) 

        log_max_retries_reached(max_retries)
        final_infeasible_paths_data = self._get_infeasible_paths() 

        if self.structurally_infeasible_paths:
            raise StructurallyInfeasiblePathError(self.structurally_infeasible_paths)

        if final_infeasible_paths_data:
            self.cfg.statically_infeasible_test_paths = [path for path, core in final_infeasible_paths_data]
            log_builder_event('warning', f"Check/fix loop completed, but {len(final_infeasible_paths_data)} paths remain statically infeasible. Details logged previously.")
        else:
             self.cfg.statically_infeasible_test_paths = []

    @log_phase("Loop Termination Finalization")
    def _finalize_and_terminate_loops_step(self) -> None: 
        """Invoke LoopTerminator to inject loop termination operations"""
        loop_terminator = LoopTerminator(self.cfg, self.app_config)
        # Pass the loop info dictionary, handling None case
        loop_info_dict = self.cfg.analysis.get_loop_info()
        loop_terminator.ensure_all_loops_terminate()
        log_builder_event('info', "LoopTerminator finished processing loops.")

    @log_phase("Parameter Initialization")
    def _initialize_parameters(self) -> None:
        self.cfg.context.ensure_parameters_initialized()

    @log_phase("Branch Pair Identification")
    def _identify_branch_pairs(self) -> None:
        """Identify nodes with exactly two successors (branch points) and map edges by label"""
        self._branch_edge_pairs = {}
        for node in self.cfg.graph.nodes():
            successors = list(self.cfg.graph.successors(node))
            if len(successors) == 2:
                # Initialize label map for this branch node
                branch_map: Dict[str, Optional[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]]] = {
                    'true': None, 'false': None, 'loop': None, 'exit': None
                }
                has_valid_label = False
                # Populate the map based on edge labels
                for succ in successors:
                    edge = (node, succ)
                    lbl = self.cfg.get_edge_label(node, succ)
                    if lbl in branch_map:
                        if branch_map[lbl] is not None:
                            logger.warning(f"Node {node.node_id}: Multiple edges found for label '{lbl}'. Overwriting {branch_map[lbl]} with {edge}.")
                        branch_map[lbl] = edge
                        has_valid_label = True
                    else:
                        logger.warning(f"Node {node.node_id}: Successor {succ.node_id} has unrecognized label '{lbl}'.")

                if has_valid_label:
                    self._branch_edge_pairs[node] = branch_map
                    # Log the found pairs for debugging/confirmation
                    found_labels = {lbl: edge for lbl, edge in branch_map.items() if edge is not None}
                else:
                    logger.warning(f"Node {node.node_id} is a branch but has no successors with recognized labels ('true', 'false', 'loop', 'exit'). Skipping.")

    @log_phase("Initial Condition Assignment")
    def _assign_initial_conditions(self) -> None:
        """Assign initial conditions to branch edges purely based on edge labels ('true'/'false', 'loop'/'exit')"""
        edge_conditions: Dict[Tuple[CFGBasicBlockNode, CFGBasicBlockNode], EdgeCondition] = {}
        assigned_count = 0

        # Positive labels indicate the branch taken when the jump condition is TRUE
        positive_labels = {'true', 'loop'}
        # Negative labels indicate the branch taken when the jump condition is FALSE
        negative_labels = {'false', 'exit'}

        for branch_node, mapping in self._branch_edge_pairs.items():
            # Pick a fresh random cond + its negation
            cond = self.expression_factory.create_condition_expression()
            neg = cond.negate()

            # Identify which edge is the "true"/positive branch and which is "false"/negative
            true_edge: Optional[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]] = mapping.get('true') or mapping.get('loop')
            false_edge: Optional[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]] = mapping.get('false') or mapping.get('exit')

            if not true_edge or not false_edge:
                logger.warning(
                    f"Node {branch_node.node_id}: incomplete labels, falling back to positional assignment."
                )
                succs = list(self.cfg.graph.successors(branch_node))
                # Ensure there are exactly two successors for positional assignment
                if len(succs) == 2:
                    true_edge = (branch_node, succs[0])
                    false_edge = (branch_node, succs[1])
                    logger.debug(f"  Fallback: TrueEdge -> {succs[0].node_id}, FalseEdge -> {succs[1].node_id}")
                else:
                    logger.error(
                        f"Node {branch_node.node_id}: Expected 2 successors for positional fallback, found {len(succs)}. Skipping assignment."
                    )
                    continue 

            # Assign jump_expr on the node itself so that "if (jump_expr) ..." takes me to the positive edge
            self._update_jump_expression(branch_node, cond)

            # Label the edges in the graph
            edge_conditions[true_edge] = cond
            edge_conditions[false_edge] = neg
            assigned_count += 2

            # Log loop details if applicable (mostly for verification)
            is_loop_header = self.cfg.analysis and branch_node in (self.cfg.analysis.get_loop_info().keys() if self.cfg.analysis.get_loop_info() else set())
            if is_loop_header:
                true_label = 'true' if mapping.get('true') else 'loop'
                false_label = 'false' if mapping.get('false') else 'exit'
                notes = f"True Branch Label: '{true_label}', False Branch Label: '{false_label}'"
                print_loop_condition_assignment_details(
                    self.console, branch_node, true_edge, cond, false_edge, neg, cond, "Initial Label-Based", notes
                )

            logger.debug(
                f"[AssignInit] Node {branch_node.node_id}: "
                f"TrueEdge={true_edge[0].node_id}->{true_edge[1].node_id}, "
                f"FalseEdge={false_edge[0].node_id}->{false_edge[1].node_id}, "
                f"Cond={cond.to_c()}, Neg={neg.to_c()}, Jump={cond.to_c()}"
            )

        # Set edge attributes in the graph
        nx.set_edge_attributes(self.cfg.graph,
            {edge: {"condition": c} for edge, c in edge_conditions.items()}
        )
        log_initial_conditions_assigned(assigned_count)
        print_initial_jump_conditions(self.console, self.cfg)

    def _update_jump_expression(self, origin_node: CFGBasicBlockNode, condition: EdgeCondition) -> None:
        """
        Assign a condition to a CFG node's jump instruction
        """
        origin_node.jump.expression = condition # type: ignore[attr-defined]
        log_jump_condition_set(origin_node.node_id, condition.to_c())
        origin_node.logical_operations_count += 1

    def _get_infeasible_paths(self) -> List[Tuple[TestPath, List[EdgeCondition]]]: 
        """Checks the defined test paths and returns infeasible ones using direct access"""
        infeasible_path_data = []
        defined_test_paths = self.cfg.test_paths

        if not defined_test_paths:
             log_builder_event('warning', "cfg.test_paths is empty or None. No paths to check.")
             return []

        # Get loop analysis results ONCE to pass down
        loop_analysis: Optional[LoopAnalysisInfo] = None
        loop_analysis = self.cfg.analysis.get_loop_info()

        log_builder_event('info', f"Checking feasibility of {len(defined_test_paths)} paths (Loop-aware: {loop_analysis is not None})...")

        for test_path in defined_test_paths:
            path_id_for_log = test_path.path_id
            test_path.update_conditions_from_cfg(self.cfg, loop_analysis_results=loop_analysis) 
            is_feasible, core_conditions = test_path._check_feasibility_for_builder()

            if not is_feasible:
                log_path_infeasible(path_id_for_log, [n.node_id for n in test_path.nodes])
                conflicting_core = core_conditions if core_conditions is not None else []
                infeasible_path_data.append((test_path, conflicting_core))

        return infeasible_path_data

    def _fix_infeasible_paths(self, infeasible_path_data: List[Tuple[TestPath, List[EdgeCondition]]]) -> None:
        """Fix the first infeasible path by regenerating a condition"""
        if not infeasible_path_data:
            return
        path_to_fix, core_conditions = infeasible_path_data[0]
        nodes_str = ' -> '.join([str(n.node_id) for n in path_to_fix.nodes])
        log_fix_attempt_start(nodes_str)

        loop_headers: Set[CFGBasicBlockNode] = set()
        loop_info = self.cfg.analysis.get_loop_info()
        loop_analysis = loop_info 
        if loop_info:
            loop_headers = set(loop_info.keys())


        target_edge: Optional[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]] = None
        origin_node: Optional[CFGBasicBlockNode] = None
        reason = ""

        # Prioritize core, handle loop edges within core before path fallback
        log_fix_strategy_info(f"Analyzing infeasible path: {' -> '.join([str(n.node_id) for n in path_to_fix.nodes])}")

        # Identify all loop control edges (header branches or exit edges) 
        all_loop_exit_edges: Set[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]] = set()
        if loop_info:
            for structure in loop_info.values():
                loop_exits_info = cast(dict, structure.get("exits"))
                all_loop_exit_edges.update(loop_exits_info.keys())

        def is_loop_control_edge(edge: Tuple[CFGBasicBlockNode, CFGBasicBlockNode]) -> bool:
            u, v = edge
            is_header_branch = u in loop_headers
            is_exit_edge = edge in all_loop_exit_edges
            return is_header_branch or is_exit_edge

        # Strategy A: Core-guided selection
        if core_conditions:
            log_fix_strategy_info(f"Using unsat core ({len(core_conditions)} conditions)...")
            core_branch_edges: List[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]] = []
            edge_map = path_to_fix.edge_conditions_map # Cache for lookup

            # Find edges in the path corresponding to core conditions
            core_condition_set = set(core_conditions)
            candidate_edges_in_core: List[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]] = []
            path_edges_in_order = list(zip(path_to_fix.nodes[:-1], path_to_fix.nodes[1:]))

            for edge in path_edges_in_order:
                 cond = edge_map.get(edge)
                 if cond and cond in core_condition_set:
                     candidate_edges_in_core.append(edge)


            log_fix_strategy_info(f"Found {len(candidate_edges_in_core)} edges in path corresponding to core conditions.")

            # Separate core edges based on whether they are loop control
            non_loop_core_edges: List[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]] = []
            loop_core_edges: List[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]] = []

            for edge in candidate_edges_in_core:
                u, v = edge
                if u in self._branch_edge_pairs: 
                    if is_loop_control_edge(edge):
                        loop_core_edges.append(edge)
                        log_fix_strategy_info(f"Core edge {u.node_id}->{v.node_id} is loop control.")
                    else:
                        non_loop_core_edges.append(edge)
                        log_fix_strategy_info(f"Core edge {u.node_id}->{v.node_id} is non-loop control.")

            # Select target: pick only those non-loop edges whose bounds allow a solution,
            # otherwise fall back to loop-control edges.
            good_nonloop = [e for e in non_loop_core_edges
                            if self._can_fix_edge(e, path_to_fix)]
            if good_nonloop:
                target_edge = good_nonloop[0]
                origin_node = target_edge[0]
                reason = "unsat core on non-loop-control branch"
                log_fix_strategy_info(f"Selected core edge: {origin_node.node_id}->{target_edge[1].node_id} ({reason})")
            elif loop_core_edges:
                target_edge = loop_core_edges[0]
                origin_node = target_edge[0]
                reason = "unsat core on loop-control branch (core fallback)"
                log_fix_strategy_info(f"Selected core edge: {origin_node.node_id}->{target_edge[1].node_id} ({reason})")
            else:
                log_fix_strategy_info("No branching edges found for core conditions.")

        # Strategy B: Path Structure - First non-loop-control branch
        if not target_edge:
            log_fix_strategy_info("Core-based selection failed or skipped. Selecting first non-loop-control branch...")
            path_edges = list(zip(path_to_fix.nodes[:-1], path_to_fix.nodes[1:]))
            for u, v in path_edges:
                current_edge = (u, v)
                if u in self._branch_edge_pairs:
                    if is_loop_control_edge(current_edge):
                        log_fix_strategy_info(f"Skipping loop control branch {u.node_id}->{v.node_id}.")
                        continue
                    # Found a non-loop control branch
                    target_edge = current_edge
                    origin_node = u
                    reason = "first non-loop-control branching edge in path"
                    log_fix_strategy_info(f"Selected: {u.node_id}->{v.node_id} ({reason})")
                    # Found target
                    break 

        # Strategy C: Path Structure - First loop-control branch (fallback)
        if not target_edge:
            log_fix_strategy_info("No non-loop-control branch found. Selecting first loop-control branch (fallback)...")
            path_edges = list(zip(path_to_fix.nodes[:-1], path_to_fix.nodes[1:]))
            for u, v in path_edges:
                current_edge = (u, v)
                if u in self._branch_edge_pairs:
                    # At this point, any branch must be a loop control one if Strategy B failed
                    target_edge = current_edge
                    origin_node = u
                    reason = "first loop-control branching edge in path (fallback)"
                    log_fix_strategy_info(f"Selected: {u.node_id}->{v.node_id} ({reason})")
                    # Found target
                    break 

        # Check if a target edge was selected
        if not target_edge or not origin_node:
            log_fix_no_suitable_edge(nodes_str)
            self.structurally_infeasible_paths.add(path_to_fix)
            return

        # Regenerate condition for the selected edge and its sibling
        log_fix_condition_regenerate(origin_node.node_id, target_edge[1].node_id, reason)

        # Get the branch mapping for the origin node
        mapping = self._branch_edge_pairs[origin_node]
        true_edge: Optional[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]] = mapping.get('true') or mapping.get('loop')
        false_edge: Optional[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]] = mapping.get('false') or mapping.get('exit')

        # Ensure both edges were found (should be guaranteed if origin_node is in _branch_edge_pairs)
        if not true_edge or not false_edge:
            logger.error(f"Logic error: Could not find true/false edges for branch node {origin_node.node_id} during fix.")
            return

        # Generate new condition and its negation
        # Quick top-down lookahead: ensure branch condition respects all downstream constraints
        try:
            # Collect the suffix of edges after this branch in the test path
            nodes = path_to_fix.nodes
            idx = nodes.index(origin_node)
            remaining_edges = list(zip(nodes[idx:-1], nodes[idx+1:]))

            # Build a suffix solver with downstream edge conditions
            from z3 import Solver, sat
            suffix_solver = Solver()
            for u, v in remaining_edges:
                cond = self.cfg.graph.edges[u, v].get("condition")
                if cond is not None:
                    suffix_solver.add(cond.to_z3())

            if suffix_solver.check() == sat:
                model = suffix_solver.model()
                # Grab first variable assignment from the model
                decl = model.decls()[0]
                val = model[decl].as_long()

                # Use that same variable to build a consistent threshold
                var = next(iter(cond.get_variables()))
                from src.core.cfg_content.expression import ComparisonExpression, VariableExpression, Constant
                from src.core.cfg_content.operator import ComparisonOperator

                if target_edge == true_edge:
                    new_condition = ComparisonExpression(
                        VariableExpression(var),
                        ComparisonOperator.GREATER_EQUAL,
                        Constant(val)
                    )
                else:
                    new_condition = ComparisonExpression(
                        VariableExpression(var),
                        ComparisonOperator.LESS,
                        Constant(val)
                    )
            else:
                new_condition = self.expression_factory.create_condition_expression(edge=target_edge)
        except Exception:
            new_condition = self.expression_factory.create_condition_expression(edge=target_edge)

        # Negate for the sibling edge
        negated_new = new_condition.negate()

        # Always assign the new condition to the jump and true edge,
        # and its negation to the false edge
        is_true_fix = (target_edge == true_edge)

        if is_true_fix:
            # fixing the true‐branch: new on true, neg on false
            jump_cond, true_cond, false_cond = new_condition, new_condition, negated_new
        else:
            # fixing the false‐branch: new on false, neg on true
            jump_cond, true_cond, false_cond = negated_new, negated_new, new_condition

        log_builder_event('debug', f"Node {origin_node.node_id} Fix (Target was {'TRUE' if is_true_fix else 'FALSE'} edge): "
                                   f"Jump='{jump_cond.to_c()}', TrueEdgeCond='{true_cond.to_c()}', FalseEdgeCond='{false_cond.to_c()}'")

        # Set the overall jump condition on the node
        self._update_jump_expression(origin_node, jump_cond)

        # Prepare edge conditions dictionary for graph update
        edge_conditions = {
            true_edge: true_cond,
            false_edge: false_cond
        }

        # Update edge attributes in the graph
        nx.set_edge_attributes(self.cfg.graph,
            {e: {"condition": c} for e, c in edge_conditions.items()}
        )

        # Log details if it was a loop header that was fixed
        loop_analysis = self.cfg.analysis.get_loop_info() if self.cfg.analysis else None
        if origin_node in (loop_analysis.keys() if loop_analysis else set()):
            is_loop_header = True
            notes = f"Fixing loop header ({reason}). True branch: {true_edge[0].node_id}->{true_edge[1].node_id}"
            # Re-fetch loop info for detailed notes if needed
            if loop_analysis: # Check again to be safe
                loop_body_nodes = self.cfg.analysis.get_loop_body(origin_node) or set()
                loop_exits_info = self.cfg.analysis.get_loop_exits(origin_node)
                exit_edges = set(loop_exits_info.keys()) if loop_exits_info else set()

                true_is_body = true_edge[1] in loop_body_nodes
                false_is_body = false_edge[1] in loop_body_nodes
                true_is_exit = true_edge in exit_edges
                false_is_exit = false_edge in exit_edges
                notes = f"Fixing Loop Header ({reason}): True->Body: {true_is_body}, False->Body: {false_is_body}, True->Exit: {true_is_exit}, False->Exit: {false_is_exit}"

            print_loop_condition_assignment_details(
                self.console,
                origin_node,
                true_edge, edge_conditions[true_edge],
                false_edge, edge_conditions[false_edge],
                jump_cond,
                "Fix",
                notes
            )
        else:
            # Log non-loop fix details if needed
            logger.debug(f"Fixed Non-Loop Branch at Node {origin_node.node_id} ({reason})")
            logger.debug(f"  True Edge: {true_edge[0].node_id}->{true_edge[1].node_id} = {edge_conditions[true_edge].to_c()}")
            logger.debug(f"  False Edge: {false_edge[0].node_id}->{false_edge[1].node_id} = {edge_conditions[false_edge].to_c()}")
            logger.debug(f"  Jump Condition: {jump_cond.to_c()}")


        log_fix_condition_update(f"{true_edge[0].node_id}->{true_edge[1].node_id}", edge_conditions[true_edge].to_c())
        log_fix_condition_update(f"{false_edge[0].node_id}->{false_edge[1].node_id}", edge_conditions[false_edge].to_c())

    def _can_fix_edge(
        self,
        edge: Tuple[CFGBasicBlockNode, CFGBasicBlockNode],
        path: TestPath
    ) -> bool:
        """
        Quick check whether there _exists_ any integer that can satisfy
        all of the other conditions on this path if we were to re‐assign `edge`
        We do a simple bound‐propagation on the single variable involved
        """
        from math import inf
        from src.core.cfg_content.expression import ComparisonExpression, VariableExpression, Constant # Added import
        from src.core.cfg_content.operator import ComparisonOperator # Added import
        lb, ub = -inf, inf
        var_name = None

        # gather all other edges' conditions
        all_edges = list(zip(path.nodes[:-1], path.nodes[1:]))
        for (u, v) in all_edges:
            if (u, v) == edge:
                continue
            cond = self.cfg.graph.edges[u, v].get("condition")
            if not isinstance(cond, ComparisonExpression):
                continue
            assert isinstance(cond.left, VariableExpression)
            assert isinstance(cond.right, Constant)
            var = cond.left.variable
            if var_name is None:
                var_name = var.name
            elif var_name != var.name:
                # multiple different vars: bail out to be safe
                return True
            val = cond.right.value
            op = cond.operator
            # tighten bounds
            if op in (ComparisonOperator.GREATER, ComparisonOperator.GREATER_EQUAL):
                # var > val  → lb = max(lb, val+1)
                # var ≥ val → lb = max(lb, val)
                new_lb = val + (1 if op is ComparisonOperator.GREATER else 0)
                lb = max(lb, new_lb)
            elif op in (ComparisonOperator.LESS, ComparisonOperator.LESS_EQUAL):
                # var < val  → ub = min(ub, val-1)
                # var ≤ val → ub = min(ub, val)
                new_ub = val - (1 if op is ComparisonOperator.LESS else 0)
                ub = min(ub, new_ub)
            else:
                continue
            if lb > ub:
                return False
        return True