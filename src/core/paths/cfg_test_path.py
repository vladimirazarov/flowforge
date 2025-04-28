"""Defines the TestPath class, representing a concrete execution path through a CFG,
used for feasibility analysis and test input generation
"""
from __future__ import annotations
from typing import TYPE_CHECKING, cast

import itertools
from z3 import Solver, BoolRef,  sat, unsat, Context, Bool, ModelRef, IntNumRef

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Set, cast

from src.utils.logging_helpers import (
    log_test_path_created,
    log_test_path_update_cond_start,
    log_test_path_update_cond_end,
    log_test_path_builder_check_start,
    log_test_path_builder_check_result,
    log_test_path_full_check_start,
    log_test_path_full_check_result,
    log_test_path_z3_call,
    log_test_path_z3_unknown,
    log_test_path_no_conditions,
    log_test_path_loop_exit_decision
)

from src.core.cfg_content import Expression
from src.core.paths.path import Path
from src.core.nodes.cfg_basic_block_node import CFGBasicBlockNode

if TYPE_CHECKING:
    from src.core.cfg.cfg import CFG
    from src.core.cfg.cfg_analysis import LoopAnalysisInfo

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

_test_path_id_counter = itertools.count()
@dataclass(eq=False)
class TestPath(Path):
    """Represents a specific path through a Control Flow Graph (CFG)

    Stores the sequence of nodes, conditions associated with traversed edges,
    and the results of feasibility analysis using a Z3 solver

    Attributes:
        nodes: List of CFGBasicBlockNode objects representing the path sequence
        edge_conditions_map: Maps edge tuples (u, v) to their corresponding Expression condition
        effective_conditions: List of conditions considered for feasibility analysis (e.g., excluding loop exit conditions under certain strategies)
        test_inputs: Dictionary of variable assignments (test inputs) if the path is feasible and a model is found
        is_feasible: Boolean flag indicating whether the path is determined to be feasible
        _id: Unique identifier for the TestPath instance
    """
    nodes: List[CFGBasicBlockNode] = field(default_factory=list)
    edge_conditions_map: Dict[Tuple[CFGBasicBlockNode, CFGBasicBlockNode], Optional[Expression]] = field(default_factory=dict)
    effective_conditions: List[Expression] = field(default_factory=list)
    test_inputs: Optional[Dict[str, Any]] = field(default=None)
    is_feasible: bool = field(default=True)
    _id: int = field(init=False)

    def __post_init__(self):
        """Initializes the unique path ID and logs the creation event"""
        self._id = next(_test_path_id_counter)
        log_test_path_created(self._id)

    @property
    def path_id(self) -> int:
        """Returns the unique identifier of this test path"""
        return self._id

    def __hash__(self) -> int:
        """Computes the hash based on the unique path ID"""
        return hash(self._id)

    def __eq__(self, other) -> bool:
        """Checks equality with another TestPath based on their unique IDs"""
        other_path = cast(TestPath, other)
        return self._id == other_path._id

    def update_conditions_from_cfg(
        self,
        cfg: CFG,
        loop_analysis_results: Optional[LoopAnalysisInfo] = None
    ) -> None:
        """Populates edge conditions and identifies effective conditions for the path

        Iterates through the path's edges in the provided CFG, extracting conditions
        It populates `edge_conditions_map` with all edge conditions and `effective_conditions`
        with a subset relevant for primary feasibility checks. A condition on a loop exit edge
        is excluded from `effective_conditions` ONLY if the path has entered the corresponding
        loop body before taking that exit

        Args:
            cfg: The Control Flow Graph containing the path
            loop_analysis_results: Optional analysis results about loops in the CFG
        """
        self.edge_conditions_map.clear()
        self.effective_conditions.clear()
        log_test_path_update_cond_start(self.path_id)

        loop_data: Dict[CFGBasicBlockNode, Dict[str, Any]] = {}
        loop_exit_edges_map: Dict[Tuple[CFGBasicBlockNode, CFGBasicBlockNode], CFGBasicBlockNode] = {}
        # Store loop body nodes excluding headers for efficient check later
        loop_bodies_no_headers: Dict[CFGBasicBlockNode, Set[CFGBasicBlockNode]] = {}
        if loop_analysis_results:
            loop_data = loop_analysis_results
            for header, loop_info in loop_analysis_results.items():
                 exits_data = loop_info.get("exits", {})
                 for exit_edge in cast(Dict, exits_data).keys():
                     loop_exit_edges_map[exit_edge] = header 
                 # Store body nodes excluding the header itself
                 body = cast(Set[CFGBasicBlockNode], loop_info.get('body', set()))
                 loop_bodies_no_headers[header] = body - {header}


        if len(self.nodes) < 2:
            return

        edges = list(zip(self.nodes[:-1], self.nodes[1:]))
        visited_nodes_in_path: List[CFGBasicBlockNode] = [] 

        for i, edge in enumerate(edges):
            u, v = edge
            # Add source node of the *previous* edge (or first node) before processing current edge
            if i == 0:
                visited_nodes_in_path.append(u)

            edge_data = cfg.graph.get_edge_data(u, v)
            cond = edge_data.get("condition") if edge_data else None
            self.edge_conditions_map[edge] = cond

            if cond is None:
                # Still need to record the node traversal even if no condition
                visited_nodes_in_path.append(v)
                continue

            skip_condition = False
            is_loop_exit = edge in loop_exit_edges_map
            loop_was_entered = False

            if is_loop_exit:
                header_node = loop_exit_edges_map[edge]
                # Check if any node visited so far
                # belongs to the loop body (excluding the header)
                if header_node in loop_bodies_no_headers:
                     body_to_check = loop_bodies_no_headers[header_node]
                     # Check nodes visited up to and including the source 'u' of the current edge
                     for visited_node in visited_nodes_in_path:
                         if visited_node in body_to_check:
                             loop_was_entered = True
                             break

                if loop_was_entered:
                    skip_condition = True
                log_test_path_loop_exit_decision(self.path_id, edge, cond.to_c(), skipped=skip_condition, entered_loop=loop_was_entered)

            if not skip_condition:
                self.effective_conditions.append(cond)

            # Record target node of current edge *after* processing the edge condition
            visited_nodes_in_path.append(v)

        log_test_path_update_cond_end(self.path_id, len(self.edge_conditions_map), len(self.effective_conditions))

    def _check_feasibility_for_builder(self) -> Tuple[bool, Optional[List[Expression]]]:
        """Performs a preliminary Z3 check for path feasibility using effective conditions

        This method is primarily intended for internal use by path generation logic
        to quickly discard infeasible paths during construction
        It uses the `effective_conditions` list

        Returns:
            A tuple (is_feasible, conflicting_conditions):
            - bool: True if the path appears feasible based on effective conditions, False otherwise
            - Optional[List[Expression]]: A list of conflicting conditions if found (unsat core),
              otherwise None
        """
        if not self.effective_conditions:
            log_test_path_no_conditions(self.path_id, "builder_check")
            return True, None

        log_test_path_builder_check_start(self.path_id)
        ctx = Context()
        solver = Solver(ctx=ctx)
        solver.set("unsat_core", True)

        z3_exprs: List[BoolRef] = []
        for cond in self.effective_conditions:
             zexpr = cond.to_z3(context=ctx)
             if zexpr is not None:
                 z3_exprs.append(cast(BoolRef, zexpr))


        if not z3_exprs:
            return True, None

        tracker_map: Dict[str, Expression] = {}
        formula_constraints: List[BoolRef] = []

        for i, cond in enumerate(self.effective_conditions):
             z3_ref = cond.to_z3(context=ctx)
             if z3_ref is not None:
                 tracker_name = f"track_eff_{id(cond)}_{i}"
                 tracker_ref = Bool(tracker_name, ctx=ctx)
                 solver.assert_and_track(cast(BoolRef, z3_ref), tracker_ref)
                 tracker_map[tracker_name] = cond
                 formula_constraints.append(cast(BoolRef, z3_ref))

        if not formula_constraints:
            return True, None

        log_test_path_z3_call(self.path_id, "builder_check", len(formula_constraints))
        result = solver.check()

        if result == sat:
            log_test_path_builder_check_result(self.path_id, True, None)
            return True, None
        elif result == unsat:
            conflicting_conditions: List[Expression] = []
            core_refs = solver.unsat_core()
            for core_ref in core_refs:
                name = str(core_ref)
                if name in tracker_map:
                    conflicting_conditions.append(tracker_map[name])
            core_size = len(conflicting_conditions) if conflicting_conditions else 0
            log_test_path_builder_check_result(self.path_id, False, core_size)
            return False, conflicting_conditions if conflicting_conditions else None
        else:
            log_test_path_z3_unknown(self.path_id, "builder_check")
            log_test_path_builder_check_result(self.path_id, False, None)
            return False, None

    def check_feasibility_and_find_inputs(self) -> bool:
        """Performs a full Z3 check for path feasibility and attempts to generate test inputs

        Checks the satisfiability of the path's `effective_conditions` using Z3
        Updates the `is_feasible` attribute
        If the path is feasible (SAT), it attempts to extract a model and populate
        the `test_inputs` attribute

        Returns:
            True if the path is feasible, False otherwise
        """
        self.test_inputs = None
        self.is_feasible = False

        conditions_to_check: List[Expression] = self.effective_conditions
        context_str = "effective_formula_check"

        if not conditions_to_check:
             self.is_feasible = True
             log_test_path_no_conditions(self.path_id, context_str)
             return True
        log_test_path_full_check_start(self.path_id)
        ctx = Context()
        solver = Solver(ctx=ctx)
        solver.set("unsat_core", True)
        solver.set("model", True)

        tracker_map: Dict[str, Expression] = {}
        formula_constraints: List[BoolRef] = []

        for i, cond in enumerate(conditions_to_check):
             z3_ref = cond.to_z3(context=ctx)
             if z3_ref is not None:
                 tracker_name = f"track_path_{self.path_id}_cond_{i}"
                 tracker_ref = Bool(tracker_name, ctx=ctx)
                 solver.assert_and_track(cast(BoolRef, z3_ref), tracker_ref)
                 tracker_map[tracker_name] = cond
                 formula_constraints.append(cast(BoolRef, z3_ref))

        if not formula_constraints:
            self.is_feasible = True
            log_test_path_full_check_result(self.path_id, True, self.test_inputs is not None)
            return True

        log_test_path_z3_call(self.path_id, context_str, len(formula_constraints))
        result = solver.check()

        if result == sat:
            self.is_feasible = True
            model = solver.model()
            if model:
                self.test_inputs = self._extract_inputs_from_model(model)
            log_test_path_full_check_result(self.path_id, True, self.test_inputs is not None)
            return True
        elif result == unsat:
            self.is_feasible = False
            log_test_path_full_check_result(self.path_id, False, False)
            return False
        else:
            self.is_feasible = False
            log_test_path_z3_unknown(self.path_id, context_str)
            log_test_path_full_check_result(self.path_id, False, False)
            return False

    def to_str(self) -> str:
        """Generates a detailed string representation of the TestPath

        Includes the sequence of node IDs, the conditions associated with each edge
        (marking conditions omitted from `effective_conditions`), the feasibility status,
        and the generated test inputs (if available)

        Returns:
            A formatted string describing the test path
        """
        node_ids = [str(node.node_id) for node in self.nodes]
        path_str = " -> ".join(node_ids)
        conditions_str = "\n  Conditions:\n"
        conditions_count = 0
        effective_cond_set = set(self.effective_conditions)

        for i, (u, v) in enumerate(zip(self.nodes[:-1], self.nodes[1:])):
            edge = (u, v)
            cond = self.edge_conditions_map.get(edge)
            if cond:
                 conditions_count += 1
                 cond_c = cond.to_c()
                 omitted_marker = "" if cond in effective_cond_set else " [yellow](omitted)[/yellow]"
                 conditions_str += f"    {i}: ({u.node_id} -> {v.node_id}): {cond_c}{omitted_marker}\n"

        if conditions_count == 0:
             conditions_str = "\n  Conditions: None\n"

        feasible_str = f"Feasible: {self.is_feasible}"
        inputs_str = ""
        if self.test_inputs is not None:
             inputs_str = "\n  Inputs: " + ", ".join(f"{k}={v}" for k, v in self.test_inputs.items())
        elif self.is_feasible:
             inputs_str = "\n  Inputs: (No specific inputs required or found)"


        if self.effective_conditions:
            effective_cond_c_list = [eff_cond.to_c() for eff_cond in self.effective_conditions]
            effective_cond_str = "\n  Effective Formula: " + " AND ".join(effective_cond_c_list)
        else:
            effective_cond_str = "\n  Effective Formula: None"

        return f"TestPath ID: {self.path_id}\n  Nodes: {path_str}{conditions_str}{effective_cond_str}\n  {feasible_str}{inputs_str}"

    def _extract_inputs_from_model(self, model: ModelRef) -> Optional[Dict[str, int]]:
        """Extracts integer variable assignments from a Z3 model

        Iterates through the model declarations, extracting names and their
        corresponding integer interpretations

        Args:
            model: The Z3 ModelRef obtained from a satisfiable check

        Returns:
            A dictionary mapping variable names to their integer values,
            or None if the model is None or contains no integer assignments
        """
        inputs: Dict[str, int] = {}
        if model is None:
            return None

        for decl in model.decls():
            var_name = decl.name()
            val_ref = model.get_interp(decl)

            if val_ref is None: continue

            if isinstance(val_ref, IntNumRef):
                inputs[var_name] = cast(IntNumRef, val_ref).as_long()


        return inputs if inputs else None

    def get_effective_formula_str(self) -> str:
        """Returns the effective path conditions as a single string joined by AND"""
        cond_strs = [cond.to_c() for cond in self.effective_conditions if hasattr(cond, 'to_c')]
        return " AND ".join(cond_strs)

    def generate_inputs_for_display(self) -> Optional[Dict[str, Any]]:
        """Returns a copy of the test inputs dictionary

        Currently, this method simply returns a copy of the `test_inputs` dictionary
        It's intended as a potential place for future formatting or simplification
        of test inputs for display purposes

        Returns:
            A dictionary containing the test inputs, or None if `test_inputs` is None
        """
        if self.test_inputs is None:
            return None

        display_inputs: Dict[str, Any] = {}
        for name, value in self.test_inputs.items():
             display_inputs[name] = value

        return display_inputs