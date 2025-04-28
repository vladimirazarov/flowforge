"""
Tunes CFG complexity (SHTV) by adding operations while preserving path feasibility

Avoids costly re-solving by using heuristics to add operations that don't alter
control flow decisions based on initial input parameter values

Heuristics:
1.  **Read-Only Params + State:** Input parameters (read-only) control branching
    State variables (modifiable) absorb complexity operations. Guarantees feasibility
2.  **Delayed Input Mod + State:** Input parameters can be modified *after* being
    used (or if unused) for the immediate branch decision from a block. State
    variables are freely modified. Balances complexity addition with dynamic code
    appearance while preserving feasibility

The Delayed Input Modification heuristic is preferred. Loop termination constraints
are respected
"""

from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING, Set, Dict, Tuple, Callable
from enum import Enum, auto
import networkx as nx
from dataclasses import dataclass, field
from collections import deque

from rich.console import Console
from loguru import logger

from src.builders.oper_builder.operations_random_factory import OperationsBuilder
from src.core.nodes.cfg_basic_block_node import CFGBasicBlockNode
from src.core.cfg_content.operation import Operation, ArithmeticOperation, LogicalOperation, UnaryOperation, ReturnOperation, DeclarationOperation
from src.core.nodes.must_be import MustBeSet
from src.core.cfg_content.expression import Expression, VariableExpression, Constant
from src.core.cfg_content.variable import Variable
from src.config.config import AppConfig

from src.utils.logging_helpers import (
    log_phase, 
    print_tuning_start_info, 
    print_tuning_results,
    print_complexity_tuning_node_details,
    log_with_node,
    log_tuner_attempt,
    log_tuner_select_node,
    log_tuner_constraints,
    log_tuner_rhs_vars,
    log_tuner_lhs_vars,
    log_tuner_op_creation_attempt,
    log_tuner_op_created,
    log_tuner_op_skipped,
    log_tuner_stats_update,
    log_tuner_must_be_add,
    print_code,
    log_tuner_state_vars_created,
    log_tuner_downstream_check_start,
    log_tuner_downstream_param_modified,
    log_tuner_downstream_param_used_cond,
    log_tuner_downstream_path_exited,
    log_tuner_downstream_cycle_detected,
    log_tuner_downstream_check_safe,
    log_tuner_downstream_check_unsafe
)

if TYPE_CHECKING:
    from src.core.cfg.cfg import CFG


__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

DEFAULT_MAX_TOTAL_OPS = 100
DEFAULT_MAX_TOTAL_ATTEMPTS = 1000

class OperationType(Enum):
    ASSIGNMENT = auto()
    ARITHMETIC = auto()

@dataclass
class ComplexityTuner:
    """Adds operations to a CFG to increase SHTV complexity while preserving path feasibility

    Uses the configured heuristics and `OperationsBuilder` to iteratively add
    operations to CFG nodes until the target SHTV complexity is met or
    other limits are reached

    Attributes:
        cfg: The Control Flow Graph to tune
        app_config: Application configuration settings
        operation_factory: Factory instance used to create operations
        console: Rich console instance for output
    """
    cfg: CFG
    app_config: AppConfig

    operation_factory: OperationsBuilder = field(init=False)
    console: Console = field(init=False)

    def __post_init__(self):
        self.operation_factory = OperationsBuilder(app_config=self.app_config)
        self.console = Console()
        self._create_initial_state_variables()
        self._create_and_declare_variables()
        self.cfg.context.create_result_variable() 

    def _create_initial_state_variables(self):
        """Create initial state variables based on configuration

        Creates variables in the CFG context according to the number specified
        in `app_config.complexity_tuner.num_state_vars`. Declarations are handled
        separately
        """
        num_state_vars = self.app_config.complexity_tuner.num_state_vars
        created_vars = []
        if num_state_vars > 0:
            for _ in range(num_state_vars):
                new_var = self.cfg.context.create_state_variable()
                if new_var:
                    created_vars.append(new_var.name)
            log_tuner_state_vars_created(created_vars)
        else:
            log_tuner_state_vars_created([])

    def _create_and_declare_variables(self):
        """Ensure 'result' variable exists and add declarations to entry node

        Checks for the 'result' variable in the CFG context, creating it if
        necessary. Gathers all non-parameter variables from the context and
        prepends `DeclarationOperation`s for them to the CFG entry node's
        instruction list if not already declared there
        """

        self.cfg.context.create_result_variable()
        
        entry_node = self.cfg.entry_node
            
        # Gather all non-parameter variables from context
        context = self.cfg.context
        variables_to_declare = [ 
            var for var in context.used_variables 
            if var not in context.used_parameters
        ]
        # Consisten order for choice to be detemenistic
        variables_to_declare.sort(key=lambda v: v.name) 

        # Create DeclarationOperation for each and prepend to entry node ops
        # Prepend in reverse order to maintain sorted declaration order at the start
        decl_ops_added_count = 0
        for var in reversed(variables_to_declare):
            already_declared = False
            for existing_op in entry_node.instructions.operations:
                 if isinstance(existing_op, DeclarationOperation) and existing_op.variable == var:
                      already_declared = True
                      break
            
            if not already_declared:
                decl_op = DeclarationOperation(variable=var, expression=Constant(0))
                entry_node.instructions.operations.insert(0, decl_op)
                log_tuner_op_created(entry_node.node_id, decl_op.to_c(), "DECLARE")
                decl_ops_added_count += 1
                
        if decl_ops_added_count > 0:
             logger.info(f"Added {decl_ops_added_count} variable declarations to entry node {entry_node.node_id}.")

    @log_phase("Complexity Tuning")
    def tune_complexity(self,
                        max_total_ops: int = DEFAULT_MAX_TOTAL_OPS,
                        max_total_attempts: int = DEFAULT_MAX_TOTAL_ATTEMPTS) -> Tuple[int, int, float]:
        """Add operations to the CFG to meet the target SHTV complexity

        Iteratively selects nodes and adds operations using `_add_operations_loop`
        until the target SHTV (`cfg.max_shtv`) is reached or limits on operations
        or attempts are exceeded. Adds a final return operation

        Args:
            max_total_ops (int): Maximum number of operations to add
            max_total_attempts (int): Maximum number of attempts to find and add operations

        Returns:
            Tuple[int, int, float]: Number of operations added, number of attempts made,
                                    and the final SHTV complexity of the CFG
        """
        initial_shtv = self.cfg.shtv
        target_max_shtv = self.cfg.max_shtv
        all_nodes = [node for node in self.cfg.graph.nodes()] 
        print_tuning_start_info(self.console, initial_shtv, target_max_shtv)
        print_complexity_tuning_node_details(self.console, all_nodes, title="Initial Node SHTV Status")

        self._populate_must_be_constraints()
        ops_added, attempts, _ = self._add_operations_loop(max_total_ops, max_total_attempts) 
        
        self._add_return_operation()

        final_shtv = self.cfg.shtv # Recalculate SHTV after adding return
        all_nodes_final = [node for node in self.cfg.graph.nodes()]
        print_tuning_results(self.console, ops_added, attempts, final_shtv, self.cfg.max_shtv)
        print_complexity_tuning_node_details(self.console, all_nodes_final, title="Final Node SHTV Status")
        print_code(self.cfg, console=self.console)
        return ops_added, attempts, final_shtv

    def _populate_must_be_constraints(self):
        """Populate MustBeSet constraints for each node using BFS and edge conditions

        Initializes `MustBeSet` for each node. Performs a Breadth-First Search
        starting from the entry node. For nodes with a single predecessor, it adds
        the condition from the incoming edge (if any) to the node's `MustBeSet`
        """
        for node in self.cfg.graph.nodes():
            if isinstance(node, CFGBasicBlockNode):
                node.must_be_set = MustBeSet()
                node.used_variables = set()
                node.arithmetic_operations_count = 0
                node.logical_operations_count = 0

        graph = self.cfg.graph
        entry_node = self.cfg.entry_node

        queue = list(nx.bfs_tree(graph, source=entry_node))

        for node in queue:
            if not isinstance(node, CFGBasicBlockNode) or node == entry_node:
                continue

            predecessors = list(graph.predecessors(node))
            if len(predecessors) == 1:
                pred = predecessors[0]
                edge_data = graph.get_edge_data(pred, node)
                if edge_data and isinstance(edge_data['condition'], Expression):
                    condition = edge_data['condition']
                    node.must_be_set.add(condition)
                    log_tuner_must_be_add(node.node_id, condition.to_c())

    def _get_constrained_variables(self, node: CFGBasicBlockNode) -> Set[Variable]:
        """Get all variables mentioned in a node's MustBeSet constraints

        Args:
            node (CFGBasicBlockNode): The node to check

        Returns:
            Set[Variable]: All variables present in the node's `must_be_set`
        """
        constrained = set()
        assert node.must_be_set is not None
        for constraint in node.must_be_set:
            constrained.update(constraint.get_variables())
        return constrained

    def _add_operations_loop(self, max_total_ops: int, max_total_attempts: int) -> Tuple[int, int, float]:
        """Add operations iteratively until SHTV target is met or limits are reached

        Args:
            max_total_ops (int): Maximum number of operations to add
            max_total_attempts (int): Maximum attempts to find/add operations

        Returns:
            Tuple[int, int, float]: Number of operations added, attempts made, final SHTV
        """
        all_nodes = [node for node in self.cfg.graph.nodes()]

        ops_added = 0
        attempts = 0
        allowed_var_idx = 0
        node_last_op_type = {}

        while self.cfg.shtv < self.cfg.max_shtv and ops_added < max_total_ops and attempts < max_total_attempts:
            attempts += 1
            log_tuner_attempt(attempts, max_total_attempts, self.cfg.shtv, self.cfg.max_shtv)

            target_node = self._select_node_to_tune(all_nodes)
            if not target_node:
                log_tuner_op_skipped(0, "No eligible nodes found for tuning")
                break

            log_tuner_select_node(target_node.node_id, target_node.max_shtv - target_node.shtv)

            constrained_vars, context_vars, parameters = self._gather_node_constraints(target_node)
            outgoing_condition_vars = self._get_outgoing_edge_condition_vars(target_node)
            log_tuner_constraints(target_node.node_id, constrained_vars, self._get_loop_termination_constraints(target_node), outgoing_condition_vars, context_vars, parameters)
            
            generated_op, chosen_op_type, next_allowed_var_idx = self._try_create_operation(
                target_node,
                constrained_vars,
                context_vars,
                parameters,
                node_last_op_type,
                allowed_var_idx
            )

            allowed_var_idx = next_allowed_var_idx

            if generated_op and chosen_op_type:
                target_node.instructions.operations.append(generated_op)
                ops_added += 1
                node_last_op_type[target_node.node_id] = chosen_op_type
                self._update_node_stats(target_node, generated_op)
                log_tuner_op_created(target_node.node_id, generated_op.to_c(), chosen_op_type.name)
            else:
                log_tuner_op_skipped(target_node.node_id, "Operation factory failed to produce an operation.")

        return ops_added, attempts, self.cfg.shtv

    def _select_node_to_tune(self, eligible_nodes: List[CFGBasicBlockNode]) -> Optional[CFGBasicBlockNode]:
        """Select the next node to add an operation to based on SHTV gap

        Finds the node with the largest difference between its maximum possible
        SHTV (`max_shtv`) and its current SHTV (`shtv`)

        Args:
            eligible_nodes (List[CFGBasicBlockNode]): Nodes to consider

        Returns:
            Optional[CFGBasicBlockNode]: The node with the largest SHTV gap, or None
        """
        nodes_below_max = [node for node in eligible_nodes if node.shtv < node.max_shtv]
        if not nodes_below_max:
            return None
        return max(nodes_below_max, key=lambda node: node.max_shtv - node.shtv)

    def _gather_node_constraints(self, node: CFGBasicBlockNode) -> Tuple[Set[Variable], List[Variable], Set[Variable]]:
        """Gather MustBeSet, loop termination constraints, context vars, and parameters

        Args:
            node (CFGBasicBlockNode): The node to gather constraints for

        Returns:
            Tuple[Set[Variable], List[Variable], Set[Variable]]: Constrained variables,
                all context variables, and input parameters
        """
        must_be_set_vars = self._get_constrained_variables(node)
        critical_term_vars = self._get_loop_termination_constraints(node)
        total_constrained_vars = must_be_set_vars.union(critical_term_vars)
        context_vars, parameters = self._get_context_variables_and_parameters(node.node_id)
        return total_constrained_vars, context_vars, parameters

    def _get_loop_termination_constraints(self, node: CFGBasicBlockNode) -> Set[Variable]:
        """Get critical termination variables if the node is part of a loop

        Checks the pre-computed loop analysis info (`cfg.analysis`) to find if
        the node belongs to any loop body and returns the set of variables marked
        as critical for that loop's termination

        Args:
            node (CFGBasicBlockNode): The node to check

        Returns:
            Set[Variable]: Variables critical for termination in loops containing this node
        """
        critical_term_vars = set()
        if self.cfg.analysis is None:
            return critical_term_vars

        loop_analysis_info = self.cfg.analysis.get_loop_info()
        if not loop_analysis_info:
             return critical_term_vars

        for _, loop_info in loop_analysis_info.items():
            loop_body = loop_info.get('body')
            if isinstance(loop_body, set) and node in loop_body:
                criticals_raw = loop_info.get('critical_termination_vars')
                if isinstance(criticals_raw, set):
                    critical_vars = {v for v in criticals_raw if isinstance(v, Variable)}
                    critical_term_vars.update(critical_vars)

        return critical_term_vars

    def _get_context_variables_and_parameters(self, node_id: int) -> Tuple[List[Variable], Set[Variable]]:
        """Retrieve used variables and parameters from the CFG context

        Args:
            node_id (int): The ID of the node (used for potential future context)

        Returns:
            Tuple[List[Variable], Set[Variable]]: List of all used variables and set
                of input parameter variables from the context
        """
        context = self.cfg.context
        context_vars = list(context.used_variables)
        parameters = context.used_parameters
        return context_vars, parameters

    def _get_outgoing_edge_condition_vars(self, node: CFGBasicBlockNode) -> Set[Variable]:
        """Get all variables used in conditions of edges leaving this node

        Args:
            node (CFGBasicBlockNode): The node to check outgoing edges from

        Returns:
            Set[Variable]: Variables used in the conditions of edges originating from the node
        """
        outgoing_vars = set()
        for successor in self.cfg.graph.successors(node):
            edge_data = self.cfg.graph.get_edge_data(node, successor)
            if edge_data and isinstance(edge_data['condition'], Expression):
                outgoing_vars.update(edge_data['condition'].get_variables())
        return outgoing_vars

    def _get_prioritized_operation_types(
        self,
        node_id: int,
        node_last_op_type: Dict[int, OperationType]
    ) -> List[OperationType]:
        """Determine the prioritized list of operation types to try

        Prioritizes Assignment, then Arithmetic. If a type was added last time
        it's moved to the end of the priority list to encourage variety

        Args:
            node_id (int): ID of the target node
            node_last_op_type (Dict[int, OperationType]): Tracks the last type added per node

        Returns:
            List[OperationType]: Prioritized list of operation types
        """
        op_types_priority = [
            OperationType.ASSIGNMENT,
            OperationType.ARITHMETIC
        ]
        
        last_type = node_last_op_type.get(node_id)
        if last_type:
            op_types_priority.remove(last_type)
            op_types_priority.append(last_type)
                
        return op_types_priority

    def _attempt_operation_creation_loop(
        self,
        prioritized_types: List[OperationType],
        factory_methods: Dict[OperationType, Callable[[], Optional[Operation]]]
    ) -> Tuple[Optional[Operation], Optional[OperationType]]:
        """Iterate through prioritized types and attempt to create an operation

        Args:
            prioritized_types (List[OperationType]): Ordered list of types to attempt
            factory_methods (Dict[OperationType, Callable]): Callables to create ops

        Returns:
            Tuple[Optional[Operation], Optional[OperationType]]: The created operation
                and its type, or (None, None) if none could be created
        """
        for current_op_type in prioritized_types:
            log_tuner_op_creation_attempt(self.target_node_id_for_logging, current_op_type.name, prioritized_types)
            try_create_func = factory_methods.get(current_op_type)
            if try_create_func:
                created_op = try_create_func()
                if created_op:
                    return created_op, current_op_type

        return None, None

    def _is_param_used_conditionally_downstream(
        self,
        start_node: CFGBasicBlockNode,
        param_to_check: Variable
    ) -> bool:
        """
        Checks if param_to_check is used in any edge condition on any path
        starting from a successor of start_node, before being modified or exiting.
        Returns True if a conditional use is found before modification/exit, False otherwise
        """
        start_node_id = start_node.node_id
        param_name = param_to_check.name
        log_tuner_downstream_check_start(start_node_id, param_name)

        queue = deque()
        # Initialize queue with (node_to_explore, visited_on_this_path_set)
        for initial_successor in self.cfg.graph.successors(start_node):
            queue.append((initial_successor, {start_node, initial_successor}))

        exit_node = self.cfg.exit_node
        overall_used_downstream = False

        while queue:
            current_node, path_visited = queue.popleft()
            current_node_id = current_node.node_id

            if current_node == exit_node:
                log_tuner_downstream_path_exited(current_node_id, param_name)
                continue # Path reached exit safely for this parameter

            # 1. Check if param is modified *within* current_node
            modified_here = False
            if current_node.instructions:
                for op in current_node.instructions.operations:
                    modified_var = op.get_modified_variable()
                    if modified_var == param_to_check:
                        modified_here = True
                        log_tuner_downstream_param_modified(current_node_id, param_name)
                        break # Param modified, this path is safe from here onwards

            if modified_here:
                continue # Don't explore successors from here; modification occurred

            # 2. Check edge conditions and explore successors
            for successor in self.cfg.graph.successors(current_node):
                successor_id = successor.node_id
                edge_data = self.cfg.graph.get_edge_data(current_node, successor)
                if edge_data: 
                    condition = edge_data['condition']
                    if param_to_check in condition.get_variables():
                        log_tuner_downstream_param_used_cond(current_node_id, successor_id, param_name)
                        overall_used_downstream = True 

                # If successor already visited on this specific path, skip
                if successor in path_visited:
                    # Cycle detected
                    log_tuner_downstream_cycle_detected(current_node_id, param_name)
                    continue

                # Add successor to queue for further exploration
                new_path_visited = path_visited | {successor}
                queue.append((successor, new_path_visited))

        # After checking all paths
        if overall_used_downstream:
            log_tuner_downstream_check_unsafe(start_node_id, param_name)
            return True
        else:
            log_tuner_downstream_check_safe(start_node_id, param_name)
            return False

    def _try_create_operation(
        self,
        target_node: CFGBasicBlockNode,
        constrained_vars: Set[Variable],
        context_vars: List[Variable],
        parameters: Set[Variable],
        node_last_op_type: Dict[int, OperationType],
        allowed_var_idx: int
    ) -> Tuple[Optional[Operation], Optional[OperationType], int]:
        """Attempt to create an operation for the node, respecting constraints

        Determines valid LHS and RHS variables based on constraints (MustBeSet,
        loop termination, outgoing edge conditions). Prioritizes operation types
        and uses the `OperationsBuilder` to generate an operation

        Args:
            target_node: The node to add an operation to
            constrained_vars: Variables that cannot be modified (LHS)
            context_vars: All variables available in the context
            parameters: Subset of context_vars that are input parameters
            node_last_op_type: History of the last operation type added to this node
            allowed_var_idx: Index for rotating RHS variables

        Returns:
            Tuple[Optional[Operation], Optional[OperationType], int]: The generated
                operation, its type, and the next index for RHS rotation
        """
        self.target_node_id_for_logging = target_node.node_id
        outgoing_condition_vars = self._get_outgoing_edge_condition_vars(target_node)
        
        state_vars = {v for v in context_vars if v not in parameters}
        result_var = self.cfg.context.get_variable("result")
        
        assignable_state_vars = {v for v in state_vars if v not in constrained_vars}
        if result_var and result_var not in constrained_vars:
            assignable_state_vars.add(result_var)

        # Modifiable for ARITHMETIC (includes params meeting local/immediate/downstream conditions)
        modifiable_arith_lhs_set = set(assignable_state_vars) # Start with state/result vars
        log = log_with_node(target_node.node_id)
        for param in parameters:
            param_name = param.name
            is_constrained = param in constrained_vars
            is_used_outgoing = param in outgoing_condition_vars
            is_used_cond_downstream = False 

            if not is_constrained and not is_used_outgoing:
                is_used_cond_downstream = self._is_param_used_conditionally_downstream(target_node, param)

            if not is_constrained and not is_used_outgoing and not is_used_cond_downstream:
                modifiable_arith_lhs_set.add(param)
                log.trace(f"Param '{param_name}' OK for Arithmetic LHS (Not constrained, not outgoing, not downstream cond use)")
            else:
                reasons = []
                if is_constrained: reasons.append("constrained (MustBe/LoopTerm)")
                if is_used_outgoing: reasons.append("outgoing condition")
                if is_used_cond_downstream: reasons.append("downstream conditional use")
                log.trace(f"Param '{param_name}' EXCLUDED from Arithmetic LHS. Reasons: [{', '.join(reasons)}]")

        modifiable_arith_lhs_list = sorted(list(modifiable_arith_lhs_set), key=lambda v: v.name)

        # Modifiable for ASSIGNMENT (STRICTLY excludes parameters)
        modifiable_assign_lhs_list = sorted(list(assignable_state_vars), key=lambda v: v.name)

        log_tuner_lhs_vars(target_node.node_id, modifiable_arith_lhs_list, "Arithmetic")
        log_tuner_lhs_vars(target_node.node_id, modifiable_assign_lhs_list, "Assignment")

        # Check if any LHS is possible before proceeding
        if not modifiable_arith_lhs_list and not modifiable_assign_lhs_list:
            log_tuner_op_skipped(target_node.node_id, "No modifiable LHS variables found for any operation type.")
            return None, None, allowed_var_idx
        
        available_rhs_vars = list(context_vars)
        available_rhs_vars.sort(key=lambda var: var.name)
        log_tuner_rhs_vars(target_node.node_id, available_rhs_vars)
        
        num_rhs_allowed = len(available_rhs_vars)
        current_rotation_idx = allowed_var_idx % num_rhs_allowed
        rotated_rhs_vars = available_rhs_vars[current_rotation_idx:] + available_rhs_vars[:current_rotation_idx]
        next_allowed_var_idx = (allowed_var_idx + 1) % num_rhs_allowed

        prioritized_types = self._get_prioritized_operation_types(target_node.node_id, node_last_op_type)

        factory_methods: Dict[OperationType, Callable[[], Optional[Operation]]] = {}
        if modifiable_assign_lhs_list:
            factory_methods[OperationType.ASSIGNMENT] = lambda: self.operation_factory._create_assignment(
                modifiable_lhs_vars=modifiable_assign_lhs_list
            )
        if modifiable_arith_lhs_list:
            factory_methods[OperationType.ARITHMETIC] = lambda: self.operation_factory._create_arithmetic(
                modifiable_lhs_vars=modifiable_arith_lhs_list,
                available_rhs_vars=rotated_rhs_vars,
                input_parameters=parameters
            )

        if not factory_methods:
            log_tuner_op_skipped(target_node.node_id, "No valid operation types possible with current LHS variable constraints.")
            return None, None, allowed_var_idx

        generated_op, chosen_op_type = self._attempt_operation_creation_loop(
            prioritized_types,
            factory_methods
        )

        return generated_op, chosen_op_type, next_allowed_var_idx

    def _update_node_stats(self, node: CFGBasicBlockNode, operation: Operation):
        """Update node complexity counts and used variables after adding an operation

        Increments arithmetic/logical operation counts on the node based on the
        added operation type. Adds the operation's used variables (LHS and RHS)
        to the node's `used_variables` set

        Args:
            node (CFGBasicBlockNode): The node that was modified
            operation (Operation): The operation that was added
        """
        modified_var = operation.get_modified_variable()
        rhs_vars = operation.get_rhs_variables() or set()

        if isinstance(operation, ArithmeticOperation):
            node.arithmetic_operations_count += 1
        if isinstance(operation, LogicalOperation):
            node.logical_operations_count +=1

        vars_to_add = rhs_vars.copy()
        vars_to_add.add(modified_var)
        node.used_variables.update(vars_to_add)

        log_tuner_stats_update(node.node_id, node.arithmetic_operations_count, node.logical_operations_count, len(node.used_variables))

    def _add_return_operation(self):
        """Add a 'return result;' operation to the designated CFG exit node

        Retrieves the designated `exit_node` from the CFG. If it exists and
        doesn't already end with a `ReturnOperation`, appends a new
        `ReturnOperation` that returns the 'result' variable from the CFG context
        """
        result_var = self.cfg.context.get_variable("result")
        if not result_var:
            logger.error("Could not find 'result' variable in CFG context. Cannot add return statement.")
            return

        exit_node = self.cfg.exit_node 
        
            
        return_op = ReturnOperation(variable=result_var, expression=VariableExpression(result_var))
        
        # Check if the last operation is already a return
        last_op = exit_node.instructions.operations[-1] if exit_node.instructions.operations else None
        if isinstance(last_op, ReturnOperation):
            logger.trace(f"Exit node {exit_node.node_id} already ends with a return operation. Skipping.")
        else:
            exit_node.instructions.operations.append(return_op)
            log_tuner_op_created(exit_node.node_id, return_op.to_c(), "RETURN") 
            logger.info(f"Added 'return {result_var.name};' to exit node {exit_node.node_id}.")