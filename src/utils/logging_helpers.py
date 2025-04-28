from __future__ import annotations
from functools import wraps
from typing import Callable, Any, TYPE_CHECKING, Optional, List, Set, cast, Tuple, Dict

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.rule import Rule
import rich.box
from rich.tree import Tree
from rich.panel import Panel
from rich.syntax import Syntax

from src.core.nodes.cfg_basic_block_node import CFGBasicBlockNode
from src.core.cfg_content.variable import Variable
from src.core.cfg_content.expression import Expression

if TYPE_CHECKING:
    from src.core.cfg.cfg import CFG
    from src.core.paths.cfg_test_path import TestPath
    from src.core.cfg.cfg_analysis import LoopAnalysisInfo
    from src.core.fragments.fragment_forest import FragmentForest
    from src.core.fragments.cfg_fragment import Fragment
    import networkx as nx 

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"


class _NullConsole:
    def print(self, *args, **kwargs): pass
    def log(self, *args, **kwargs): pass
    def table(self, *args, **kwargs):pass
    def tree(self, *args, **kwargs):pass
    def panel(self, *args, **kwargs):pass
    def syntax(self, *args, **kwargs):pass
    def rule(self, *args, **kwargs):pass
    def box(self, *args, **kwargs):pass
    @property
    def size(self):
        class _Size:
            width = 80  
        return _Size()
    def __getattr__(self, name):
        return self.print

def log_phase(phase_name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger.info(f"🔄 {phase_name} started")
            result = func(*args, **kwargs)
            logger.info(f"✅ {phase_name} completed")
            return result
        return wrapper
    return decorator

def log_with_node(node_id: int):
    return logger.bind(node_id=node_id)


def print_tuning_results(console: Console, ops_added: int, attempts: int, final_shtv: float, target_max_shtv: float):
    """Prints the final results table for complexity tuning"""
    console.print(Rule("[bold blue]🏁 Complexity Tuning Finished[/bold blue]"))
    table = Table(title="Tuning Results", show_header=True, header_style="bold magenta", box=rich.box.MINIMAL)
    table.add_column("Metric", style="dim cyan", width=25)
    table.add_column("Value", style="green")
    table.add_row("Total Operations Added", str(ops_added))
    table.add_row("Total Attempts", str(attempts))
    table.add_row("Final CFG SHTV", f"{final_shtv:.2f}")
    table.add_row("Target Max SHTV", f"{target_max_shtv:.2f}")
    console.print(table)
    console.print(Rule())

def print_tuning_start_info(console: Console, initial_shtv: float, target_max_shtv: float):
    """Prints the initial setup table for complexity tuning"""
    console.print(Rule("[bold blue]🔩 Starting Complexity Tuning[/bold blue]"))
    table = Table(title="Tuner Setup", show_header=True, header_style="bold magenta", box=rich.box.MINIMAL)
    table.add_column("Parameter", style="dim cyan", width=25)
    table.add_column("Value", style="green")
    table.add_row("Initial CFG SHTV", f"{initial_shtv:.2f}")
    table.add_row("Target Max SHTV", f"{target_max_shtv:.2f}")
    console.print(table)
    console.print(Rule())


def print_complexity_tuning_node_details(console: Console, nodes: List[CFGBasicBlockNode], title: str = "Node SHTV Status"):
    """Prints a table detailing the SHTV and its contributing factors for each node"""
    console.print(Rule(f"[bold blue]{title}[/bold blue]"))
    table = Table(title=title, show_header=True, header_style="bold magenta", box=rich.box.MINIMAL)
    table.add_column("Node ID", style="dim cyan", width=8, justify="right")
    table.add_column("Depth", style="blue", width=5, justify="right")
    table.add_column("Position", style="blue", width=8, justify="right")
    table.add_column("Logic Ops", style="cyan", width=9, justify="right")
    table.add_column("Arith Ops", style="cyan", width=9, justify="right")
    table.add_column("Used Vars", style="cyan", width=9, justify="right")
    table.add_column("Current SHTV", style="green", width=12, justify="right")
    table.add_column("Max SHTV", style="yellow", width=10, justify="right")

    # Filter for CFGBasicBlockNode instances and sort by ID
    block_nodes = [n for n in nodes if isinstance(n, CFGBasicBlockNode)]
    sorted_nodes = sorted(block_nodes, key=lambda n: n.node_id)

    for node in sorted_nodes:
        # Direct access per pythoncoding.mdc rules (assume attrs exist)
        node_id_str = str(node.node_id)
        depth_str = str(node.depth)
        position_str = str(node.position)
        logic_ops_str = str(node.logical_operations_count)
        arith_ops_str = str(node.arithmetic_operations_count)
        used_vars_str = str(len(node.used_variables))
        current_shtv_str = f"{node.shtv:.2f}"
        max_shtv_str = f"{node.max_shtv:.2f}"

        table.add_row(
            node_id_str,
            depth_str,
            position_str,
            logic_ops_str,
            arith_ops_str,
            used_vars_str,
            current_shtv_str, 
            max_shtv_str
        )

    if not sorted_nodes:
        console.print("[yellow]No CFGBasicBlockNodes found to display SHTV details.[/yellow]")
    else:
        console.print(table)
    console.print(Rule())


# Constraint Logging 
def log_must_be_constraints(console: Console, cfg: 'CFG'):
    """Prints a table summarizing MustBeSet constraints derived from dominator logic"""
    table = Table(title="Populated MustBeSet Constraints (Dominator Logic)", show_header=True, header_style="bold magenta", box=rich.box.ROUNDED)
    table.add_column("Node ID", style="dim cyan", width=10)
    table.add_column("Constraints", style="green")

    for node in cfg.graph.nodes():
        if isinstance(node, CFGBasicBlockNode) and hasattr(node, 'must_be_set'):
            must_be_set_attr = getattr(node, 'must_be_set', None)
            if must_be_set_attr is not None:
                if isinstance(must_be_set_attr, (list, set, tuple)):
                    constraints_str = " AND ".join(f"({c.to_c()})" for c in must_be_set_attr)
                    if constraints_str:
                        table.add_row(str(node.node_id), constraints_str)
                else:
                    logger.warning(f"Node {node.node_id} has non-iterable must_be_set attribute ({type(must_be_set_attr)}). Skipping in log.")
            else:
                logger.trace(f"Node {node.node_id} must_be_set is None.")

    console.print(table)


def log_updated_vars(node: CFGBasicBlockNode, newly_added: Set[Variable]):
    """Log information about newly added variables to a node"""
    log = log_with_node(node.node_id)
    log.trace(f"Updated vars: {[v.name for v in newly_added]} (total: {len(node.used_variables)}) -> Node used_vars: {[v.name for v in node.used_variables]}")

def log_node_constraints(node: CFGBasicBlockNode, constraints: Set[Variable]):
    """Log information about constrained variables for a node"""
    log = log_with_node(node.node_id)
    log.trace(f"Constrained vars: {[v.name for v in constraints]}")


def print_edge_cond_start_info(console: Console, max_retries: int, num_paths: Optional[int]):
    """Prints the initial setup table for the Edge Condition Builder"""
    console.print(Rule("[bold blue]🚦 Starting Edge Condition Builder[/bold blue]"))
    start_table = Table(title="Initial Setup", show_header=True, header_style="bold magenta", box=rich.box.MINIMAL)
    start_table.add_column("Parameter", style="dim cyan", width=25)
    start_table.add_column("Value", style="green")
    start_table.add_row("Max Feasibility Retries", str(max_retries))
    if num_paths is not None:
        start_table.add_row("Paths to Check", str(num_paths))
    else:
        start_table.add_row("Paths to Check", "[yellow]Unavailable[/yellow]")
    console.print(start_table)
    console.print(Rule())

def print_loop_analysis_summary(console: Console, loop_analysis_results: Optional[LoopAnalysisInfo], graph: 'nx.DiGraph'):
    """Prints a table summarizing the loop analysis results"""
    if not loop_analysis_results:
        logger.info("No loop analysis results to summarize.")
        return

    logger.info("--- Loop Analysis Summary --- [Rich Table]")
    table = Table(title="Loop Analysis Summary", show_header=True, header_style="bold magenta", box=rich.box.ROUNDED)
    table.add_column("Loop #", style="dim cyan", width=8)
    table.add_column("Header", style="cyan", width=8)
    table.add_column("Body Nodes", style="green")
    table.add_column("Latch Nodes", style="yellow")
    table.add_column("Exit Edges (Edge, Cond)", style="red")

    for loop_id, (header, loop_info) in enumerate(loop_analysis_results.items()):
        header_id = header.node_id
        body_nodes = cast(Set[CFGBasicBlockNode], loop_info.get('body', set()))
        latch_nodes = cast(Set[CFGBasicBlockNode], loop_info.get('latch_nodes', set()))
        exit_edge_conditions = loop_info.get('exits')

        body_ids_str = ", ".join(map(str, sorted([n.node_id for n in body_nodes])))
        latch_ids_str = ", ".join(map(str, sorted([n.node_id for n in latch_nodes])))
        
        exit_details = []
        if isinstance(exit_edge_conditions, dict):
            for (u, v), condition in exit_edge_conditions.items():
                u_id = getattr(u, 'node_id', 'N/A')
                v_id = getattr(v, 'node_id', 'N/A')
                cond_str = getattr(condition, 'to_c', lambda: "[Invalid]")() if condition else "[None]"
                exit_details.append(f"({u_id}->{v_id}, {cond_str})")
        elif exit_edge_conditions:
             logger.warning(f"Loop {loop_id} has unexpected 'exits' format: {type(exit_edge_conditions)}. Skipping exit details.")
             
        exit_str = "; ".join(exit_details) if exit_details else "None"

        table.add_row(str(loop_id), str(header_id), body_ids_str, latch_ids_str, exit_str)

    console.print(table)
    logger.info("--- End Loop Analysis Summary --- [Rich Table]")

def print_edge_cond_finish_info(console: Console, final_infeasible_paths: List['TestPath']):
    """Prints the final status table and details of remaining infeasible paths for the Edge Condition Builder"""
    console.print(Rule("[bold blue]🏁 Edge Condition Builder Finished[/bold blue]"))
    final_summary_table = Table(title="Final Status", show_header=True, header_style="bold magenta", box=rich.box.MINIMAL)
    final_summary_table.add_column("Metric", style="dim cyan", width=25)
    final_summary_table.add_column("Result", style="green")

    num_remaining_infeasible = len(final_infeasible_paths)
    status = "[bold green]Success[/bold green]" if num_remaining_infeasible == 0 else "[bold yellow]Warning[/bold yellow]"
    final_summary_table.add_row("Overall Status", status)
    final_summary_table.add_row("Remaining Infeasible Paths", str(num_remaining_infeasible))

    console.print(final_summary_table)

    if num_remaining_infeasible > 0:
        console.print(Rule("[bold yellow]⚠️ Details of Remaining Infeasible Paths[/bold yellow]"))
        for path in final_infeasible_paths:
            path_details_str = path.to_str()
            console.print(Panel(path_details_str, title=f"Path ID: {path.path_id}", border_style="yellow"))
            console.print()
    else:
        console.print(Rule("[bold green]✅ All paths resolved.[/bold green]"))

def print_all_test_path_details(console: Console, all_test_paths: Optional[List['TestPath']], cfg: 'CFG'):
    """Prints the details of all defined test paths"""
    console.print(Rule("[bold blue]📊 Final Test Path Details[/bold blue]"))
    if not all_test_paths:
        console.print("[yellow]No test paths defined in CFG.[/yellow]")
        console.print(Rule())
        return

    console.print("Checking feasibility and extracting inputs for detailed display...")
    # Run feasibility check again here if needed for updated inputs/status
    for path in all_test_paths:
         path.check_feasibility_and_find_inputs() # Pass cfg here

    details_table = Table(title="All Defined Test Paths", show_header=True, header_style="bold magenta", box=rich.box.ROUNDED)
    details_table.add_column("Path ID", style="dim cyan", width=8)
    details_table.add_column("Nodes", style="cyan")
    details_table.add_column("Feasible", style="green", width=10)
    details_table.add_column("Inputs", style="yellow")

    for path in sorted(all_test_paths, key=lambda p: p.path_id):
        path_nodes_str = " -> ".join([str(n.node_id) for n in path.nodes])
        feasibility_str = str(path.is_feasible)
        inputs = path.generate_inputs_for_display() # Get potentially renamed inputs
        inputs_str = str(inputs) if inputs is not None else "(None)"
        details_table.add_row(
            str(path.path_id),
            path_nodes_str,
            feasibility_str,
            inputs_str
        )

    console.print(details_table)
    console.print(Rule())


def log_builder_event(level: str, message: str):
    """Generic logger for simple info/warning/error messages"""
    getattr(logger, level.lower())(message)

def log_feasibility_check_attempt(attempt: int, max_retries: int):
    logger.info(f"Feasibility check attempt {attempt}/{max_retries}...")

def log_all_paths_feasible():
    logger.success("All defined test paths are feasible (considering loop exits).")

def log_infeasible_paths_found(count: int):
    logger.warning(f"Found {count} infeasible paths. Attempting to fix...")

def log_max_retries_reached(max_retries: int):
    logger.warning(f"Maximum retries ({max_retries}) reached.")

def log_structural_infeasibility_error(count: int, paths: Set['TestPath']):
    logger.error(f"Detected {count} structurally infeasible paths that could not be resolved.")
    path_details = []
    for path in paths:
         path_id_str = getattr(path, 'path_id', f"Path starting {path.nodes[0].node_id}") 
         nodes_str = ' -> '.join([str(n.node_id) for n in path.nodes])
         path_details.append(f"  - {path_id_str}: [{nodes_str}]")
    logger.error("Structurally infeasible paths details:\n" + "\n".join(path_details))

def log_branch_pair_found(node_id: int, edge1_target: int, edge2_target: int):
    logger.debug(f"Found branch at node {node_id}: edges ->{edge1_target} and ->{edge2_target}")

def log_condition_assignment(node_id: int, target_edge_str: str, cond_str_primary: str, cond_str_secondary: Optional[str], is_loop_header: bool, type: str):
    """Logs details about condition assignment (loop, non-loop, ambiguous)"""
    prefix = f"Node {node_id}" if not is_loop_header else f"Loop Header {node_id}"
    if type == "loop_body_exit":
        log_str = f"  {prefix}: Body edge gets {cond_str_primary}, Exit edge gets {cond_str_secondary or '[MISSING]'}"
        logger.trace(log_str)
    elif type == "non_loop":
        log_str = f"{prefix} is not a loop header. Assigning conditions randomly: {target_edge_str} gets {cond_str_primary}"
        logger.trace(log_str)
    elif type == "ambiguous_loop":
        log_str = f"Could not definitively identify body/exit edges for {prefix}. Using fallback assignment (body=cond, exit=negated). Details: {cond_str_primary}"
        logger.warning(log_str)

def log_initial_conditions_assigned(count: int):
    logger.info(f"Initial conditions assigned to {count} edges.")

def log_jump_condition_set(node_id: int, cond_str: str):
    logger.debug(f"Set jump condition in node {node_id} to: {cond_str}")

def log_jump_condition_fail(node_id: int, reason: str):
    logger.warning(f"Failed to set jump condition for node {node_id}: {reason}")

def print_initial_jump_conditions(console: Console, cfg: CFG):
    """Prints a table showing the initial jump condition assigned to each branch node"""
    console.print(Rule("[bold blue]📊 Initial Jump Conditions[/bold blue]"))
    table = Table(title="Initial Node Jump Conditions", show_header=True, header_style="bold magenta", box=rich.box.ROUNDED)
    table.add_column("Node ID", style="dim cyan", width=10)
    table.add_column("Jump Condition", style="green")

    branch_nodes = []
    for node in cfg.graph.nodes():
        if node.jump is not None:
            branch_nodes.append(node)
    
    sorted_nodes = sorted(branch_nodes, key=lambda n: n.node_id)

    for node in sorted_nodes:
        condition_expr = node.jump.expression
        condition_str = condition_expr.to_c() if condition_expr is not None else "[bold yellow]None[/bold yellow]"
        table.add_row(str(node.node_id), condition_str)

    if not sorted_nodes:
        console.print("[yellow]No branching nodes with conditions found.[/yellow]")
    else:
        console.print(table)
    console.print(Rule())

def print_loop_condition_assignment_details(console: Console, header_node: CFGBasicBlockNode, 
                                        edge1: Tuple[CFGBasicBlockNode, CFGBasicBlockNode], edge1_cond: Expression, 
                                        edge2: Tuple[CFGBasicBlockNode, CFGBasicBlockNode], edge2_cond: Expression, 
                                        jump_cond: Expression, context: str, notes: Optional[str] = None):
    """Prints a table detailing the condition assignments for a loop header's branches"""
    header_id = header_node.node_id
    edge1_str = f"{edge1[0].node_id}->{edge1[1].node_id}"
    edge2_str = f"{edge2[0].node_id}->{edge2[1].node_id}"
    edge1_cond_str = edge1_cond.to_c()
    edge2_cond_str = edge2_cond.to_c()
    jump_cond_str = jump_cond.to_c()

    console.print(Rule(f"[bold cyan]Loop Header {header_id} Condition Assignment ({context})[/bold cyan]"))
    table = Table(show_header=True, header_style="bold magenta", box=rich.box.MINIMAL_HEAVY_HEAD, title=f"Assignment Details ({context})")
    table.add_column("Element", style="dim cyan", width=15)
    table.add_column("Details", style="green")

    table.add_row("Loop Header", str(header_id))
    table.add_row("Edge 1", edge1_str)
    table.add_row("  └ Condition", edge1_cond_str)
    table.add_row("Edge 2", edge2_str)
    table.add_row("  └ Condition", edge2_cond_str)
    table.add_row("Jump Condition", jump_cond_str)
    if notes:
        table.add_row("Notes", notes)

    console.print(table)
    console.print(Rule())

def log_path_check_error(path_id: Any, error_msg: str, is_attribute_error: bool = False):
    msg = f"Error processing test path {path_id}: {error_msg}."
    if is_attribute_error:
        msg += " Missing required TestPath attribute/method?"
    logger.error(msg, exc_info=True)

def log_path_infeasible(path_id: Any, node_ids: List[int]):
    logger.warning(f"Defined test path {path_id} (Nodes: {node_ids}) is infeasible.")

def log_fix_attempt_start(path_nodes_str: str):
    logger.debug(f"Attempting to fix path (nodes: {path_nodes_str})")

def log_fix_strategy_info(message: str):
    logger.debug(message)

def log_fix_condition_regenerate(origin_node_id: int, target_node_id: int, reason: str):
    logger.info(f"Regenerating condition for edge ({origin_node_id}->{target_node_id}) [{reason}].")

def log_fix_condition_update(edge_str: str, cond_str: str):
    logger.info(f"Edge {edge_str} condition updated to: {cond_str}")

def log_fix_no_branch_found(path_nodes_str: str):
     logger.warning(f"Could not find any branching edge in path {path_nodes_str} during fallback.")
     
def log_fix_no_suitable_edge(path_nodes_str: str):
     logger.warning(f"No suitable branching edge found to fix for path {path_nodes_str}. Path remains infeasible.")

def log_fix_sibling_error(origin_node_id: int):
    logger.error(f"Origin node {origin_node_id} was in branch pairs but key missing? Cannot find sibling edge.")

def log_loop_terminator_issue(message: str, error: Optional[Exception] = None):
    if error:
        logger.error(f"LoopTerminator failed: {message}", exc_info=error)
    else:
        logger.error(message)


def log_test_path_created(path_id: int):
    logger.trace(f"TestPath {path_id} created.")

def log_test_path_update_cond_start(path_id: int):
    logger.trace(f"TestPath {path_id}: Updating conditions from CFG...")

def log_test_path_update_cond_end(path_id: int, num_edge_conds: int, num_effective_conds: int):
    logger.trace(f"TestPath {path_id}: Conditions updated. Total edge conditions: {num_edge_conds}, Effective conditions for feasibility check: {num_effective_conds}.")

def log_test_path_builder_check_start(path_id: int):
    logger.trace(f"TestPath {path_id}: Starting feasibility check for builder...")

def log_test_path_builder_check_result(path_id: int, is_feasible: bool, core_size: Optional[int]):
    status = "Feasible" if is_feasible else "Infeasible"
    core_info = f" (Unsat Core Size: {core_size})" if core_size is not None else ""
    logger.debug(f"TestPath {path_id}: Builder feasibility check result: {status}{core_info}.")

def log_test_path_full_check_start(path_id: int):
    logger.trace(f"TestPath {path_id}: Starting full feasibility check and input generation...")

def log_test_path_full_check_result(path_id: int, is_feasible: bool, has_inputs: bool):
    status = "Feasible" if is_feasible else "Infeasible"
    input_info = "Inputs found." if has_inputs else ("No inputs generated." if is_feasible else "")
    logger.debug(f"TestPath {path_id}: Full feasibility check result: {status}. {input_info}")

def log_test_path_z3_call(path_id: int, context: str, num_constraints: int):
    logger.trace(f"TestPath {path_id} ({context}): Calling Z3 solver with {num_constraints} constraints.")

def log_test_path_z3_unknown(path_id: int, context: str):
    logger.warning(f"TestPath {path_id} ({context}): Z3 returned 'unknown'. Treating as infeasible.")

def log_test_path_no_conditions(path_id: int, context: str):
    logger.trace(f"TestPath {path_id} ({context}): No conditions to check, trivially feasible.")

def log_test_path_loop_exit_decision(path_id: int, edge: Tuple[CFGBasicBlockNode, CFGBasicBlockNode], condition_str: str, skipped: bool, entered_loop: bool):
    """Logs the decision on whether to include a loop exit condition"""
    u_id = edge[0].node_id
    v_id = edge[1].node_id
    action = "Skipped" if skipped else "Kept"
    reason = "loop body was entered" if entered_loop else "loop body was NOT entered"
    logger.trace(f"TestPath {path_id}: Edge ({u_id}->{v_id}) is loop exit. Condition '{condition_str}' {action} because {reason}.")

# CFG Renderer 

def print_cfg_rich(cfg: 'CFG') -> None:
    """
    Render the control flow graph (CFG) as a rich-formatted tree in the terminal
    """
    console = Console()
    visited_paths = set() # Store (node, parent_tree_id) tuples to detect loops within a branch
    visited_nodes = set() # Store nodes visited globally

    def render_node_recursive(node: CFGBasicBlockNode, parent_tree: Tree, edge_prefix: str = ""):
        node_key_for_branch_path = (node, id(parent_tree))
        
        is_revisited_in_branch = node_key_for_branch_path in visited_paths
        is_revisited_globally = node in visited_nodes

        node_label = f"[bold]{node.node_id}[/bold]"
        if node.is_entry:
            node_label += " [green](entry)[/green]"
        if node.is_exit:
            node_label += " [red](exit)[/red]"
        node_label += f" [dim]({type(node.jump).__name__})[/dim]" if node.jump else ""

        full_label = edge_prefix + node_label 

        if is_revisited_in_branch:
            parent_tree.add(f"{full_label} [yellow loop](Loop back)[/yellow loop]")
            return
        
        visited_paths.add(node_key_for_branch_path)
        visited_nodes.add(node)

        if is_revisited_globally and not is_revisited_in_branch:
            current_branch = parent_tree.add(f"{full_label} [dim](...)[/dim]")
        else:
            current_branch = parent_tree.add(full_label)
            successors = sorted(cfg.graph.successors(node), key=lambda n: n.node_id)
            for succ_node in successors:
                edge_data = {} 
                retrieved_label = None
                try:
                    edge_data = cfg.graph.edges[node, succ_node]
                    retrieved_label = cfg.get_edge_label(node, succ_node)
                except KeyError:
                     print(f"DEBUG RENDER: Edge {node.node_id}->{succ_node.node_id} not found in graph.edges during rendering?")

                prefix_label_part = f"\[{retrieved_label}] " if retrieved_label else "" 
                next_edge_prefix = f"{prefix_label_part}-> "
                render_node_recursive(succ_node, current_branch, next_edge_prefix)

    tree = Tree("CFG Structure")
    
    if cfg.entry_node:
        render_node_recursive(cfg.entry_node, tree, "") 
    else:
        tree.add("[yellow]No entry node found in CFG.[/yellow]")
        
    console.print(tree)
    console.print(Rule())

def print_code(cfg, console: Console, language: str = "c", theme: str = "ansi_dark"):
    """Prints the generated code from cfg.code with syntax highlighting and terminal adaptation"""
    console.print(Rule("[bold blue]📄 Generated Code[/bold blue]"))

    code_str = cfg.code
    if not code_str:
        console.print("[yellow]No code generated or FragmentForest not available.[/yellow]")
        console.print(Rule())
        return

    terminal_width = console.size.width
    code_margin = 4 

    syntax = Syntax(
        code_str,
        language,
        theme=theme,
        line_numbers=True,
        word_wrap=False,
        indent_guides=True,
        code_width=terminal_width - code_margin
    )

    console.print(syntax)
    console.print(Rule())


def log_tuner_attempt(attempt: int, max_attempts: int, current_shtv: float, target_shtv: float):
    logger.trace(f"Tuner Attempt {attempt}/{max_attempts}. Current CFG SHTV: {current_shtv:.2f}/{target_shtv:.2f}")

def log_tuner_select_node(node_id: int, shtv_gap: float):
    log_with_node(node_id).debug(f"Selected for tuning (SHTV gap: {shtv_gap:.2f})")

def log_tuner_constraints(node_id: int, must_be: Set[Variable], term: Set[Variable], outgoing: Set[Variable], context_vars: List[Variable], params: Set[Variable]):
    log = log_with_node(node_id)
    log.trace(f" Constraints: MustBe={{{', '.join(v.name for v in must_be)}}}, Term={{{', '.join(v.name for v in term)}}}, Outgoing={{{', '.join(v.name for v in outgoing)}}}")
    log.trace(f" Context: Vars={{{', '.join(v.name for v in context_vars)}}}, Params={{{', '.join(v.name for v in params)}}}")

def log_tuner_rhs_vars(node_id: int, available_rhs: List[Variable]):
     log_with_node(node_id).trace(f"Available RHS vars: [{', '.join(v.name for v in available_rhs)}] (Count: {len(available_rhs)})")

def log_tuner_lhs_vars(node_id: int, modifiable_lhs: List[Variable], context: str = ""):
     context_str = f" ({context})" if context else ""
     log_with_node(node_id).trace(f"Modifiable LHS vars{context_str}: [{', '.join(v.name for v in modifiable_lhs)}] (Count: {len(modifiable_lhs)})")

def log_tuner_op_creation_attempt(node_id: int, op_type: str, prioritized_types: List[Any]):
    priority_str = ' -> '.join([t.name for t in prioritized_types])
    log_with_node(node_id).trace(f"Attempting to create {op_type}. Priority: [{priority_str}]")

def log_tuner_op_created(node_id: int, op_str: str, op_type: str):
    log_with_node(node_id).debug(f"Added Operation ({op_type}): {op_str}")

def log_tuner_op_skipped(node_id: int, reason: str):
    log_with_node(node_id).trace(f"Skipped adding operation: {reason}")

def log_tuner_stats_update(node_id: int, arith_count: int, logic_count: int, total_used: int):
    log_with_node(node_id).trace(f"Node stats updated: Arith={arith_count}, Logic={logic_count}, UsedVars={total_used}")


def log_tuner_must_be_add(node_id: int, condition_str: str):
    log_with_node(node_id).trace(f"Added MustBeSet constraint: {condition_str}")

def log_tuner_state_vars_created(var_names: List[str]):
    """Logs the names of state variables created by the tuner"""
    if var_names:
        logger.info(f"Complexity Tuner created state variables: {', '.join(var_names)}")
    else:
        logger.info("Complexity Tuner: No initial state variables configured or created.")

def print_shtv_assignment_start_info(console: Console, global_max_shtv: float, alpha: float, beta: float):
    """Prints the initial parameter table for SHTV ceiling assignment"""
    console.print(Rule("[bold blue]📊 Assigning SHTV Ceilings[/bold blue]"))
    table = Table(title="Assigner Parameters", show_header=True, header_style="bold magenta", box=rich.box.MINIMAL)
    table.add_column("Parameter", style="dim cyan", width=20)
    table.add_column("Value", style="green")

    table.add_row("Global Max SHTV", str(global_max_shtv))
    table.add_row("Alpha (Depth Decay)", str(alpha))
    table.add_row("Beta (Position Decay)", str(beta))
    
    console.print(table)
    console.print(Rule())

def print_shtv_assignment_results(console: Console, cfg: 'CFG'):
    """Prints the final node ceiling assignment results table"""
    console.print(Rule("[bold blue]🏁 SHTV Ceiling Assignment Finished[/bold blue]"))
    
    node_ceiling_table = Table(title="Node Ceiling Assignments", show_header=True, header_style="bold magenta", box=rich.box.ROUNDED)
    node_ceiling_table.add_column("Node ID", style="cyan", justify="right")
    node_ceiling_table.add_column("Assigned max_shtv", style="green", justify="right")

    basic_blocks = [node for node in cfg if isinstance(node, CFGBasicBlockNode)]
    sorted_nodes = sorted(basic_blocks, key=lambda n: n.node_id)

    nodes_processed = 0 
    if sorted_nodes:
        for node in sorted_nodes:
            ceiling_value = getattr(node, 'max_shtv', 'N/A') 
            formatted_value = f"{ceiling_value:.2f}" if isinstance(ceiling_value, (int, float)) else str(ceiling_value)
            node_ceiling_table.add_row(str(node.node_id), formatted_value)
            nodes_processed += 1
        
        node_ceiling_table.add_row("[bold]Total Nodes[/bold]", f"[bold]{nodes_processed}[/bold]")
    else:
        node_ceiling_table.add_row("No CFGBasicBlockNodes found", "N/A")

    console.print(node_ceiling_table)
    console.print(Rule())

# Test Path Finder Logging 

def print_test_path_finder_start_info(console: Console, coverage_criterion: str, num_nodes: int, num_edges: int):
    """Prints the initial setup table for the Test Paths Finder"""
    console.print(Rule(f"[bold blue]🗺️ Finding Test Paths (Criterion: {coverage_criterion})[/bold blue]"))
    
    start_table = Table(title="Finder Setup", show_header=True, header_style="bold magenta", box=rich.box.MINIMAL)
    start_table.add_column("Parameter", style="dim cyan", width=25)
    start_table.add_column("Value", style="green")

    start_table.add_row("Coverage Criterion", coverage_criterion)
    start_table.add_row("CFG Nodes", str(num_nodes))
    start_table.add_row("CFG Edges", str(num_edges))

    console.print(start_table)
    console.print(Rule())

# Flow Builder Logging 

def print_flow_builder_start_info(console: Console, target_cc: int, max_depth: int, nesting_probability: float, seed: Optional[int]):
    """Prints the initial setup table for the FlowBuilder"""
    console.print(Rule("[bold blue]🚀 Starting FlowBuilder[/bold blue]"))
    config_table = Table(title="Builder Configuration", show_header=True, header_style="bold magenta", box=rich.box.MINIMAL)
    config_table.add_column("Parameter", style="dim cyan", width=30)
    config_table.add_column("Value", style="green")
    config_table.add_row("Target Cyclomatic Complexity", str(target_cc))
    config_table.add_row("Max Nesting Depth", str(max_depth))
    config_table.add_row("Nesting Probability", f"{nesting_probability:.2f}")
    config_table.add_row("Seed", str(seed) if seed is not None else "Random")
    console.print(config_table)
    console.print(Rule())

def print_flow_builder_target_cc_one(console: Console):
    """Prints a message indicating the build process is skipped because target CC is 1"""
    console.print("[yellow]Target CC is 1, skipping build process.[/yellow]")

def print_flow_builder_finish_info(console: Console, iteration: int, final_cc: int):
    """Prints the final summary table for the FlowBuilder"""
    console.print(Rule("[bold blue]📊 FlowBuilder Finished[/bold blue]"))
    summary_table = Table(title="Build Summary", show_header=True, header_style="bold magenta", box=rich.box.MINIMAL)
    summary_table.add_column("Metric", style="dim cyan", width=30)
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Iterations", str(iteration))
    summary_table.add_row("Final Cyclomatic Complexity", str(final_cc))
    console.print(summary_table)
    console.print(Rule())

def print_flow_builder_shtv_assigned(console: Console):
    """Prints a confirmation message that SHTV ceilings have been assigned"""
    console.print("[green]SHTV Ceilings Assigned.[/green]")


def log_tuner_downstream_check_start(start_node_id: int, param_name: str):
    """Log the start of the downstream conditional usage check"""
    log_with_node(start_node_id).debug(f"Downstream Check START for Param '{param_name}'")

def log_tuner_downstream_param_modified(current_node_id: int, param_name: str):
    """Log when the param is found modified within a downstream node"""
    log_with_node(current_node_id).debug(f"  -> Param '{param_name}' modified in this node. Path safe from here.")

def log_tuner_downstream_param_used_cond(edge_u_id: int, edge_v_id: int, param_name: str):
    """Log when the param is found used in a downstream edge condition"""
    # Use the source node ID for binding the log message
    log_with_node(edge_u_id).debug(f"  -> Param '{param_name}' USED in condition on edge ->{edge_v_id}. UNSAFE.")

def log_tuner_downstream_path_exited(last_node_id: int, param_name: str):
    """Log when a path exploration reaches the exit node safely"""
    # Use the last node ID before exit for binding
    log_with_node(last_node_id).debug(f"  -> Path reached exit safely for Param '{param_name}'.")

def log_tuner_downstream_cycle_detected(node_id: int, param_name: str):
    """Log when a cycle is detected during the downstream check"""
    log_with_node(node_id).debug(f"  -> Cycle involving node {node_id} detected while checking Param '{param_name}'.")

def log_tuner_downstream_check_safe(start_node_id: int, param_name: str):
    """Log when the downstream check completes without finding conditional use"""
    log_with_node(start_node_id).debug(f"Downstream Check END for Param '{param_name}': SAFE")

def log_tuner_downstream_check_unsafe(start_node_id: int, param_name: str):
    """Log when the downstream check completes having found conditional use"""
    log_with_node(start_node_id).debug(f"Downstream Check END for Param '{param_name}': UNSAFE")

def print_fragment_forest_rich(forest: FragmentForest, console: Console) -> None:
    """
    Render the fragment forest as a rich-formatted tree in the terminal
    """
    tree = Tree("Fragment Forest Structure")
    visited: Set[int] = set()

    def render_fragment(frag_id: int, parent_tree: Tree):
        if frag_id in visited:
            return
        visited.add(frag_id)

        fragment = forest.get_fragment(frag_id)
        if not fragment:
            parent_tree.add(f"[red]Fragment {frag_id} not found[/red]")
            return

        label = f"[bold]ID: {fragment.fragment_id}[/bold] [dim]({fragment.fragment_role.name})[/dim]"
        current_branch = parent_tree.add(label)

        # Recursively render children in their stored order
        children = forest.get_children(frag_id)
        for child_fragment in children:
            render_fragment(child_fragment.fragment_id, current_branch)

    # Start rendering from each root in the forest
    for root_id in forest._roots:
        render_fragment(root_id, tree)

    if not forest._roots:
        console.print("[yellow]Fragment Forest is empty.[/yellow]")
    else:
        console.print(tree)
    console.print(Rule())

def print_fragment_gen_start(console: Console, frag_id: int, role: str, indent: int):
    """Prints a starting rule for fragment code generation"""
    console.print(Rule(f"[cyan]Generating Fragment ID: {frag_id} | Role: {role} | Indent: {indent}[/cyan]"))

def print_fragment_gen_details(console: Console, fragment: Fragment):
    """Prints a table with details about the fragment being generated"""
    from src.core.fragments import FragmentRole

    table = Table(title=f"Fragment {fragment.fragment_id} Details", box=rich.box.MINIMAL, show_header=False)
    table.add_column("Attribute", style="dim")
    table.add_column("Value")

    def get_node_id_str(node: Optional[CFGBasicBlockNode]) -> str:
        return str(node.node_id) if node else "[dim]None[/dim]"

    entry_node = getattr(fragment, 'entry_node', None)
    exit_node = getattr(fragment, 'exit_node', None)
    related_nodes = getattr(fragment, 'related_nodes', [])

    table.add_row("Entry Node", get_node_id_str(entry_node))
    table.add_row("Exit Node", get_node_id_str(exit_node))
    table.add_row("Related Nodes", ", ".join(get_node_id_str(n) for n in related_nodes))
    role = getattr(fragment, 'fragment_role', None)
    if role == FragmentRole.IF or role == FragmentRole.IF_ELSE:
        true_node = getattr(fragment, 'true_branch_node', None)
        table.add_row("True Branch Node", get_node_id_str(true_node))
    if role == FragmentRole.IF_ELSE:
        false_node = getattr(fragment, 'false_branch_node', None)
        table.add_row("False Branch Node", get_node_id_str(false_node))
    if role in [FragmentRole.WHILE, FragmentRole.FOR, FragmentRole.FOR_EARLY_EXIT]:
        body_node = getattr(fragment, 'body_node', None)
        table.add_row("Body Node", get_node_id_str(body_node))
    if role == FragmentRole.FOR_EARLY_EXIT:
        break_node = getattr(fragment, 'break_node', None)
        table.add_row("Break/Latch Node", get_node_id_str(break_node))
    console.print(table)

def print_fragment_gen_end(console: Console, frag_id: int):
    """Prints an ending rule for fragment code generation"""
    console.print(Rule(f"[cyan]Finished Fragment ID: {frag_id}[/cyan]"))

def log_fragment_gen_ops(node_id: Optional[int], context: str):
    """Logs when operations code is generated for a specific node"""
    node_str = str(node_id) if node_id is not None else "Unknown"
    logger.trace(f"[GenCode] Generating ops for Node {node_str} (Context: {context})")

def log_fragment_gen_if_else_targets(frag_id: int, true_target: Optional[int], false_target: Optional[int]):
    """Logs the identified true and false target node IDs for an IF_ELSE"""
    true_str = str(true_target) if true_target is not None else "None"
    false_str = str(false_target) if false_target is not None else "None"
    logger.debug(f"[GenCode] Fragment {frag_id} (IF_ELSE): True Target Node ID = {true_str}, False Target Node ID = {false_str}")

def log_fragment_gen_if_else_map(frag_id: int, child_map: Dict[int, str]):
    """Logs the created map of child entry node IDs to generated code snippets"""
    map_str = "{" + ", ".join(f"{k}: [code snippet len={len(v)}]" for k, v in child_map.items()) + "}"
    logger.debug(f"[GenCode] Fragment {frag_id} (IF_ELSE): Child Code Map = {map_str}")

def log_fragment_gen_if_else_lookup(frag_id: int, branch: str, target_id: Optional[int], found: bool):
    """Logs the result of looking up a child fragment for a specific branch target"""
    target_str = str(target_id) if target_id is not None else "None"
    status = "Found" if found else "NOT Found"
    logger.debug(f"[GenCode] Fragment {frag_id} (IF_ELSE): Lookup for {branch} branch (Target ID: {target_str}): {status}")


def print_test_gen_file_written(filename: str):
    """Prints a success message for file generation using rich"""
    console = Console(stderr=True, force_terminal=True)
    console.print(f"[green]✔[/green]  {filename} written")

def print_test_gen_build_success(so_name: str, bin_name: str):
    """Prints a success message for build completion using rich"""
    console = Console(stderr=True, force_terminal=True)
    console.print(f"[green]✔[/green]  built [cyan]{so_name}[/cyan] and [cyan]{bin_name}[/cyan]")

def print_test_gen_build_failure(exc: Exception):
    """Prints a build failure message using rich"""
    console = Console(stderr=True, force_terminal=True)
    console.print(f"[red]✖[/red]  compilation failed: [bold red]{exc}[/bold red]")

def print_test_gen_usage():
    """Prints the usage message for generate_tests.py using rich"""
    console = Console(stderr=True, force_terminal=True)
    console.print("[yellow]Usage:[/yellow] python -m src.utils.generate_tests <run_folder>")

def print_test_gen_fatal_error(message: str):
    """Prints a fatal error message and exits"""
    console = Console(stderr=True, force_terminal=True)
    console.print(f"[bold red]✖ Error:[/bold red] {message}")

# --- Loop Terminator Logging ---

def print_loop_termination_summary(assignments: List[Tuple[int, str]]):
    """Prints a summary table of operations added by LoopTerminator"""
    console = Console(stderr=True, force_terminal=True) 
    console.print(Rule("[bold blue]📊 Loop Termination Summary[/bold blue]"))
    
    if not assignments:
        console.print("[yellow]No termination operations were added.[/yellow]")
        console.print(Rule())
        return

    table = Table(title="Added Termination Operations", show_header=True, header_style="bold magenta", box=rich.box.ROUNDED)
    table.add_column("Latch Node ID", style="dim cyan", width=15, justify="right")
    table.add_column("Added Operation", style="green")

    sorted_assignments = sorted(assignments, key=lambda x: x[0])
    for node_id, op_str in sorted_assignments:
        table.add_row(str(node_id), op_str)

    console.print(table)
    console.print(Rule())

