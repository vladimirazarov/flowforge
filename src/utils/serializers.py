"""
Serialization functions for converting internal data structures to file formats
"""
from __future__ import annotations
import json
from typing import TYPE_CHECKING, Optional, List, Any, Dict, Tuple
from dataclasses import asdict
import yaml
from loguru import logger

from src.core.nodes.cfg_jump_node import (
    CFGIfJump, CFGSwitchJump, CFGWhileLoopJump, CFGForLoopJump
)

if TYPE_CHECKING:
    from src.core.cfg.cfg import CFG
    from src.config.config import AppConfig

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

def test_suite_values(path_data: Dict[str, Any]) -> Dict[str, int]:
    """
    For each parameter (e.g. 'A'), pick the test_inputs entry whose
    SSA‐suffix (u,v) corresponds to the *first branch* edge along this path
    Branch edges are typically labeled 'true', 'false', 'case_*', or 'default'
    Loop exits ('exit') and continuations ('loop') are skipped
    """
    logger.debug(f"Extracting lowest suffix values for path with nodes: {path_data.get('nodes', 'N/A')}")
    model = path_data.get("test_inputs", {})
    result: Dict[str, int] = {}

    if not model:
        logger.warning("Path has no model inputs, returning empty result.")
        return result

    # for every var in the model, extract its base name
    bases = set()
    for name in model:
         # Ensure splitting works even if there's no underscore
         parts = name.split('_', 1)
         if len(parts) > 0 and parts[0]:
             bases.add(parts[0])
         else:
             logger.warning(f"Could not determine base variable for model key: '{name}'")

    logger.debug(f"Base variables found in model: {bases}")

    for base in bases:
        logger.debug(f"Processing base variable: '{base}'")
        relevant_keys: List[Tuple[Optional[int], str]] = []

        for key in model:
            if key.startswith(f"{base}_"):
                suffix_str = key[len(base)+1:]
                suffix_int: Optional[int] = None
                try:
                    suffix_int = int(suffix_str)
                    relevant_keys.append((suffix_int, key))
                    logger.debug(f"  Found key '{key}' with integer suffix {suffix_int}.")
                except ValueError:
                    relevant_keys.append((None, key))
                    logger.debug(f"  Found key '{key}' with non-integer suffix '{suffix_str}'.")
            elif key == base:
                # Handle case where the key is exactly the base name (no suffix)
                relevant_keys.append((None, key))
                logger.debug(f"  Found key '{key}' matching base name directly.")


        if not relevant_keys:
            logger.warning(f"  No keys found for base variable '{base}'. Assigning default 0.")
            result[base] = 0
            continue

        relevant_keys.sort(key=lambda item: (item[0] is None, item[0] if item[0] is not None else float('inf'), item[1]))
        logger.debug(f"  Sorted relevant keys for '{base}': {relevant_keys}")

        chosen_suffix, chosen_key = relevant_keys[0]
        chosen_value = model.get(chosen_key, 0) 
        result[base] = chosen_value
        logger.debug(f"    Selected key '{chosen_key}' (Suffix: {chosen_suffix}) with value {chosen_value} for base '{base}'.")


    logger.debug(f"Final selected input values: {result}")
    return result


def serialize_test_paths(cfg: CFG) -> Tuple[str, Dict[str, Any]]:
    """
    Serialize the CFG's test paths into the JSON structure expected by generate_tests

    Returns:
        A tuple of (json_string, parsed_dict) where parsed_dict matches
        the schema:
        {
          total_paths: int,
          function_parameters: [str, ...],
          test_paths: [
            {
              path_id: int,
              node_count: int,
              nodes: [int, ...],
              effective_formula: str,
              edge_labels: { "<u>_<v>": "<label>", ... },
              test_inputs: { "<Var>_<label>_<u>_<v>": int, ... }
            },
            ...
          ]
        }
    """
    paths: List[Dict[str, Any]] = []
    for tp in getattr(cfg, "test_paths", []):
        node_ids = [n.node_id for n in tp.nodes]
        edge_labels: Dict[str, str] = {}
        for u, v in zip(tp.nodes[:-1], tp.nodes[1:]):
            key = f"{u.node_id}_{v.node_id}"
            edge_labels[key] = cfg.get_edge_label(u, v) or "unconditional"

        inputs = tp.generate_inputs_for_display() or {}

        paths.append({
            "path_id": tp.path_id,
            "node_count": len(node_ids),
            "nodes": node_ids,
            "effective_formula": tp.get_effective_formula_str(),
            "edge_labels": edge_labels,
            "test_inputs": inputs,
        })

    func_params: List[str] = []
    ctx = getattr(cfg, "context", None)
    if ctx and hasattr(ctx, "used_parameters"):
        func_params = sorted(p.name for p in ctx.used_parameters)
    else:
        logger.warning("CFG context or used_parameters not found. Parameters will be inferred later if possible.")

    output = {
        "total_paths": len(paths),
        "function_parameters": func_params,
        "test_paths": paths
    }
    json_str = json.dumps(output, indent=2)
    return json_str, output


def serialize_config(config: AppConfig, timestamp: str, run_number: Optional[int]) -> str:
    """Serialize the run configuration to a YAML string"""
    config_dict = asdict(config)
    config_dict['timestamp'] = timestamp
    config_dict['run_number'] = run_number
    return yaml.dump(config_dict, indent=2, default_flow_style=False)


def serialize_cfg_structure(cfg: CFG) -> str:
    """Serialize the CFG structure (nodes, edges, metadata) to a JSON string"""
    nodes_data = []
    for node in cfg.graph.nodes():
        node_info = {
            "node_id": node.node_id,
            "depth": node.depth,
            "position": node.position,
            "shtv": node.shtv,
            "logical_ops_count": node.logical_operations_count,
            "arithmetic_ops_count": node.arithmetic_operations_count,
            "jump_type": node.jump.__class__.__name__ if node.jump else None,
            "jump_condition": node.jump.expression.to_c() if node.jump and hasattr(node.jump, 'expression') and node.jump.expression else None,
            "operations": [op.to_c() for op in node.instructions.operations] if node.instructions else []
        }
        nodes_data.append(node_info)

    edges_data = []
    for u, v, data in cfg.graph.edges(data=True):
        condition = data.get("condition")
        edge_info = {
            "source": u.node_id,
            "target": v.node_id,
            "condition": condition.to_c() if condition else None,
        }
        edges_data.append(edge_info)

    cfg_data = {
        "nodes": nodes_data,
        "edges": edges_data,
        "metadata": {
            "total_nodes": cfg.graph.number_of_nodes(),
            "total_edges": cfg.graph.number_of_edges(),
            "entry_node_id": cfg.entry_node.node_id if cfg.entry_node else None,
            "exit_node_id": cfg.exit_node.node_id if cfg.exit_node else None,
            "shtv": cfg.shtv,
            "cc": cfg.cc,
            "prime_paths_count": cfg.prime_paths_count,
        }
    }
    return json.dumps(cfg_data, indent=2)


def format_shtv_details(cfg: CFG, config: AppConfig) -> str:
    """Format SHTV calculation details into a human-readable text string"""
    lines = ["--- SHTV Calculation Details ---\n"]

    lines.append("--- Global CFG Parameters ---")
    total_shtv = cfg.shtv
    max_shtv = cfg.max_shtv
    cc = cfg.cc
    prime_paths = cfg.prime_paths_count
    total_vars = len(cfg.context.used_variables)
    lines.append(f"Calculated Total SHTV: {total_shtv:.2f}")
    lines.append(f"Target Max SHTV: {max_shtv:.2f}")
    lines.append(f"Cyclomatic Complexity (CC): {cc}")
    lines.append(f"Prime Paths Count: {prime_paths}")
    lines.append(f"Total Variables Used: {total_vars}\n")

    lines.append("--- Weights Used ---")
    weights = config.weights
    for key, value in weights.items():
        lines.append(f"  Weight \'{key}\': {value}")
    lines.append("")

    lines.append("--- Node-Specific SHTV Components ---")
    node_shtv_sum = 0.0
    for node in cfg:
        lines.append(f"Node ID: {node.node_id}")
        has_condition = isinstance(node.jump, (CFGIfJump, CFGSwitchJump))
        has_loop = isinstance(node.jump, (CFGWhileLoopJump, CFGForLoopJump))
        logical_ops = node.logical_operations_count
        arithmetic_ops = node.arithmetic_operations_count
        variables_used = len(node.used_variables)
        depth = node.depth
        position = node.position
        node_shtv = node.shtv
        node_shtv_sum += node_shtv

        lines.append(f"  has_condition: {has_condition} (Weight: {weights.get('a', 0)})")
        lines.append(f"  has_loop: {has_loop} (Weight: {weights.get('b', 0)})")
        lines.append(f"  logical_ops: {logical_ops} (Weight: {weights.get('c', 0)})")
        lines.append(f"  arithmetic_ops: {arithmetic_ops} (Weight: {weights.get('d', 0)})")
        lines.append(f"  nesting_depth: {depth} (Weight: {weights.get('e', 0)})")
        lines.append(f"  variables_used: {variables_used} (Weight: {weights.get('f', 0)})")
        lines.append(f"  position: {position} (Weight: {weights.get('k', 0)})")
        lines.append(f"  Calculated Node SHTV: {node_shtv:.2f}\n")

    lines.append("--- SHTV Calculation Summary ---")
    lines.append(f"Sum of all Node SHTVs: {node_shtv_sum:.2f}")
    global_contrib = total_shtv - node_shtv_sum
    lines.append(f"Contribution from Global Factors (CC, Prime Paths, Total Vars): {global_contrib:.2f}")
    lines.append(f"Total SHTV (Node Sum + Global Contribution): {total_shtv:.2f}")

    return "\n".join(lines)


def format_with_loops(nodes: list[int]) -> str:
    """
    Recursively detects the shortest repeated segment (i.e. a loop),
    wraps it in parentheses with ^+, and joins everything with arrows.
    """
    n = len(nodes)
    # find the shortest i<j such that nodes[i] == nodes[j]
    best_i = best_j = None
    best_len = n + 1
    for i in range(n):
        for j in range(i + 1, n):
            if nodes[i] == nodes[j]:
                length = j - i
                if length < best_len:
                    best_len, best_i, best_j = length, i, j

    # no loop found, just print flat
    if best_i is None:
        return " -> ".join(map(str, nodes))

    assert best_i is not None
    assert best_j is not None
    # if the loop covers the entire sequence, don't recurse forever
    if best_i == 0 and best_j == n - 1:
        return " -> ".join(map(str, nodes))

    # split into before, loop, after
    before = nodes[:best_i]
    loop    = nodes[best_i : best_j + 1]
    after   = nodes[best_j + 1 :]

    parts: list[str] = []
    if before:
        parts.append(format_with_loops(before))
    parts.append(f"({ ' -> '.join(map(str, loop)) })⁺")
    if after:
        parts.append(format_with_loops(after))

    return " -> ".join(parts)


def format_test_suites(paths_json_data: Dict[str, Any]) -> str:
    """Generate test suites text based on parsed test path JSON data"""
    function_parameters = paths_json_data.get("function_parameters", [])
    paths_data = paths_json_data.get("test_paths", [])

    if not function_parameters:
        logger.warning("Function parameters not found in JSON. Inferring for test suites.")
        inferred_params = set()
        for p_data in paths_data:
            earliest_inputs = test_suite_values(p_data)
            inferred_params.update(earliest_inputs.keys())
        if inferred_params:
             function_parameters = sorted(list(inferred_params))
             logger.info(f"Inferred parameters for test suites: {function_parameters}")
        else:
             logger.error("Could not determine function parameters for test suites text.")
             return "# Error: Could not determine function parameters."

    output_lines = []
    output_lines.append(f"# Test Suites for {len(paths_data)} paths")
    output_lines.append(f"# Parameters Order: {', '.join(function_parameters)}")
    output_lines.append("---")

    for p_data in paths_data:
        # Skip paths that are marked as infeasible (missing test_inputs)
        test_inputs_model = p_data.get("test_inputs")
        if not test_inputs_model: 
            path_id = p_data.get("path_id", "N/A")
            logger.info(f"Skipping Path {path_id} for test suites generation: No test inputs found (likely infeasible).")
            continue 

        path_id = p_data.get("path_id", "N/A")
        earliest_inputs = test_suite_values(p_data) 
        node_ids = p_data.get("nodes", [])
        path_str = format_with_loops(node_ids) if node_ids else "[No Nodes]"

        param_values = []
        for name in function_parameters:
            param_values.append(earliest_inputs.get(name, 0))

        values_str = repr(tuple(param_values))
        output_lines.append(f"Path {path_id}: [{path_str}] : {values_str}")

    return "\n".join(output_lines)
