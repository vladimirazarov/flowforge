"""Assigns SHTV ceilings to nodes in a Control Flow Graph (CFG)

The ceiling for each node is determined based on its depth and position within the graph,
using an exponential decay function. This module uses the 'rich' library for formatted output
"""
from __future__ import annotations
import math
from src.config.config import config
from typing import TYPE_CHECKING
from src.core.cfg.cfg_analysis import find_shortest_path
from src.utils.logging_helpers import print_shtv_assignment_start_info, print_shtv_assignment_results
from rich.console import Console

if TYPE_CHECKING:
    from src.core.cfg.cfg import CFG

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

# n.max_shtv = max_shtv · (weight_n / sum_{v∈CFG} weight_v)
# weight_n = e^(-α·depth_n) · e^(-β·position_n)

def assign_shtv_cellings(cfg: CFG):
    """Calculates and assigns max_shtv ceilings to all nodes in the CFG

    Distributes the global max SHTV based on node weights, which are
    calculated using depth and position decay factors (alpha, beta)
    Prints assignment details using helper functions

    Args:
        cfg (CFG): The Control Flow Graph to process
    """
    console = Console() 
    max_shtv = config.global_max_shtv
    alpha = config.shtv_alpha
    beta = config.shtv_beta
    
    # Print start info using helper
    print_shtv_assignment_start_info(console, max_shtv, alpha, beta)

    update_node_positions(cfg)
    # Calculate weights for all nodes
    node_weights = {}
    for node in cfg:
        # Calculate weight using formula: e^(-α·depth_n) · e^(-β·position_n)
        weight = math.exp(-alpha * node.depth) * math.exp(-beta * node.position)
        node_weights[node] = weight
    
    # Calculate sum of all weights
    total_weight = sum(node_weights.values()) if node_weights else 1.0 # Avoid division by zero
    if total_weight == 0:
        total_weight = 1.0 # Prevent division by zero if all weights are somehow zero
    
    # Assign max_shtv to each node based on their weight proportion
    for node in cfg:
        # n.max_shtv = max_shtv * (weight_n / sum_{v∈CFG} weight_v)
        assigned_value = max_shtv * (node_weights.get(node, 0) / total_weight)
        node.max_shtv = assigned_value

    print_shtv_assignment_results(console, cfg)
          
def update_node_positions(cfg: CFG):
    """Calculates and updates the position attribute for each node in the CFG

    The position is defined as the length (number of edges) of the shortest
    path from the CFG entry node to the current node

    Args:
        cfg (CFG): The Control Flow Graph to process
    """
    assert cfg.entry_node is not None
    for node in cfg:
        # Get shortest path from entry to this node
        path = find_shortest_path(cfg, cfg.entry_node, node)
        node.position = len(path.nodes) - 1