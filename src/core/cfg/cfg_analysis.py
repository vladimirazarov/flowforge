"""Analyzes Control Flow Graphs (CFGs)

Provides capabilities for CFG analysis, including loop detection, path finding
(simple, prime, shortest), and edge pair extraction
"""
from __future__ import annotations
from collections import deque
from typing import Dict, Set, List, Tuple, Optional, Union, TYPE_CHECKING, cast

import networkx as nx

from src.core.paths.path import Path, PrimePath, SimplePath
from src.core.nodes.cfg_basic_block_node import CFGBasicBlockNode
from src.core.cfg_content import Expression
from src.core.cfg_content.variable import Variable

if TYPE_CHECKING:
    from src.core.cfg.cfg import CFG 

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"


LoopExitInfo = Dict[Tuple[CFGBasicBlockNode, CFGBasicBlockNode], Optional[Expression]]
LoopAnalysisInfo = Dict[CFGBasicBlockNode, Dict[str, Union[Set[CFGBasicBlockNode], LoopExitInfo, Set[Variable], int]]]

class CFGAnalysis:
    """Performs and stores analysis results for a CFG

    Focuses on loop identification and related properties

    Attributes:
        cfg: The CFG instance being analyzed
        graph: The underlying NetworkX DiGraph of the CFG
        immediate_doms: A dictionary mapping nodes to their immediate dominators
        loop_info: Stores detailed information about detected loops
    """
    def __init__(self, cfg: CFG):
        """Initializes the analysis object for a given CFG

        Args:
            cfg (CFG): The control flow graph to analyze
        """
        self.cfg: CFG = cfg
        self.graph: nx.DiGraph = cfg.graph
        assert self.cfg.entry_node is not None
        self.immediate_doms: Dict[CFGBasicBlockNode, CFGBasicBlockNode] = nx.immediate_dominators(self.graph, self.cfg.entry_node)
        self.loop_info: LoopAnalysisInfo = self._analyze_loops()

    def _analyze_loops(self) -> LoopAnalysisInfo:
        """Helper to identify loops and their properties

        Detects loop headers, body nodes, latch nodes, and exit edges using
        dominator analysis and graph traversal

        Returns:
            LoopAnalysisInfo: A dictionary mapping loop header nodes to their
                detailed loop information (body nodes, exits, latch nodes)
        """
        loop_info: LoopAnalysisInfo = {}
        potential_headers = set()
        back_edges: Dict[CFGBasicBlockNode, List[CFGBasicBlockNode]] = {}

        # Find potential headers and back edges using pre-computed immediate dominators
        for u, v in self.graph.edges():
            # Check if v dominates u by traversing up the immediate dominator tree from u
            dominates = False
            current = u
            # Traverse up using the pre-computed immediate dominators map
            while current in self.immediate_doms:
                immediate_dominator = self.immediate_doms[current]
                if immediate_dominator == v:
                    dominates = True
                    break
                if immediate_dominator == current: 
                    break
                current = immediate_dominator

            if dominates:
                potential_headers.add(v)
                back_edges.setdefault(v, []).append(u)

        # Analyze each potential loop
        for header in potential_headers:
            loop_body: Set[CFGBasicBlockNode] = {header}
            # Use BFS starting from predecessors of the header (latch nodes)
            # to find all nodes belonging to the loop body.
            queue = deque(back_edges.get(header, []))
            processed: Set[CFGBasicBlockNode] = {header}

            while queue:
                node = queue.popleft()
                # Explore nodes backwards from latch nodes only if they haven't been processed
                # and are part of the loop (i.e., reachable from inside the loop body without passing the header)
                # This correctly identifies the natural loop body.
                if node != header and node not in processed:
                    processed.add(node)
                    loop_body.add(node)
                    for pred in self.graph.predecessors(node):
                         # Add predecessors to the queue to continue the backward search
                         # Ensure only relevant nodes are added.
                        if pred not in processed:
                            queue.append(pred)


            exit_edges: LoopExitInfo = {}
            latch_nodes: Set[CFGBasicBlockNode] = set(back_edges.get(header, []))

            # Identify exit edges: edges from a loop body node to a node outside the loop
            for body_node in loop_body:
                for successor in self.graph.successors(body_node):
                    if successor not in loop_body:
                        edge = (body_node, successor)
                        edge_data = self.graph.get_edge_data(body_node, successor, default={})
                        condition = edge_data.get('condition')
                        exit_edges[edge] = condition

            loop_info[header] = {
                'body': loop_body,
                'exits': exit_edges,
                'latch_nodes': latch_nodes,
                'critical_termination_vars': set()
            }

        return loop_info

    def get_loop_info(self) -> LoopAnalysisInfo:
        """Returns the computed loop analysis information

        Returns:
            LoopAnalysisInfo: The dictionary containing loop details
        """
        return self.loop_info

    def get_loop_body(self, header: CFGBasicBlockNode) -> Optional[Set[CFGBasicBlockNode]]:
        """Returns the set of nodes in the loop body for a given header

        Args:
            header (CFGBasicBlockNode): The header node of the loop

        Returns:
            Optional[Set[CFGBasicBlockNode]]: The set of body nodes, or None if
                the header does not identify a known loop
        """
        loop_data = self.loop_info.get(header)
        if loop_data:
            return cast(Optional[Set[CFGBasicBlockNode]], loop_data.get('body'))
        return None

    def get_loop_exits(self, header: CFGBasicBlockNode) -> Optional[LoopExitInfo]:
        """Returns the exit edges and conditions for a given loop header

        Args:
            header (CFGBasicBlockNode): The header node of the loop

        Returns:
            Optional[LoopExitInfo]: A dictionary mapping exit edges (node pairs)
                to their optional conditions, or None if the header does not
                identify a known loop
        """
        loop_data = self.loop_info.get(header)
        if loop_data:
            return cast(Optional[LoopExitInfo], loop_data.get('exits'))
        return None

    def get_latch_nodes(self, header: CFGBasicBlockNode) -> Optional[Set[CFGBasicBlockNode]]:
        """Returns the latch nodes for a given loop header

        Latch nodes are nodes within the loop body that have a back edge to the header

        Args:
            header (CFGBasicBlockNode): The header node of the loop

        Returns:
            Optional[Set[CFGBasicBlockNode]]: The set of latch nodes, or None if
                the header does not identify a known loop
        """
        loop_data = self.loop_info.get(header)
        if loop_data:
            return cast(Optional[Set[CFGBasicBlockNode]], loop_data.get('latch_nodes'))
        return None


def find_prime_paths(cfg) -> Set[PrimePath]:
    """
    Identifies all prime paths in the CFG according to the formal definition:
      1) Collect every simple path (including simple cycles)
      2) Remove any path that is a proper subpath of another simple path
    Returns a set of PrimePath instances
    """
    G = cfg.graph

    # 1) Gather all simple paths (candidates)
    candidates: Set[Tuple[CFGBasicBlockNode, ...]] = set()
    def dfs(path: List[CFGBasicBlockNode]):
        last = path[-1]
        for succ in G.successors(last):
            if succ not in path:
                new_path = path + [succ]
                candidates.add(tuple(new_path))
                dfs(new_path)

    for n in G.nodes():
        dfs([n])

    for cycle in nx.simple_cycles(G):
        candidates.add(tuple(cycle + [cycle[0]]))

    def is_subpath(p: Tuple, q: Tuple) -> bool:
        if len(p) >= len(q):
            return False
        for i in range(len(q) - len(p) + 1):
            if q[i : i + len(p)] == p:
                return True
        return False

    prime_tuples = set(candidates)
    for p in candidates:
        for q in candidates:
            if p is not q and is_subpath(p, q):
                prime_tuples.discard(p)
                break

    # 3) Dedupe by node‐ID tuple and build exactly one PrimePath per sequence
    prime_map: Dict[Tuple[int, ...], PrimePath] = {}
    for tup in prime_tuples:
        key = tuple(node.node_id for node in tup)
        # Only create one PrimePath per unique key
        if key not in prime_map:
            prime_map[key] = PrimePath(nodes=list(tup))

    return set(prime_map.values())

def find_simple_paths(cfg: CFG, start: CFGBasicBlockNode, end: CFGBasicBlockNode) -> List[SimplePath]:
    """Finds all simple paths between two nodes

    A simple path has no repeated nodes, except possibly the start and end nodes

    Args:
        cfg (CFG): The CFG instance
        start (CFGBasicBlockNode): The starting node for paths
        end (CFGBasicBlockNode): The ending node for paths

    Returns:
        List[SimplePath]: A list of all simple paths found
    """
    path_generator = nx.all_simple_paths(cfg.graph, start, end)
    return [SimplePath(nodes=path) for path in path_generator]

def find_edge_pairs(cfg: CFG) -> Set[
    Tuple[Tuple[CFGBasicBlockNode, CFGBasicBlockNode], Tuple[CFGBasicBlockNode, CFGBasicBlockNode]]
]:
    """Extracts all consecutive edge pairs (u->v->w) from the CFG

    Useful for edge-pair coverage criteria in testing

    Args:
        cfg (CFG): The CFG instance

    Returns:
        Set[Tuple[Tuple[Node, Node], Tuple[Node, Node]]]: A set of edge pairs,
            where each pair is represented as ((u, v), (v, w))
    """
    edge_pairs = set()
    for (u, v) in cfg.graph.edges():
        for w in cfg.graph.successors(v):
            edge_pairs.add(((u, v), (v, w)))
    return edge_pairs

def find_shortest_path(cfg: CFG, start: CFGBasicBlockNode, end: CFGBasicBlockNode) -> Path:
    """Finds the shortest path between two nodes using edge count

    Args:
        cfg (CFG): The CFG instance
        start (CFGBasicBlockNode): The starting node
        end (CFGBasicBlockNode): The ending node

    Returns:
        Path: A Path object containing the nodes of the shortest path Returns
            an empty Path if no path exists
    """
    node_list = nx.shortest_path(cfg.graph, start, end)
    return Path(nodes=list(node_list))
