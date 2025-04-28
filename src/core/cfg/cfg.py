"""
Defines the main Control Flow Graph (CFG) class and its core functionalities

Provides the CFG structure based on networkx.DiGraph, manages nodes, edges,
context, analysis results, fragment forest, and properties like complexity
and test paths
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Set, Tuple, Iterator, TYPE_CHECKING

import networkx as nx
from loguru import logger

if TYPE_CHECKING:
    from src.core.fragments import FragmentForest
    from src.core.paths import TestPath, PrimePath, SimplePath

from src.core.nodes.cfg_basic_block_node import CFGBasicBlockNode
from src.core.fragments import FragmentForest
from src.config.config import config
from src.core.cfg.cfg_analysis import find_prime_paths, CFGAnalysis
from src.core.cfg.cfg_context import CFGContext


__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"


@dataclass
class CFG:
    """
    Represents a directed Control Flow Graph (CFG) for an entire function. Uses a 
    networkx.DiGraph with CFGBasicBlockNode objects as nodes

    Attributes:
        graph: The underlying directed graph
        context: A CFGContext providing variable, parameter, and loop-related info
        analysis: An instance of CFGAnalysis holding loop and potentially other analyses
        fragment_forest: Optional container for fragment-level data/code generation
        _entry_node: The designated entry node, if any
        _exit_node: The designated exit node, if any
        max_shtv: An integer storing a global maximum for structural coverage constraints
        _test_paths: Cached test path objects for coverage
        _prime_paths: Cached prime paths objects for coverage
        _simple_paths: Cached simple paths
        _cc: Cached cyclomatic complexity value
        _code: Cached generated code string
        statically_infeasible_test_paths: Paths determined infeasible by analysis
    """

    graph: nx.DiGraph = field(init=False)
    context: CFGContext = field(init=False)
    analysis: Optional[CFGAnalysis] = field(default=None, init=False)
    fragment_forest: Optional[FragmentForest] = field(init=False)
    max_shtv: int = field(init=False)

    _entry_node: Optional[CFGBasicBlockNode] = field(default=None, init=False)
    _exit_node: Optional[CFGBasicBlockNode] = field(default=None, init=False)
    _test_paths: Optional[Set[TestPath]] = field(default=None, init=False)
    _prime_paths: Optional[Set[PrimePath]] = field(default=None, init=False)
    _simple_paths: Optional[Set[SimplePath]] = field(default=None, init=False)
    _cc: Optional[int] = field(default=None, init=False)
    _code: Optional[str] = field(default=None, init=False)

    statically_infeasible_test_paths: Optional[List[TestPath]] = field(default=None, init=False)

    def __post_init__(self):
        self.context = CFGContext(cfg=self)
        self.graph = nx.DiGraph()
        self.max_shtv = config.global_max_shtv

        if self.__class__ == CFG:
            self.fragment_forest = FragmentForest()
        else:
            self.fragment_forest = None

        # Force CC recalculation
        self._cc = None  
        self.fragment_forest.set_cfg_reference(self)

    def run_analysis(self):
        """Runs the CFG analysis (e.g., loop detection) and stores the results"""
        logger.info("Running CFG analysis...")
        self.analysis = CFGAnalysis(self)
        logger.info("CFG analysis complete.")

    @property
    def entry_node(self) -> Optional[CFGBasicBlockNode]:
        """
        Return the entry node for the CFG, searching the graph if not set
        """
        if self._entry_node is None:
            logger.debug("Entry node not set, searching graph...")
            for node in self.graph:
                if node.is_entry:
                    logger.debug(f"Found entry node {node.node_id} during search")
                    self._entry_node = node
                    break
        return self._entry_node

    @entry_node.setter
    def entry_node(self, node: CFGBasicBlockNode) -> None:
        """
        Set the entry node for the CFG and mark it appropriately
        """
        node.is_entry = True
        self._entry_node = node

    @property
    def exit_node(self) -> Optional[CFGBasicBlockNode]:
        """
        Return the exit node for the CFG, searching the graph if not set
        """
        if self._exit_node is None:
            logger.debug("Exit node not set, searching graph...")
            for node in self.graph:
                if node.is_exit:
                    logger.debug(f"Found exit node {node.node_id} during search")
                    self._exit_node = node
                    break
        return self._exit_node

    @exit_node.setter
    def exit_node(self, node: CFGBasicBlockNode) -> None:
        """
        Set the exit node for the CFG and mark it appropriately
        """
        node.is_exit = True
        self._exit_node = node

    @property
    def test_paths(self) -> Optional[Set[TestPath]]:
        """
        Return the set of test paths for coverage, if any
        """
        return self._test_paths

    @test_paths.setter
    def test_paths(self, paths: Set[TestPath]) -> None:
        """
        Set the test paths for this CFG, if they are not already set
        """
        if self._test_paths is None:
            self._test_paths = set(paths)
        else:
            raise ValueError("Test paths already set")

    @property
    def prime_paths(self) -> Set[PrimePath]:
        """
        Return or compute the prime paths for the CFG
        """
        if self._prime_paths is None:
            raw_paths = find_prime_paths(self)
            self._prime_paths = set()
            for path in raw_paths:
                self._prime_paths.add(path)
        return self._prime_paths

    @prime_paths.setter
    def prime_paths(self, paths: Set[PrimePath]) -> None:
        """
        Set the prime paths for this CFG, if not already set
        """
        if self._prime_paths is None:
            self._prime_paths = set()
            for path in paths:
                self._prime_paths.add(path)
        else:
            raise ValueError("Prime paths already set")

    @property
    def prime_paths_count(self) -> int:
        """
        Return the count of prime paths for the CFG
        """
        if self._prime_paths is None:
            self._prime_paths = find_prime_paths(self)
        assert self._prime_paths is not None
        return len(self._prime_paths)

    @property
    def cc(self) -> int:
        """
        Return the cyclomatic complexity for the CFG
        """
        if self._cc is None:
            edges = self.graph.number_of_edges()
            nodes = self.graph.number_of_nodes()
            connected_components = 1
            self._cc = edges - nodes + 2 * connected_components
        assert self._cc is not None
        return self._cc

    @cc.setter
    def cc(self, cc: int) -> None:
        """
        Set the cyclomatic complexity once if it is not already set
        """
        if self._cc is None:
            self._cc = cc
        else:
            raise ValueError("CC already set")

    @property
    def shtv(self) -> float:
        """
        Compute and return the SHTV (some structural coverage metric) for the CFG
        Involves node-level and global factors (prime paths, cyclomatic complexity, variables)
        """
        weights = config.weights
        node_shtv_sum = sum(node.shtv for node in self)

        prime_paths = self.prime_paths_count
        cyclomatic_complexity = self.cc
        variables = len(self.context.used_variables)

        global_factors = (
            weights['g'] * prime_paths +
            weights['h'] * cyclomatic_complexity +
            weights['i'] * variables
        )
        return float(node_shtv_sum + global_factors)

    @property
    def code(self) -> str:
        """
        Generate code from the fragment forest if available
        """
        if self.fragment_forest is None:
            logger.warning("Cannot generate code: FragmentForest is not initialized")
            return ""
        return self.fragment_forest.generate_code(self.context)

    def add_node(self, node: CFGBasicBlockNode) -> None:
        """
        Add a CFGBasicBlockNode to the CFG
        
        Args:
            block: The basic block node to add
        """
        if node in self.graph:
            raise ValueError(f"Node {node.node_id} already exists in graph")
        node.cfg = self
        self.graph.add_node(node)

    def erase_node(self, block: CFGBasicBlockNode) -> None:
        """
        Remove a node and clear references to allow garbage collection

        Args:
            block: The node to remove
        """
        if block.is_entry:
            self._entry_node = None
        if block.is_exit:
            self._exit_node = None

        if block not in self.graph:
            raise ValueError(f"Node {block.node_id} not found in graph and cannot be removed")

        self._prime_paths = None
        self._test_paths = None
        self._simple_paths = None
        self._cc = None

        if self.fragment_forest is not None:
            self.fragment_forest.update_fragment_related_nodes(block)

        self.graph.remove_node(block)
        block.cleanup()
        logger.info(f"Node {block.node_id} completely removed from CFG and cleared for garbage collection")

    def add_edge(self, from_block: CFGBasicBlockNode, to_block: CFGBasicBlockNode) -> None:
        """
        Add an edge from one block to another

        Args:
            from_block: Source node
            to_block: Target node
        """
        if self.graph.has_edge(from_block, to_block):
            raise ValueError(f"Edge from {from_block.node_id} to {to_block.node_id} already exists")

        self.graph.add_edge(from_block, to_block)
        # Force CC recalculation
        self._cc = None  

    def init_entry_node(self) -> CFGBasicBlockNode:
        """
        Initialize and return a new entry node for the CFG
        """
        block = CFGBasicBlockNode(jump=None)
        block.is_entry = True
        self.add_node(block)
        self.entry_node = block
        return block

    def init_exit_node(self) -> CFGBasicBlockNode:
        """
        Initialize and return a new exit node for the CFG
        """
        block = CFGBasicBlockNode(jump=None)
        block.is_exit = True
        self.add_node(block)
        self.exit_node = block
        return block

    def get_node_children(self, node: CFGBasicBlockNode) -> List[CFGBasicBlockNode]:
        """
        Return all child nodes (successors) of the specified node

        Args:
            node: The node whose children are needed

        Returns:
            A list of child nodes
        """
        if node not in self.graph:
            logger.warning(f"Attempted to get children of non-existent node {node.node_id}")
            return []
        return list(self.graph.successors(node))

    def get_node_parents(self, node: CFGBasicBlockNode) -> List[CFGBasicBlockNode]:
        """
        Return all parent nodes (predecessors) of the specified node

        Args:
            node: The node whose parents are needed

        Returns:
            A list of parent nodes
        """
        if node not in self.graph:
            logger.warning(f"Attempted to get parents of non-existent node {node.node_id}")
            return []
        return list(self.graph.predecessors(node))

    def nodes_bfs(self) -> Iterator[CFGBasicBlockNode]:
        """
        Perform a BFS traversal from the entry node (or the first node if no entry)
        
        Yields:
            CFGBasicBlockNode objects in BFS order
        """
        start = self.entry_node
        if start is None:
            all_nodes = list(self.graph.nodes())
            start = all_nodes[0] if all_nodes else None

        if not start:
            return

        visited = set()
        for block in nx.bfs_tree(self.graph, start):
            if block not in visited:
                visited.add(block)
                yield block

    def nodes_dfs(self) -> Iterator[CFGBasicBlockNode]:
        """
        Perform a DFS traversal from the entry node (or the first node if no entry)

        Yields:
            CFGBasicBlockNode objects in DFS order
        """
        start = self.entry_node
        if start is None:
            all_nodes = list(self.graph.nodes())
            start = all_nodes[0] if all_nodes else None

        if not start:
            return

        visited = set()
        for block in nx.dfs_preorder_nodes(self.graph, start):
            if block not in visited:
                visited.add(block)
                yield block

    def __len__(self):
        """Return the number of nodes in the CFG"""
        return self.graph.number_of_nodes()

    def __iter__(self) -> Iterator[CFGBasicBlockNode]:
        """Iterate over all nodes in the CFG"""
        return iter(self.graph.nodes())

    def edges(self) -> Iterator[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]]:
        """
        Return an iterator over all edges in the CFG as (from_node, to_node) tuples
        """
        return iter(self.graph.edges())

    def get_edges(self) -> List[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]]:
        """
        Return a list of all edges in the CFG as (from_node, to_node) tuples
        """
        return list(self.graph.edges())

    def __str__(self) -> str:
        """Return a string representation of the entry and exit node"""
        return str(self.entry_node) + str(self.exit_node)

    def remove_edge(self, from_block: CFGBasicBlockNode, to_block: CFGBasicBlockNode) -> None:
        """
        Remove an edge from one node to another in the CFG

        Args:
            from_block: The source node
            to_block: The target node
        """
        if not self.graph.has_edge(from_block, to_block):
            raise ValueError(f"Edge from {from_block.node_id} to {to_block.node_id} does not exist")

        self.graph.remove_edge(from_block, to_block)
        # Force CC recalculation
        self._cc = None  

    def set_edge_label(self, from_node: CFGBasicBlockNode, to_node: CFGBasicBlockNode, label: str) -> None:
        """
        Set a label (e.g., 'true', 'false') on a specific edge in the graph.

        Args:
            from_node: The source node of the edge
            to_node: The target node of the edge
            label: The string label to assign to the edge
        """
        self.graph.edges[from_node, to_node]['label'] = label
        logger.debug(f"Set label '{label}' on edge {from_node.node_id} -> {to_node.node_id}")

    def get_edge_label(self, from_node: CFGBasicBlockNode, to_node: CFGBasicBlockNode) -> Optional[str]:
        """
        Get the label of a specific edge in the graph

        Args:
            from_node: The source node of the edge
            to_node: The target node of the edge

        Returns:
            The label string if set, otherwise None
        """
        return self.graph.edges[from_node, to_node].get('label')  