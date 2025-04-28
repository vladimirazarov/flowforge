"""Find test paths in a Control Flow Graph based on coverage criteria

This module provides the `TestPathsFinder` class, which analyzes a CFG and
generates a list of test paths sufficient to satisfy specified structural
coverage criteria (Node, Edge, Edge-Pair, Prime Path)

It uses the networkx library for graph analysis and incorporates strategies
to handle loops and ensure path validity. The main entry point is the
`find_and_set_test_paths` function
"""

from __future__ import annotations

from collections import Counter
from loguru import logger
import networkx as nx
from typing import Set, List, Tuple, Optional, cast

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
import rich.box

from src.core.paths import TestPath
from src.core.nodes.cfg_basic_block_node import CFGBasicBlockNode
from src.core.cfg.cfg_analysis import find_prime_paths, find_edge_pairs
from src.config.config import AppConfig
from src.utils.logging_helpers import print_test_path_finder_start_info

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

class TestPathsFinder:
    __test__ = False          # pytest: do NOT treat this as a test class
    """Find test paths in a Control Flow Graph (CFG)

    Finds test paths sufficient to satisfy a specified coverage criterion
    (Node Coverage, Edge Coverage, Edge Pair Coverage, Prime Path Coverage)
    The generated paths are stored in the `test_paths` attribute of the CFG

    Attributes:
        cfg: The Control Flow Graph object
        app_config: Configuration settings for the application
        test_paths: List storing the generated `TestPath` objects
        coverage_criterion: The coverage criterion to satisfy
        loop_headers: Set of nodes identified as loop entry points
        visited_edges: Set to track edges visited during path finding (unused)
        console: Rich console object for formatted output
        max_loop_iters: Maximum times a loop header can be visited in a path
    """
    def __init__(self, cfg, app_config: AppConfig):
        """Initialize the TestPathsFinder

        Args:
            cfg: The CFG object to analyze
            app_config (AppConfig): Application configuration settings
        """
        self.cfg = cfg
        self.test_paths: List[TestPath] = []
        self.app_config = app_config
        self.coverage_criterion = self.app_config.test_paths_finder.coverage_criterion
        self.loop_headers = self._identify_loop_headers()
        self.visited_edges = set()
        self.console = Console()
        self.max_loop_iters = 2

    def _identify_loop_headers(self) -> Set[CFGBasicBlockNode]:
        """Helper to identify nodes that start cycles (loop headers)

        Uses `networkx.simple_cycles` to find all simple cycles in the CFG
        The first node of each cycle found is considered a loop header

        Returns:
            Set[CFGBasicBlockNode]: A set containing all identified loop header nodes
        """
        loop_headers = set()

        for cycle in nx.simple_cycles(self.cfg.graph):
            if cycle:
                loop_headers.add(cycle[0])

        return loop_headers

    def _is_loop_header(self, node: CFGBasicBlockNode) -> bool:
        """Helper to check if a node is a loop header

        Args:
            node (CFGBasicBlockNode): The node to check

        Returns:
            bool: True if the node is in the set of loop headers, False otherwise
        """
        return node in self.loop_headers

    def _get_loop_body(self, header: CFGBasicBlockNode) -> Set[CFGBasicBlockNode]:
        """Helper to get all nodes belonging to a specific loop

        Performs a Depth First Search (DFS) starting from the successors of the
        loop header to identify all nodes that can reach the header back

        Args:
            header (CFGBasicBlockNode): The loop header node

        Returns:
            Set[CFGBasicBlockNode]: A set of nodes constituting the loop body,
                excluding the header itself. Returns an empty set if the header
                is not a valid loop header
        """
        if not self._is_loop_header(header):
            return set()

        loop_body = set()
        visited = set()

        def dfs(node):
            if node in visited:
                return
            visited.add(node)

            for succ in self.cfg.get_node_children(node):
                if succ == header:
                    loop_body.add(node)
                    return True

                if dfs(succ):
                    loop_body.add(node)
                    return True
            return False

        for succ in self.cfg.get_node_children(header):
            dfs(succ)

        return loop_body

    def _convert_node_list_to_test_path(self, nodes: List[CFGBasicBlockNode]) -> TestPath:
        """Helper to convert a node list into a TestPath object

        Args:
            nodes (List[CFGBasicBlockNode]): The sequence of nodes forming the path

        Returns:
            TestPath: A TestPath object representing the input node sequence
        """
        test_path = TestPath()
        for node in nodes:
            test_path.nodes.append(node)
        return test_path

    def find_test_paths(self) -> List[TestPath]:
        """Find test paths satisfying the configured coverage criterion

        Selects the appropriate path finding strategy based on the
        `self.coverage_criterion`. It then validates the generated paths for
        continuity and ensures they start at the CFG entry and end at the CFG
        exit. Valid paths are stored in `self.cfg.test_paths`

        Returns:
            List[TestPath]: A list of validated TestPath objects satisfying the
                coverage criterion

        Raises:
            ValueError: If the CFG lacks an entry or exit node when required by
                a coverage strategy
        """

        num_nodes = len(list(self.cfg.graph.nodes()))
        num_edges = len(list(self.cfg.graph.edges()))
        print_test_path_finder_start_info(self.console, str(self.coverage_criterion), num_nodes, num_edges)

        if self.coverage_criterion == "NC":
            paths = self._find_node_coverage_paths()
        elif self.coverage_criterion == "EC":
            paths = self._find_edge_coverage_paths()
        elif self.coverage_criterion == "EPC":
            paths = self._find_edge_pair_coverage_paths()
        else:
            paths = self._find_test_paths_covering_prime_paths()

        test_paths: List[TestPath] = []
        for path in paths:
            if not path or not path.nodes:
                continue

            path_nodes = path.nodes

            valid = True
            for i in range(len(path_nodes) - 1):
                if not self.cfg.graph.has_edge(path_nodes[i], path_nodes[i + 1]):
                    logger.warning(f"Invalid path found (discontinuity): {path_nodes}")
                    valid = False
                    break

            if not valid:
                continue

            entry_node = self.cfg.entry_node
            exit_node = self.cfg.exit_node

            if not entry_node or not exit_node or path_nodes[0] != entry_node or path_nodes[-1] != exit_node:
                logger.warning(f"Path does not start at entry or end at exit: {path_nodes}")
                continue

            test_paths.append(path)

        # remove any path where any loop‐header appears >2 timesd
        filtered: List[TestPath] = []
        for p in test_paths:
            # count how often each node appears
            cnt = Counter(p.nodes)
            # if any header-node is visited more than twice (i.e. >1 iteration), drop
            if any(cnt[hdr] > 2 for hdr in self.loop_headers):
                continue
            filtered.append(p)

        self.cfg.test_paths = filtered

        self.console.print(Rule("[bold blue]🏁 Test Path Finding Finished[/bold blue]"))

        if filtered:
            paths_table = Table(title="Generated Test Paths", show_header=True, header_style="bold cyan", box=rich.box.SIMPLE)
            paths_table.add_column("Path #", style="dim", width=6)
            paths_table.add_column("Node Sequence", style="white")

            for i, path_obj in enumerate(filtered):
                node_str = "->".join(str(node.node_id) for node in path_obj.nodes)
                paths_table.add_row(str(i + 1), node_str)

            self.console.print(paths_table)

        self.console.print(Rule())

        return filtered

    def _find_all_paths(self, start: CFGBasicBlockNode, end: CFGBasicBlockNode) -> List[List[CFGBasicBlockNode]]:
        """Helper to find all simple paths between two nodes, limiting loop traversals

        Uses a custom DFS (`_find_all_simple_paths_one_loop`) to find paths,
        ensuring loop headers are not visited more than `self.max_loop_iters` times

        Args:
            start (CFGBasicBlockNode): The starting node
            end (CFGBasicBlockNode): The ending node

        Returns:
            List[List[CFGBasicBlockNode]]: A list of paths, where each path is
                a list of nodes adhering to the loop traversal limit
        """
        return self._find_all_simple_paths_one_loop(start, end)

    def _find_all_simple_paths_one_loop(self, start: CFGBasicBlockNode, end: CFGBasicBlockNode) -> List[List[CFGBasicBlockNode]]:
        """Internal helper to find paths using DFS with loop visit constraints

        Args:
            start (CFGBasicBlockNode): The starting node
            end (CFGBasicBlockNode): The ending node

        Returns:
            List[List[CFGBasicBlockNode]]: A list of paths found
        """
        visits = Counter()
        result: List[List[CFGBasicBlockNode]] = []

        def dfs(node: CFGBasicBlockNode, path: List[CFGBasicBlockNode]):
            if node == end:
                result.append(path.copy())
                return
            for succ in self.cfg.get_node_children(node):
                vcount = visits[succ]
                # if it's a loop header, only allow up to self.max_loop_iters visits
                if self._is_loop_header(succ):
                    if vcount >= self.max_loop_iters:
                        continue
                # for non‐loop nodes, only allow once
                else:
                    if vcount >= 1:
                        continue

                visits[succ] += 1
                path.append(succ)
                dfs(succ, path)
                path.pop()
                visits[succ] -= 1

        visits[start] = 1
        dfs(start, [start])
        return result

    def _find_node_coverage_paths(self) -> List[TestPath]:
        """Helper to find paths satisfying Node Coverage (NC)

        Generates a set of test paths such that every node in the CFG is visited
        at least once. It uses shortest paths to connect the entry node, target
        uncovered nodes, and the exit node iteratively

        Returns:
            List[TestPath]: A list of TestPath objects covering all nodes

        Raises:
            ValueError: If the CFG does not have an entry or exit node defined
        """
        logger.info("[TestPathsFinder] Finding node coverage paths")
        all_nodes = set(self.cfg)
        covered_nodes = set()
        paths: List[TestPath] = []

        entry_node = self.cfg.entry_node
        exit_node = self.cfg.exit_node

        if not entry_node or not exit_node:
            raise ValueError("CFG must have entry and exit nodes")

        if entry_node:
            covered_nodes.add(entry_node)
        if exit_node:
            covered_nodes.add(exit_node)

        _initial_path_nodes_raw = nx.shortest_path(self.cfg.graph, entry_node, exit_node)
        initial_path_nodes = cast(List[CFGBasicBlockNode], _initial_path_nodes_raw)
        if initial_path_nodes:
             initial_test_path = self._convert_node_list_to_test_path(initial_path_nodes)
             paths.append(initial_test_path)
             covered_nodes.update(initial_path_nodes)

        remaining_nodes = all_nodes - covered_nodes
        while remaining_nodes:
            target_node = remaining_nodes.pop()

            path_nodes = self._find_path_through_node(entry_node, target_node, exit_node)

            if path_nodes:
                test_path = self._convert_node_list_to_test_path(path_nodes)
                paths.append(test_path)
                newly_covered = set(path_nodes)
                covered_nodes.update(newly_covered)
                remaining_nodes -= newly_covered
            else:
                logger.warning(f"Could not find path to cover node {target_node}")

        unique_paths_dict = {tuple(p.nodes): p for p in paths}

        return list(unique_paths_dict.values())

    def _find_path_with_loop_iterations(self, start: CFGBasicBlockNode, header: CFGBasicBlockNode, end: CFGBasicBlockNode, iterations: int) -> Optional[List[CFGBasicBlockNode]]:
        """Helper to find a path traversing a loop a specific number of times

        Constructs a path from `start` to `end` that passes through the loop
        defined by `header` exactly `iterations` times. It finds shortest paths
        to the header, within the loop body back to the header, and from the
        header to the end node

        Args:
            start (CFGBasicBlockNode): The starting node of the path
            header (CFGBasicBlockNode): The header node of the loop to traverse
            end (CFGBasicBlockNode): The ending node of the path
            iterations (int): The desired number of times to traverse the loop body

        Returns:
            Optional[List[CFGBasicBlockNode]]: A list of nodes representing the
                constructed path, or None if such a path cannot be constructed
                (e.g., no path exists between segments, or iterations > 0 but
                no loop body path is found)
        """
        _path_to_raw = nx.shortest_path(self.cfg.graph, start, header)
        path_to = cast(List[CFGBasicBlockNode], _path_to_raw)

        loop_body = self._get_loop_body(header)
        if not loop_body and iterations > 0:
            return None

        loop_path = None
        if iterations > 0:
            for body_node in loop_body:
                _body_to_header_raw = nx.shortest_path(self.cfg.graph, body_node, header)
                body_to_header = cast(List[CFGBasicBlockNode], _body_to_header_raw)
                _header_to_body_raw = nx.shortest_path(self.cfg.graph, header, body_node)
                header_to_body = cast(List[CFGBasicBlockNode], _header_to_body_raw)

                if header_to_body:
                    loop_path = header_to_body[:-1] + body_to_header
                    break

        _path_from_raw = nx.shortest_path(self.cfg.graph, header, end)
        path_from = cast(List[CFGBasicBlockNode], _path_from_raw)

        if not isinstance(path_to, list):
             logger.error(f"Path 'path_to' is not a list: {path_to}")
             return None
        result: List[CFGBasicBlockNode] = path_to.copy()

        if loop_path and iterations > 0:
            if not isinstance(loop_path, list):
                 logger.error(f"Path 'loop_path' is not a list: {loop_path}")
                 return None
            for _ in range(iterations):
                result.extend(loop_path[1:])

        if not isinstance(path_from, list):
            logger.error(f"Path 'path_from' is not a list: {path_from}")
            return None
        result.extend(path_from[1:])

        return result

    def _find_edge_coverage_paths(self) -> List[TestPath]:
        """Helper to find paths satisfying Edge Coverage (EC)

        Generates a set of test paths such that every edge in the CFG is
        traversed at least once. It uses shortest paths to connect the entry node,
        target uncovered edges, and the exit node iteratively

        Returns:
            List[TestPath]: A list of TestPath objects covering all edges

        Raises:
            ValueError: If the CFG does not have an entry or exit node defined
        """
        logger.info("[TestPathsFinder] Finding edge coverage paths")
        all_edges = set(self.cfg.edges())
        covered_edges = set()
        paths: List[TestPath] = []

        entry_node = self.cfg.entry_node
        exit_node = self.cfg.exit_node

        if not entry_node or not exit_node:
            raise ValueError("CFG must have entry and exit nodes")

        _initial_path_nodes_raw = nx.shortest_path(self.cfg.graph, entry_node, exit_node)
        initial_path_nodes = cast(List[CFGBasicBlockNode], _initial_path_nodes_raw)
        if initial_path_nodes:
             initial_test_path = self._convert_node_list_to_test_path(initial_path_nodes)
             paths.append(initial_test_path)
             initial_edges = set(zip(initial_path_nodes[:-1], initial_path_nodes[1:]))
             covered_edges.update(initial_edges)

        remaining_edges = all_edges - covered_edges
        while remaining_edges:
            target_edge = remaining_edges.pop()

            path_nodes = self._find_path_through_edge(entry_node, target_edge, exit_node)

            if path_nodes:
                test_path = self._convert_node_list_to_test_path(path_nodes)
                paths.append(test_path)
                newly_covered_edges = set(zip(path_nodes[:-1], path_nodes[1:]))
                covered_edges.update(newly_covered_edges)
                remaining_edges -= newly_covered_edges
            else:
                logger.warning(f"Could not find path to cover edge {target_edge}")

        unique_paths_dict = {tuple(p.nodes): p for p in paths}
        return list(unique_paths_dict.values())

    def _find_path_through_edge(self, start: CFGBasicBlockNode, edge: Tuple[CFGBasicBlockNode, CFGBasicBlockNode], end: CFGBasicBlockNode) -> Optional[List[CFGBasicBlockNode]]:
        """Helper to find a shortest path traversing a specific edge

        Constructs a path from `start` to `end` that includes the directed edge
        `(u, v)` by finding the shortest path from `start` to `u` and the
        shortest path from `v` to `end`

        Args:
            start (CFGBasicBlockNode): The starting node
            edge (Tuple[CFGBasicBlockNode, CFGBasicBlockNode]): The edge (u, v)
                that the path must traverse
            end (CFGBasicBlockNode): The ending node

        Returns:
            Optional[List[CFGBasicBlockNode]]: A list of nodes representing the
                shortest path through the edge, or None if no such path exists
        """
        u, v = edge
        path_to_u: Optional[List[CFGBasicBlockNode]] = None
        path_from_v: Optional[List[CFGBasicBlockNode]] = None

        # Find path to u
        _path_to_u_raw = nx.shortest_path(self.cfg.graph, start, u)
        path_to_u = cast(List[CFGBasicBlockNode], _path_to_u_raw)

        # Find path from v to end
        _path_from_v_raw = nx.shortest_path(self.cfg.graph, v, end)
        path_from_v = cast(List[CFGBasicBlockNode], _path_from_v_raw)

        # Combine paths - ensure both are lists before concatenation
        # The path includes the edge (u, v) by construction
        # path_to_u ends in u, path_from_v starts with v
        if path_to_u is not None and path_from_v is not None:
            # Correct combination: path to u + path from v (starting with v)
            return path_to_u + path_from_v

    def _find_edge_pair_coverage_paths(self) -> List[TestPath]:
        """Helper to find paths satisfying Edge Pair Coverage (EPC)

        Generates test paths covering all feasible edge pairs (sequences of two
        adjacent edges) in the CFG. It uses `find_edge_pairs` to determine the
        required pairs and then iteratively finds shortest paths through each
        uncovered pair

        Returns:
            List[TestPath]: A list of TestPath objects covering all edge pairs

        Raises:
            ValueError: If the CFG does not have an entry or exit node defined
        """
        logger.info("[TestPathsFinder] Finding edge pair coverage paths")

        edge_pairs_to_cover = find_edge_pairs(self.cfg)
        covered_edge_pairs = set()
        paths: List[TestPath] = []

        entry_node = self.cfg.entry_node
        exit_node = self.cfg.exit_node

        if not entry_node or not exit_node:
            raise ValueError("CFG must have entry and exit nodes")

        _initial_path_nodes_raw = nx.shortest_path(self.cfg.graph, entry_node, exit_node)
        initial_path_nodes = cast(List[CFGBasicBlockNode], _initial_path_nodes_raw)
        if initial_path_nodes:
             initial_test_path = self._convert_node_list_to_test_path(initial_path_nodes)
             paths.append(initial_test_path)
             initial_edges = list(zip(initial_path_nodes[:-1], initial_path_nodes[1:]))
             initial_edge_pairs = set(zip(initial_edges[:-1], initial_edges[1:]))
             covered_edge_pairs.update(initial_edge_pairs)

        remaining_pairs = edge_pairs_to_cover - covered_edge_pairs
        max_depth = len(list(self.cfg)) * 4

        while remaining_pairs:
            target_pair = remaining_pairs.pop()
            path_nodes = self._find_path_through_edge_pair(entry_node, target_pair, exit_node, max_depth)

            if path_nodes:
                test_path = self._convert_node_list_to_test_path(path_nodes)
                paths.append(test_path)
                edges_in_path = list(zip(path_nodes[:-1], path_nodes[1:]))
                edge_pairs_in_path = set(zip(edges_in_path[:-1], edges_in_path[1:]))
                newly_covered = edge_pairs_in_path & edge_pairs_to_cover
                covered_edge_pairs.update(newly_covered)
                remaining_pairs -= newly_covered
                logger.debug(f"[TestPathsFinder] Found path covering edge pair {target_pair}: {path_nodes}")
            else:
                 logger.warning(f"Could not find path to cover edge pair {target_pair}")

        missing = edge_pairs_to_cover - covered_edge_pairs
        if missing:
            logger.warning(f"Could not cover all required edge pairs. Missing: {missing}")

        unique_paths_dict = {tuple(p.nodes): p for p in paths}
        return list(unique_paths_dict.values())

    def _find_path_through_edge_pair(self, start: CFGBasicBlockNode, edge_pair: Tuple[Tuple[CFGBasicBlockNode, CFGBasicBlockNode], Tuple[CFGBasicBlockNode, CFGBasicBlockNode]],
                                   end: CFGBasicBlockNode, max_depth: int) -> Optional[List[CFGBasicBlockNode]]:
        """Helper to find a shortest path traversing a specific edge pair

        Constructs a path from `start` to `end` that includes the edge pair
        `((u, v), (v, w))`. It finds the shortest path from `start` to `u` and
        the shortest path from `w` to `end`, then concatenates them with node `v`
        in between. Includes a depth check to prevent excessively long paths

        Args:
            start (CFGBasicBlockNode): The starting node
            edge_pair (Tuple[Tuple[CFGBasicBlockNode, CFGBasicBlockNode], Tuple[CFGBasicBlockNode, CFGBasicBlockNode]]):
                The edge pair ((u, v), (v, w)) to traverse
            end (CFGBasicBlockNode): The ending node
            max_depth (int): The maximum allowed length for the generated path

        Returns:
            Optional[List[CFGBasicBlockNode]]: A list of nodes for the path,
                or None if no such path exists, if the middle nodes of the pair
                don't match, or if the path exceeds `max_depth`
        """
        (u, v), (v2, w) = edge_pair
        if v != v2:
             logger.error(f"Invalid edge pair provided: {edge_pair}. Middle nodes do not match.")
             return None

        path_to_u: Optional[List[CFGBasicBlockNode]] = None
        path_from_w: Optional[List[CFGBasicBlockNode]] = None

        # First find path to u
        _path_to_u_raw = nx.shortest_path(self.cfg.graph, start, u)
        path_to_u = cast(List[CFGBasicBlockNode], _path_to_u_raw)

        # Then find path from w to end
        _path_from_w_raw = nx.shortest_path(self.cfg.graph, w, end)
        path_from_w = cast(List[CFGBasicBlockNode], _path_from_w_raw)

        # Combine paths: path to u -> v -> w -> path from w
        # Ensure all components are lists before combining
        if path_to_u is not None and path_from_w is not None:
             # path_to_u ends in u. path_from_w starts with w.
             # We need path_to_u + [v] + path_from_w
             path = path_to_u + [v] + path_from_w


        if len(path) > max_depth:
             logger.warning(f"Generated path for edge pair {edge_pair} exceeds max depth {max_depth}. Path: {path}")
             return None # Path too long, likely due to complex loop structure not fully handled by shortest path
             
        return path

    def _find_test_paths_covering_prime_paths(self) -> List[TestPath]:
        """Helper to find paths satisfying Prime Path Coverage (PPC)

        Generates test paths covering all prime paths in the CFG. It uses
        `find_prime_paths` to get the prime paths and then constructs a full
        test path (entry to exit) for each prime path by finding shortest paths
        from the CFG entry to the prime path's start and from the prime path's
        end to the CFG exit

        Returns:
            List[TestPath]: A list of TestPath objects covering all prime paths

        Raises:
            ValueError: If the CFG does not have an entry or exit node defined
        """
        logger.info("[TestPathsFinder] Finding test paths covering prime paths")
        paths: List[TestPath] = []

        entry_node = self.cfg.entry_node
        exit_node = self.cfg.exit_node

        prime_paths_set = find_prime_paths(self.cfg)
        
        # For each prime path, find a test path that covers it
        for prime_path in prime_paths_set:
            nodes = prime_path.nodes
            if not nodes or len(nodes) < 1:
                logger.debug(f"Skipping empty or invalid prime path: {prime_path}")
                continue

            prime_start_node = nodes[0]
            prime_end_node = nodes[-1]

            # Find path from entry to start of prime path
            path_to_start: Optional[List[CFGBasicBlockNode]] = None
            if entry_node == prime_start_node:
                path_to_start = [entry_node]
            else:
                _path_to_start_raw = nx.shortest_path(self.cfg.graph, entry_node, prime_start_node)
                path_to_start = cast(List[CFGBasicBlockNode], _path_to_start_raw)

            # Find path from end of prime path to exit
            path_from_end: Optional[List[CFGBasicBlockNode]] = None
            if prime_end_node == exit_node:
                path_from_end = [exit_node]
            else:
                _path_from_end_raw = nx.shortest_path(self.cfg.graph, prime_end_node, exit_node)
                path_from_end = cast(List[CFGBasicBlockNode], _path_from_end_raw)

            # Combine paths: path_to_start (excluding last) + prime_path_nodes + path_from_end (excluding first)
            if path_to_start is not None and path_from_end is not None:
                # Handle overlaps: if prime path starts at entry or ends at exit
                combined_nodes: List[CFGBasicBlockNode] = []
                if len(path_to_start) > 1:
                    combined_nodes.extend(path_to_start[:-1])

                combined_nodes.extend(nodes)

                if len(path_from_end) > 1:
                     # Only extend if path_from_end is cfg_content than just the exit node
                     # And avoid duplicating the last node of the prime path if it's the start of path_from_end
                     if combined_nodes[-1] == path_from_end[0]:
                          combined_nodes.extend(path_from_end[1:])
                     else:
                          logger.error("Mismatch between prime path end and path_from_end start - unexpected graph structure?")
                          combined_nodes.extend(path_from_end)
                elif len(path_from_end) == 1 and combined_nodes[-1] != path_from_end[0]:
                    combined_nodes.append(path_from_end[0])


                test_path = self._convert_node_list_to_test_path(combined_nodes)
                paths.append(test_path)


        unique_paths_dict = {tuple(p.nodes): p for p in paths}
        return list(unique_paths_dict.values())

    def _is_subpath(self, subpath: List[CFGBasicBlockNode] | Tuple[CFGBasicBlockNode, ...], path: List[CFGBasicBlockNode] | Tuple[CFGBasicBlockNode, ...]) -> bool:
        """Helper to check if a sequence of nodes is a subpath of another

        Args:
            subpath (List[CFGBasicBlockNode] | Tuple[CFGBasicBlockNode, ...]):
                The potential subpath
            path (List[CFGBasicBlockNode] | Tuple[CFGBasicBlockNode, ...]):
                The full path to check against

        Returns:
            bool: True if `subpath` is found within `path`, False otherwise
        """
        sub_str = ','.join(str(node.node_id) for node in subpath)
        path_str = ','.join(str(node.node_id) for node in path)
        return sub_str in path_str

    def _find_path_through_node(self, start: CFGBasicBlockNode, through: CFGBasicBlockNode, end: CFGBasicBlockNode) -> Optional[List[CFGBasicBlockNode]]:
        """Helper to find a shortest path passing through a specific node

        Constructs a path from `start` to `end` that includes the `through` node
        by finding the shortest path from `start` to `through` and the shortest
        path from `through` to `end`

        Args:
            start (CFGBasicBlockNode): The starting node
            through (CFGBasicBlockNode): The node the path must pass through
            end (CFGBasicBlockNode): The ending node

        Returns:
            Optional[List[CFGBasicBlockNode]]: A list of nodes representing the
                shortest path through the node, or None if no such path exists
        """
        _path_to_raw = nx.shortest_path(self.cfg.graph, start, through)
        path_to = cast(List[CFGBasicBlockNode], _path_to_raw)

        _path_from_raw = nx.shortest_path(self.cfg.graph, through, end)
        path_from = cast(List[CFGBasicBlockNode], _path_from_raw)

        if path_to is not None and path_from is not None:
            return path_to + path_from[1:]
        else:
            return None

def find_and_set_test_paths(cfg, app_config: AppConfig):
    """Find and set test paths for a CFG based on configuration

    Instantiates a `TestPathsFinder` and calls its `find_test_paths` method
    to generate and store test paths directly onto the provided CFG object

    Args:
        cfg: The Control Flow Graph object to process
        app_config (AppConfig): The application configuration object

    Returns:
        List[TestPath]: The list of generated test paths (also stored in
            `cfg.test_paths`)
    """
    finder = TestPathsFinder(cfg, app_config=app_config)
    finder.find_test_paths()
    return cfg.test_paths