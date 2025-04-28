"""
Path module with comprehensive path types for CFG utility

Provides Path, PrimePath, and SimplePath classes for representing sequences
of nodes in a Control Flow Graph
"""
from __future__ import annotations
from typing import List, Iterator
from typing import Collection
from dataclasses import dataclass, field
from abc import ABC
from z3.z3 import Context

from src.core.nodes import CFGBasicBlockNode

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

Z3_SHARED_CONTEXT = Context()
@dataclass
class Path(ABC):
    """
    Represent a base class for paths in a Control Flow Graph (CFG)

    Paths are ordered sequences of nodes. This abstract base class defines
    common attributes and methods for different path types

    Attributes:
        nodes: A list of CFGBasicBlockNode objects representing the path
    """
    nodes: List[CFGBasicBlockNode] = field(default_factory=list)
    
    def __len__(self) -> int:
        """Return the number of nodes in the path"""
        return len(self.nodes)
        
    def __eq__(self, other: object) -> bool:
        """Compare this path to another object for equality

        Two paths are considered equal if they are both Path instances and
        contain the exact same sequence of node IDs

        Args:
            other (object): The object to compare against

        Returns:
            bool: True if the paths are equal, False otherwise
        """
        if not isinstance(other, Path):
            return False
        
        if len(self) != len(other):
            return False
            
        return all(n1.node_id == n2.node_id for n1, n2 in zip(self.nodes, other.nodes))
    
    def __hash__(self) -> int:
        """Generate a hash value for the path

        The hash is based on the tuple of node IDs in the path sequence,
        making paths with the same nodes in the same order hash identically

        Returns:
            int: The hash value of the path
        """
        return hash(tuple(node.node_id for node in self.nodes))
    
    def __iter__(self) -> Iterator[CFGBasicBlockNode]:
        """Return an iterator over the nodes in the path"""
        for node in self.nodes:
            yield node
            
    def append(self, node: CFGBasicBlockNode) -> None:
        """Append a single node to the end of the path

        Args:
            node (CFGBasicBlockNode): The node to add
        """
        self.nodes.append(node)
    
    def extend(self, nodes: List[CFGBasicBlockNode]) -> None:
        """Extend the path by appending nodes from a list

        Args:
            nodes (List[CFGBasicBlockNode]): A list of nodes to append
        """
        self.nodes.extend(nodes)
    
    def get_nodes(self) -> List[CFGBasicBlockNode]:
        """Return the list of nodes constituting the path"""
        return self.nodes
    
    def is_subpath_of(self, other: Path) -> bool:
        """Check if this path is a subpath of another path

        Compares the sequence of node IDs in this path against all possible
        contiguous subsequences of the same length in the other path

        Args:
            other (Path): The potential superpath to check against

        Returns:
            bool: True if this path is a subpath of the other, False otherwise
        """
        if len(self) > len(other):
            return False
            
        self_ids = [node.node_id for node in self.nodes]
        other_ids = [node.node_id for node in other.nodes]
        
        for i in range(len(other_ids) - len(self_ids) + 1):
            if self_ids == other_ids[i:i+len(self_ids)]:
                return True
                
        return False

@dataclass(eq=False)
class PrimePath(Path):
    """
    Represent a prime path in the CFG

    A prime path is a simple path that is not a proper subpath of any other
    simple path within the graph. It represents a maximal simple sequence
    of control flow
    """
    def is_prime(self, all_paths: Collection[Path]) -> bool:
        """Verify if this path is prime relative to a collection of paths

        Checks if this path is a proper subpath of any other path in the
        provided collection. Assumes this path itself is simple

        Args:
            all_paths (Collection[Path]): A collection of paths (typically
                simple paths) from the same graph to compare against

        Returns:
            bool: True if this path is not a proper subpath of any other path
                in the collection, False otherwise
        """
        for other in all_paths:
            if other != self and self.is_subpath_of(other):
                return False
        return True



@dataclass 
class SimplePath(Path):
    """
    Represent a simple path in the CFG

    A simple path contains no repeated nodes, with the possible exception
    that the first and last nodes may be identical (forming a cycle)
    """
    def is_simple(self) -> bool:
        """Check if the path adheres to the definition of a simple path

        Verifies that no node appears more than once in the path, unless
        it is the start and end node being the same

        Returns:
            bool: True if the path is simple, False otherwise
        """
        if len(self.nodes) <= 1:
            return True
            
        # Check uniqueness across all nodes, allowing first==last
        all_node_ids = [node.node_id for node in self.nodes]
        if len(all_node_ids) == len(set(all_node_ids)):
            return True
        
        # If first != last and duplicates exist, it's not simple
        if self.nodes[0].node_id == self.nodes[-1].node_id:
            # Check if the middle part is unique among itself AND doesn't contain the end node ID
            middle_nodes = self.nodes[1:-1]
            middle_ids = [node.node_id for node in middle_nodes]
            end_node_id = self.nodes[0].node_id
            return len(middle_ids) == len(set(middle_ids)) and end_node_id not in middle_ids
        
        return False
