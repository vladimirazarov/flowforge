"""Defines the CFGBasicBlockNode, the core block of the CFG

A `CFGBasicBlockNode` represents a basic block, a sequence of instructions
ending in a single control flow transfer (jump). It holds instructions,
the jump condition (if any), and metadata/analysis results

See Also:
    `CFGInstructionsNode`: Holds the linear sequence of operations
    `CFGJumpNode`: Represents the control flow exit from the block
"""

from __future__ import annotations
import itertools
import networkx as nx
from typing import Optional, Set, TYPE_CHECKING, cast
from dataclasses import dataclass, field

from src.core.nodes.cfg_instructions_node import CFGInstructionsNode
from src.core.nodes.cfg_jump_node import (
    CFGJumpNode, CFGIfJump, CFGSwitchJump, CFGWhileLoopJump, CFGForLoopJump
)
from src.config.config import config
from src.core.nodes.must_be import MustBeSet
from src.core.cfg_content.variable import Variable

if TYPE_CHECKING:
    from src.core.cfg.cfg import CFG
    from src.core.nodes.must_be import MustBeSet

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"


_id_counter = itertools.count()

@dataclass()
class CFGBasicBlockNode:
    """Represents a basic block node within a Control Flow Graph

    Contains instructions, an optional jump condition, and CFG metadata

    Attributes:
        cfg: Reference to the parent CFG
        instructions: Node containing the sequence of operations
        jump: Node representing the control flow exit (e.g., If, While, Switch jump)
        origin_fragment_id: ID of the Fragment this node originated from
        depth: Depth of the node in the CFG (from entry)
        position: Position metric (e.g., shortest path length from entry)
        is_entry: True if this is the CFG entry node
        is_exit: True if this is a CFG exit node
        is_loop_exit: True if this node is the designated exit path from a loop fragment
        used_variables: Set of variables used within this block
        must_be_set: Constraints that must hold true at this node
        _shtv: Cached SHTV calculation result
        max_shtv: Maximum SHTV ceiling assigned to this node
        _id: Unique identifier for the node instance
        arithmetic_operations_count: Count of arithmetic ops in the block
        logical_operations_count: Count of logical ops in the block
        _dominators: Cached set of dominator nodes
    """

    cfg: Optional[CFG] = None
    instructions: Optional[CFGInstructionsNode] = None
    jump: Optional[CFGJumpNode] = None
    origin_fragment_id: Optional[int] = None

    depth: int = field(default=0) 
    position: int = 0
    _dominators: Optional[Set[CFGBasicBlockNode]] = field(default=None, init=False)
    is_entry: bool = False
    is_exit: bool = False
    is_loop_exit: bool = False
    used_variables: Set[Variable] = field(default_factory=set)
    must_be_set: Optional[MustBeSet] = None
    _shtv: float = 0
    max_shtv: float = 0
    _id: int = field(init = False)
    arithmetic_operations_count: int = 0
    logical_operations_count: int = 0


    def __post_init__(self):
        self._id = next(_id_counter)
        if self.instructions is None:
            self.instructions = CFGInstructionsNode()
        
    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, other) -> bool:
        """Compares nodes based on their unique ID and type."""
        return type(self) is type(other) and self._id == cast(CFGBasicBlockNode, other)._id

    @property
    def node_id(self) -> int:
        """Returns the unique identifier of this node"""
        return self._id

    @property
    def dominators(self) -> Set[CFGBasicBlockNode]:
        """Returns the set of nodes that dominate this node

        Dominators are computed on first access and cached

        Returns:
            Set[CFGBasicBlockNode]: The set of dominator nodes
        """
        if self._dominators is None:
            assert self.cfg is not None
            assert self.cfg.entry_node is not None

            immediate_doms = nx.immediate_dominators(self.cfg.graph, self.cfg.entry_node)

            all_dominators: Set[CFGBasicBlockNode] = set()
            current = self

            while current in immediate_doms:
                immediate_dominator = immediate_doms[current]
                all_dominators.add(immediate_dominator)
                current = immediate_dominator

            self._dominators = all_dominators
        return self._dominators

    @property
    def shtv(self) -> float:
        """Calculates and returns the SHTV (Shortest History Test Vector) value for this node

        Uses configured weights and node properties (loops, conditions, ops,
        variables, depth, position) to compute the SHTV score

        Returns:
            float: The calculated SHTV value for this node
        """
        weights = config.weights
        has_loop = isinstance(self.jump, (CFGWhileLoopJump, CFGForLoopJump))
        has_condition = isinstance(self.jump, (CFGIfJump, CFGSwitchJump))
        logical_ops = self.logical_operations_count
        arithmetic_ops = self.arithmetic_operations_count
        variables_used = len(self.used_variables)
        depth = 0 if self.depth is None else self.depth
        position = 0 if self.position is None else self.position

        self._shtv = (
                weights['a'] * has_condition +
                weights['b'] * has_loop +
                weights['c'] * logical_ops +
                weights['d'] * arithmetic_ops +
                weights['e'] * depth +
                weights['f'] * variables_used +
                weights['k'] * position
        )
        return self._shtv

    def __repr__(self) -> str:
        if self.jump is None:
            return str(self.instructions) + "\nDepth: " + str(self.depth) + "\nFragment: " + str(self.origin_fragment_id)
        else:
            return str(self.jump) + "\nDepth: " + str(self.depth) + "\nFragment: " + str(self.origin_fragment_id)

    def __str__(self) -> str:
        if self.jump is None:
            return str(self.instructions) + " " + str(self.depth) 
        else:
            return str(self.jump) + " " + str(self.depth) 
  

    @property
    def code(self) -> str:
        if self.jump is None:
            instruction_code = ""
            if self.instructions and hasattr(self.instructions, "operations"):
                for op in self.instructions.operations:
                    instruction_code += str(op) + ";\n"
            return instruction_code
        else:
            return f""
            
    def cleanup(self) -> None:
        """Clears internal references to aid garbage collection

        Nullifies references to jump node, instruction operations, variable sets,
        constraint sets, and cached dominators
        """
        if self.instructions and hasattr(self.instructions, "operations"):
            self.instructions.operations = []
        
        self.used_variables.clear()
        self.jump = None
        self.instructions = None
        self.must_be_set = None
        self._dominators = None