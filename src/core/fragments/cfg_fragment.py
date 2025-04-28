"""
Defines CFG Fragment classes representing common control flow structures

Includes a base `Fragment` class and specific subclasses for structures
like If/Else, While/For loops, and Switch statements. Each fragment holds
a minimal sub-CFG representing its structure

A factory pattern using `Fragment.__new__` allows creation of the correct
subclass based on the provided `FragmentRole`
"""

from __future__ import annotations
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from loguru import logger

from src.core.nodes.cfg_instructions_node import (
    CFGInstructionsNode, CFGTrueBranch, CFGFalseBranch,
    CFGWhileLoopBody, CFGForLoopBody
)
from src.core.nodes.cfg_jump_node import (
    CFGIfJump, CFGWhileLoopJump, CFGForLoopJump,
    CFGForLoopBreakJump, CFGSwitchJump
)
from src.core.nodes.cfg_basic_block_node import CFGBasicBlockNode
from src.core.fragments.fragment_types import FragmentRole

if TYPE_CHECKING:
    from src.core.cfg.cfg import CFG


__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"


_next_fragment_id = itertools.count()

def _create_cfg_instance() -> CFG:
    """
    Create a new CFG instance at runtime to avoid circular imports
    """
    from src.core.cfg.cfg import CFG
    return CFG()

@dataclass(eq=False)
class Fragment(ABC):
    """
    Base class for all CFG fragments. A fragment contains:
      - A sub-CFG holding the minimal structure for a given control construct
      - A unique fragment_id assigned globally
      - A fragment_role (e.g., IF, WHILE, etc.)
      - Optional entry_node and exit_node references in its sub-CFG
    """
    fragment_role: FragmentRole
    fragment_id: int = field(init=False)
    parent_fragment_id: Optional[int] = None
    sub_cfg: Optional[CFG] = field(default_factory=_create_cfg_instance, init=False)
    entry_node: Optional[CFGBasicBlockNode] = None
    exit_node: Optional[CFGBasicBlockNode] = None
    is_nested: bool = False

    @property
    def cc(self) -> int:
        """
        Return the cyclomatic complexity of this fragment's sub-CFG
        """
        return self.sub_cfg.cc if self.sub_cfg else 0

    @property
    def related_nodes(self) -> list[CFGBasicBlockNode]:
        """
        Return main CFG nodes this fragment depends on (entry, exit, etc.)
        Filters out None values if nodes were deleted or not assigned
        """
        nodes = []
        if self.entry_node is not None:
            nodes.append(self.entry_node)
        if self.exit_node is not None:
            nodes.append(self.exit_node)
        return [n for n in nodes if n is not None]


    def __hash__(self) -> int:
        """
        Make the fragment hashable by its unique ID
        """
        return hash(self.fragment_id)

    def __eq__(self, other) -> bool:
        """
        Consider fragments equal if they share the same fragment_id
        """
        if not isinstance(other, Fragment):
            return False
        return self.fragment_id == other.fragment_id

    def __new__(cls, fragment_role: FragmentRole, **kwargs):
        """
        Factory method to create appropriate subclass based on role
        """
        if cls is not Fragment:
            instance = super().__new__(cls)
            return instance

        fragment_map = {
            FragmentRole.IF: IfFragment,
            FragmentRole.IF_ELSE: IfElseFragment,
            FragmentRole.WHILE: WhileDoFragment,
            FragmentRole.FOR: ForLoopFragment,
            FragmentRole.FOR_EARLY_EXIT: ForLoopEarlyExitFragment,
            FragmentRole.SWITCH: SwitchFragment
        }
        fragment_class = fragment_map.get(fragment_role)
        if fragment_class is None:
            raise ValueError(f"Unknown fragment role: {fragment_role}")

        # Instantiate the correct subclass.
        if fragment_class is SwitchFragment:
            num_of_cases = kwargs.get('num_of_cases', 3)
            instance = super().__new__(fragment_class) # type: ignore[arg-type]
            instance.num_of_cases = num_of_cases # type: ignore
        else:
            instance = super().__new__(fragment_class)

        return instance


    @classmethod
    def get_next_id(cls) -> int:
        """
        Return the next unique fragment ID from a global counter
        """
        new_id = next(_next_fragment_id)
        logger.info(f"[Fragment] Generated new fragment ID: {new_id}")
        return new_id

    @abstractmethod
    def wire_edges(self, main_cfg: CFG, decision_node: CFGBasicBlockNode) -> None:
        pass

    def introduce_into(self, cfg: CFG):
        """
        Merge this fragment's sub-CFG nodes and edges into the main CFG.
        Assumes the sub-CFG exists.
        """
        for node_obj in self.sub_cfg: # type: ignore
            cfg.add_node(node_obj)

        for (u, v) in self.sub_cfg.get_edges(): # type: ignore
            cfg.add_edge(u, v)


@dataclass(eq=False)
class IfFragment(Fragment):
    """
    Represents an if statement without an else branch. Sub-CFG:
    start(if_jump) -> true_branch -> end
                 `-> end
    """

    def __post_init__(self):
        self.fragment_id = Fragment.get_next_id()
        self.fragment_role = FragmentRole.IF

        start = CFGBasicBlockNode(
            instructions=CFGInstructionsNode(),
            jump=CFGIfJump(),
            origin_fragment_id=self.fragment_id
        )
        true_branch = CFGBasicBlockNode(
            instructions=CFGTrueBranch(),
            depth=1,
            origin_fragment_id=self.fragment_id
        )
        end = CFGBasicBlockNode(
            instructions=CFGInstructionsNode(),
            origin_fragment_id=self.fragment_id
        )

        self.sub_cfg.add_node(start)
        self.sub_cfg.add_node(true_branch)
        self.sub_cfg.add_node(end)
        self.sub_cfg.add_edge(start, true_branch)
        self.sub_cfg.add_edge(start, end)
        self.sub_cfg.add_edge(true_branch, end)

        self.entry_node = start
        self.exit_node = end
        self._related_nodes = [start, true_branch, end]

        logger.info(f"[IfFragment] Created fragment {self.fragment_id} (Role: {self.fragment_role}) with entry node {self.entry_node.node_id} and exit node {self.exit_node.node_id}")

    @property
    def related_nodes(self) -> list[CFGBasicBlockNode]:
        """Return all nodes specific to this fragment type, filtering out Nones"""
        return [n for n in getattr(self, '_related_nodes', []) if n is not None]

    def wire_edges(self, main_cfg: CFG, decision_node: CFGBasicBlockNode) -> None:
        children = main_cfg.get_node_children(decision_node)
        for child in children:
            if isinstance(child.instructions, CFGTrueBranch):
                main_cfg.set_edge_label(decision_node, child, "true")
            else:
                main_cfg.set_edge_label(decision_node, child, "false")


@dataclass(eq=False)
class IfElseFragment(Fragment):
    """
    Represents an if statement with an else branch. Sub-CFG:
    start(if_jump) -> true_branch -> end
                 `-> false_branch -> end
    """

    def __post_init__(self):
        self.fragment_id = Fragment.get_next_id()
        self.fragment_role = FragmentRole.IF_ELSE

        start = CFGBasicBlockNode(
            instructions=CFGInstructionsNode(),
            jump=CFGIfJump(),
            origin_fragment_id=self.fragment_id
        )
        true_branch = CFGBasicBlockNode(
            instructions=CFGTrueBranch(),
            depth=1,
            origin_fragment_id=self.fragment_id
        )
        false_branch = CFGBasicBlockNode(
            instructions=CFGFalseBranch(),
            depth=1,
            origin_fragment_id=self.fragment_id
        )
        end = CFGBasicBlockNode(
            instructions=CFGInstructionsNode(),
            origin_fragment_id=self.fragment_id
        )

        self.sub_cfg.add_node(start)
        self.sub_cfg.add_node(true_branch)
        self.sub_cfg.add_node(false_branch)
        self.sub_cfg.add_node(end)

        self.sub_cfg.add_edge(start, true_branch)
        self.sub_cfg.add_edge(start, false_branch)
        self.sub_cfg.add_edge(true_branch, end)
        self.sub_cfg.add_edge(false_branch, end)

        self.entry_node = start
        self.exit_node = end
        self._related_nodes = [start, true_branch, false_branch, end]

        logger.info(f"[IfElseFragment] Created fragment {self.fragment_id} (Role: {self.fragment_role}) with entry node {self.entry_node.node_id} and exit node {self.exit_node.node_id}")

    @property
    def related_nodes(self) -> list[CFGBasicBlockNode]:
        """Return all nodes specific to this fragment type, filtering out Nones"""
        return [n for n in getattr(self, '_related_nodes', []) if n is not None]

    def wire_edges(self, main_cfg: CFG, decision_node: CFGBasicBlockNode) -> None:
        children = main_cfg.get_node_children(decision_node)
        for child in children:
            if isinstance(child.instructions, CFGTrueBranch):
                label = "true"
            elif isinstance(child.instructions, CFGFalseBranch):
                label = "false"
            else:
                continue
            main_cfg.set_edge_label(decision_node, child, label)


@dataclass(eq=False)
class WhileDoFragment(Fragment):
    """
    Represents a while loop construct. Sub-CFG:
    start(while_jump) -> body -> start (loop back)
                    `-> end (exit loop)
    """
    body_node: Optional[CFGBasicBlockNode] = field(init=False, default=None)

    def __post_init__(self):
        self.fragment_id = Fragment.get_next_id()
        self.fragment_role = FragmentRole.WHILE

        start = CFGBasicBlockNode(
            instructions=CFGInstructionsNode(),
            jump=CFGWhileLoopJump(),
            origin_fragment_id=self.fragment_id
        )
        body = CFGBasicBlockNode(
            instructions=CFGWhileLoopBody(),
            depth=1,
            origin_fragment_id=self.fragment_id
        )
        end = CFGBasicBlockNode(
            instructions=CFGInstructionsNode(),
            origin_fragment_id=self.fragment_id
        )

        self.sub_cfg.add_node(start)
        self.sub_cfg.add_node(body)
        self.sub_cfg.add_node(end)

        self.sub_cfg.add_edge(start, body)
        self.sub_cfg.add_edge(body, start)
        self.sub_cfg.add_edge(start, end)

        self.entry_node = start
        self.exit_node = end
        self.body_node = body
        self._related_nodes = [start, body, end]
        self.exit_node.is_loop_exit = True

        logger.info(f"[WhileDoFragment] Created fragment {self.fragment_id} (Role: {self.fragment_role}) with entry node {self.entry_node.node_id}, body node {body.node_id}, and exit node {self.exit_node.node_id}")

    @property
    def related_nodes(self) -> list[CFGBasicBlockNode]:
        """Return all nodes specific to this fragment type, filtering out Nones"""
        return [n for n in getattr(self, '_related_nodes', []) if n is not None]

    def wire_edges(self, main_cfg: CFG, decision_node: CFGBasicBlockNode) -> None:
        children = main_cfg.get_node_children(decision_node)
        for child in children:
            if isinstance(child.instructions, CFGWhileLoopBody):
                label = "loop"
            else:
                label = "exit"
            main_cfg.set_edge_label(decision_node, child, label)


@dataclass(eq=False)
class ForLoopEarlyExitFragment(Fragment):
    """
    Represents a for loop with a possible early exit (break) point. Sub-CFG:
    start(for_jump) -> body -> possible_break(break_jump) -> start (continue)
                  `-> end (loop finish)                 `-> end (break)
    """
    body_node: Optional[CFGBasicBlockNode] = field(init=False, default=None)
    break_node: Optional[CFGBasicBlockNode] = field(init=False, default=None)

    def __post_init__(self):
        self.fragment_id = Fragment.get_next_id()
        self.fragment_role = FragmentRole.FOR_EARLY_EXIT

        # Loop initialization/condition check
        start = CFGBasicBlockNode(
            instructions=CFGInstructionsNode(), 
            jump=CFGForLoopJump(),
            origin_fragment_id=self.fragment_id
        )
        # Main loop body
        body = CFGBasicBlockNode(
            instructions=CFGForLoopBody(),     
            depth=1,
            origin_fragment_id=self.fragment_id
        )
        # Loop increment/update
        # Conditional break jump
        possible_break = CFGBasicBlockNode(
            instructions=CFGInstructionsNode(), 
            jump=CFGForLoopBreakJump(),      
            depth=1,
            origin_fragment_id=self.fragment_id
        )
        # Code after loop
        end = CFGBasicBlockNode(
            instructions=CFGInstructionsNode(), 
            origin_fragment_id=self.fragment_id
        )

        self.sub_cfg.add_node(start)
        self.sub_cfg.add_node(body)
        self.sub_cfg.add_node(possible_break)
        self.sub_cfg.add_node(end)

        self.sub_cfg.add_edge(start, body)
        self.sub_cfg.add_edge(body, possible_break)
        self.sub_cfg.add_edge(possible_break, start)
        self.sub_cfg.add_edge(possible_break, end)
        self.sub_cfg.add_edge(start, end)

        self.entry_node = start
        self.exit_node = end
        self.body_node = body
        self.break_node = possible_break
        self._related_nodes = [start, body, possible_break, end]
        self.exit_node.is_loop_exit = True

        logger.info(f"[ForLoopEarlyExitFragment] Created fragment {self.fragment_id} (Role: {self.fragment_role}) with entry node {self.entry_node.node_id}, body node {body.node_id}, break node {possible_break.node_id}, and exit node {self.exit_node.node_id}")

    @property
    def related_nodes(self) -> list[CFGBasicBlockNode]:
        """Return all nodes specific to this fragment type, filtering out Nones"""
        return [n for n in getattr(self, '_related_nodes', []) if n is not None]

    def wire_edges(self, main_cfg: CFG, decision_node: CFGBasicBlockNode) -> None:
        children = main_cfg.get_node_children(decision_node)
        for child in children:
            if isinstance(child.instructions, CFGForLoopBody):
                label = "loop"
            elif isinstance(child.jump, CFGForLoopBreakJump):
                label = "break"
            else:
                label = "exit"
            main_cfg.set_edge_label(decision_node, child, label)


@dataclass(eq=False)
class ForLoopFragment(Fragment):
    """
    Represents a standard for loop construct (no explicit break node). Sub-CFG:
    start(for_jump) -> body -> start (loop back)
                  `-> end (exit loop)
    """
    body_node: Optional[CFGBasicBlockNode] = field(init=False, default=None)

    def __post_init__(self):
        self.fragment_id = Fragment.get_next_id()
        self.fragment_role = FragmentRole.FOR

        # Init/condition
        start = CFGBasicBlockNode(
            instructions=CFGInstructionsNode(), 
            jump=CFGForLoopJump(),
            origin_fragment_id=self.fragment_id
        )
        # Loop body / increment
        body = CFGBasicBlockNode(
            instructions=CFGForLoopBody(),      
            depth=1,
            origin_fragment_id=self.fragment_id
        )
        # Code after loop
        end = CFGBasicBlockNode(
            instructions=CFGInstructionsNode(), 
            origin_fragment_id=self.fragment_id
        )

        self.sub_cfg.add_node(start)
        self.sub_cfg.add_node(body)
        self.sub_cfg.add_node(end)

        self.sub_cfg.add_edge(start, body)
        self.sub_cfg.add_edge(body, start)
        self.sub_cfg.add_edge(start, end)

        self.entry_node = start
        self.exit_node = end
        self.body_node = body
        self._related_nodes = [start, body, end]
        self.exit_node.is_loop_exit = True

        logger.info(f"[ForLoopFragment] Created fragment {self.fragment_id} (Role: {self.fragment_role}) with entry node {self.entry_node.node_id}, body node {body.node_id}, and exit node {self.exit_node.node_id}")

    @property
    def related_nodes(self) -> list[CFGBasicBlockNode]:
        """Return all nodes specific to this fragment type, filtering out Nones"""
        return [n for n in getattr(self, '_related_nodes', []) if n is not None]

    def wire_edges(self, main_cfg: CFG, decision_node: CFGBasicBlockNode) -> None:
        children = main_cfg.get_node_children(decision_node)
        for child in children:
            if isinstance(child.instructions, CFGForLoopBody):
                label = "loop"
            else:
                label = "exit"
            main_cfg.set_edge_label(decision_node, child, label)


@dataclass(eq=False)
class SwitchFragment(Fragment):
    """
    Represents a switch-case construct. Sub-CFG:
    start(switch_jump) -> case_block_1 -> end
                       -> case_block_2 -> end
                       ...
                       -> case_block_n -> end
                       -> default_block -> end (implicit edge to end)
    """
    num_of_cases: int = field(default=3)
    case_nodes: list[CFGBasicBlockNode] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.fragment_id = Fragment.get_next_id()
        self.fragment_role = FragmentRole.SWITCH
        # Switch expression
        start = CFGBasicBlockNode(
            instructions=CFGInstructionsNode(), 
            jump=CFGSwitchJump(),
            origin_fragment_id=self.fragment_id
        )
        # Merge point
        end = CFGBasicBlockNode(
            instructions=CFGInstructionsNode(), 
            origin_fragment_id=self.fragment_id
        )

        self.sub_cfg.add_node(start)
        self.sub_cfg.add_node(end)

        self.case_nodes = []
        # Case body
        for i in range(self.num_of_cases):
            case_block = CFGBasicBlockNode(
                instructions=CFGInstructionsNode(), 
                depth=1,
                origin_fragment_id=self.fragment_id
            )
            self.sub_cfg.add_node(case_block)
            self.sub_cfg.add_edge(start, case_block)
            self.sub_cfg.add_edge(case_block, end)
            self.case_nodes.append(case_block)

        self.entry_node = start
        self.exit_node = end
        self._related_nodes = [start] + self.case_nodes + [end]

        logger.info(f"[SwitchFragment] Created fragment {self.fragment_id} (Role: {self.fragment_role}) with {self.num_of_cases} cases, entry node {self.entry_node.node_id}, and exit node {self.exit_node.node_id}")


    @property
    def related_nodes(self) -> list[CFGBasicBlockNode]:
        """Return all nodes specific to this fragment type, filtering out Nones"""
        return [n for n in getattr(self, '_related_nodes', []) if n is not None]

    def wire_edges(self, main_cfg: CFG, decision_node: CFGBasicBlockNode) -> None:
        children = main_cfg.get_node_children(decision_node)
        for idx, child in enumerate(children):
            main_cfg.set_edge_label(decision_node, child, f"case_{idx}")

