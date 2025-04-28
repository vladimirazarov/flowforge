"""Defines node types that represent conditional and unconditional jumps in a Control Flow Graph (CFG)

Includes a base class `CFGJumpNode` and specialized subclasses for various control flow
constructs like if/else, switch, and different types of loops
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional 
from src.core.cfg_content import Expression

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

@dataclass()
class CFGJumpNode:
    """Base class for nodes representing control flow jumps

    Attributes:
        expression (Optional[Expression]): The condition associated with the jump,
                                            if it is conditional
    """
    expression: Optional[Expression] = None

    def __str__(self) -> str:
        return self.__class__.__name__[3:-4] if self.__class__.__name__.endswith('Jump') else self.__class__.__name__


@dataclass ()
class CFGIfJump(CFGJumpNode):
    """Represents a conditional jump for an if statement"""
    def __str__(self) -> str:
        return "If"


@dataclass
class CFGSwitchJump(CFGJumpNode):
    """Represents a multi-way jump for a switch statement"""
    def __str__(self) -> str:
        return "Switch"


@dataclass
class CFGWhileLoopJump(CFGJumpNode):
    """Represents the conditional jump controlling a while loop"""
    def __str__(self) -> str:
        return "While"


@dataclass
class CFGForLoopJump(CFGJumpNode):
    """Represents the conditional jump controlling a for loop"""
    def __str__(self) -> str:
        return "For"


@dataclass
class CFGForLoopBreakJump(CFGJumpNode):
    """Represents the jump associated with a potential break from a for loop"""
    def __str__(self) -> str:
        return "For Break"
