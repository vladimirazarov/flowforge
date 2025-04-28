"""Defines instruction container nodes for CFG basic blocks

Includes a base `CFGInstructionsNode` and specialized subclasses for common
control flow structures like loops and branches
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from src.core.cfg_content import Operation

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

@dataclass()
class CFGInstructionsNode:
    """Base container for a sequence of operations within a CFG node

    Attributes:
        operations (List[Operation]): The list of operations
    """
    operations: List[Operation] = field(default_factory=list)

    @property
    def arithmetic_operations_count(self) -> int:
        """Returns the total count of arithmetic operations"""
        return sum(op.arithmetic_operations_count for op in self.operations)

    @property
    def logical_operations_count(self) -> int:
        """Returns the total count of logical operations"""
        return sum(op.logical_operations_count for op in self.operations)

    def add_operation(self, operation: Operation) -> None:
        """Appends an operation to the list

        Args:
            operation (Operation): The operation to add
        """
        self.operations.append(operation)

    def remove_operation(self, operation: Operation) -> None:
        """Removes a specific operation from the list

        Args:
            operation (Operation): The operation to remove
        """
        self.operations.remove(operation)

    def generate_random_operation(self, operation_factory) -> bool:
        """Generates and adds a random operation using a factory

        Args:
            operation_factory (Any): A factory object capable of creating
                                     operations (expects create_random_operation method)

        Returns:
            bool: True if an operation was successfully created and added, False otherwise
        """
        new_operation = operation_factory.create_random_operation(node=self)
        if new_operation:
            self.add_operation(new_operation)
            return True
        return False

    def __str__(self) -> str:
        return "Instructions"

@dataclass
class CFGCaseNode(CFGInstructionsNode):
    """Instruction node specifically for a switch case body"""
    def __str__(self) -> str:
        return f"Case"

@dataclass
class CFGLoopBody(CFGInstructionsNode):
    """Instruction node specifically for a generic loop body"""
    def __str__(self) -> str:
        return f"Loop Body"

@dataclass
class CFGWhileLoopBody(CFGLoopBody):
    """Instruction node specifically for a while loop body"""
    def __str__(self) -> str:
        return f"While Body"

@dataclass
class CFGForLoopBody(CFGLoopBody):
    """Instruction node specifically for a for loop body"""
    def __str__(self) -> str:
        return f"For Body"

@dataclass
class CFGTrueBranch(CFGInstructionsNode):
    """Instruction node specifically for the true branch of a conditional"""
    def __str__(self) -> str:
        return f"True Branch"

@dataclass
class CFGFalseBranch(CFGInstructionsNode):
    """Instruction node specifically for the false branch of a conditional"""
    def __str__(self) -> str:
        return f"False Branch"

