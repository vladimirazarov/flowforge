"""
Defines the Variable class and related enumerations for CFG analysis

Includes:
- Variable class: Represents program variables with name, type, state, value,
  and usage tracking. Supports conversion to Z3 and C representations
- VariableType enum: Supported variable types (INT, BOOL)
- VariableState enum: Lifecycle states (DECLARED, INITIALIZED, MODIFIED)
"""
from __future__ import annotations
from enum import Enum, unique
from typing import Any, Optional, Union, Dict
from dataclasses import dataclass, field

import z3
from z3.z3 import Context

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

@unique
class VariableState(Enum):
    """
    Possible states for a variable in the program
    """
    DECLARED = "declared"
    INITIALIZED = "initialized"
    MODIFIED = "modified"


@unique
class VariableType(Enum):
    """
    Supported variable types with corresponding Z3 sorts
    """
    INT = "int"
    BOOL = "bool"

    def to_z3_sort(self):
        """
        Return the corresponding Z3 sort for this type
        """
        sort_map = {
            VariableType.INT: z3.IntSort(),
            VariableType.BOOL: z3.BoolSort()
        }
        return sort_map[self]


@dataclass
class Variable:
    """
    Represents a program variable, optionally associated with a value/state for analysis

    Attributes:
        name: Name of the variable
        type: VariableType indicating int or bool
        state: Current VariableState
        value: Current value if any
        uses: Number of times this variable is referenced
        modifications: Number of times this variable is modified
        scope_level: Nesting level or block in which the variable is declared
    """
    name: str
    type: VariableType = VariableType.INT
    state: Optional[VariableState] = None
    value: Optional[Union[int, bool]] = None
    uses: int = field(default=0, repr=False)
    modifications: int = field(default=0, repr=False)
    scope_level: int = field(default=0, repr=False)

    _z3_var: Optional[z3.ExprRef] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Ensure the name is valid and is a string"""
        if not self.name:
            raise ValueError("Variable name cannot be empty")
        if not isinstance(self.name, str):
            raise TypeError("Variable name must be a string")

    def declare(self) -> None:
        """
        Mark this variable as declared if it has no current state
        """
        if self.state is None:
            self.state = VariableState.DECLARED

    def initialize(self, value: Any) -> None:
        """
        Initialize the variable with a given value

        Args:
            value: The initial value
        """
        self.value = value
        self.state = VariableState.INITIALIZED

    def modify(self, value: Any) -> None:
        """
        Modify the variable with a new value

        Args:
            value: The new value
        """
        self.value = value
        self.state = VariableState.MODIFIED
        self.modifications += 1

    def use(self) -> None:
        """Increment usage count by one"""
        self.uses += 1

    def is_initialized(self) -> bool:
        """
        Check if the variable is initialized or modified

        Returns:
            True if the variable has been initialized or modified
        """
        return self.state in (VariableState.INITIALIZED, VariableState.MODIFIED)

    def to_z3(self, var_map: Optional[Dict[str, z3.ExprRef]] = None,
              context: Optional[Context] = None, *, _suffix: str = "") -> z3.ExprRef:
        """
        Create a Z3 variable with an optional context and suffix
        Note: This currently creates a fresh Z3 variable each time called

        Args:
            var_map: Not used in this implementation
            context: Optional Z3 context
            _suffix: Optional suffix to append to the Z3 variable name

        Returns:
            Z3 expression for this variable
        """
        z3_name = f"{self.name}{_suffix}"
        if self.type == VariableType.INT:
            return z3.Int(z3_name, ctx=context)
        elif self.type == VariableType.BOOL:
            return z3.Bool(z3_name, ctx=context)
        return z3.Int(z3_name, ctx=context)

    def to_c(self) -> str:
        """
        Convert this variable to its C representation

        Returns:
            The variable name
        """
        return self.name

    def __str__(self) -> str:
        """String representation is the variable name"""
        return self.name

    def __hash__(self) -> int:
        """Hash based on variable name and type"""
        return hash((self.name, self.type))

    def __eq__(self, other: Any) -> bool:
        """Check equality by matching name and type"""
        if not isinstance(other, Variable):
            return False
        return (self.name == other.name and self.type == other.type)