"""
Defines strongly-typed enumerations for operators used in expressions

Includes:
- ArithmeticOperator (+, -, *, /, %, =)
- ComparisonOperator (>, >=, <, <=, ==, !=)
- LogicalOperator (&&, ||, !)
- UnaryOperator (++, --, -, !)

Provides mapping to Z3 functions and negation logic where applicable
"""

from __future__ import annotations
from enum import Enum, unique

import z3

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"


class Operator(Enum):
    """
    Base class for all operators, enforcing unique enum values
    and providing common string representations
    """
    def __str__(self) -> str:
        """Return the operator's symbol for code generation"""
        return self.value


@unique
class ArithmeticOperator(Operator):
    """
    Arithmetic operators for numeric operations

    Supported:
      - ASSIGN (=)
      - ADD (+)
      - SUB (-)
      - MUL (*)
      - DIV (/)
      - MOD (%)
    """
    ASSIGN = "="
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"

    def to_z3(self):
        """
        Return a function that takes two Z3 expressions and applies
        the corresponding arithmetic operation
        """
        op_map = {
            ArithmeticOperator.ADD: lambda x, y: x + y,
            ArithmeticOperator.SUB: lambda x, y: x - y,
            ArithmeticOperator.MUL: lambda x, y: x * y,
            ArithmeticOperator.DIV: lambda x, y: x / y,
            ArithmeticOperator.MOD: lambda x, y: x % y,
            ArithmeticOperator.ASSIGN: lambda x, y: y
        }
        return op_map[self]


@unique
class ComparisonOperator(Operator):
    """
    Comparison operators for conditional expressions

    Supported:
      - GREATER (>)
      - GREATER_EQUAL (>=)
      - LESS (<)
      - LESS_EQUAL (<=)
      - EQUAL (==)
      - NOT_EQUAL (!=)
    """
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="

    def to_z3(self):
        """
        Return a function that takes two Z3 expressions and applies
        the corresponding comparison operation
        """
        op_map = {
            ComparisonOperator.GREATER: lambda x, y: x > y,
            ComparisonOperator.GREATER_EQUAL: lambda x, y: x >= y,
            ComparisonOperator.LESS: lambda x, y: x < y,
            ComparisonOperator.LESS_EQUAL: lambda x, y: x <= y,
            ComparisonOperator.EQUAL: lambda x, y: x == y,
            ComparisonOperator.NOT_EQUAL: lambda x, y: x != y
        }
        return op_map[self]

    def negate(self) -> ComparisonOperator:
        """
        Return the logical negation of this operator, e.g. > becomes <=
        """
        negation_map = {
            ComparisonOperator.GREATER: ComparisonOperator.LESS_EQUAL,
            ComparisonOperator.GREATER_EQUAL: ComparisonOperator.LESS,
            ComparisonOperator.LESS: ComparisonOperator.GREATER_EQUAL,
            ComparisonOperator.LESS_EQUAL: ComparisonOperator.GREATER,
            ComparisonOperator.EQUAL: ComparisonOperator.NOT_EQUAL,
            ComparisonOperator.NOT_EQUAL: ComparisonOperator.EQUAL
        }
        return negation_map[self]


@unique
class LogicalOperator(Operator):
    """
    Logical operators for boolean operations:
      - AND (&&)
      - OR (||)
      - NOT (!)
    """
    AND = "&&"
    OR = "||"
    NOT = "!"

    def to_z3(self):
        """
        Return a function that applies the corresponding logical operation
        in Z3
        """
        op_map = {
            LogicalOperator.AND: z3.And,
            LogicalOperator.OR: z3.Or,
            LogicalOperator.NOT: z3.Not
        }
        return op_map[self]


@unique
class UnaryOperator(Operator):
    """
    Unary operators for single-operand operations:
      - INCREMENT (++)
      - DECREMENT (--)
      - NEGATE (-)
      - NOT (!)
    """
    INCREMENT = "++"
    DECREMENT = "--"
    NEGATE = "-"
    NOT = "!"
