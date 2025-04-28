"""
Defines classes representing different types of expressions in a program

Includes constants, variable references, arithmetic, comparison, and logical
expressions. Each class supports conversion to C code and Z3 representation
for constraint solving
"""
from __future__ import annotations
from typing import Optional, Dict, Union, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from loguru import logger

import z3
from z3.z3 import ExprRef, And, Or, Context
import copy

from .operator import ComparisonOperator, ArithmeticOperator, LogicalOperator
from .variable import Variable, VariableType

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

@dataclass
class Expression(ABC):
    """
    Abstract base class for all expressions with Z3 integration
    
    Each expression encapsulates a Z3 expression for constraint solving
    and provides methods for code generation
    """
    _z3_expr: Optional[ExprRef] = field(default=None, init=False)
    
    @abstractmethod
    def get_variables(self) -> Set[Variable]:
        """Get all variables used in this expression"""
        pass

    def to_z3(self, var_map: Optional[Dict[str, ExprRef]] = None, context: Optional[Context] = None, *, _suffix: str = "") -> ExprRef:
        """
        Get or create the Z3 expression representation using the provided context
        Passes the _suffix down to the creation method
        
        Args:
            var_map: Optional mapping of variable names to Z3 variables
            context: The Z3 context to use for creating expressions
            _suffix: Optional suffix for variable names in Z3
            
        Returns:
            ExprRef: Z3 expression representing this expression
        """
        if self._z3_expr is None:
            self._z3_expr = self._create_z3_expr(var_map, context, _suffix=_suffix)
        elif context is not None and self._z3_expr.ctx != context:
            logger.trace(f"Context mismatch for cached expression {self}! Regenerating.")
            self._z3_expr = self._create_z3_expr(var_map, context, _suffix=_suffix)
        return self._z3_expr
    
    @abstractmethod
    def _create_z3_expr(self, var_map: Optional[Dict[str, ExprRef]], context: Optional[Context], *, _suffix: str = "") -> ExprRef:
        """Create the Z3 expression representation using the provided context and suffix"""
        pass

    @abstractmethod
    def to_c(self) -> str:
        """Convert to C code representation"""
        pass

    @abstractmethod
    def negate(self) -> Expression:
        """Return the logical negation of this expression"""
        pass

    def clone(self) -> Expression:
        """Create a deep copy of the expression"""
        return copy.deepcopy(self)

@dataclass
class Constant(Expression):
    """
    Represents constant values in expressions
    
    Attributes:
        value: The constant value (integer or boolean)
    """
    value: Union[int, bool]
    
    def get_variables(self) -> Set[Variable]:
        """Constants have no variables"""
        return set()
        
    def _create_z3_expr(self, var_map: Optional[Dict[str, ExprRef]] = None, context: Optional[Context] = None, *, _suffix: str = "") -> ExprRef:
        """Create Z3 constant using the provided context. Suffix is ignored"""
        if isinstance(self.value, bool):
            return z3.BoolVal(self.value, ctx=context)
        elif isinstance(self.value, int):
            return z3.IntVal(self.value, ctx=context)
        else:
            raise TypeError(f"Unsupported constant type for Z3: {type(self.value)}")
            
    def to_c(self) -> str:
        """Convert to C code representation"""
        if isinstance(self.value, bool):
            # Represent boolean as 0 or 1 in C
            return str(int(self.value))
        return str(self.value)

    def negate(self) -> Expression:
        """Negate the constant. Only meaningful for booleans"""
        if isinstance(self.value, bool):
            return Constant(not self.value)
        return LogicalExpression(LogicalOperator.NOT, operands=[self])

    def __eq__(self, other) -> bool:
        if not isinstance(other, Constant):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        return hash((Constant, self.value))

@dataclass
class VariableExpression(Expression):
    """
    Represents a variable reference in an expression
    
    Attributes:
        variable: The referenced variable
    """
    variable: Variable
    
    def get_variables(self) -> Set[Variable]:
        """Return the referenced variable"""
        return {self.variable}
        
    def _create_z3_expr(self, var_map: Optional[Dict[str, ExprRef]] = None, context: Optional[Context] = None, *, _suffix: str = "") -> ExprRef:
        """
        Create Z3 variable reference using the variable's own Z3 conversion and the context

        `_suffix` is appended by the TestPath builder to create
        a fresh SSA-like instance for every node visit
        """
        return self.variable.to_z3(var_map, context=context, _suffix=_suffix)
            
    def to_c(self) -> str:
        """Convert to C code representation"""
        return self.variable.to_c()

    def negate(self) -> Expression:
        """Negate the variable expression by wrapping in Logical NOT"""
        return LogicalExpression(LogicalOperator.NOT, operands=[self])

    def __eq__(self, other) -> bool:
        if not isinstance(other, VariableExpression):
            return NotImplemented
        # Comparison relies on Variable's __eq__
        return self.variable == other.variable

    def __hash__(self) -> int:
        # Hash relies on Variable implementing __hash__ correctly
        return hash((VariableExpression, self.variable))

@dataclass
class ComparisonExpression(Expression):
    """
    Represents comparison expressions (e.g., x > 5)
    
    Attributes:
        left: Left operand expression (generalized from Variable)
        operator: Comparison operator
        right: Right operand expression (generalized from int)
    """
    left: Expression # Changed from Variable
    operator: ComparisonOperator
    right: Expression # Changed from int (value)

    def get_variables(self) -> Set[Variable]:
        """Return variables from both sides of the comparison"""
        return self.left.get_variables().union(self.right.get_variables())
        
    def _create_z3_expr(self, var_map: Optional[Dict[str, ExprRef]] = None, context: Optional[Context] = None, *, _suffix: str = "") -> ExprRef:
        """Create Z3 comparison expression using the context, passing down suffix"""
        try:
            left_z3 = self.left.to_z3(var_map, context=context, _suffix=_suffix)
            right_z3 = self.right.to_z3(var_map, context=context, _suffix=_suffix)
            
            # Ensure operands are from the correct context before comparing
            if context is not None:
                if left_z3.ctx != context:
                    logger.error(f"Context mismatch in Comparison: Left operand {self.left} context {left_z3.ctx} != expected {context}")
                    return z3.BoolVal(False, ctx=context)
                if right_z3.ctx != context:
                    logger.error(f"Context mismatch in Comparison: Right operand {self.right} context {right_z3.ctx} != expected {context}")
                    return z3.BoolVal(False, ctx=context)

            logger.trace(f"Creating Z3 comparison (ctx={context}): Left={repr(left_z3)} (ctx: {left_z3.ctx}), Op={self.operator}, Right={repr(right_z3)} (ctx: {right_z3.ctx})")
            
            op_func = self.operator.to_z3()
            result = op_func(left_z3, right_z3)
            
            if not isinstance(result, z3.BoolRef):
                logger.error(f"Z3 comparison '{self.to_c()}' did not produce BoolRef! Got {type(result)}. Inputs: L={repr(left_z3)}, R={repr(right_z3)}. Returning False.")
                return z3.BoolVal(False, ctx=context)

            logger.trace(f"  -> Z3 comparison result: {repr(result)} (ctx: {result.ctx})")
            
            return result
        except Exception as e:
            logger.error(f"Error creating Z3 for ComparisonExpression ({self.to_c()}) using context {context}: {e}", exc_info=True)
            return z3.BoolVal(False, ctx=context)
            
    def to_c(self) -> str:
        """Convert to C code representation"""
        return f"{self.left.to_c()} {self.operator.value} {self.right.to_c()}"

    def negate(self) -> Expression:
        """Negate the comparison by negating the operator"""
        negated_operator = self.operator.negate()
        return ComparisonExpression(self.left, negated_operator, self.right)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ComparisonExpression):
            return NotImplemented
        # Recursively check operands
        return (self.operator == other.operator and
                self.left == other.left and 
                self.right == other.right)

    def __hash__(self) -> int:
        # Hash based on type, operator, and operands' hashes
        return hash((ComparisonExpression, self.operator, self.left, self.right))


@dataclass
class ArithmeticExpression(Expression):
    """
    Represents arithmetic expressions (e.g., x + y, a * b + c)
    
    Attributes:
        left: Left operand expression
        operator: Arithmetic operator
        right: Right operand expression
    """
    left: Expression
    operator: ArithmeticOperator
    right: Expression
    
    def get_variables(self) -> Set[Variable]:
        """Get variables from both operands"""
        return self.left.get_variables().union(self.right.get_variables())
        
    def _create_z3_expr(self, var_map: Optional[Dict[str, z3.ExprRef]] = None, context: Optional[Context] = None, *, _suffix: str = "") -> z3.ExprRef:
        """Create Z3 arithmetic expression using the context, passing down suffix."""
        left_z3 = self.left.to_z3(var_map, context=context, _suffix=_suffix)
        right_z3 = self.right.to_z3(var_map, context=context, _suffix=_suffix)
        
        # Ensure operands are from the correct context
        if context is not None:
            if left_z3.ctx != context:
                logger.error(f"Context mismatch in Arithmetic: Left operand {self.left} context {left_z3.ctx} != expected {context}")
                return z3.IntVal(0, ctx=context)
            if right_z3.ctx != context:
                logger.error(f"Context mismatch in Arithmetic: Right operand {self.right} context {right_z3.ctx} != expected {context}")
                return z3.IntVal(0, ctx=context)

        op_func = self.operator.to_z3()
        result = op_func(left_z3, right_z3)
        logger.trace(f"Arithmetic Z3 result: {result} (ctx: {result.ctx})")
        return result
            
    def to_c(self) -> str:
        """Convert to C code representation"""
        left_c = self.left.to_c()
        right_c = self.right.to_c()

        # Add parentheses around left operand if it's an arithmetic expression
        if isinstance(self.left, (ArithmeticExpression, LogicalExpression)):
             left_c = f"({left_c})"

        # Add parentheses around right operand if it's an arithmetic expression
        # or if it's a negative constant being subtracted or added
        if isinstance(self.right, (ArithmeticExpression, LogicalExpression)):
             right_c = f"({right_c})"
        elif isinstance(self.right, Constant) and isinstance(self.right.value, int) and self.right.value < 0 and self.operator in [ArithmeticOperator.SUB, ArithmeticOperator.ADD]:
            right_c = f"({right_c})"

        return f"{left_c} {self.operator.value} {right_c}"

    def negate(self) -> Expression:
        """Negate an arithmetic expression by wrapping in Logical NOT"""
        return LogicalExpression(LogicalOperator.NOT, operands=[self])

    def __eq__(self, other) -> bool:
        if not isinstance(other, ArithmeticExpression):
            return NotImplemented
        # Recursively check operands
        return (self.operator == other.operator and
                self.left == other.left and 
                self.right == other.right)

    def __hash__(self) -> int:
        # Hash based on type, operator, and operands' hashes
        return hash((ArithmeticExpression, self.operator, self.left, self.right))


@dataclass
class LogicalExpression(Expression):
    """
    Represents logical expressions (e.g., a && b, !c)
    
    Handles unary (NOT) and binary (AND, OR) logical operations
    
    Attributes:
        operator: The logical operator (AND, OR, NOT)
        operands: A list containing one (for NOT) or two (for AND, OR) operand Expressions
    """
    operator: LogicalOperator
    operands: list[Expression] 

    def __post_init__(self):
        """Validate operands based on operator"""
        num_ops = len(self.operands)
        if self.operator == LogicalOperator.NOT:
            if num_ops != 1:
                raise ValueError(f"Logical NOT requires 1 operand, got {num_ops}")
        elif self.operator in [LogicalOperator.AND, LogicalOperator.OR]:
            if num_ops != 2:
                raise ValueError(f"Logical {self.operator.name} requires 2 operands, got {num_ops}")
        else:
            # Should not happen with Enum
            raise ValueError(f"Unknown logical operator: {self.operator}")
        
        for op in self.operands:
            if not isinstance(op, Expression):
                raise TypeError(f"Logical expression operands must be Expression instances, got {type(op)}")


    def get_variables(self) -> Set[Variable]:
        """Get variables from all operands"""
        variables = set()
        for operand in self.operands:
            variables.update(operand.get_variables())
        return variables

    def _create_z3_expr(self, var_map: Optional[Dict[str, z3.ExprRef]] = None, context: Optional[Context] = None, *, _suffix: str = "") -> z3.ExprRef:
        """Create Z3 logical expression using the context, passing down suffix"""
        z3_operands = [op.to_z3(var_map, context=context, _suffix=_suffix) for op in self.operands]
        
        # Ensure all operands are from the correct context
        if context is not None:
            for i, op_z3 in enumerate(z3_operands):
                if op_z3.ctx != context:
                    logger.error(f"Context mismatch in Logical operand {i} ({self.operands[i]}): context {op_z3.ctx} != expected {context}")
                    return z3.BoolVal(False, ctx=context)

        logger.trace(f"Creating Z3 logical expression (ctx={context}): Op={self.operator}, Operands={[repr(op) for op in z3_operands]}")
        
        op_func = self.operator.to_z3()
        
        # Z3 functions handle context
        if self.operator == LogicalOperator.NOT:
            return op_func(z3_operands[0])
        elif self.operator == LogicalOperator.AND:
            return And(*z3_operands)
        elif self.operator == LogicalOperator.OR:
            return Or(*z3_operands)
        else:
             raise ValueError(f"Unsupported logical operator {self.operator}")

    def to_c(self) -> str:
        """Convert to C code representation"""
        c_operands = []
        for op in self.operands:
             # Wrap operand in parentheses if needed
            c_ops = f"({op.to_c()})" if _c_needs_paren(op) else op.to_c()
            c_operands.append(c_ops)

        if self.operator == LogicalOperator.NOT:
            return f"!{c_operands[0]}"
        elif self.operator == LogicalOperator.AND:
            return f"{c_operands[0]} {self.operator.value} {c_operands[1]}"
        elif self.operator == LogicalOperator.OR:
            return f"{c_operands[0]} {self.operator.value} {c_operands[1]}"
        # Should not be reached due to __post_init__ check
        return "" 

    def negate(self) -> Expression:
        """Negate the logical expression"""
        if self.operator == LogicalOperator.NOT:
            # Negation of NOT p is just p
            return self.operands[0] 
        elif self.operator in [LogicalOperator.AND, LogicalOperator.OR]:
             # Apply De Morgan's laws
             negated_operands = [op.negate() for op in self.operands]
             new_operator = LogicalOperator.OR if self.operator == LogicalOperator.AND else LogicalOperator.AND
             return LogicalExpression(new_operator, negated_operands)
        else:
            raise ValueError(f"Cannot negate unknown logical operator: {self.operator}")

    def __eq__(self, other) -> bool:
        if not isinstance(other, LogicalExpression):
            return NotImplemented
        # Compare operator and operands (order matters for binary ops)
        return (self.operator == other.operator and
                self.operands == other.operands) 

    def __hash__(self) -> int:
        # Convert list to tuple for hashing
        return hash((LogicalExpression, self.operator, tuple(self.operands)))


# Helper function to determine if parentheses are needed for C output
def _c_needs_paren(expr: Expression) -> bool:
    # Constants and Variables are atomic
    if isinstance(expr, (Constant, VariableExpression)):
        return False
    # Other expression types might need parentheses depending on context
    return True

# Factory function to create expressions directly from Z3 expressions (Needs update for LogicalExpression)
def from_z3(z3_expr: ExprRef, variables: Dict[str, Variable], *, _suffix: str = "") -> Expression:
    """
    Create an Expression from a Z3 expression
    Note: Needs careful handling of variable names and suffixes
    The current implementation might not correctly handle suffixed variables
    when converting *from* Z3 back to Expression if the suffix isn't stripped
    This likely requires adjusting the variable lookup/creation logic
    
    Args:
        z3_expr: Z3 expression to convert.
        variables: Dictionary mapping variable names to Variable objects.
        _suffix: Suffix used when creating Z3 variables (may need stripping)
        
    Returns:
        Expression: Equivalent Expression object
        
    Note: Handles basic constants, variables, comparisons, arithmetic (+,-,*), 
          and logical (AND, OR, NOT). Assumes binary arithmetic/logic for simplicity
    """
    decl = z3_expr.decl()
    kind = decl.kind()
    num_args = z3_expr.num_args()

    # Handle Boolean Constants
    if kind == z3.Z3_OP_TRUE:
        return Constant(True)
    if kind == z3.Z3_OP_FALSE:
        return Constant(False)

    # Handle Integer/Boolean Variables
    if num_args == 0 and kind == z3.Z3_OP_UNINTERPRETED:
        var_name = str(z3_expr) 
        base_var_name = var_name 
        if base_var_name not in variables:
            sort = z3_expr.sort()
            var_type = VariableType.BOOL if sort == z3.BoolSort(ctx=z3_expr.ctx) else VariableType.INT
            variables[base_var_name] = Variable(base_var_name, type=var_type)
            logger.trace(f"Created new variable during from_z3: {variables[base_var_name]}")
        return VariableExpression(variables[base_var_name])
        
    # Handle Integer Constants
    if num_args == 0 and z3.is_int_value(z3_expr):
         return Constant(z3_expr.as_long())

    # Handle Comparisons (covers >, >=, <, <=, ==, !=)
    if kind in [z3.Z3_OP_LT, z3.Z3_OP_LE, z3.Z3_OP_GT, z3.Z3_OP_GE, z3.Z3_OP_EQ, z3.Z3_OP_DISTINCT]:
        if num_args == 2:
            # Pass suffix down recursively
            left_expr = from_z3(z3_expr.arg(0), variables, _suffix=_suffix)
            right_expr = from_z3(z3_expr.arg(1), variables, _suffix=_suffix)
            
            op_map = {
                z3.Z3_OP_LT: ComparisonOperator.LESS,
                z3.Z3_OP_LE: ComparisonOperator.LESS_EQUAL,
                z3.Z3_OP_GT: ComparisonOperator.GREATER,
                z3.Z3_OP_GE: ComparisonOperator.GREATER_EQUAL,
                z3.Z3_OP_EQ: ComparisonOperator.EQUAL,
                z3.Z3_OP_DISTINCT: ComparisonOperator.NOT_EQUAL
            }
            operator = op_map[kind]
            return ComparisonExpression(left_expr, operator, right_expr)

    # Handle Arithmetic Operations (+, -, *, / - DIV/MOD removed for now)
    if kind in [z3.Z3_OP_ADD, z3.Z3_OP_SUB, z3.Z3_OP_MUL]:
        # Handle binary case
        if num_args >= 2:
           # Pass suffix down recursively
           left_expr = from_z3(z3_expr.arg(0), variables, _suffix=_suffix)
           right_expr = from_z3(z3_expr.arg(1), variables, _suffix=_suffix)
           op_map = {
               z3.Z3_OP_ADD: ArithmeticOperator.ADD,
               z3.Z3_OP_SUB: ArithmeticOperator.SUB,
               z3.Z3_OP_MUL: ArithmeticOperator.MUL,
               # z3.Z3_OP_IDIV: ArithmeticOperator.DIV,
               # z3.Z3_OP_MOD: ArithmeticOperator.MOD,
           }
           operator = op_map[kind]
           return ArithmeticExpression(left_expr, operator, right_expr)

    # Handle Logical Operations (AND, OR, NOT)
    if kind in [z3.Z3_OP_AND, z3.Z3_OP_OR, z3.Z3_OP_NOT]:
        op_map = {
            z3.Z3_OP_AND: LogicalOperator.AND,
            z3.Z3_OP_OR: LogicalOperator.OR,
            z3.Z3_OP_NOT: LogicalOperator.NOT,
        }
        operator = op_map[kind]
        
        # Pass suffix down recursively
        if operator == LogicalOperator.NOT and num_args == 1:
             operand_expr = from_z3(z3_expr.arg(0), variables, _suffix=_suffix)
             return LogicalExpression(operator, [operand_expr])
        elif operator in [LogicalOperator.AND, LogicalOperator.OR] and num_args >= 2:
             left_expr = from_z3(z3_expr.arg(0), variables, _suffix=_suffix)
             right_expr = from_z3(z3_expr.arg(1), variables, _suffix=_suffix)
             return LogicalExpression(operator, [left_expr, right_expr])

    raise ValueError(f"Unsupported Z3 expression type: {z3_expr} (Kind: {kind}, Args: {num_args})")
