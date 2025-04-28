"""
Defines classes representing different types of operations within CFG nodes

Includes abstract base `Operation` and concrete subclasses for:
- Arithmetic (`ArithmeticOperation`, `CompoundAssignmentOperation`)
- Logical (`LogicalOperation`)
- Array Access (`ArrayAccessOperation`)
- Unary (`UnaryOperation`)
- Control Flow (`ReturnOperation`, `DeclarationOperation`)

Each operation supports conversion to C code and potentially Z3 representation
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Optional, Set

import z3
from loguru import logger

from src.core.cfg_content.expression import Expression, Constant
from src.core.cfg_content.operator import ArithmeticOperator, LogicalOperator, UnaryOperator
from src.core.cfg_content.variable import Variable

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

@dataclass
class Operation(ABC):
    """
    Abstract base class for all program operations with Z3 integration

    An operation represents a single atomic state transformation in the program,
    consisting of a target variable and an expression that modifies it
    """
    variable: Variable
    expression: Expression
    _z3_expr: Optional[z3.ExprRef] = field(default=None, init=False)
    step: Optional[int] = field(default=None, init=False)

    def __post_init__(self):
        """Validate operation components"""
        if not isinstance(self.variable, Variable):
            raise TypeError("Operation target must be a Variable")
        if not isinstance(self.expression, Expression):
            raise TypeError("Operation source must be an Expression")

    def get_modified_variable(self) -> Variable:
        """
        Get the variable modified by this operation

        Returns:
            The target variable of this operation
        """
        return self.variable

    def affects_variable(self, variable: Variable) -> bool:
        """
        Check if this operation affects a given variable

        Args:
            variable: The variable to check

        Returns:
            True if this operation modifies the specified variable
        """
        return self.variable == variable

    def get_used_variables(self) -> Set[Variable]:
        """
        Get all variables used in this operation

        Returns:
            A set of all variables involved in this operation
        """
        result = {self.variable}
        result.update(self.expression.get_variables())
        return result

    def get_rhs_variables(self) -> Set[Variable]:
        """
        Get variables used only on the right-hand side (in the expression)

        Returns:
            Set[Variable]: Set of variables referenced only in the expression part
        """
        return self.expression.get_variables()

    def get_constant_value(self) -> Optional[int]:
        """
        Get the constant value if this is a direct assignment of a constant

        Returns:
            The integer value if it's a constant assignment, or None otherwise
        """
        if isinstance(self.expression, Constant):
            value = self.expression.value
            return value if isinstance(value, int) else None
        return None

    def to_z3(self, var_map: Optional[dict] = None) -> z3.ExprRef:
        """
        Create or return the cached Z3 expression representing this operation

        Args:
            var_map: Optional dictionary mapping variable names to Z3 variables

        Returns:
            The Z3 ExprRef corresponding to this operation
        """
        if self._z3_expr is None:
            self._z3_expr = self._create_z3_expr(var_map)
        return self._z3_expr

    @abstractmethod
    def _create_z3_expr(self, var_map: Optional[dict] = None) -> z3.ExprRef:
        """
        Create the Z3 representation for this operation

        Args:
            var_map: Optional dictionary mapping variable names to Z3 variables

        Returns:
            A Z3 expression for the operation
        """
        pass

    @abstractmethod
    def is_loop_breaking_operation(self) -> bool:
        """
        Check if this operation is typically used to break loops

        Returns:
            True if this operation can terminate a loop, otherwise False
        """
        pass

    @abstractmethod
    def to_c(self) -> str:
        """
        Convert to C code representation

        Returns:
            A string containing valid C code
        """
        pass

    def clone(self) -> Operation:
        """
        Create a new identical operation instance, handling subclass fields

        Returns:
            A deep copy of the current operation
        """
        cls = self.__class__
        init_fields = {f.name: getattr(self, f.name) for f in fields(cls) if f.init}

        if 'expression' in init_fields and hasattr(self.expression, 'clone'):
            init_fields['expression'] = self.expression.clone()
        
        if isinstance(self, ArrayAccessOperation) and 'index' in init_fields and hasattr(self.index, 'clone'):
            init_fields['index'] = self.index.clone()

        return cls(**init_fields)

    def __str__(self) -> str:
        """Convert to string using the C representation"""
        return self.to_c()

    def __eq__(self, other: Any) -> bool:
        """
        Check if two operations are equal by variable and expression string representation
        Note: Comparing expression string forms might not be fully robust
        Subclasses should override if more fields are needed for equality
        """
        if not isinstance(other, type(self)):
            return False
        return (self.variable == other.variable and
                str(self.expression) == str(other.expression))

    def __hash__(self) -> int:
        """Generate a hash from the variable and expression string representation"""
        return hash((type(self), str(self.variable), str(self.expression)))

    @property
    def logical_operations_count(self) -> int:
        """Return 1 if this is a logical operation, otherwise 0"""
        return 1 if isinstance(self, LogicalOperation) else 0

    @property
    def arithmetic_operations_count(self) -> int:
        """Return 1 if this is an arithmetic operation, otherwise 0"""
        return 1 if isinstance(self, ArithmeticOperation) else 0


@dataclass
class ArithmeticOperation(Operation):
    """
    Operation that performs arithmetic (e.g., x = x + 1)

    Attributes:
        operator: An ArithmeticOperator indicating which operation is performed
    """
    operator: ArithmeticOperator

    def is_loop_breaking_operation(self) -> bool:
        """Arithmetic operations like ADD/SUB can be used in loops, but not to directly break them"""
        return self.operator in (ArithmeticOperator.ADD, ArithmeticOperator.SUB)

    def _create_z3_expr(self, var_map: Optional[dict] = None) -> z3.ExprRef:
        """Create the Z3 representation for this arithmetic operation"""
        var_map = var_map if var_map is not None else {}
        var_name = self.variable.name

        if var_name in var_map:
            z3_var = var_map[var_name]
        else:
            z3_var = z3.Int(var_name)
            var_map[var_name] = z3_var

        expr_z3 = self.expression.to_z3(var_map=var_map)

        if self.operator == ArithmeticOperator.ASSIGN:
            result = expr_z3
        elif self.operator == ArithmeticOperator.ADD:
            result = z3_var + expr_z3
        elif self.operator == ArithmeticOperator.SUB:
            result = z3_var - expr_z3
        elif self.operator == ArithmeticOperator.MUL:
            result = z3_var * expr_z3
        elif self.operator == ArithmeticOperator.DIV:
            result = z3_var / expr_z3
        elif self.operator == ArithmeticOperator.MOD:
            result = z3_var % expr_z3
        else:
            logger.error(f"Unsupported arithmetic operator for Z3: {self.operator}")
            raise ValueError(f"Unsupported operator: {self.operator}")

        # The operation represents an equality constraint: var == result
        return z3_var == result

    def to_c(self) -> str:
        """Convert this arithmetic operation to C code"""
        var_c = self.variable.to_c()
        expr_c = self.expression.to_c()

        if self.operator == ArithmeticOperator.ASSIGN:
            return f"{var_c} = {expr_c};"
        elif self.operator == ArithmeticOperator.ADD:
            return f"{var_c} = {var_c} + {expr_c};"
        elif self.operator == ArithmeticOperator.SUB:
            return f"{var_c} = {var_c} - {expr_c};"
        elif self.operator == ArithmeticOperator.MUL:
            return f"{var_c} = {var_c} * {expr_c};"
        elif self.operator == ArithmeticOperator.DIV:
            return f"{var_c} = {var_c} / {expr_c};"
        elif self.operator == ArithmeticOperator.MOD:
            return f"{var_c} = {var_c} % {expr_c};"
        else:
            logger.warning(f"Unrecognized ArithmeticOperator {self.operator} during C conversion. Defaulting to assignment.")
            return f"{var_c} = {expr_c};"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ArithmeticOperation):
             return False
        return super().__eq__(other) and self.operator == other.operator

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.operator))


@dataclass
class CompoundAssignmentOperation(ArithmeticOperation):
    """
    Operation that performs compound assignment (e.g., x += 1, x *= y)

    Inherits from ArithmeticOperation. Operator indicates the core operation (e.g., ADD for +=)
    Expression holds the right-hand side
    """

    def __post_init__(self):
        """Validate operation components"""
        pass

    def _create_z3_expr(self, var_map: Optional[dict] = None) -> z3.ExprRef:
        """
        Create the Z3 representation, equivalent to var = var op expr
        """
        # Logic is same as ArithmeticOperation: var == var + expr (for +=), etc.
        return super()._create_z3_expr(var_map)

    def to_c(self) -> str:
        """Convert this compound assignment operation to C code"""
        var_c = self.variable.to_c()
        expr_c = self.expression.to_c()
        op_symbol = ""

        if self.operator == ArithmeticOperator.ADD:
            op_symbol = "+="
        elif self.operator == ArithmeticOperator.SUB:
            op_symbol = "-="
        elif self.operator == ArithmeticOperator.MUL:
            op_symbol = "*="
        elif self.operator == ArithmeticOperator.DIV:
            op_symbol = "/="
        elif self.operator == ArithmeticOperator.MOD:
            op_symbol = "%="
        else:
            assert self.operator in [
                ArithmeticOperator.ADD, ArithmeticOperator.SUB,
                ArithmeticOperator.MUL, ArithmeticOperator.DIV,
                ArithmeticOperator.MOD
            ], f"Unsupported operator {self.operator} for compound assignment C code."

        return f"{var_c} {op_symbol} {expr_c};"


@dataclass
class LogicalOperation(Operation):
    """
    Operation that performs logical (boolean) operations (e.g., x = a && b)

    Attributes:
        operator: A LogicalOperator indicating which operation is performed
    """
    operator: LogicalOperator

    def is_loop_breaking_operation(self) -> bool:
        """Logical operations do not inherently break loops"""
        return False

    def _create_z3_expr(self, var_map: Optional[dict] = None) -> z3.ExprRef:
        """
        Create the Z3 representation for this logical operation as an equality:
        variable == (operator(variable, expression)). Assumes Boolean vars
        """
        var_map = var_map if var_map is not None else {}
        var_name = self.variable.name

        if var_name in var_map:
            z3_var = var_map[var_name]
        else:
            z3_var = z3.Bool(var_name)
            var_map[var_name] = z3_var

        expr_z3 = self.expression.to_z3(var_map=var_map)

        if self.operator == LogicalOperator.AND:
            result = z3.And(z3_var, expr_z3)
        elif self.operator == LogicalOperator.OR:
            result = z3.Or(z3_var, expr_z3)
        elif self.operator == LogicalOperator.NOT:
            result = z3.Not(expr_z3)
        else:
            logger.error(f"Unsupported logical operator for Z3: {self.operator}")
            raise ValueError(f"Unsupported operator: {self.operator}")

        return z3_var == result

    def to_c(self) -> str:
        """Convert this logical operation to C code"""
        var_c = self.variable.to_c()
        expr_c = self.expression.to_c()

        if self.operator == LogicalOperator.AND:
            return f"{var_c} = {var_c} && {expr_c};"
        elif self.operator == LogicalOperator.OR:
            return f"{var_c} = {var_c} || {expr_c};"
        elif self.operator == LogicalOperator.NOT:
            return f"{var_c} = !{expr_c};"
        else:
            logger.warning(f"Unrecognized LogicalOperator {self.operator} during C conversion. Defaulting to assignment.")
            return f"{var_c} = {expr_c};"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, LogicalOperation):
            return False
        return super().__eq__(other) and self.operator == other.operator

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.operator))


@dataclass
class ArrayAccessOperation(Operation):
    """
    Operation that performs array element access (e.g., arr[i] = x)

    Attributes:
        index: Expression representing the array index
    """
    index: Expression

    def __post_init__(self):
        """Validate operation components, including index"""
        super().__post_init__()
        if not isinstance(self.index, Expression):
            raise TypeError("Array index must be an Expression")

    def is_loop_breaking_operation(self) -> bool:
        """Array operations do not inherently break loops"""
        return False

    def _create_z3_expr(self, var_map: Optional[dict] = None) -> z3.ExprRef:
        """
        Create a Z3 representation of array assignment using Z3 Arrays
        Requires target variable to be a Z3 Array. Ensures index is mapped
        """
        var_map = var_map if var_map is not None else {}
        array_name = self.variable.name

        if array_name in var_map:
            z3_array = var_map[array_name]
        else:
            # Assume Array Int -> Int if type info not available
            logger.warning(f"Creating default Z3 Array (Int -> Int) for {array_name}. Specify types for precision.")
            z3_array = z3.Array(array_name, z3.IntSort(), z3.IntSort())
            var_map[array_name] = z3_array

        idx_z3 = self.index.to_z3(var_map=var_map)
        val_z3 = self.expression.to_z3(var_map=var_map)

        # Z3 array assignment: Store(array, index, value)
        updated_array = z3.Store(z3_array, idx_z3, val_z3)
        return z3_array == updated_array

    def to_c(self) -> str:
        """Convert this array assignment to C code"""
        var_c = self.variable.to_c()
        expr_c = self.expression.to_c()
        idx_c = self.index.to_c()
        return f"{var_c}[{idx_c}] = {expr_c};"

    def get_used_variables(self) -> Set[Variable]:
        """
        Return all variables used by this operation, including the index expression
        """
        vars_used = super().get_used_variables()
        vars_used.update(self.index.get_variables())
        return vars_used

    def __eq__(self, other: Any) -> bool:
        """Check equality based on variable, expression, and index"""
        if not isinstance(other, ArrayAccessOperation):
            return False
        return (super().__eq__(other) and
                str(self.index) == str(other.index))

    def __hash__(self) -> int:
        """Generate hash based on variable, expression, and index string reps"""
        return hash((super().__hash__(), str(self.index)))


@dataclass
class UnaryOperation(Operation):
    """
    Operation that performs unary operations (e.g., ++x or x++)

    Attributes:
        operator: A UnaryOperator
        is_prefix: True for prefix (e.g., ++x), False for postfix (x++) default True
    """
    operator: UnaryOperator
    is_prefix: bool = True

    def is_loop_breaking_operation(self) -> bool:
        """
        Increment/decrement can affect loops, but do not directly break them
        """
        return self.operator in (UnaryOperator.INCREMENT, UnaryOperator.DECREMENT)

    def _create_z3_expr(self, var_map: Optional[dict] = None) -> z3.ExprRef:
        """
        Create a Z3 representation for a unary operation as equality:
        x++ => x = x + 1, x-- => x = x - 1, !x => x = !x (assuming Bool), etc
        """
        var_map = var_map if var_map is not None else {}
        var_name = self.variable.name

        if var_name in var_map:
            z3_var = var_map[var_name]
        elif self.operator == UnaryOperator.NOT:
             z3_var = z3.Bool(var_name)
             var_map[var_name] = z3_var
        else:
            z3_var = z3.Int(var_name)
            var_map[var_name] = z3_var

        if self.operator == UnaryOperator.INCREMENT:
            result = z3_var + 1
        elif self.operator == UnaryOperator.DECREMENT:
            result = z3_var - 1
        elif self.operator == UnaryOperator.NEGATE:
            assert isinstance(z3_var, z3.ArithRef), "NEGATE requires arithmetic Z3 var"
            result = -z3_var
        elif self.operator == UnaryOperator.NOT:
            assert isinstance(z3_var, z3.BoolRef), "NOT requires boolean Z3 var"
            result = z3.Not(z3_var)
        else:
            logger.error(f"Unsupported unary operator for Z3: {self.operator}")
            raise ValueError(f"Unsupported operator: {self.operator}")

        return z3_var == result

    def to_c(self) -> str:
        """Convert this unary operation to C code"""
        var_c = self.variable.to_c()
        op_str = ""

        if self.operator == UnaryOperator.INCREMENT:
            op_str = "++"
        elif self.operator == UnaryOperator.DECREMENT:
            op_str = "--"
        elif self.operator == UnaryOperator.NEGATE:
            return f"{var_c} = -{var_c};"
        elif self.operator == UnaryOperator.NOT:
            return f"{var_c} = !{var_c};"
        else:
             logger.warning(f"Unrecognized UnaryOperator {self.operator} during C conversion.")
             return f"{var_c};"

        if self.operator in (UnaryOperator.INCREMENT, UnaryOperator.DECREMENT):
            return f"{op_str}{var_c};" if self.is_prefix else f"{var_c}{op_str};"
        else:
            return f"{var_c};"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, UnaryOperation):
            return False
        return (super().__eq__(other) and
                self.operator == other.operator and
                self.is_prefix == other.is_prefix)

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.operator, self.is_prefix))


@dataclass
class ReturnOperation(Operation):
    """
    Represents a return statement 

    The 'variable' field is conceptually the value being returned
    """

    def __post_init__(self):
        assert isinstance(self.variable, Variable), "ReturnOperation requires a Variable" 
        pass

    def get_modified_variable(self) -> Variable:
        # Return doesn't modify, it uses the value.
        return self.variable

    def get_used_variables(self) -> Set[Variable]:
        return {self.variable}

    def get_rhs_variables(self) -> Set[Variable]:
        return {self.variable}

    def _create_z3_expr(self, var_map: Optional[dict] = None) -> z3.ExprRef:
        return z3.BoolVal(True)

    def is_loop_breaking_operation(self) -> bool:
        return True

    def to_c(self) -> str:
        """Convert to C return statement"""
        return f"return {self.variable.to_c()};"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ReturnOperation):
            return False
        return self.variable == other.variable

    def __hash__(self) -> int:
        return hash((ReturnOperation, self.variable))

    @property
    def logical_operations_count(self) -> int:
        return 0

    @property
    def arithmetic_operations_count(self) -> int:
        return 0


@dataclass
class DeclarationOperation(Operation):
    """
    Represents a variable declaration (e.g., int result;)
    Primarily for code generation; no Z3 meaning or runtime effect
    Expression field is unused
    """

    def __post_init__(self):
        assert isinstance(self.variable, Variable), "DeclarationOperation requires a Variable"
        # Set expression to a placeholder, as it's not used
        self.expression = Constant(0)
        pass

    def get_modified_variable(self) -> Variable:
        return self.variable

    def get_used_variables(self) -> Set[Variable]:
        return {self.variable}

    def get_rhs_variables(self) -> Set[Variable]:
        return set()

    def _create_z3_expr(self, var_map: Optional[dict] = None) -> z3.ExprRef:
        return z3.BoolVal(True)

    def is_loop_breaking_operation(self) -> bool:
        return False

    def to_c(self) -> str:
        """Convert to C declaration statement"""
        c_type = self.variable.type.value if self.variable.type else "int"
        return f"{c_type} {self.variable.to_c()};"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DeclarationOperation):
            return False
        return self.variable == other.variable

    def __hash__(self) -> int:
        return hash((DeclarationOperation, self.variable))

    @property
    def logical_operations_count(self) -> int:
        return 0

    @property
    def arithmetic_operations_count(self) -> int:
        return 0