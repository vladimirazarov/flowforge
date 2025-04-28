"""Factory for creating `Operation` objects based on configuration and context

Purpose:
    Factory for creating `Operation` objects. Used by:
    - `ComplexityTuner`: To get basic assignment, arithmetic, and dummy
      operations based on allowed variables and parameter constraints
    - `LoopTerminator`: To get specific loop-breaking operations
    Operations are generated based on configuration settings (value range,
    complexity) and context provided by callers (e.g., allowed variables)

Core Functionality & Parameter Handling:
    Relies on callers (`ComplexityTuner`) to provide correctly filtered
    variable lists (`modifiable_lhs_vars`, `available_rhs_vars`) respecting
    heuristic rules. This module focuses on object creation, not heuristic logic
"""

from __future__ import annotations
import random
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass, field

from src.core.cfg_content.operation import Operation, ArithmeticOperation, CompoundAssignmentOperation
from src.core.cfg_content.operator import ArithmeticOperator, ComparisonOperator
from src.core.cfg_content.expression import Constant, VariableExpression, Expression, ArithmeticExpression, ComparisonExpression
from src.core.cfg_content.variable import Variable, VariableType
from src.config.config import AppConfig
from loguru import logger

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

@dataclass
class OperationsBuilder:
    """Generate `Operation` instances for CFG construction

    Creates assignment, arithmetic, dummy, and loop-breaking operations as
    requested by callers like `ComplexityTuner` or `LoopTerminator`

    Attributes:
        app_config (AppConfig): The application configuration object
        value_range (Tuple[int, int]): Range [min, max] for constant values
        complexity (int): Target complexity level affecting generation
        _operand_nesting_probability (float): Probability of creating a nested expression
    """
    app_config: AppConfig

    value_range: Tuple[int, int] = field(init=False)
    complexity: int = field(init=False) 
    _operand_nesting_probability: float = field(init=False)

    def __post_init__(self):
        """Initialize fields derived from app_config"""

        config_range = self.app_config.expression_initializer.value_range[:2]
        self.value_range = (config_range[0], config_range[1]) 
        self.complexity = self.app_config.test_input_complexity
        self._operand_nesting_probability = self.app_config.complexity_tuner.operand_nesting_probability
        logger.debug(f"OperationsBuilder initialized with nesting prob: {self._operand_nesting_probability}")

    def create_loop_breaking_operation(self, target_variable: Variable, loop_condition: Expression) -> Optional[Operation]:
        """
        Create an operation that helps break a loop condition by modifying the target variable
        Called by LoopTerminator
        
        Args:
            target_variable: The loop variable to modify
            loop_condition: The loop continuation condition (Expression)
            
        Returns:
            Operation: An operation that modifies the loop variable, or None on error
        """
        try:
            logger.debug(f"Creating loop breaking operation for var '{target_variable.name}' based on condition: {loop_condition.to_c()}")

            # Determine operator (ADD/SUB) and a small random step value
            op_enum = ArithmeticOperator.ADD # Default to increment
            step_value = random.choice([1, 2, 3]) # Small random step

            if isinstance(loop_condition, ComparisonExpression):
                op = loop_condition.operator
                # If condition is var > val or var >= val, decrement the variable
                if op in (ComparisonOperator.GREATER, ComparisonOperator.GREATER_EQUAL):
                    logger.debug(f"Condition suggests decrementing {target_variable.name} by {step_value}")
                    op_enum = ArithmeticOperator.SUB
                # If condition is var < val or var <= val, increment the variable (default)
                elif op in (ComparisonOperator.LESS, ComparisonOperator.LESS_EQUAL):
                     logger.debug(f"Condition suggests incrementing {target_variable.name} by {step_value}")
                     op_enum = ArithmeticOperator.ADD 
                # For == or !=, or complex conditions, stick with default ADD/SUB
                elif op in (ComparisonOperator.EQUAL, ComparisonOperator.NOT_EQUAL):
                     logger.debug(f"Condition is == or !=, using default op ({op_enum.name}) with step {step_value} for {target_variable.name}")
            else:
                 logger.warning(f"Loop condition is not a simple ComparisonExpression ({type(loop_condition)}), using default op ({op_enum.name}) with step {step_value}.")

            # Create the arithmetic expression: target_var op step_value
            var_expr = VariableExpression(target_variable)
            step_const_expr = Constant(step_value)
            arith_expr = ArithmeticExpression(var_expr, op_enum, step_const_expr)
            
            # Return the assignment operation: target_var = target_var op step_value
            terminating_op = ArithmeticOperation(target_variable, arith_expr, ArithmeticOperator.ASSIGN)
            terminating_op.step = step_value 
            return terminating_op

        except Exception as e:
            logger.error(f"Error creating loop breaking operation for var '{target_variable.name}': {e}", exc_info=True)
            return None 

    def _create_assignment(self, modifiable_lhs_vars: List[Variable]) -> ArithmeticOperation:
        """
        Create a simple assignment operation: var = constant
        Assumes the caller has provided a list of LHS variables that
        are safe to modify according to the chosen complexity tuning heuristic

        Args:
            modifiable_lhs_vars: List of allowed variables for the LHS

        Returns:
            ArithmeticOperation: Assignment operation
        """
        target_var = random.choice(modifiable_lhs_vars)
        constant_value = random.randint(*self.value_range)
        # Fixed Constant constructor call
        constant_expr = Constant(constant_value)

        return ArithmeticOperation(
            target_var,
            constant_expr,
            ArithmeticOperator.ASSIGN
        )

    def _create_operand(
        self,
        available_rhs_vars: List[Variable],
        input_parameters: Set[Variable],
        allow_nesting: bool = True,
        param_priority_prob: float = 0.70
    ) -> Expression:
        """Creates a single operand: Constant, Variable, or optionally a nested ArithmeticExpression
        Prioritizes using input parameters on the RHS if available
        """
        
        # Decide whether to create a nested expression
        if allow_nesting and random.random() < self._operand_nesting_probability:
            # Create a nested expression (depth 1)
            # Pass input_parameters down recursively
            nested_op1 = self._create_operand(available_rhs_vars, input_parameters, allow_nesting=False)
            nested_op2 = self._create_operand(available_rhs_vars, input_parameters, allow_nesting=False)
            nested_operator = random.choice([ArithmeticOperator.ADD, ArithmeticOperator.SUB, ArithmeticOperator.MUL])
            return ArithmeticExpression(nested_op1, nested_operator, nested_op2)
        else:
            # Create a simple operand (Variable or Constant)
            
            # Separate available RHS vars into parameters and state variables
            available_params = [v for v in available_rhs_vars if v in input_parameters]
            available_state = [v for v in available_rhs_vars if v not in input_parameters]
            
            use_param = False
            use_state = False

            if available_params:
                # Prioritize parameters if they exist
                if random.random() < param_priority_prob:
                    use_param = True
                else:
                    # If not choosing param, decide between state var or constant
                    use_state = random.random() < 0.7 and available_state # 70% chance for state 
            elif available_state:
                 # No params available, choose between state vars and constants
                 use_state = random.random() < 0.7 # 70% chance for state var

            if use_param:
                operand_var = random.choice(available_params)
                return VariableExpression(operand_var)
            elif use_state:
                operand_var = random.choice(available_state)
                return VariableExpression(operand_var)
            else:
                # Use constant
                const_value = random.randint(*self.value_range)
                return Constant(const_value)

    def _create_arithmetic(
            self, 
            modifiable_lhs_vars: List[Variable], 
            available_rhs_vars: List[Variable],
            input_parameters: Set[Variable]
        ) -> Optional[ArithmeticOperation]:
        """
        Create an arithmetic operation: lhs_var = rhs_operand1 op rhs_operand2
        Assumes the caller has provided lists of LHS and RHS variables that
        are safe to use according to the chosen complexity tuning heuristic
        
        If the LHS is an input parameter, ensures at least one RHS operand is a variable
        to maintain dependency

        Args:
            modifiable_lhs_vars: List of allowed variables for the LHS
            available_rhs_vars: List of variables available for use on the RHS
            input_parameters: Set of variables considered input parameters

        Returns:
            ArithmeticOperation: Arithmetic operation assignment, or None if constraints violated
        """
        if not modifiable_lhs_vars or not available_rhs_vars:
            logger.warning("Cannot create arithmetic op: empty modifiable LHS or available RHS vars.")
            return None
            
        allowed_params_for_lhs = [v for v in modifiable_lhs_vars if v in input_parameters]
        if allowed_params_for_lhs:
            lhs_var = random.choice(allowed_params_for_lhs)
        else:
            lhs_var = random.choice(modifiable_lhs_vars)
            
        lhs_is_param = lhs_var in input_parameters

        operators = [
            ArithmeticOperator.ADD,
            ArithmeticOperator.SUB,
            ArithmeticOperator.MUL,
        ]
        operator = random.choice(operators)

        # Determine RHS operands using the helper method, passing input_parameters
        operand1 = self._create_operand(available_rhs_vars, input_parameters, allow_nesting=True)
        operand2 = self._create_operand(available_rhs_vars, input_parameters, allow_nesting=True)

        rhs_expression = ArithmeticExpression(operand1, operator, operand2)
            
        # Final check: If LHS is a parameter, RHS must contain at least one variable
        if lhs_is_param and not rhs_expression.get_variables():
            logger.debug(f"Rejected arithmetic op for param {lhs_var.name}: RHS {rhs_expression.to_c()} contains no variables.")
            return None # Reject this attempt

        if random.random() < 0.5: 
            # Pattern: lhs = lhs op expr
            if rhs_expression.left == VariableExpression(lhs_var):
                # Ensure operator is not ASSIGN 
                if operator != ArithmeticOperator.ASSIGN:
                    # Simplify rhs to just the right part
                    simplified_rhs_expr = rhs_expression.right 
                    return CompoundAssignmentOperation(lhs_var, simplified_rhs_expr, operator)

            # Pattern: lhs = expr op lhs
            elif rhs_expression.right == VariableExpression(lhs_var):
                if operator in [ArithmeticOperator.ADD, ArithmeticOperator.MUL]: 
                    simplified_rhs_expr = rhs_expression.left 
                    return CompoundAssignmentOperation(lhs_var, simplified_rhs_expr, operator)

        return ArithmeticOperation(lhs_var, rhs_expression, ArithmeticOperator.ASSIGN)

    def _create_dummy_operation(self) -> ArithmeticOperation:
        """
        Create a dummy assignment operation (dummy_op_var = 0)
        Used when no other operations can be generated by ComplexityTuner

        Returns:
            ArithmeticOperation: Simple assignment operation (dummy = 0)
        """
        dummy_var = Variable("dummy_op_var", type=VariableType.INT)
        constant_expr = Constant(0)

        return ArithmeticOperation(
            dummy_var,
            constant_expr,
            ArithmeticOperator.ASSIGN
        )