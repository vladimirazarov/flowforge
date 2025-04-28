"""
Factory for generating random expressions for CFG edge conditions

Centralizes expression generation using CFG context and configuration
Ensures variety and uses strategies like variable rotation
"""

from __future__ import annotations
import random
from typing import  List, Optional, Tuple, TYPE_CHECKING, Iterator
import itertools
from dataclasses import dataclass, field

from loguru import logger

from src.core.nodes.cfg_basic_block_node import CFGBasicBlockNode
from src.core.cfg_content.expression import (
    Constant, 
    Expression,
    ComparisonExpression,
    VariableExpression,
    LogicalExpression 
)

from src.core.cfg_content.variable import Variable, VariableType
from src.core.cfg_content.operator import ComparisonOperator, LogicalOperator

from src.config.config import AppConfig

if TYPE_CHECKING:
    from src.core.cfg.cfg import CFG

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

@dataclass
class ExpressionRandomFactory:
    """Generates random `Expression` objects for CFG edge conditions

    Uses CFG input parameters and configuration settings to create expressions
    Employs a round-robin strategy for selecting variables to improve variety

    Attributes:
        cfg: The control flow graph instance
        app_config: Application configuration settings
        test_input_complexity: Complexity level for generated expressions
        allowed_expressions: Types of expressions allowed ('simple', 'complex')
        _available_variables: Cache of input variables from the CFG context
        _sorted_variables: Sorted list of available variables for iteration
        _variable_iterator: Iterator for round-robin variable selection
        _last_variable_set_tuple: Cache to detect changes in available variables
    """
    cfg: CFG
    app_config: AppConfig

    test_input_complexity: int = field(init=False)
    allowed_expressions: List[str] = field(default_factory=list, init=False)
    _available_variables: List[Variable] = field(default_factory=list, init=False)
    _sorted_variables: List[Variable] = field(default_factory=list, init=False)
    _variable_iterator: Optional[Iterator[Variable]] = field(default=None, init=False)
    _last_variable_set_tuple: Optional[Tuple[Variable, ...]] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize derived fields and populate variable cache"""
        self.test_input_complexity = self.app_config.test_input_complexity
        self.allowed_expressions = self.app_config.expression_initializer.allowed_expressions
        self._available_variables = []
        self._sorted_variables = []
        self._update_variable_cache() 

    def _update_variable_cache(self):
        """Helper to refresh available variables from CFG input parameters

        Updates the internal cache (`_available_variables`, `_sorted_variables`)
        only with variables marked as input parameters in `cfg.context`
        Resets the variable iterator (`_variable_iterator`) if the set of
        input parameters changes
        """
        current_vars_set = set()
        input_params = set()
        if hasattr(self.cfg, 'context') and hasattr(self.cfg.context, 'used_parameters'):
            input_params = {p for p in self.cfg.context.used_parameters if isinstance(p, Variable)}
            if not input_params:
                 logger.warning("No input parameters found in CFG context during factory initialization.")
        else:
             logger.error("CFG context or used_parameters not found. Cannot select variables.")

        # Filter context variables to only include those that are also input parameters
        # This ensures edge conditions only use input parameters
        current_vars_set = input_params 
        current_vars_tuple = tuple(sorted(current_vars_set, key=lambda var: var.name))

        # Check if the set of available variables has changed
        if current_vars_tuple != self._last_variable_set_tuple:
            logger.debug(f"Input Parameter set changed. Old: {[v.name for v in self._last_variable_set_tuple or []]}, New: {[v.name for v in current_vars_tuple]}")
            self._available_variables = list(current_vars_tuple) 
            self._sorted_variables = self._available_variables
            self._variable_iterator = None 
            self._last_variable_set_tuple = current_vars_tuple
            if not self._available_variables:
                 logger.warning("Resetting variable cache, but no input parameters are available.")
            else:
                 logger.info(f"Updated variable cache with input parameters and reset iterator. Available: {[v.name for v in self._sorted_variables]}")


    def create_condition_expression(self, edge: Optional[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]] = None) -> Expression:
        """Creates a random expression suitable for an edge condition

        Ensures the variable cache is up-to-date before generation
        Selects between simple or complex expression based on configuration

        Args:
            edge (Optional[Tuple[CFGBasicBlockNode, CFGBasicBlockNode]]): Contextual
                edge for which the condition is being generated (optional, currently unused)

        Returns:
            Expression: A randomly generated comparison or logical expression
        """
        self._update_variable_cache()

        if not self._available_variables:
             logger.error("No variables available to create expression. Returning fallback.")
             fallback_var = Variable("fallback_var", VariableType.INT) 
             fallback_var_expr = VariableExpression(fallback_var)
             return ComparisonExpression(fallback_var_expr, ComparisonOperator.EQUAL, Constant(0))

        # Can add cfg_content complex logic based on self.allowed_expressions
        sorted_allowed_expressions = sorted(self.allowed_expressions)
        choice = random.choice(sorted_allowed_expressions)

        if 'complex' in choice: 
             expr = self.generate_complex_expression()
        else:
             expr = self.generate_simple_comparison()
        
        logger.debug(f"Generated condition for edge {edge}: {expr.to_c()}")
        return expr

    def _get_safe_variable(self) -> Variable:
        """Helper to get the next variable using round-robin rotation

        Cycles through the sorted list of available input parameters
        Initializes the cycle iterator if needed

        Returns:
            Variable: The next variable from the rotation

        Raises:
            ValueError: If no input parameter variables are available in the cache
        """
        if not self._sorted_variables: # Check the sorted list used by iterator
            logger.error("Attempted to get variable when none are available. Raising error.")
            raise ValueError("No variables available in the factory's cache.")
            
        # Initialize or re-initialize the iterator if it's None (e.g., first call or cache reset)
        if self._variable_iterator is None:
            logger.debug(f"Initializing variable iterator with: {[v.name for v in self._sorted_variables]}")
            self._variable_iterator = itertools.cycle(self._sorted_variables)

        # Get the next variable from the cycle
        chosen_var = next(self._variable_iterator)
        logger.debug(f"Selected safe variable via rotation: {chosen_var.name}")
        return chosen_var

    def generate_simple_comparison(self) -> Expression:
        """Generates a simple comparison expression (e.g., var < constant)

        Selects a variable using rotation, a random comparison operator
        (excluding ==, !=), and a random constant within the configured range

        Returns:
            ComparisonExpression: The generated simple comparison
        """
        var = self._get_safe_variable() 
        allowed_operators = [
            op for op in ComparisonOperator 
            if op not in (ComparisonOperator.EQUAL, ComparisonOperator.NOT_EQUAL)
        ]
        comparison_operators_list = sorted(allowed_operators, key=lambda op: op.name)

        op = random.choice(comparison_operators_list)
        
        value_range = self.app_config.expression_initializer.value_range
        value = random.randint(value_range[0], value_range[1]) 
        
        right_expr = Constant(value)
        left_expr = VariableExpression(var) 

        expr = ComparisonExpression(left=left_expr, operator=op, right=right_expr)
        logger.debug(f"Generated simple comparison: {expr.to_c()}")
        return expr


    def generate_complex_expression(self) -> Expression:
        """Generates a complex logical expression (e.g., expr1 AND expr2)

        Combines two simple comparison expressions with a random logical
        operator (AND/OR). Attempts to use different variables in the two
        sub-expressions by calling `generate_simple_comparison` twice
        Falls back to a simple comparison if fewer than two variables are available

        Returns:
            Expression: The generated logical or comparison expression
        """
        # Requires at least 2 variables for a meaningful complex expression involving different vars
        if len(self._available_variables) < 2:
            logger.warning("Not enough variables for complex expression, generating simple comparison instead.")
            return self.generate_simple_comparison()
            
        # Try to get two different variables using the rotation
        expr1 = self.generate_simple_comparison()
        expr2 = self.generate_simple_comparison() 
        
        # Check: if the variable used in expr1 and expr2 is the same, maybe generate expr2 again
        # Note: expr1/expr2 are ComparisonExpressions, access their left operand (VariableExpression) then the variable itself
        if isinstance(expr1, ComparisonExpression) and isinstance(expr2, ComparisonExpression):
             if isinstance(expr1.left, VariableExpression) and isinstance(expr2.left, VariableExpression):
                  if expr1.left.variable == expr2.left.variable:
                       logger.debug("Complex expression used same variable twice, regenerating second part.")
                       expr2 = self.generate_simple_comparison() # Try again to get the next variable


        logical_operators_list = sorted([LogicalOperator.AND, LogicalOperator.OR], key=lambda op: op.name)
        logical_op = random.choice(logical_operators_list)
        
        complex_expr = LogicalExpression(operator=logical_op, operands=[expr1, expr2])
        logger.debug(f"Generated complex expression: {complex_expr.to_c()}")
        return complex_expr