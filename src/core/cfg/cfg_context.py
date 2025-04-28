"""
Manages the context and state associated with a Control Flow Graph (CFG)

This includes tracking used variables (parameters, locals, loop indices, state),
providing unique variable creation, and interfacing with Z3 for variable representation
"""



from __future__ import annotations
from typing import Set, Dict, Any, Optional
from loguru import logger

import z3

from src.config.config import config
from src.core.cfg_content.variable import Variable, VariableType

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

Z3_SHARED_CONTEXT = None 

class CFGContext:
    """
    Maintains per-CFG state such as used variables, parameters, and loop variables
    Provides Z3 variable creation and ensures consistent naming/usage of variables
    """

    def __init__(self, cfg):
        """
        Initialize the CFG context with a required CFG instance

        Args:
            cfg: The control flow graph (required)
        """
        self.cfg = cfg
        self._used_variables: Set[Variable] = set()
        self._used_parameters: Set[Variable] = set()
        self._max_variables = config.expression_initializer.num_func_params
        self._used_loop_names = set()
        self._loop_variables: Dict[int, Variable] = {}
        self._z3_variables: Dict[str, Any] = {}
        self._next_state_var_idx = 0

    def get_z3_variable(self, var_name: str) -> Any:
        """
        Return a Z3 Int variable for the given var_name, creating it if needed

        Args:
            var_name: The variable name to convert into a Z3 variable

        Returns:
            The z3 variable object
        """
        z3_var = self._z3_variables.get(var_name)
        if z3_var is not None:
            return z3_var

        z3_var = z3.Int(var_name)
        self._z3_variables[var_name] = z3_var
        return z3_var

    def get_variable(self, var_name: str) -> Optional[Variable]:
        """
        Return a Variable object from the used variables by name

        Args:
            var_name: The name of the variable to find

        Returns:
            The matching Variable object if found, None otherwise
        """
        return next((var for var in self._used_variables if var.name == var_name), None)

    def ensure_parameters_initialized(self) -> bool:
        """
        Ensure parameters are initialized exactly once based on config

        Reads parameter names and count from the global config. Declares
        the required parameters and associated variables if not already done

        Returns:
            True if new parameters were created, False if they already existed
        """
        if self._used_parameters:
            return False

        params_names = config.expression_initializer.params_names
        num_params = config.expression_initializer.num_func_params

        selected_params_names = params_names[:num_params]

        for name in selected_params_names:
            var = Variable(name, VariableType.INT)
            self.declare_parameter(var)

        return True

    def declare_variable(self, var: Variable):
        """
        Declare a variable in the CFG context if not already present by name and type

        Adds the variable to the internal set and ensures a corresponding Z3 variable exists

        Args:
            var: The Variable object to declare
        """
        if any(v.name == var.name and v.type == var.type for v in self._used_variables):
            return

        self._used_variables.add(var)
        self.get_z3_variable(str(var.name))

    def declare_parameter(self, var: Variable):
        """
        Declare a parameter variable in the context, adding it to _used_parameters

        Args:
            var: The parameter Variable object
        """
        if var not in self._used_variables:
             logger.warning(f"Declaring parameter {var.name} which was not previously declared as a variable. Declaring it now.")
             self.declare_variable(var)

        self._used_parameters.add(var)

    def create_result_variable(self, var_name: str = "result", var_type: VariableType = VariableType.INT) -> Variable:
        """
        Creates and declares the standard 'result' variable for the function's return value

        Args:
            var_name: The name for the result variable (defaults to "result")
            var_type: The type for the result variable (defaults to INT)

        Returns:
            The newly created and declared result Variable
        """
        # Check if it already exists
        existing_var = self.get_variable(var_name)
        if existing_var and existing_var.type == var_type:
            logger.debug(f"Result variable '{var_name}' already exists.")
            return existing_var
        elif existing_var:
            logger.warning(f"Result variable '{var_name}' exists but with wrong type ({existing_var.type}). Re-creating with type {var_type}.")

        var = Variable(var_name, var_type)
        self.declare_variable(var)
        logger.info(f"Created and declared result variable '{var.name}' (type: {var.type})")
        return var

    def create_state_variable(self, var_type: VariableType = VariableType.INT) -> Variable:
        """
        Creates a new uniquely named state variable (e.g., 'state_0', 'state_1')

        Ensures the name is unique within the current context and declares
        the variable using `declare_variable`

        Args:
            var_type: The type for the new state variable (default: INT)

        Returns:
            The newly created and declared state Variable
        """
        while True:
            name = f"state_{self._next_state_var_idx}"
            if not any(v.name == name for v in self._used_variables):
                var = Variable(name, var_type)
                self.declare_variable(var)
                self._next_state_var_idx += 1
                logger.info(f"Created and declared state variable '{name}' (type: {var_type})")
                return var
            self._next_state_var_idx += 1

    def create_loop_variable(self, node_id: int) -> Variable:
        """
        Create a new loop variable (e.g., 'i', 'j', 'k') for a specific node ID

        Uses the first available name from a predefined list or generates an indexed name
        (e.g., 'i0', 'i1') if the standard names are taken. Declares the variable globally

        Args:
            node_id: The node ID requiring a loop variable (used for mapping)

        Returns:
            The newly created loop Variable
        """
        loop_var_names = ['i', 'j', 'k', 'm', 'n']

        for name in loop_var_names:
            if name not in self._used_loop_names:
                var = Variable(name, VariableType.INT)
                self._used_loop_names.add(name)
                self._loop_variables[node_id] = var
                self.declare_variable(var)
                return var

        idx = len(self._used_loop_names)
        name = f"i{idx}"
        while name in self._used_loop_names:
             idx += 1
             name = f"i{idx}"

        var = Variable(name, VariableType.INT)
        self._used_loop_names.add(name)
        self._loop_variables[node_id] = var
        self.declare_variable(var)
        return var

    # 1000 - 7 ??? 

    def get_loop_variable(self, node_id: int) -> Variable:
        """
        Return the loop variable associated with the given node ID, creating one if needed

        Args:
            node_id: The node ID requiring a loop variable

        Returns:
            The existing or newly created loop Variable
        """
        var = self._loop_variables.get(node_id)
        if var is not None:
            return var
        else:
            return self.create_loop_variable(node_id)


    def is_loop_variable(self, var: Variable) -> bool:
        """
        Check if a given variable is one of the context's loop variables

        Args:
            var: The variable to check

        Returns:
            True if it is a loop variable, False otherwise
        """
        return var in self._loop_variables.values()

    def clear_parameters(self):
        """
        Clear all parameter variables from this context (_used_parameters)
        Crucially, this also removes them from the general _used_variables set
        """
        params_to_remove = self._used_parameters.copy()
        if not params_to_remove:
            return

        self._used_parameters.clear()
        self._used_variables -= params_to_remove


    def create_random_parameters(self):
        """
        Re-initialize parameters based on the current configuration

        This first clears any existing parameters and then calls the
        deterministic initialization method
        """
        self.clear_parameters()
        self.ensure_parameters_initialized()

    @property
    def used_variables(self) -> Set[Variable]:
        """
        Return the set of all variables currently declared in this context
        """
        return self._used_variables

    @property
    def used_parameters(self) -> Set[Variable]:
        """
        Return the set of variables currently marked as parameters in this context
        """
        return self._used_parameters
     