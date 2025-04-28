"""
Module for representing sets of logical constraints (Expressions)
associated with specific program points (e.g., CFG nodes)

These constraint sets (like MustBeSet) are intended to be populated
based on path conditions and used by other components (e.g., ComplexityTuner)
to guide decisions like operation generation or placement, ensuring
that added logic is consistent with the established path constraints
"""

from __future__ import annotations

from abc import ABC
from collections.abc import MutableSet
from typing import Set

from loguru import logger

from src.core.cfg_content.expression import Expression

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

class ConstraintSet(MutableSet, ABC):
    """Base implementation of a constraint set
    
    Stores Expression objects
    """
    
    def __init__(self):
        """Initialize the constraint set"""
        self._constraints: Set[Expression] = set()
        
    def __contains__(self, item: Expression) -> bool:
        return item in self._constraints
    
    def __iter__(self):
        return iter(self._constraints)
    
    def __len__(self) -> int:
        return len(self._constraints)
    
    def add(self, constraint: Expression) -> None:
        """Add a constraint to the set"""
        if not hasattr(constraint, 'to_c') or not callable(getattr(constraint, 'to_c')):
             logger.warning(f"Attempted to add object of type {type(constraint)} which does not seem to be a valid Expression to ConstraintSet. Skipping.")
             return
            
        self._constraints.add(constraint)
        
    def clear(self) -> None:
        """Remove all constraints from the set"""
        self._constraints.clear()

    def discard(self, constraint: Expression) -> None:
        """Remove a constraint if it exists in the set"""
        if constraint in self._constraints:
            self._constraints.remove(constraint)

    def __str__(self) -> str:
        """String representation of the constraint set"""
        constraints_str = ", ".join(str(c) for c in self._constraints)
        return f"{{{constraints_str}}}"

class MustBeSet(ConstraintSet):
    """Set of constraints that must be true at a specific program point (node)
    
    Intended to be populated based on incoming path conditions and used by
    other builders (like ComplexityTuner) to validate or guide the addition
    of new operations
    """
    
    def __init__(self):
        """Initialize a must-be constraint set"""
        super().__init__()

