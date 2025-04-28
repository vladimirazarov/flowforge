"""Defines roles for CFG fragments (e.g., If, While, For)"""
from enum import Enum

class FragmentRole(Enum):
    IF = "if_fragment"
    IF_ELSE = "if_else_fragment"
    WHILE = "while_fragment"
    FOR = "for_fragment"
    FOR_EARLY_EXIT = "for_fragment_early_exit"
    SWITCH = "switch_fragment" 