"""Manages application configuration using dataclasses and YAML loading"""
from __future__ import annotations
import yaml
import random
from string import ascii_lowercase

from loguru import logger
from typing import Dict, Any, Optional, List, ClassVar
from pathlib import Path
from dataclasses import dataclass, field, asdict
from functools import lru_cache

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

logger.remove() 

@dataclass(frozen=True)
class DeveloperOptions:
    """Developer options configuration"""
    debug: bool = True
    logging_levels: List[str] = field(
        default_factory=lambda: ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    )
    show_shtv: bool = True
    log_node_ids_in_code: bool = False
    log_branch_labels_in_code: bool = False
    flow_builder_log: bool = False
    log_fragment_forest: bool = False
    log_edge_cond_builder: bool = False
    log_loop_term: bool = False
    log_test_path: bool = False
    validate_outputs: bool = False
    show_rich: bool = False


@dataclass(frozen=True)
class VisualizationConfig:
    """Visualization configuration"""
    traversal_method: str = "bfs"
    node_labels: List[str] = field(default_factory=lambda: ["type"])
    show_edge_conditions: bool = False


@dataclass(frozen=True)
class CFGBuilderConfig:
    """CFG Builder configuration"""
    total_cyclomatic_complexity: Optional[int] = None
    nesting_probability: Optional[float] = None
    max_nesting_depth: Optional[int] = None
    required_fragments: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        all_fragment_types = ["if", "ifelse", "while", "for"]
        if not self.required_fragments:
            object.__setattr__(self, "required_fragments", all_fragment_types.copy())
        else:
            valid_fragments = [f for f in self.required_fragments if f in all_fragment_types]
            object.__setattr__(self, "required_fragments", valid_fragments)


@dataclass(frozen=True)
class TestPathsFinderConfig:
    __test__ = False          
    """Test Paths Finder configuration"""
    coverage_criterion: str = "EPC"


@dataclass(frozen=True)
class ExpressionInitializerConfig:
    """Expression Initializer configuration"""
    num_func_params: Optional[int] = None  
    value_range: List[int] = field(default_factory=lambda: [-100, 100])
    params_names: List[str] = field(default_factory=lambda: [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
    ])
    allowed_comparison_operators: List[str] = field(default_factory=lambda: [">", ">=", "<", "<=", "==", "!="])
    allowed_expressions: List[str] = field(default_factory=lambda: ["logical_comparison", "compound_condition"])


@dataclass(frozen=True)
class ComplexityTunerConfig:
    """Complexity Tuner configuration"""
    # Number of dedicated state variables to create 
    num_state_vars: int = 3 
    # Probability (0.0-1.0) of nesting operands in expressions
    operand_nesting_probability: float = 0.3
    allowed_operators: List[str] = field(default_factory=lambda: ["+", "-", "*", "/", "%"])
    allowed_operations: List[str] = field(default_factory=lambda: [
        "increment", "decrement", "prefix_increment", "prefix_decrement",
        "assignment", "addition_assign", "subtraction_assign", "multiplication_assign",
        "division_assign", "modulo_assign", "complex_arithmetic", "simple_arithmetic",
        "mixed_arithmetic", "logical_and", "logical_or", "logical_not", "complex_logical"
    ])
    # Pool of variable names to choose from for generated 'for' loops
    for_loop_variable_names: List[str] = field(default_factory=lambda: list(ascii_lowercase))


@dataclass
class AppConfig:
    """
    Main application configuration
    
    This class centralizes all configuration settings and handles
    loading from external files
    """
    # Global parameters
    seed: Optional[int] = None
    test_input_complexity: int = 1
    weights: Dict[str, float] = field(default_factory=lambda: {
        "a": 1, "b": 3, "c": 1, "d": 1, "e": 3,
        "f": 1, "g": 1, "h": 1, "i": 5, "k": 2
    })
    
    # SHTV Ceiling Assignment Parameters
    shtv_alpha: float = 0.40  # Decay factor for node depth
    shtv_beta: float = 0.50   # Decay factor for node position
    
    # Nested configurations - immutable after initialization
    developer_options: DeveloperOptions = field(default_factory=DeveloperOptions)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    cfg_builder: CFGBuilderConfig = field(default_factory=CFGBuilderConfig)
    test_paths_finder: TestPathsFinderConfig = field(default_factory=TestPathsFinderConfig)
    expression_initializer: ExpressionInitializerConfig = field(default_factory=ExpressionInitializerConfig)
    complexity_tuner: ComplexityTunerConfig = field(default_factory=ComplexityTunerConfig)
    
    # Constants defined as class variables
    DEFAULT_CONFIG_PATHS: ClassVar[List[str]] = [
        "config.yaml",
        "src/config.yaml",
        "src/config/config.yaml",
        "../config.yaml",
    ]
    
    DEFAULT_COSTS_PATHS: ClassVar[List[str]] = [
        "costs.yaml",
        "src/model/CFGFactory/costs.yaml",
    ]
    
    def __post_init__(self) -> None:
        """Initialize configuration after instance creation"""
        self._load_config()
        logger.info(f"Configuration initialized. Using seed: {self.seed}")
    
    def _find_file(self, potential_paths: List[str]) -> Optional[str]:
        """Find the first existing file from a list of potential paths"""
        module_dir = Path(__file__).parent
        all_paths = [
            Path(path) for path in potential_paths
        ] + [
            module_dir / f"../../{path}" for path in potential_paths
        ]
        
        for path in all_paths:
            if path.exists():
                return str(path)
        return None
    
    def _load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file"""
        config_data: Dict[str, Any] = {}
        try:
            # Determine config path
            if config_path:
                self._config_path = Path(config_path)
            else:
                found_path = self._find_file(self.DEFAULT_CONFIG_PATHS)
                if found_path:
                    self._config_path = Path(found_path)
                else:
                    logger.warning("Configuration file not found, using defaults")
                    self._config_path = None # Indicate no file was loaded

            if self._config_path and self._config_path.exists():
                with open(self._config_path, "r", encoding='utf-8') as f:
                    loaded_config_data = yaml.safe_load(f)
                if not loaded_config_data:
                    logger.warning("Empty configuration file, using defaults")
                else:
                    config_data = loaded_config_data 
            elif not self._config_path:
                 logger.warning("Configuration file not found, using defaults")
            
        except FileNotFoundError:
             logger.warning(f"Configuration file not found at path: {self._config_path}. Using defaults.")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}. Using default values.")
            config_data = {} 

        self._update_from_dict(config_data)
        logger.info(f"Loaded config_data from YAML: {config_data}")

    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration using values from loaded dictionary"""
        # Update top-level simple fields first (needed for dependent defaults)
        if 'test_input_complexity' in config_data:
            self.test_input_complexity = config_data['test_input_complexity']
        if 'seed' in config_data:
            self.seed = config_data['seed']
        if 'shtv_alpha' in config_data:
            self.shtv_alpha = float(config_data['shtv_alpha'])
        if 'shtv_beta' in config_data:
            self.shtv_beta = float(config_data['shtv_beta'])
        
        if 'weights' in config_data:
            self.weights.update(config_data['weights'])
            
        self._update_nested_config('developer_options', DeveloperOptions, config_data)
        self._update_nested_config('visualization', VisualizationConfig, config_data)
        self._update_nested_config('cfg_builder', CFGBuilderConfig, config_data)
        self._update_nested_config('test_paths_finder', TestPathsFinderConfig, config_data)
        self._update_nested_config('expression_initializer', ExpressionInitializerConfig, config_data)
        self._update_nested_config('complexity_tuner', ComplexityTunerConfig, config_data)

        #  Apply dependent defaults  
        self._update_dependent_defaults(source="initial load", config_data_for_init=config_data)

        # Generate random seed only if not loaded and not set by default
        if getattr(self, 'seed', None) is None:
            generated_seed = random.randint(1, 1000000)
            self.seed = generated_seed
            logger.info(f"Seed not found in config file, generated random seed: {self.seed}")

    def _update_dependent_defaults(self, source: str = "update", config_data_for_init: Optional[Dict[str, Any]] = None) -> None:
        """
        Recalculates and sets default values for configurations that depend 
        on test_input_complexity
        Uses object.__setattr__ for frozen nested dataclasses
        Logs which defaults are being set and why

        Args:
            source: Indicates the trigger ('initial load' or 'CLI override')
            config_data_for_init: The loaded YAML data, used only during initial load
                                    to check if a value was explicitly set
        """
        logger.debug(f"Updating dependent defaults (triggered by {source}) based on test_input_complexity={self.test_input_complexity}")
        
        # total_cyclomatic_complexity 
        cc_map = {1: 3, 2: 5, 3: 7, 4: 9}
        default_cc = cc_map.get(self.test_input_complexity, cc_map[1])
        current_cc = getattr(self.cfg_builder, 'total_cyclomatic_complexity', None)
        
        apply_default = False
        if source == "CLI override":
            apply_default = True # Always apply default on CLI override
            if current_cc is not None and current_cc != default_cc:
                 logger.info(f"[{source}] Overwriting existing total_cyclomatic_complexity ({current_cc}) with default {default_cc} for level {self.test_input_complexity}.")
        elif source == "initial load" and current_cc is None:
             apply_default = True # Apply if not set during initial load
             logger.info(f"[{source}] Setting default total_cyclomatic_complexity to {default_cc} for level {self.test_input_complexity}.")
             
        if apply_default:
            object.__setattr__(self.cfg_builder, 'total_cyclomatic_complexity', default_cc)
        elif source == "initial load" and current_cc is not None:
             logger.debug(f"[{source}] Kept explicitly set total_cyclomatic_complexity ({current_cc}).")

        # nesting_probability 
        prob_map = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6}
        default_prob = prob_map.get(self.test_input_complexity, prob_map[1])
        current_prob = getattr(self.cfg_builder, 'nesting_probability', None)
        
        apply_default = False
        if source == "CLI override":
            apply_default = True
            if current_prob is not None and current_prob != default_prob:
                 logger.info(f"[{source}] Overwriting existing nesting_probability ({current_prob:.1f}) with default {default_prob:.1f} for level {self.test_input_complexity}.")
        elif source == "initial load" and current_prob is None:
             apply_default = True
             logger.info(f"[{source}] Setting default nesting_probability to {default_prob:.1f} for level {self.test_input_complexity}.")

        if apply_default:
             object.__setattr__(self.cfg_builder, 'nesting_probability', default_prob)
        elif source == "initial load" and current_prob is not None:
             logger.debug(f"[{source}] Kept explicitly set nesting_probability ({current_prob:.1f}).")

        #  max_nesting_depth 
        depth_map = {1: 1, 2: 2, 3: 3, 4: 3}
        default_depth = depth_map.get(self.test_input_complexity, 1)
        current_depth = getattr(self.cfg_builder, 'max_nesting_depth', None)
        
        apply_default = False
        if source == "CLI override":
            apply_default = True
            if current_depth is not None and current_depth != default_depth:
                 logger.info(f"[{source}] Overwriting existing max_nesting_depth ({current_depth}) with default {default_depth} for level {self.test_input_complexity}.")
        elif source == "initial load" and current_depth is None:
             apply_default = True
             logger.info(f"[{source}] Setting default max_nesting_depth to {default_depth} for level {self.test_input_complexity}.")

        if apply_default:
             object.__setattr__(self.cfg_builder, 'max_nesting_depth', default_depth)
        elif source == "initial load" and current_depth is not None:
             logger.debug(f"[{source}] Kept explicitly set max_nesting_depth ({current_depth}).")

        max_available_params = len(self.expression_initializer.params_names)
        max_sensible_params = 4
        upper_bound = min(max_available_params, max_sensible_params)
        default_num_params = max(1, min(upper_bound, self.test_input_complexity))
        current_params = getattr(self.expression_initializer, 'num_func_params', None)
        
        apply_default = False
        if source == "CLI override":
            apply_default = True
            if current_params is not None and current_params != default_num_params:
                 logger.info(f"[{source}] Overwriting existing num_func_params ({current_params}) with default {default_num_params} for level {self.test_input_complexity}.")
        elif source == "initial load" and current_params is None:
             apply_default = True
             logger.info(f"[{source}] Setting default num_func_params to {default_num_params} for level {self.test_input_complexity}.")

        if apply_default:
             object.__setattr__(self.expression_initializer, 'num_func_params', default_num_params)
        elif source == "initial load" and current_params is not None:
             logger.debug(f"[{source}] Kept explicitly set num_func_params ({current_params}).")

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Applies CLI overrides to the configuration and recalculates dependent defaults
        """
        if not overrides:
            return

        logger.info("Applying CLI overrides...")
        updated = False
        complexity_changed = False 

        if 'seed' in overrides and overrides['seed'] is not None:
             self.seed = overrides['seed']
             random.seed(self.seed)
             logger.info(f"CLI Override: Applied seed {self.seed}")
             updated = True
             
        if 'test_input_complexity' in overrides and overrides['test_input_complexity'] is not None:
             new_complexity = overrides['test_input_complexity']
             if self.test_input_complexity != new_complexity:
                  self.test_input_complexity = new_complexity
                  logger.info(f"CLI Override: Set test_input_complexity to {self.test_input_complexity}")
                  updated = True 
                  complexity_changed = True 
             else:
                  logger.debug(f"CLI Override: test_input_complexity already set to {new_complexity}, no change.")

        if 'coverage_criterion' in overrides and overrides['coverage_criterion'] is not None:
            new_criterion = overrides['coverage_criterion']
            if self.test_paths_finder.coverage_criterion != new_criterion:
                 object.__setattr__(self.test_paths_finder, 'coverage_criterion', new_criterion)
                 logger.info(f"CLI Override: Set coverage_criterion to {new_criterion}")
                 updated = True 
            else:
                 logger.debug(f"CLI Override: coverage_criterion already set to {new_criterion}, no change.")

        # Recalculate dependent defaults IF complexity changed 
        if complexity_changed:
            logger.info("Recalculating dependent defaults due to test_input_complexity override...")
            self._update_dependent_defaults(source="CLI override")
        
        if updated:
             logger.info("CLI overrides applied successfully.")
        else:
             logger.info("No applicable CLI overrides found or values matched current config.")

    def _update_nested_config(self, key: str, config_class: type, config_data: Dict[str, Any]) -> None:
        """Update a nested configuration by creating a new instance with updated values"""
        current_values = asdict(config_class()) 
        if key in config_data and isinstance(config_data[key], dict):
            current_values.update(config_data[key])
        logger.debug(f"Creating {config_class.__name__} instance with values: {current_values}")
        setattr(self, key, config_class(**current_values))
    
    @property
    def global_max_shtv(self) -> int:
        """Get the maximum SHTV for the current test input oper_builder level"""
        return {
            1: 250,
            2: 500,
            3: 750,
            4: 1000
        }.get(self.test_input_complexity, 250)
    
    def get_construction_config(self, cfg: Any) -> Dict[str, Any]:
        """Create a construction configuration context for a given CFG"""
        return {
            "cfg": cfg,
            "arithmetic_operators": self.complexity_tuner.allowed_operators,
            "comparison_operators": self.expression_initializer.allowed_comparison_operators,
            "value_range": self.expression_initializer.value_range,
            "allowed_operations": self.complexity_tuner.allowed_operations,
        }
    
    def get_node_label(self, node_id: int, default_label: str, node: Optional[Any] = None) -> str:
        """Get node label based on labeling method configuration"""
        # Start with the default label
        label = default_label
        
        if self.developer_options.show_shtv and node and hasattr(node, 'shtv') and hasattr(node, 'max_shtv'):
            label = f"{label}\nSHTV: {node.shtv:.2f}\nMax: {node.max_shtv:.2f}"
        
        return label
    
    def reload(self, config_path: Optional[str] = None) -> 'AppConfig':
        """Reload configuration from file"""
        self._load_config(config_path)
        return self

@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Get the singleton AppConfig instance"""
    instance = AppConfig()
    return instance

config = get_config()

if config.seed is not None:
    random.seed(config.seed)
    logger.info(f"Applied global random seed: {config.seed}")

