from __future__ import annotations
import os
import sys
import subprocess
from loguru import logger
from typing import Optional, List

from src.core.cfg.cfg import CFG
from src.builders.flow_builder.flow_builder import FlowBuilder
from src.core.paths.test_paths_finder import find_and_set_test_paths
from src.builders.edge_cond_builder.edge_cond_builder import EdgeCondBuilder
from src.builders.edge_cond_builder.expression_random_factory import ExpressionRandomFactory
from src.builders.oper_builder.complexity_tuner import ComplexityTuner
from src.utils.program_output import ProgramOutput
from src.config.config import AppConfig

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

def run_command(cmd_list: List[str]) -> bool:
    """Runs a command using subprocess and logs the outcome"""
    cmd_str = ' '.join(cmd_list)
    logger.info(f"Executing validation command: {cmd_str}")
    try:
        result = subprocess.run(
            cmd_list, 
            check=True, 
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True, 
            timeout=300
        )
        logger.success(f"Command successful: {cmd_str}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {cmd_str}")
        logger.error(f"Return code: {e.returncode}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {cmd_str}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred executing {cmd_str}: {e}")
        return False

def reset_cfg() -> CFG:
    """Initialize a new CFG with an entry node

    Returns:
        CFG: The newly initialized CFG object
    """
    logger.info("Initializing new CFG...")
    cfg = CFG()
    cfg.init_entry_node()
    return cfg


def generate_random_structure(cfg: CFG, app_config: AppConfig) -> bool:
    """Generate a random control flow graph structure

    Uses the FlowBuilder to populate the provided CFG object with a randomly
    generated structure based on the application configuration

    Args:
        cfg (CFG): The CFG object to populate
        app_config (AppConfig): The application configuration settings

    Returns:
        bool: True indicating successful structure generation
    """
    logger.info("Generating random CFG structure...")
    builder = FlowBuilder(app_config=app_config)
    builder.build_random_cfg(cfg)
    return True


def find_test_paths(cfg: CFG, app_config: AppConfig) -> bool:
    """Find and set test paths for the CFG based on a coverage criterion

    Skips path finding if the CFG has 2 or fewer nodes Logs the number
    of paths found

    Args:
        cfg (CFG): The CFG object to analyze
        app_config (AppConfig): The application configuration, specifying the
            coverage criterion

    Returns:
        bool: True if test paths were found and set, False otherwise (e.g,
              if the CFG is too small)
    """
    if len(cfg.graph) <= 2:
        return False

    find_and_set_test_paths(cfg, app_config=app_config)
    path_count = len(cfg.test_paths) if cfg.test_paths else 0
    logger.info(f"Found {path_count} test paths")
    return True


def build_conditions(cfg: CFG, app_config: AppConfig) -> bool:
    """Build edge conditions for the CFG and check path feasibility

    Uses an EdgeCondBuilder and ExpressionRandomFactory to assign conditions
    to CFG edges Then, checks the feasibility of each test path and generates
    test inputs where possible Logs feasibility and input generation statistics

    Args:
        cfg (CFG): The CFG object containing test paths
        app_config (AppConfig): The application configuration

    Returns:
        bool: True if conditions were built successfully, False if no test paths
              exist or the builder fails
    """
    if not cfg.test_paths:
        return False

    expression_factory = ExpressionRandomFactory(cfg, app_config=app_config)
    edge_cond_builder = EdgeCondBuilder(cfg, expression_factory, app_config=app_config)
    
    if not edge_cond_builder.build_conditions():
        return False

    feasible = infeasible = inputs_generated = 0
    
    for path in cfg.test_paths:
        if path.check_feasibility_and_find_inputs():
            feasible += 1
            if path.test_inputs:
                inputs_generated += 1
        else:
            infeasible += 1
    
    logger.info(f"Inputs: {len(cfg.test_paths)} paths, {feasible} feasible, {inputs_generated} with inputs")
    
    display_inputs = 0
    for path in cfg.test_paths:
        if len(path.nodes) == len(set(path.nodes)):
            result = path.generate_inputs_for_display()
            if result is not None:
                path.test_inputs = result
                display_inputs += 1
    
    logger.info(f"Display inputs for {display_inputs} non-loopy paths")
    return True


def build_operations(cfg: CFG, app_config: AppConfig) -> bool:
    """Build operations within CFG nodes using complexity tuning

    Applies the ComplexityTuner to add operations to basic block nodes, aiming
    to meet complexity targets defined in the application configuration Logs
    the final SHTV values

    Args:
        cfg (CFG): The CFG object with defined structure and potentially
                   conditions
        app_config (AppConfig): The application configuration containing tuning
                           parameters

    Returns:
        bool: True if operations were built successfully, False if no test paths
              exist
    """
    if not cfg.test_paths:
        return False

    tuner = ComplexityTuner(cfg, app_config=app_config)
    tuner.tune_complexity()
    logger.info(f"SHTV: {cfg.shtv:.2f}/{cfg.max_shtv:.2f}")
    return True


def save_outputs(cfg: CFG, config: AppConfig, output_dir: Optional[str] = None) -> bool:
    """Save the generated CFG, code, and related outputs to a directory

    Uses the ProgramOutput class to bundle and save the generated artifacts
    Logs the path where the outputs were saved

    Args:
        # cfg (CFG): The fully generated CFG object
        config (AppConfig): The application configuration
        output_dir (Optional[str]): The directory to save outputs to If None,
                                   a default or configured directory is used

    Returns:
        bool: True if the outputs were saved successfully, False otherwise
    """
    program_output = ProgramOutput(cfg=cfg, config=config)
    saved_path = program_output.save_to_directory(output_dir=output_dir)
    
    if not saved_path:
        return False
        
    logger.info(f"Saved to: {saved_path}")

    return True


def validate_saved_outputs(config: AppConfig, saved_path: str) -> bool:
    """Validates the saved outputs by running generate_tests and pytest"""
    logger.info(f"--- Running Output Validation for: {saved_path} ---")
    generate_tests_cmd = [
        sys.executable,
        "-m", "src.utils.generate_tests",
        saved_path
    ]
    pytest_cmd = [
        sys.executable,
        "-m", "pytest",
        os.path.join(saved_path, "test_suite.py")
    ]

    verification_passed = False
    if run_command(generate_tests_cmd):
        if run_command(pytest_cmd):
            logger.success("Output validation successful.")
            verification_passed = True
        else:
            logger.error("Pytest verification failed.")
    else:
        logger.error("generate_tests script failed.")

    return verification_passed 