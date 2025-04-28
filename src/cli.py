"""Provide a command-line interface for the FlowForge CFG pipeline

Exposes the 'pipeline' command to run generation steps sequentially
Allows overriding configuration parameters like seed and complexity
For interactive control, use 'tui.py'
"""
from __future__ import annotations
import argparse
import sys
from loguru import logger
from typing import Optional

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
import rich.box

from src.core.cfg.cfg import CFG
from src.utils.pipeline_logic import reset_cfg, generate_random_structure, find_test_paths, build_conditions, build_operations, validate_saved_outputs
from src.utils.program_output import ProgramOutput
from src.config.config import config

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

def handle_pipeline(args: argparse.Namespace):
    """Run the CFG generation pipeline sequentially based on CLI arguments

    Applies configuration overrides from args, executes pipeline stages
    up to the specified point, and saves the results

    Args:
        args (argparse.Namespace): Parsed command-line arguments
    """
    console = Console()
    console.print("[bold blue]Starting full pipeline execution...[/bold blue]")
    overrides = {}
    if args.seed is not None:
        overrides['seed'] = args.seed
    if args.test_input_complexity is not None:
        overrides['test_input_complexity'] = args.test_input_complexity
    if args.coverage_criterion is not None:
        overrides['coverage_criterion'] = args.coverage_criterion

    logger.debug(f"CLI: Before overrides - config ID: {id(config)}, Complexity: {config.test_input_complexity}, CC: {config.cfg_builder.total_cyclomatic_complexity}, Seed: {config.seed}")

    initial_dependent_values = {
        'total_cyclomatic_complexity': config.cfg_builder.total_cyclomatic_complexity,
        'nesting_probability': config.cfg_builder.nesting_probability,
        'max_nesting_depth': config.cfg_builder.max_nesting_depth,
        'num_func_params': config.expression_initializer.num_func_params
    }
    initial_complexity = config.test_input_complexity

    if overrides:
        console.print(Rule("[cyan]Processing CLI Overrides[/cyan]"))
        config.apply_overrides(overrides)

        logger.debug(f"CLI: After overrides - config ID: {id(config)}, Complexity: {config.test_input_complexity}, CC: {config.cfg_builder.total_cyclomatic_complexity}, Seed: {config.seed}")

        if 'seed' in overrides: console.print(f"  [magenta]Applied Override:[/magenta] Seed = {config.seed}")
        if 'test_input_complexity' in overrides: console.print(f"  [magenta]Applied Override:[/magenta] Complexity Level = {config.test_input_complexity}")
        if 'coverage_criterion' in overrides: console.print(f"  [magenta]Applied Override:[/magenta] Coverage Criterion = {config.test_paths_finder.coverage_criterion}")
        console.print(Rule())

    final_dependent_values = {
        'total_cyclomatic_complexity': config.cfg_builder.total_cyclomatic_complexity,
        'nesting_probability': config.cfg_builder.nesting_probability,
        'max_nesting_depth': config.cfg_builder.max_nesting_depth,
        'num_func_params': config.expression_initializer.num_func_params
    }
    final_complexity = config.test_input_complexity

    changed_params = []
    for key, final_val in final_dependent_values.items():
        initial_val = initial_dependent_values.get(key)

        if isinstance(initial_val, float):
            initial_val_str = f"{initial_val:.1f}"
        else:
            initial_val_str = str(initial_val)

        if isinstance(final_val, float):
            final_val_str = f"{final_val:.1f}"
        else:
            final_val_str = str(final_val)

        if initial_val_str != final_val_str:
            changed_params.append((key.replace('_', ' ').title(), initial_val_str, final_val_str))

    if changed_params or initial_complexity != final_complexity:
        title = f"⚙️ Settings Derived from Complexity Level [bold cyan]{final_complexity}[/bold cyan]"
        changes_table = Table(title=title, show_header=True, header_style="bold magenta", box=rich.box.ROUNDED)
        changes_table.add_column("Parameter", style="dim cyan", width=30)
        changes_table.add_column("Initial Value", style="yellow")
        changes_table.add_column("Final Value", style="green")

        if not changed_params:
             changes_table.add_row("[i]No dependent parameters changed value.[/i]", "-", "-")
        else:
            for name, initial, final in changed_params:
                changes_table.add_row(name, initial, final)

        console.print(changes_table)
        console.print(Rule())

    cfg: Optional[CFG] = None
    pipeline_success = True

    console.print(Rule("[cyan]Stage 1: Reset CFG[/cyan]"))
    cfg = reset_cfg()
    if not cfg:
        console.print("[bold red]Pipeline aborted: Failed to reset CFG.[/bold red]")
        pipeline_success = False
        return
    console.print("[green]  -> CFG Reset Complete.[/green]")
    if args.upto == 'reset':
        console.print(f"[yellow]Pipeline finished after 'reset' stage as requested.[/yellow]")
        pipeline_success = False

    if pipeline_success:
        console.print(Rule("[cyan]Stage 2: Generate Structure[/cyan]"))
        success_gen = generate_random_structure(cfg, config)
        if not success_gen:
            console.print("[bold red]Pipeline aborted: Failed to generate structure.[/bold red]")
            pipeline_success = False
        else:
            console.print("[green]  -> Structure Generation Complete.[/green]")
        if args.upto == 'generate':
            console.print(f"[yellow]Pipeline finished after 'generate' stage as requested.[/yellow]")
            pipeline_success = False

    if pipeline_success:
        console.print(Rule("[cyan]Stage 3: Find Paths[/cyan]"))
        success_paths = find_test_paths(cfg, config)
        if not success_paths:
            console.print("[yellow]Warning: Failed to find test paths (error occurred or graph too simple).[/yellow]")
        elif not cfg.test_paths:
             console.print("[yellow]Warning: Test path finding complete, but no paths identified.[/yellow]")
        else:
             console.print(f"[green]  -> Found {len(cfg.test_paths)} test paths.[/green]")
        if args.upto == 'paths':
            console.print(f"[yellow]Pipeline finished after 'paths' stage as requested.[/yellow]")
            pipeline_success = False

    if pipeline_success:
        console.print(Rule("[cyan]Stage 4: Build Conditions[/cyan]"))
        success_cond = build_conditions(cfg, config)
        if not success_cond:
             if not hasattr(cfg, 'test_paths') or not cfg.test_paths:
                 console.print("[bold red]Pipeline aborted: Failed to build conditions likely due to missing test paths.[/bold red]")
                 pipeline_success = False
             else:
                 console.print("[yellow]Warning: Failed to build feasible conditions.[/yellow]")
        else:
             console.print("[green]  -> Condition Building Complete.[/green]")
        if args.upto == 'conditions':
            console.print(f"[yellow]Pipeline finished after 'conditions' stage as requested.[/yellow]")
            pipeline_success = False

    if pipeline_success and (args.upto is None or args.upto == 'operations' or args.upto == 'save'):
        console.print(Rule("[cyan]Stage 5: Build Operations[/cyan]"))
        success_ops = build_operations(cfg, config)
        if not success_ops:
            console.print("[bold red]Pipeline finished with errors: Failed to build operations.[/bold red]")
            pipeline_success = False
        else:
            console.print("[green]  -> Operation Building Complete.[/green]")
            if args.upto is None or args.upto == 'operations':
                 pipeline_success = True

    if args.upto and args.upto != 'operations':
        pipeline_success = False

    if cfg:
        console.print(Rule("[cyan]Saving Results[/cyan]"))
        output_manager = ProgramOutput(cfg, config)
        save_path = output_manager.save_to_directory(output_dir=args.output_dir if hasattr(args, 'output_dir') else None)
        if save_path:
            console.print(f"[bold green]Results saved to:[/bold green] {save_path}")
            # Conditionally validate outputs if enabled in config
            if config.developer_options.validate_outputs:
                logger.debug(f"Checking config.developer_options.validate_outputs: {config.developer_options.validate_outputs}")
                validation_result = validate_saved_outputs(config, str(save_path))
                if not validation_result:
                    logger.warning("Output validation failed after saving. Check logs for details.")
                    console.print("[bold yellow]⚠️ Warning: Output validation failed.[/bold yellow]")
        else:
            console.print("[bold red]Failed to save results.[/bold red]")

        console.print(f"  [dim]Final CFG SHTV: {cfg.shtv:.2f} (Target: {cfg.max_shtv:.2f})[/dim]")

    else:
        console.print("[yellow]Pipeline ended without a valid CFG object. Cannot save results.[/yellow]")

    console.print(Rule("[bold blue]Pipeline Summary[/bold blue]"))
    if pipeline_success:
         console.print("[bold green]✅ Pipeline process completed successfully.[/bold green]")
    else:
         console.print("[bold yellow]⚠️ Pipeline process finished with warnings, errors, or was incomplete.[/bold yellow]")

def main():
    """Parse CLI arguments and execute the requested pipeline command"""
    logger.remove()

    parser = argparse.ArgumentParser(
        description="FlowForge CFG Generation CLI. Runs pipeline steps. For interactive use, run `python -m flowforge.src.tui`.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
"""Available Commands:
  pipeline    Run the full CFG generation pipeline sequentially.

Pipeline Options (when using the 'pipeline' command):
  --upto {reset,generate,paths,conditions,operations}
              Run pipeline up to this stage (default: operations).
  --seed SEED Override random seed from config.yaml.
  --test-input-complexity LEVEL
              Override test_input_complexity (1-4) from config.yaml.
  --coverage-criterion {NC,EC,EPC,PPC}
              Override coverage_criterion from config.yaml.
  --output-dir DIR
              Specify output directory for results.
  --log       Enable detailed loguru logging to stderr.

Example: python -m src.cli pipeline --seed 42 --test-input-complexity 3 --coverage-criterion EC --log
"""
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    pipeline_stages = ['reset', 'generate', 'paths', 'conditions', 'operations']
    coverage_criteria = ['NC', 'EC', 'EPC', 'PPC']
    parser_pipeline = subparsers.add_parser(
        'pipeline',
        help='Run the full CFG generation pipeline sequentially, optionally stopping at an intermediate stage. Allows overriding key configuration parameters.'
    )
    parser_pipeline.add_argument(
        '--upto',
        choices=pipeline_stages,
        default='operations',
        help=f"Run pipeline steps sequentially up to and including this stage (default: %(default)s). Choices: {', '.join(pipeline_stages)}"
    )
    parser_pipeline.add_argument('--seed', type=int, help='Override the random seed from config.yaml for deterministic generation.')
    parser_pipeline.add_argument(
        '--test-input-complexity',
        type=int,
        choices=[1, 2, 3, 4],
        help='Override the main test_input_complexity level (1-4) from config.yaml.'
    )
    parser_pipeline.add_argument(
        '--coverage-criterion',
        type=str,
        choices=coverage_criteria,
        help=f"Override the coverage_criterion from config.yaml. Choices: {', '.join(coverage_criteria)}"
    )
    parser_pipeline.add_argument(
        '--output-dir',
        type=str,
        help='Specify a directory to save the run outputs (summary.pdf, cfg.svg, etc.). Overrides default cfgs/run<N> behavior.'
    )
    parser_pipeline.add_argument(
        '--log',
        action='store_true',
        help='Enable detailed loguru logging to stderr (in addition to rich console output).'
    )
    parser_pipeline.set_defaults(func=handle_pipeline)

    args = parser.parse_args()

    if hasattr(args, 'log') and args.log:
        print("--- Enabling Loguru Logging to stderr ---", file=sys.stderr)
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
            colorize=True
        )
        logger.info("Loguru logging enabled.")

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()