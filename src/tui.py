from __future__ import annotations

from typing import Optional
import time 
import json

from loguru import logger 
from dataclasses import asdict, is_dataclass 
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, RichLog
from textual.containers import Container
from textual import work

from src.core.cfg.cfg import CFG
from src.utils.pipeline_logic import reset_cfg, generate_random_structure, find_test_paths, build_conditions, build_operations, save_outputs
from src.config.config import config

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

class TuiApp(App):
    """The main Textual application for FlowForge TUI"""
    TITLE = "FlowForge TUI" 
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "reset_cfg", "Reset CFG"),
        ("g", "generate_cfg", "Generate Random"),
        ("f", "find_paths", "Find Paths"),
        ("c", "build_conditions", "Build Conditions"),
        ("o", "build_operations", "Build Operations"),
        ("d", "display_cfg", "Display CFG State"),
        ("s", "show_code", "Show Code"),
        ("p", "show_test_paths", "Show Paths"),
        ("x", "show_config", "Show Config"), 
        ("a", "save_outputs", "Save All Outputs"), 
    ]
    current_cfg: Optional[CFG] = None
    def compose(self) -> ComposeResult:
        """Create child widgets for the app"""
        yield Header()
        with Container(id="main-container"):
            yield Static("Pipeline Status: Idle", id="status-line")
            yield RichLog(id="output-log", auto_scroll=True, highlight=True, markup=True)
        yield Footer()

    def write_log(self, message: str, level: str = "INFO") -> None:
        """Helper to write styled messages to the RichLog widget"""
        is_debug_mode = False 
        if level == "DEBUG" and not is_debug_mode:
            return
        try:
            log_widget = self.query_one(RichLog)
            if level == "ERROR":
                log_widget.write(f"[bold red]ERROR:[/bold red] {message}")
            elif level == "WARNING":
                log_widget.write(f"[bold yellow]WARNING:[/bold yellow] {message}")
            elif level == "SUCCESS":
                 log_widget.write(f"[bold green]SUCCESS:[/bold green] {message}")
            elif level == "DEBUG":
                 log_widget.write(f"[dim blue]DEBUG:[/dim] {message}") 
            else: 
                 if message.startswith("Calling"):
                     log_widget.write(f"{message}") 
                 else:
                     log_widget.write(f"[bold]INFO:[/bold] {message}") 
        except Exception as e:
             print(f"TUI Log Error ({level}): {e} - Message: {message}")

    def set_status(self, message: str) -> None:
         """Helper to update the status line"""
         try:
              status_widget = self.query_one("#status-line", Static)
              status_widget.update(f"Pipeline Status: {message}")
         except Exception as e:
              self.write_log(f"TUI Status Error: {e}", "ERROR")

    def _format_cfg_state(self, cfg: Optional[CFG]) -> str:
        if cfg is None:
            return "No CFG loaded."
        nodes_count = len(list(cfg.graph.nodes())) if cfg.graph else 0
        edges_count = len(list(cfg.graph.edges())) if cfg.graph else 0
        cc = getattr(cfg, 'cc', 'N/A')
        shtv = getattr(cfg, 'shtv', 'N/A')
        max_shtv = getattr(cfg, 'max_shtv', 'N/A')
        shtv_str = f"{shtv:.2f}" if isinstance(shtv, (int, float)) else str(shtv)
        max_shtv_str = f"{max_shtv:.2f}" if isinstance(max_shtv, (int, float)) else str(max_shtv)
        state_str = (
            f"[bold]Current CFG State:[/bold]\n"
            f"  Nodes: {nodes_count}\n"
            f"  Edges: {edges_count}\n"
            f"  CC: {cc}\n"
            f"  SHTV: {shtv_str}\n"
            f"  Max SHTV Target: {max_shtv_str}"
        )
        return state_str

    def _format_config_dict(self, config_dict: dict, indent: int = 0) -> str:
        """Recursively formats a dictionary for display"""
        lines = []
        indent_str = "  " * indent
        for key, value in config_dict.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}[bold cyan]{key}:[/bold cyan]")
                lines.append(self._format_config_dict(value, indent + 1))
            elif is_dataclass(value) and not isinstance(value, type):
                lines.append(f"{indent_str}[bold cyan]{key}:[/bold cyan]")
                lines.append(self._format_config_dict(asdict(value), indent + 1))
            elif isinstance(value, list):
                 lines.append(f"{indent_str}[bold cyan]{key}:[/bold cyan]")
                 max_items = 10
                 list_items_str = [str(item) for item in value[:max_items]]
                 if len(value) > max_items:
                     list_items_str.append(f"... ({len(value) - max_items} cfg_content)")
                 formatted_list = ", ".join(list_items_str)
                 lines.append(f"{indent_str}  [{formatted_list}]") 
            else:
                lines.append(f"{indent_str}[bold cyan]{key}:[/bold cyan] {value}")
        return "\n".join(lines)

    def action_quit(self) -> None:
        """Called when the user presses 'q' or clicks a Quit button"""
        self.write_log("Exiting TUI.") 
        time.sleep(0.1)
        self.exit()
    @work(exclusive=True)

    async def action_reset_cfg(self) -> None:
        """Handles the 'Reset CFG' action using pipeline_logic"""
        self.set_status("Resetting CFG...")
        self.write_log("Calling reset_cfg()...") 
        try:
            self.current_cfg = reset_cfg()
            self.write_log("New empty CFG created.", "SUCCESS") 
        except Exception as e:
             self.write_log(f"Error during CFG reset: {e}", "ERROR") 
        finally:
             self.set_status("Idle")
    @work(exclusive=True)

    async def action_generate_cfg(self) -> None:
        """Handles the 'Generate Random' action using pipeline_logic"""
        if self.current_cfg is None:
            self.write_log("Cannot generate CFG. Please Reset first (press 'r').", "WARNING")
            return
        if len(self.current_cfg.graph) < 1: 
            self.write_log("Cannot generate structure: CFG not properly initialized. Please Reset first ('r').", "WARNING")
            return
        self.set_status("Generating random structure...")
        self.write_log("Calling generate_random_structure()...") 
        try:
            success = generate_random_structure(self.current_cfg, config)
            if success:
                self.write_log("Random CFG structure generated.", "SUCCESS")
            else:
                self.write_log("Failed to generate CFG structure (check console/logs for details).", "WARNING")
        except Exception as e:
            self.write_log(f"Unhandled exception during CFG generation: {e}", "ERROR")
        finally:
             self.set_status("Idle")
    @work(exclusive=True)

    async def action_find_paths(self) -> None:
        """Handles the 'Find Paths' action using pipeline_logic"""
        if self.current_cfg is None:
            self.write_log("Cannot find paths. Please Reset and Generate first.", "WARNING")
            return
        if len(self.current_cfg.graph) <= 2:
             self.write_log("CFG structure is too simple. Please Generate a cfg_content complex graph first.", "WARNING")
             return
        self.set_status("Finding test paths...")
        self.write_log("Calling find_test_paths()...") 
        try:
            success = find_test_paths(self.current_cfg, config)
            if success:
                if hasattr(self.current_cfg, 'test_paths') and self.current_cfg.test_paths:
                     self.write_log(f"Found {len(self.current_cfg.test_paths)} test paths.", "SUCCESS")
                else:
                     self.write_log("Path finding complete, but no paths were identified.", "INFO") 
            else:
                self.write_log("Failed to find test paths (check console/logs for details).", "WARNING")
        except Exception as e:
            self.write_log(f"Unhandled exception during test path finding: {e}", "ERROR")
        finally:
             self.set_status("Idle")
    @work(exclusive=True)

    async def action_build_conditions(self) -> None:
        """Handles the 'Build Conditions' action using pipeline_logic"""
        if self.current_cfg is None:
            self.write_log("Cannot build conditions. Please Reset, Generate, and Find Paths first.", "WARNING")
            return
        if not hasattr(self.current_cfg, 'test_paths') or not self.current_cfg.test_paths:
            self.write_log("Cannot build conditions. Test paths not found. Please run 'Find Paths' (f) first.", "WARNING")
            return
        self.set_status("Building edge conditions...")
        self.write_log("Calling build_conditions()...") 
        try:
            success = build_conditions(self.current_cfg, config)
            self.write_log(f"build_conditions returned: {success}", "DEBUG")
            if success:
                infeasible_paths = getattr(self.current_cfg, 'statically_infeasible_test_paths', [])
                self.write_log(f"DEBUG: Post-build check - Infeasible paths attribute value: {infeasible_paths}", "DEBUG")
                if infeasible_paths:
                    num_infeasible = len(infeasible_paths)
                    self.write_log(f"Conditions built, but {num_infeasible} paths remain infeasible (may be structural or static). Consider regenerating structure and paths (r -> g -> f -> c).", "WARNING")
                else:
                     self.write_log("Edge conditions built successfully. All required paths statically feasible.", "SUCCESS")
            else:
                self.write_log(f"Conditions built, but problems encountered (likely structural). Consider regenerating structure and paths (r -> g -> f -> c).", "WARNING")
        except Exception as e:
            self.write_log(f"Unhandled exception during edge condition building: {e}", "ERROR")
        finally:
             self.set_status("Idle")
    @work(exclusive=True)

    async def action_build_operations(self) -> None:
        """Handles the 'Build Operations' action using pipeline_logic"""
        if self.current_cfg is None:
            self.write_log("Cannot build operations. Please run previous pipeline steps first.", "WARNING")
            return
        if not hasattr(self.current_cfg, 'test_paths') or not self.current_cfg.test_paths:
            self.write_log("Test paths not found. Please run 'Find Paths' (f) or 'Build Conditions' (c) first.", "WARNING")
            return
        self.set_status("Adding operations...")
        self.write_log("Calling build_operations()...") 
        try:
            success = build_operations(self.current_cfg, config)
            if success:
                self.write_log("Operations added.", "SUCCESS")
            else:
                 self.write_log("Failed to add operations (check console/logs for details).", "WARNING")
        except Exception as e:
            self.write_log(f"Unhandled exception during operation building: {e}", "ERROR")
        finally:
             self.set_status("Idle")

    def action_display_cfg(self) -> None:
        """Displays the current CFG state summary in the log."""
        self.write_log("--- CFG State Summary ---", "INFO")
        state_info = self._format_cfg_state(self.current_cfg)
        log_widget = self.query_one(RichLog)
        log_widget.write(state_info)
        self.write_log("--- End Summary ---", "INFO")

    def action_show_code(self) -> None:
        """Displays the generated code in the log."""
        self.write_log("--- Generated Code --- ", "INFO")
        if self.current_cfg and hasattr(self.current_cfg, 'code') and self.current_cfg.code:
            log_widget = self.query_one(RichLog)
            log_widget.write("```c") 
            log_widget.write(self.current_cfg.code)
            log_widget.write("```") 
        elif self.current_cfg:
            self.write_log("No code has been generated for the current CFG yet.", "WARNING")
        else:
            self.write_log("No CFG loaded. Reset (r) and Generate (g) first.", "WARNING")
        self.write_log("--- End Code --- ", "INFO")

    def action_show_test_paths(self) -> None:
        """Displays the calculated test paths in the log with detailed format"""
        log_widget = self.query_one(RichLog)

        if self.current_cfg and hasattr(self.current_cfg, 'test_paths') and self.current_cfg.test_paths:
            log_widget.write(f"Found {len(self.current_cfg.test_paths)} test paths:")
            paths_list = list(self.current_cfg.test_paths)
            
            for i, path in enumerate(paths_list, 1):
                 log_widget.write(f"\n[bold]Path {i}:[/bold]") 

                 # Nodes
                 node_ids = [str(getattr(n, 'node_id', 'N/A')) for n in getattr(path, 'nodes', [])]
                 path_nodes_str = " -> ".join(node_ids)
                 log_widget.write(f"  Nodes: {path_nodes_str}")
                 
                 # Effective Formula
                 effective_formula = "[Formula Unavailable]"
                 if hasattr(path, 'get_effective_formula_str'):
                     effective_formula = path.get_effective_formula_str()
                 log_widget.write(f"  Effective Formula: {effective_formula}")
                 
                 # Test Inputs
                 test_inputs = getattr(path, 'test_inputs', None)
                 inputs_str = json.dumps(test_inputs, separators=(',', ':')) if test_inputs is not None else "N/A"
                 log_widget.write(f"  Test Inputs: {inputs_str}")

        elif self.current_cfg:
             self.write_log("Test paths have not been calculated yet. Run 'Find Paths' (f).", "WARNING")
        else:
             self.write_log("No CFG loaded. Run pipeline steps first.", "WARNING")


    def action_show_config(self) -> None:
        """Displays the current application configuration"""
        self.write_log("--- Current Application Configuration ---", "INFO")
        try:
            config_dict = asdict(config)
            formatted_config = self._format_config_dict(config_dict)
            log_widget = self.query_one(RichLog)
            log_widget.write(formatted_config) 
        except Exception as e:
            self.write_log(f"Error formatting or displaying config: {e}", "ERROR")
        self.write_log("--- End Configuration ---", "INFO")

    @work(exclusive=True)
    async def action_save_outputs(self) -> None:
        """Handles the 'Save Outputs' action using pipeline_logic"""
        if self.current_cfg is None:
            self.write_log("Cannot save: No CFG is currently loaded. Run pipeline steps first.", "WARNING")
            return
        self.set_status("Saving outputs...")
        self.write_log("Calling save_outputs()...", "INFO")
        try:
            success = save_outputs(cfg=self.current_cfg, config=config, output_dir=None)
            if success:
                self.write_log("Outputs saved successfully (check console/logs for path).", "SUCCESS")
            else:
                self.write_log("Failed to save outputs (check console/logs for details).", "ERROR")
        except Exception as e:
            self.write_log(f"Unhandled exception during output saving: {e}", "ERROR")
            logger.error(f"Error in TUI action_save_outputs: {e}", exc_info=True)
        finally:
             self.set_status("Idle")

    def on_mount(self) -> None:
        self.write_log("Welcome to FlowForge TUI!") 
        self.write_log("Press keys: [r] Reset, [g] Generate, [f] Find Paths, [c] Conditions, [o] Operations, [d] Display CFG, [s] Show Code, [p] Show Paths, [x] Show Config, [a] Save, [q] Quit") 

def main():
    logger.remove()
    app = TuiApp()
    app.run()

if __name__ == "__main__":
    main() 