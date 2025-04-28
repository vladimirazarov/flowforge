import subprocess
import os
import shutil
import sys
import glob
import re
from rich.console import Console

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

CONFIGURATIONS = [
    {'criterion': 'NC', 'complexity': 1, 'seed': 907753, 'target_dir': 'NC_FINAL_1'},
    {'criterion': 'PPC', 'complexity': 2, 'seed': 909667, 'target_dir': 'PPC_FINAL_2'},
    {'criterion': 'EPC', 'complexity': 3, 'seed': 342880, 'target_dir': 'EPC_FINAL_3'},
]
FINAL_OUTPUT_DIR = "../thesis/src/obrazky-figures"
DEFAULT_OUTPUT_BASE = "output"

console = Console(stderr=True) 

def run_command(cmd_list):
    """Runs a command using subprocess and returns True on success"""
    cmd_str = ' '.join(cmd_list)
    console.print(f"\n[bold blue]Executing:[/bold blue] {cmd_str}")
    try:
        result = subprocess.run(cmd_list, check=True, capture_output=True, text=True, timeout=300) # Added timeout
        console.print("[green]Command successful.[/green]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error executing command:[/bold red] {cmd_str}", style="red")
        console.print(f"[red]Return code:[/red] {e.returncode}")
        if e.stdout:
             console.print(f"[yellow]stdout:[/yellow]\n{e.stdout.strip()}")
        if e.stderr:
             console.print(f"[bold red]stderr:[/bold red]\n{e.stderr.strip()}")
        return False
    except subprocess.TimeoutExpired:
         console.print(f"[bold red]Command timed out:[/bold red] {cmd_str}", style="red")
         return False
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
        return False

def find_latest_run_dir(base_dir):
    """Finds the most recently created 'run<N>' directory in base_dir"""
    run_dirs = glob.glob(os.path.join(base_dir, 'run*'))
    latest_run_dir = None
    highest_run_num = -1

    for d in run_dirs:
        if os.path.isdir(d):
            match = re.match(r'run(\d+)$', os.path.basename(d))
            if match:
                run_num = int(match.group(1))
                if run_num > highest_run_num:
                    highest_run_num = run_num
                    latest_run_dir = d
                    
    return latest_run_dir

def main():
    """Main execution function"""
    final_output_parent = os.path.dirname(FINAL_OUTPUT_DIR)
    if final_output_parent and not os.path.exists(final_output_parent):
        console.print(f"Ensuring parent directory for final output exists: [cyan]{final_output_parent}[/cyan]")
        os.makedirs(final_output_parent, exist_ok=True)
        
    console.print(f"Final output will be placed in: [cyan]{FINAL_OUTPUT_DIR}[/cyan]")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    console.print(f"Project root detected as: [cyan]{project_root}[/cyan]")
    console.print("Changing working directory to project root.")
    os.chdir(project_root)
    console.print(f"Current working directory: [cyan]{os.getcwd()}[/cyan]")
    
    os.makedirs(DEFAULT_OUTPUT_BASE, exist_ok=True)

    for config in CONFIGURATIONS:
        criterion = config['criterion']
        complexity = config['complexity']
        seed = config['seed']
        target_dir_name = config['target_dir']
        
        console.print(f"\n[bold magenta]--- Processing Configuration: Criterion=[yellow]{criterion}[/yellow], Complexity=[yellow]{complexity}[/yellow], Seed=[yellow]{seed}[/yellow], TargetDir=[cyan]{target_dir_name}[/cyan] ---[/bold magenta]")
        
        cmd = [
            sys.executable,  
            "-m", "src.cli",
            "pipeline",
            "--test-input-complexity", str(complexity),
            "--coverage-criterion", criterion,
            "--seed", str(seed),
        ]

        if run_command(cmd):
            latest_run_dir = find_latest_run_dir(DEFAULT_OUTPUT_BASE)
            
            if latest_run_dir:
                console.print(f"Found latest run directory: [cyan]{latest_run_dir}[/cyan]")

                console.print("\n[bold]--- Running Verification ---[/bold]")
                generate_tests_cmd = [
                    sys.executable,
                    "-m", "src.utils.generate_tests",
                    latest_run_dir
                ]
                pytest_cmd = [
                    sys.executable,
                    "-m", "pytest",
                    os.path.join(latest_run_dir, "test_suite.py")
                ]

                verification_passed = False
                if run_command(generate_tests_cmd):
                    if run_command(pytest_cmd):
                        console.print("[green]Verification successful.[/green]")
                        verification_passed = True
                    else:
                        console.print("[bold red]Pytest verification failed.[/bold red]")
                else:
                    console.print("[bold red]generate_tests script failed.[/bold red]")

                if verification_passed:
                    dest_dir_path = os.path.join(FINAL_OUTPUT_DIR, target_dir_name)

                    if os.path.exists(dest_dir_path):
                        console.print(f"Removing existing directory: [cyan]{dest_dir_path}[/cyan]...")
                        try:
                            shutil.rmtree(dest_dir_path)
                            console.print("[dim]Existing directory removed.[/dim]")
                        except Exception as e:
                            console.print(f"[bold red]Error removing existing directory[/bold red] [cyan]{dest_dir_path}[/cyan]: {e}")
                            continue 
                    
                    # Copy the entire run directory
                    console.print(f"Copying entire directory '[cyan]{latest_run_dir}[/cyan]' to '[cyan]{dest_dir_path}[/cyan]'...")
                    try:
                        shutil.copytree(latest_run_dir, dest_dir_path)
                        console.print("[green]Directory copy successful.[/green]")
                    except Exception as e:
                        console.print(f"[bold red]Error copying directory:[/bold red] {e}")
                else:
                    console.print(f"[yellow]Skipping directory copy for[/yellow] [cyan]{target_dir_name}[/cyan] [yellow]due to verification failure.[/yellow]")
            else:
                console.print(f"[bold red]Error:[/bold red] Could not find the latest run directory in '[cyan]{DEFAULT_OUTPUT_BASE}[/cyan]'.")
        else:
            console.print(f"[yellow]Skipping processing for[/yellow] [cyan]{target_dir_name}[/cyan] [yellow]due to main command execution failure.[/yellow]")

    console.print("\n[bold magenta]--- All specified configurations processed ---[/bold magenta]")

if __name__ == "__main__":
    main() 