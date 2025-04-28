"""
Generates class diagrams for core FlowForge data structures using pyreverse and dot

This script automates the process of:
1. Running pyreverse (from pylint) on specified source files to generate a .dot file
2. Running dot (from Graphviz) to convert the .dot file into a PDF diagram

Generates multiple focused diagrams for different components

Requires:
- pylint (`pip install pylint`)
- Graphviz (`sudo apt-get install graphviz` or download from graphviz.org)
"""
import subprocess
import sys
from pathlib import Path

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

SRC_EXPRESSION = PROJECT_ROOT / "src/core/cfg_content/expression.py"
SRC_OPERATION = PROJECT_ROOT / "src/core/cfg_content/operation.py"
SRC_CFG = PROJECT_ROOT / "src/core/cfg/cfg.py"
SRC_NODE_BASE = PROJECT_ROOT / "src/core/nodes/cfg_basic_block_node.py"
SRC_NODE_INSTR = PROJECT_ROOT / "src/core/nodes/cfg_instructions_node.py"
SRC_NODE_JUMP = PROJECT_ROOT / "src/core/nodes/cfg_jump_node.py"
SRC_FRAG_BASE = PROJECT_ROOT / "src/core/cfg/fragments/cfg_fragment.py"
SRC_FRAG_TYPES = PROJECT_ROOT / "src/core/cfg/fragments/fragment_types.py"
SRC_FRAG_FOREST = PROJECT_ROOT / "src/core/cfg/fragments/fragment_forest.py"


DIAGRAM_CONFIGS = [
    {
        "name": "Expression",
        "output_filename": "diagram_expression.pdf",
        "project_name": "flowforge_expr",
        "source_files": [SRC_EXPRESSION],
    },
    {
        "name": "Operation",
        "output_filename": "diagram_operation.pdf",
        "project_name": "flowforge_op",
        "source_files": [SRC_OPERATION],
    },
    {
        "name": "CFG",
        "output_filename": "diagram_cfg.pdf",
        "project_name": "flowforge_cfg",
        "source_files": [SRC_CFG],
    },
    {
        "name": "Base Node",
        "output_filename": "diagram_node_base.pdf",
        "project_name": "flowforge_node_base",
        "source_files": [SRC_NODE_BASE],
    },
    {
        "name": "Instructions Node",
        "output_filename": "diagram_node_instructions.pdf",
        "project_name": "flowforge_node_instr",
        "source_files": [SRC_NODE_INSTR, SRC_NODE_BASE],
    },
    {
        "name": "Jump Node",
        "output_filename": "diagram_node_jump.pdf",
        "project_name": "flowforge_node_jump",
        "source_files": [SRC_NODE_JUMP, SRC_NODE_BASE],
    },
    {
        "name": "Fragment Forest (Solo)",
        "output_filename": "diagram_fragment_forest_solo.pdf",
        "project_name": "flowforge_forest_solo",
        "source_files": [SRC_FRAG_FOREST],
    },
]

# Output directory for all diagrams
OUTPUT_DIR = PROJECT_ROOT / "bp/obrazky-figures/implementation"


def run_command(command: list[str], cwd: Path | None = None) -> bool:
    """Runs a command using subprocess and prints output/errors"""
    print(f"--> Running: {' '.join(command)}")
    try:
        str_command = [str(part) for part in command]
        process = subprocess.run(
            str_command,
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd or PROJECT_ROOT,
        )
        print("Output:\n", process.stdout)
        if process.stderr:
            print("--> Stderr:", file=sys.stderr)
            print(process.stderr, file=sys.stderr)
        print("--> Command successful.")
        return True
    except FileNotFoundError:
        print(f"--> Error: Command '{command[0]}' not found. Is it installed and in your PATH?", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:
        print(f"--> Error: Command failed with exit code {e.returncode}", file=sys.stderr)
        print("--> Stderr:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        print("--> Stdout:", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        return False
    except Exception as e:
        print(f"--> Error: An unexpected error occurred while running command: {e}", file=sys.stderr)
        return False

def generate_dot_file(source_files: list[Path], project_name: str) -> Path | None:
    """Generates the .dot file using pyreverse for specific source files

    Args:
        source_files: List of source file Path objects
        project_name: Prefix for pyreverse output files

    Returns:
        Path to the generated .dot file, or None on failure
    """
    existing_files = [f for f in source_files if f.exists()]
    missing_files = [f for f in source_files if not f.exists()]
    if missing_files:
        print("--> Warning: The following source files were not found and will be skipped:", file=sys.stderr)
        for f in missing_files:
            print(f"    - {f.relative_to(PROJECT_ROOT)}", file=sys.stderr)
    if not existing_files:
        print("--> Error: No source files found for this diagram. Cannot generate.", file=sys.stderr)
        return None

    dot_filename = f"classes_{project_name}.dot"
    dot_file_path = PROJECT_ROOT / dot_filename

    pyreverse_cmd = [
        "pyreverse",
        "-o", "dot",
        "-p", project_name,
    ] + [str(f.relative_to(PROJECT_ROOT)) for f in existing_files]

    if run_command(pyreverse_cmd, cwd=PROJECT_ROOT):
        if dot_file_path.exists():
            return dot_file_path
        else:
            print(f"--> Error: pyreverse completed but '{dot_filename}' was not found.", file=sys.stderr)
            return None
    else:
        return None

def convert_dot_to_pdf(dot_file_path: Path, output_pdf_path: Path) -> bool:
    """Converts a specific .dot file to a PDF using Graphviz dot"""
    if not dot_file_path.exists():
        print(f"--> Error: Cannot convert. Intermediate file '{dot_file_path.relative_to(PROJECT_ROOT)}' not found.", file=sys.stderr)
        return False

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    dot_cmd = [
        "dot",
        "-Tpdf",
        str(dot_file_path),
        "-o", str(output_pdf_path)
    ]

    return run_command(dot_cmd, cwd=PROJECT_ROOT)

def cleanup_dot_file(dot_file_path: Path | None):
    """Cleans up a specific intermediate dot file"""
    if dot_file_path and dot_file_path.exists():
        try:
            print(f"--> Cleaning up intermediate file: {dot_file_path.relative_to(PROJECT_ROOT)}")
            dot_file_path.unlink()
        except OSError as e:
            print(f"--> Warning: Could not delete intermediate file {dot_file_path}: {e}", file=sys.stderr)
    elif dot_file_path:
         print(f"--> Intermediate file {dot_file_path.relative_to(PROJECT_ROOT)} not found, skipping cleanup.")

if __name__ == "__main__":
    print(f"Running diagram generation script from project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")

    success_count = 0
    failure_count = 0

    for config in DIAGRAM_CONFIGS:
        diagram_name = config["name"]
        output_filename = config["output_filename"]
        project_name = config["project_name"]
        source_files = config["source_files"]
        output_pdf_path = OUTPUT_DIR / output_filename

        print(f"\n--- Generating Diagram: {diagram_name} ---")
        print(f"---> Output: {output_pdf_path.relative_to(PROJECT_ROOT)}")
        print(f"---> Sources: {[f.relative_to(PROJECT_ROOT) for f in source_files]}")

        print(" ---> Step 1: Generating .dot file...")
        dot_file_path = generate_dot_file(source_files, project_name)

        if not dot_file_path:
            print(f" Skipping conversion for '{diagram_name}' due to .dot generation failure.")
            failure_count += 1
            continue

        print(" ---> Step 2: Converting .dot to PDF...")
        if convert_dot_to_pdf(dot_file_path, output_pdf_path):
            print(f" ==> Successfully generated: {output_pdf_path.relative_to(PROJECT_ROOT)}")
            success_count += 1
        else:
            print(f" ==> FAILED to generate: {output_pdf_path.relative_to(PROJECT_ROOT)}", file=sys.stderr)
            failure_count += 1

        print(" ---> Step 3: Cleaning up intermediate file...")
        cleanup_dot_file(dot_file_path)

    print("\n--- Diagram Generation Summary ---")
    print(f"Successfully generated: {success_count}")
    print(f"Failed: {failure_count}")

    if failure_count > 0:
        print("\nOne or cfg_content diagrams failed to generate. Please check the errors above.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll diagrams generated successfully.")
        sys.exit(0)