# Welcome to FlowForge!

FlowForge is a tool designed to generate synthetic software artifacts for testing purposes. Based on configurable parameters, it creates not just a Control Flow Graph (CFG), but also corresponding C source code, test paths with inputs, and various visualizations. You can easily adjust the complexity and structure of the generated artifacts. It comes with two ways to use it:

*   A **Command-Line Interface (CLI)** for running the whole generation process automatically.
*   A **Terminal User Interface (TUI)** for stepping through the process interactively.

## What You'll Need

*   Python (Version 3.12.3 is recommended).
*   Make sure you have `pip` (Python's package installer) and `venv` (for virtual environments).

## Getting Started (Installation)

1.  **Download the Code:**
    You should have received a link to download the `software` directory from a cloud storage service (Nextcloud).
    Download and extract the contents of the `software` directory to a location on your computer.
    Open your terminal or command prompt and navigate into that extracted `software` directory:
    ```bash
    cd path/to/your/extracted/software
    ```

2.  **Set up a Virtual Environment (Recommended):**
    This keeps the project's dependencies separate from your main Python setup.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # On Windows, use: .venv\Scripts\activate
    ```
    You should see `(.venv)` appear at the start of your terminal prompt.

3.  **Install Required Packages:**
    This project uses `pyproject.toml` to define dependencies.

    **Preferred Method (using pip):**
    The recommended way to install is using `pip` directly with the `pyproject.toml` file. This ensures you get the dependencies specified for the project.
    ```bash
    pip install .
    ```
    *Note: If you plan on making changes to the FlowForge code itself, install it in editable mode:* `pip install -e .`

    **(Alternative) Using Other Tools:**
    If you use package managers like Poetry or PDM, you can typically use their respective `install` commands (e.g., `poetry install` or `pdm install`).

    **(Alternative) Using `requirements.txt`):**
    A `requirements.txt` file is also provided for compatibility or in environments where `pip install .` might not be directly usable. If you prefer this method:
    ```bash
    pip install -r requirements.txt
    ```

## Making it Your Own (Configuration)

You can tweak how FlowForge works by editing the `config.yaml` file in the main `software` directory. Inside, you can change things like:

*   The random starting point (`seed`)
*   How complex the tests should be (`test_input_complexity`)
*   Details about the graph structure (like nesting levels)
*   The code coverage goal (`coverage_criterion`)
*   ...and more!

Check `config.yaml` for all the options.

## Running FlowForge

Make sure your virtual environment is active (`(.venv)` should be in your prompt)!

You have two main ways to run the tool:

### Option 1: The Quick Way (CLI)

Use the command line (`src/cli.py`) if you want to run the entire CFG generation process from start to finish without stopping.

**Simple Run:**

Just run the pipeline with the settings from `config.yaml`:
```bash
python -m src.cli pipeline
```

**Handy Options:**

You can override settings from the command line:

*   Use a specific random seed: `--seed 42`
*   Change complexity: `--test-input-complexity 3`
*   Pick a coverage type: `--coverage-criterion EC`
*   Save results somewhere else: `--output-dir ./my_results`
*   Stop after a certain step: `--upto generate` (Steps: `generate`, `paths`, `conditions`, `operations`)
*   See more detailed log messages: `--log`

Example combining options:
```bash
python -m src.cli pipeline --seed 123 --test-input-complexity 2 --log
```

### Option 2: Step-by-Step (TUI)

Use the terminal interface (`src/tui.py`) if you want to control each step of the pipeline yourself.

**Start the TUI:**
```bash
python -m src.tui
```

**Inside the TUI, use these keys:**

*   `r`: Start over (Reset CFG)
*   `g`: Create the basic graph structure
*   `f`: Figure out the test paths
*   `c`: Build the conditions for graph edges
*   `o`: Add operations to the graph nodes
*   `d`: Show a summary of the current graph
*   `s`: Display the generated C code (if available)
*   `p`: List the test paths found
*   `x`: Show the current settings from `config.yaml`
*   `a`: Save everything (graph, code, etc.)
*   `q`: Exit the TUI

Just follow the instructions on screen!

**Example TUI Workflow:**

The TUI allows you to inspect the state of the CFG after each pipeline stage. Here's a sample workflow:

1.  **Start the TUI:**
    ```bash
    python -m src.tui
    ```
2.  **Reset:** Press `r` to create a fresh, empty CFG.
3.  **Generate Structure:** Press `g` to build the random graph structure 
4.  **Observe & Save (Optional):**
    *   Press `d`, `s`, `p` to observe the initial structure.
    *   Press `a` to save the current state (e.g., to `output/run1`).
5.  **Find Paths:** Press `f` to run the path finding algorithm.
6.  **Observe & Save (Optional):**
    *   Press `d`, `s`, `p` to see the newly calculated test paths.
    *   Press `a` to save this state (e.g., to `output/run2`).
7.  **Build Conditions:** Press `c`. This assigns conditions to edges and tries to make paths feasible.
8.  **Observe & Save (Optional):**
    *   Press `d`, `s`, `p` (check path details/formulas).
    *   Press `a` to save this state (e.g., to `output/run3`).
9.  **Build Operations:** Press `o`. This adds operations to nodes to meet complexity targets.
10. **Observe Final State:**
    *   Press `d` to see the final complexity metrics.
    *   Press `s` to view the complete generated C code.
    *   Press `p` to review the final test paths and their inputs.
11. **Save Final:** Press `a` to save the final outputs (e.g., to `output/run4`).
12. **Quit:** Press `q` to exit.

This step-by-step approach, combined with saving incrementally after stages, is useful for understanding how each part of the pipeline transforms the CFG and the generated artifacts. You can compare the contents of the `output/run<N>` directories to see how the visualizations, code, and path data "grow" at each step.

## Where do the Outputs Go?

By default, all generated files are saved into an `output/run<N>` directory (where `<N>` is a run number like 1, 2, etc.) created within the main `software` directory where you run the tool. A typical run might produce files like:

*   `cfg.pdf`: A visual representation of the Control Flow Graph.
*   `fragment_forest.pdf`: A visualization of the internal structure used during generation.
*   `generated.c`: The synthesized C source code corresponding to the CFG.
*   `config.yaml`: A copy of the configuration parameters used for the run.
*   `test_paths.txt`: A human-readable list of the generated test paths, their node sequences, and effective formulas.
*   `test_paths.json`: A structured JSON version of the test paths, including calculated test inputs.
*   `summary.pdf`: A consolidated PDF report containing the code, CFG visualization and test paths.

If you use the CLI, you can specify a different location using the `--output-dir` option (e.g., `python -m src.cli pipeline --output-dir ./my_custom_outputs`). Otherwise, both the CLI (without `--output-dir`) and the TUI's 'Save' command will create a new `output/run<N>` folder for the results.

## Known Limitations

*   **Complex CFGs:** For highly complex Control Flow Graphs, particularly those with deeply nested loops or intricate branching, the tool may occasionally struggle to generate all required operations. This can sometimes result in the generated C code (`generated.c`) being syntactically or logically incomplete for these edge cases.

## Utility Scripts (`src/scripts/`)

This directory contains helper scripts for automating common tasks or analysis related to FlowForge. You are encouraged to add your own custom scripts here as needed!

**Example:**

*   `organize_summaries.py`: This script demonstrates how to run the FlowForge pipeline automatically using the CLI for several specific configurations defined within the script. After each run, it copies the *entire output directory* (e.g., `output/run<N>/`) to a final location (`../thesis/src/obrazky-figures/`), renaming the copied directory based on the configuration used (e.g., `EPC_FINAL_3/`).

To run this script (make sure your virtual environment is active):
```bash
python src/scripts/organize_summaries.py
```
Remember to check the script's contents for the exact configurations and output paths it uses.