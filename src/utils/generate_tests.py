"""Utility helpers to turn **test_paths.json** that FlowForge writes
into executable regression-test artefacts:

*   **pytest** parametrisation file  →  *test_suite.py*
*   tiny **C harness** that calls each path  →  *harness.c*
*   (optionally) compiles the generated sources into a shared library
    *libfoo.so* and an executable *foo_bin* so the user can run / fuzz
    the program immediately

The module is *self-contained*: call it as a script and pass the run
folder that contains **generated.c** and **test_paths.json**

Example
-------
```bash
python -m src.utils.generate_tests ./output/run94
pytest -q output/run94/test_suite.py    # dynamic check via ctypes
./output/run94/foo_bin                  # standalone executable
```
"""
from __future__ import annotations

import json
from loguru import logger
import pathlib
import re
import subprocess
import sys
import textwrap
from subprocess import CalledProcessError
from typing import Any, Dict, Tuple

from src.utils.serializers import test_suite_values
from src.utils.logging_helpers import (
    print_test_gen_file_written,
    print_test_gen_build_success,
    print_test_gen_build_failure,
    print_test_gen_usage,
    print_test_gen_fatal_error
)

__author__ = "Vladimir Azarov"
__email__ = "azarov.swe@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"





def build_pytest_suite(tp_json: pathlib.Path, out_py: pathlib.Path, lib_name: str = "libfoo.so") -> None:
    """Create *out_py* – a pytest module that loads ``foo`` via *ctypes* and
    executes every path once with the entry-time values
    """
    data = json.loads(tp_json.read_text())
    
    # Get authoritative parameter names from the JSON
    function_parameters = data.get("function_parameters", [])
    if not function_parameters:
        param_names = set()
        for p in data["test_paths"]:
            ev = test_suite_values(p)
            param_names.update(ev.keys())
        function_parameters = sorted(list(param_names))
        if not function_parameters:
             logger.warning("No parameters found in test_paths.json or inferred from paths. Test generation might fail.")
        else:
             logger.warning("Parameter names not found in test_paths.json, inferred from paths. Using: %s", function_parameters)
    
    # Prepare parameter strings for pytest
    params_list_str = ",".join(function_parameters)
    # Format for parametrize: 'A', 'B', 'C' or 'A', for single param
    params_tuple_str = "'" + "','".join(function_parameters) + "'"
    if len(function_parameters) == 1:
         params_tuple_str += "," 

    lines = [
        "import ctypes, pytest",
        "import pathlib",
        f"lib_path = pathlib.Path(__file__).parent / '{lib_name}'",
        f"lib = ctypes.CDLL(str(lib_path))",
        "foo = lib.foo",
        # Assuming all parameters are c_int for now
        f"foo.argtypes = [{ ', '.join(['ctypes.c_int'] * len(function_parameters)) }]",
        "foo.restype  = ctypes.c_int  ",
        "",
        f"@pytest.mark.parametrize(({params_tuple_str}), [", 
    ]
    # Generate parameter tuples for each path
    for p in data["test_paths"]:
        # Skip paths that are marked as infeasible (empty or missing test_inputs)
        test_inputs_model = p.get("test_inputs")
        if not test_inputs_model: 
            path_id = p.get("path_id", "N/A")
            logger.info(f"Skipping Path {path_id} for pytest suite generation: No test inputs found (likely infeasible).")
            continue 

        ev = test_suite_values(p) 
        param_values = tuple(ev.get(name, 0) for name in function_parameters)
        lines.append(f"    {repr(param_values)}," )

    lines.extend([
        "])",
        f"def test_paths({params_list_str}):",
        f"    foo({params_list_str})   ",
    ])
    out_py.write_text("\n".join(lines))


def build_c_harness(tp_json: pathlib.Path, out_c: pathlib.Path) -> None:
    """Emit a C file that calls *foo* for every path """
    data = json.loads(tp_json.read_text())

    # Get parameter names from the JSON
    function_parameters = data.get("function_parameters", [])
    if not function_parameters:
        param_names = set()
        for p in data["test_paths"]:
             ev = test_suite_values(p)
             param_names.update(ev.keys())
        function_parameters = sorted(list(param_names))
        if not function_parameters:
             logger.warning("No parameters found in test_paths.json or inferred from paths (C harness). Generation might fail.")
        else:
             logger.warning("Parameter names not found in test_paths.json, inferred from paths (C harness). Using: %s", function_parameters)

    c_param_defs = ",".join(f'int {name}' for name in function_parameters)

    fns = []
    cases_data = [] 
    for p in data["test_paths"]:
         ev = test_suite_values(p)
         cases_data.append((p["path_id"], ev))

    for pid, ev in cases_data:
        c_param_values = ",".join(str(ev.get(name, 0)) for name in function_parameters)
        fns.append(textwrap.dedent(f"""
            void test_path{pid}(void) {{
                /* TODO: inspect return value / side-effects */
                foo({c_param_values});
            }}"""))

    harness = textwrap.dedent(
        f"""
        #include <assert.h>
        extern int foo({c_param_defs});
        {'' .join(fns)}
        int main(void){{
            {'' .join(f'test_path{pid}();' for pid, _ in cases_data)}
            return 0;
        }}
        """
    )
    out_c.write_text(harness)


GCC = "gcc"  

def _run(cmd: list[str], cwd: pathlib.Path) -> None:
    try:
        subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except CalledProcessError as e:
        sys.stderr.write(e.stderr.decode(errors="ignore"))
        raise


def compile_sources(run_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Compile *generated.c* plus harness into **foo_bin** and **libfoo.so**

    Returns (shared_object_path, executable_path)
    """
    gen_c     = run_dir / "generated.c"
    tp_json   = run_dir / "test_paths.json"
    harness_c = run_dir / "harness.c"

    if not gen_c.exists() or not tp_json.exists():
        raise FileNotFoundError("generated.c or test_paths.json missing in run dir")

    # 1. create harness.c
    build_c_harness(tp_json, harness_c)

    # 2. shared library for ctypes
    so_path = run_dir / "libfoo.so"
    _run([GCC, "-std=c11", "-shared", "-fPIC", "-o", so_path.name, gen_c.name], run_dir)

    # 3. standalone executable (object + harness)
    obj_path = run_dir / "generated.o"
    _run([GCC, "-std=c11", "-c", gen_c.name, "-o", obj_path.name], run_dir)
    bin_path = run_dir / "foo_bin"
    _run([GCC, "-std=c11", obj_path.name, harness_c.name, "-o", bin_path.name], run_dir)

    return so_path, bin_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_test_gen_usage()
        sys.exit(1)

    run_folder = pathlib.Path(sys.argv[1]).resolve()
    tp_json = run_folder / "test_paths.json"
    if not tp_json.exists():
        print_test_gen_fatal_error(f"test_paths.json not found in {run_folder}")
        sys.exit(1)

    # 1. build pytest suite
    out_py_path = run_folder / "test_suite.py"
    build_pytest_suite(tp_json, out_py_path)
    print_test_gen_file_written(out_py_path.name)

    # 2. compile C artefacts
    try:
        so, bin_ = compile_sources(run_folder)
        print_test_gen_build_success(so.name, bin_.name)
    except Exception as exc:
        print_test_gen_build_failure(exc)
        sys.exit(2)