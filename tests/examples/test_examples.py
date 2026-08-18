import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
import os
import subprocess

def get_notebooks(directory):
    return [str(p) for p in Path(directory).rglob("*.ipynb") if ".ipynb_checkpoints" not in str(p)]


def get_run_scripts(directory):
    return [str(p) for p in Path(directory).rglob("run.sh")]


def _test_run_script(script_path):
    script_dir = os.path.dirname(script_path)
    try:
        result = subprocess.run(
            ["bash", os.path.basename(script_path)],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            pytest.fail(
                f"run.sh {script_path} exited with code {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"run.sh {script_path} timed out: {e}")
    except Exception as e:
        pytest.fail(f"run.sh {script_path} failed execution: {e}")


def _test_notebook(notebook_path):
    # 1. Load the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # 2. Configure the executor
    # timeout: -1 means no timeout, or set to 600 for 10 minutes
    # kernel_name: ensures it uses your python environment
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        # 3. Run the notebook
        # 'path' sets the working directory to the notebook's folder
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        
        # 4. Optional: Detection of specific output files
        # If your notebook is named 'data_processor.ipynb', check for its output
        if "processor" in notebook_path:
            assert os.path.exists("processed_data.csv"), "Notebook failed to generate output file!"
            os.remove("processed_data.csv") # Cleanup

    except Exception as e:
        pytest.fail(f"Notebook {notebook_path} failed execution: {e}")
   
    
def test_01_basic():
    for _notebook_path in get_notebooks("examples/01_basic"):
        _test_notebook(_notebook_path)
    for _script_path in get_run_scripts("examples/01_basic"):
        _test_run_script(_script_path)

def test_02_experimental():
    for _notebook_path in get_notebooks("examples/02_experimental"):
        _test_notebook(_notebook_path)
    for _script_path in get_run_scripts("examples/02_experimental"):
        _test_run_script(_script_path)

def test_03_nongeoscience():
    for _notebook_path in get_notebooks("examples/03_non_geoscience"):
        _test_notebook(_notebook_path)
    for _script_path in get_run_scripts("examples/03_non_geoscience"):
        _test_run_script(_script_path)

