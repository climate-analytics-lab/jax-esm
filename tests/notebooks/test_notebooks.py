import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
import os

def get_notebooks(directory):
    return [str(p) for p in Path(directory).rglob("*.ipynb") if ".ipynb_checkpoints" not in str(p)]


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
    for _notebook_path in get_notebooks("notebooks/01_basic"):
        _test_notebook(_notebook_path)
 
#def test_02_experimental():
#    for _notebook_path in get_notebooks("notebooks/02_experimental"):
#        _test_notebook(_notebook_path)
 
def test_03_nongeoscience():
    for _notebook_path in get_notebooks("notebooks/03_non_geoscience"):
        _test_notebook(_notebook_path)
    
