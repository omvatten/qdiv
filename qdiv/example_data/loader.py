import importlib.resources as pkg_resources
import pandas as pd

def load_example_tab():
    """Load the example abundance table as a pandas DataFrame."""
    with pkg_resources.files("qdiv.example_data").joinpath("simple_tab.csv").open("r") as f:
        return pd.read_csv(f)

def load_example_meta():
    """Load the example metadata table as a pandas DataFrame."""
    with pkg_resources.files("qdiv.example_data").joinpath("simple_meta.csv").open("r") as f:
        return pd.read_csv(f, index_col=0)
