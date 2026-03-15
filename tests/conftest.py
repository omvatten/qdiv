import os
import qdiv
import pandas as pd
import numpy as np
import pytest

# Use non-interactive backend for Matplotlib
import matplotlib
matplotlib.use('Agg')

@pytest.fixture(scope="session")
def rng():
    # Reproducible random generator
    seed = int(os.environ.get("QDIV_TEST_SEED", 42))
    return np.random.default_rng(seed)

@pytest.fixture(scope="session")
def qb():
    """Load the example MicrobiomeData object, rarefied, once per session.
    """
    obj = qdiv.MicrobiomeData.load_example("Saheb-Alam_DADA2")
    obj.rename_features(inplace=True, name_type="ASV")
    obj.tax_prefix(add=True, inplace=True)
    obj.rarefy(inplace=True)
    return obj

@pytest.fixture(scope="session")
def dis(qb):
    """A small beta-diversity distance matrix for the rarefied example.
    """
    return qdiv.diversity.naive_beta(qb, q=1)

@pytest.fixture(scope="session")
def qtoy():
    """A tiny 3×4 toy dataset with known diversity values.
    """
    table = pd.DataFrame(
        {
            "S1": [10,  0,  0,  0],
            "S2": [ 5,  5,  0,  0],
            "S3": [ 1,  1,  1,  7],
        },
        index=["OTU1", "OTU2", "OTU3", "OTU4"]
    )
    return qdiv.MicrobiomeData(table)