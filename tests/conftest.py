
import os
import random
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
    Returns a qdiv.MicrobiomeData-like object that the qdiv API can consume.
    """
    import qdiv
    obj = qdiv.MicrobiomeData.load_example("Saheb-Alam_DADA2")
    obj.rename_features(inplace=True, name_type="ASV")
    obj.tax_prefix(add=True, inplace=True)
    obj.rarefy(inplace=True)
    return obj

@pytest.fixture(scope="session")
def dis(qb):
    """A small beta-diversity distance matrix for the rarefied example.
    """
    import qdiv
    return qdiv.diversity.naive_beta(qb, q=1)
