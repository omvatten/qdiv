"""
qdiv: Microbial diversity analysis with Hill numbers
====================================================

This package provides tools for diversity analysis, including:

- **Data structures**: Easily filter and manipulate abundance tables, taxonomic data, and phylogenetic trees.
- **Diversity calculations**: Compute alpha and beta diversity metrics, focusing on the Hill number framework.
- **Statistics**: Perform multivariate tests and generate null models.
- **Visualization**: Create plots for ordinations, diversity metrics, and relative abundance.

Subpackages
-----------
.. autosummary::
    :toctree: _autosummary

    qdiv.sequences
    qdiv.stats
    qdiv.diversity
    qdiv.plot
    qdiv.model

"""

__version__ = "4.0.0"
__author__ = "Oskar Modin"
__docformat__ = "restructuredtext"

# Public API imports
from .data_object import MicrobiomeData
from .citations import citations
from . import sequences
from . import stats
from . import diversity
from . import plot
from . import model

__all__ = [
    "MicrobiomeData",
    "citations",
    "sequences",
    "stats",
    "diversity",
    "plot",
    "model"
]

