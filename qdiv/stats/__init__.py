"""
stats
=====

Provides functions for multivariate statistical tests and ordinations.
"""

from .distance_tests import mantel, permanova, gower
from .ordination_calculations import pcoa_lingoes, dbrda, summarize_dbrda
from .data_stats import corr, bootstrap_sample_matrix, phylo_signal_mantel

__all__ = [
    "mantel",
    "permanova",
    "gower",
    "pcoa_lingoes",
    "dbrda",
    "summarize_dbrda",
    "corr",
    "bootstrap_sample_matrix",
    "phylo_signal_mantel"
]
