"""
sequences
=========

Provides functions for aligning and comparing sequences.
"""

from .sequence_comparisons import sequence_distance_matrix, tree_distance_matrix, align, consensus, merge_objects

__all__ = [
    "sequence_distance_matrix",
    "tree_distance_matrix",
    "align",
    "consensus",
    "merge_objects"
]
