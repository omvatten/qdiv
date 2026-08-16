"""
sequences
=========

Provides functions for aligning and comparing sequences, 
and to calculate distance matrices from phylogenetic trees or sequences.
"""

from .sequence_comparisons import sequence_distance_matrix, tree_distance_matrix, align, consensus, merge_objects, load_compressed_matrix

__all__ = [
    "sequence_distance_matrix",
    "tree_distance_matrix",
    "load_compressed_matrix",
    "align",
    "consensus",
    "merge_objects"
]
