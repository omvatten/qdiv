# qdiv
A Python package for analyzing results from high-throughput amplicon sequencing of microbial communities (and similar data).

A compiled Windows-version of the program is available at omvatten.se/software.html

Oskar Modin, 2019-09-15

Welcome to qdiv's documentation!
================================
qdiv is a python package for analyzing results from 16S rRNA gene amplicon sequencing (or similar data).

With qdiv, you can subset the data, generate a consensus table based on several frequency tables, 
calculate alpha- and beta diversity indices (focusing on Hill-based indices), 
plot a heatmap or PCoA, carry out null model analysis using Raup-Crick extended to Hill-based dissimilarity indices, and do the Mantel test.

.. toctree::
   :maxdepth: 2
   
   Installation
   Input files
   Load and print files
   Information and statistics
   Subset and manipulate data
   Diversity calculations
   Plotting
   GUI
