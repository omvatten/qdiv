What is qdiv?
*************

qdiv is a modular Python package built for microbial ecology research. 
It offers a unified framework for loading, managing, and analyzing multi-dimensional datasets, seamlessly integrating taxonomic, phylogenetic, and functional perspectives.
At its core is the MicrobiomeData class, which organizes relative abundance or count tables, taxonomy, sequences, phylogenetic trees, 
and metadata into a coherent structure, enabling reproducible and transparent analysis.

The heart of qdiv lies in the Hill number framework  (a unified approach to quantifying diversity), which allows smooth control over how relative abundances influence diversity metrics. 
This framework is implemented across a broad range of ecological analysis methods, including diversity metrics, multivariate statistics, and null models.

Key Features
------------

- Data Management: Load and validate data from multiple formats (CSV/TSV, FASTA, Newick). Functions for subsetting, rarefaction, merging, renaming, tree pruning, and more.

- Diversity analysis: Compute multiple alpha and beta diversity metrics based on the Hill number framework.

- Statistical methods: Integrate diversity metrics using the Hill number framework into null models (Raup–Crick, NRI, NTI), ordinations (PCoA, db-RDA), PERMANOVA, and Mantel tests.

- Visualization: Heatmaps, ordinations, and diversity plots, and more. 
