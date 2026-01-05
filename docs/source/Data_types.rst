Data types
**********

qdiv uses five types of data:

- **tab**: A table with counts or relative abundances; features (i.e. species, OTUs, ASVs, MAGs, bins, etc) are row indices and samples are column headings.

- **tax**: A table with taxonomic information about the features. Column heading are typically Domain, Phylum, Class, Order, Family, Genus, Species, but other taxonomic levels also work.

- **seq**: A table with the sequence of each feature. This is typically used for amplicon sequencing data and loaded from a fasta file.

- **tree**: A table with information about all branches, nodes, and leaves of a phylogenetic tree. This is loaded from a Newick-formatted file.

- **meta**: A table with meta data about the samples. The sample names are row indices and the following columns hold information about the samples.

All data are stored as pandas dataframes in a MicrobiomeData object. They can easily be retrieved as CSV files. 
For example, if we have a MicrobiomeData object called obj, the count table can be saved using obj.tab.to_csv(“Your_chosen_file_name.csv”).