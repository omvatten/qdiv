Input files
*************
Example files are provided on the github pages_.

.. _pages: github.com/omvatten/qdiv/example_files

In summary, four types of files are used as input:

- The count table, which contains the reads counts associated with each OTU/ASV and sample. The first column contains the OTU/ASV names. The column headings are the sample names. The count table can (optionally) also contain information about the taxonomic classification of the OTUs/ASVs. If so, the right-most columns should have all or some of the following headings: Domain, Kingdom, Phylum, Class, Order, Family, Genus, Species.
- The fasta file, which contains the sequences of each OTU/ASV.
- The meta data file in which the user supplies information about the samples in the data set. The first column of the meta data contains the sample names, which must be the same as the column headings in the count table. The other columns contain information about the samples.
- A Newick tree file in case phylogenetic diversity indices are to be calculated.