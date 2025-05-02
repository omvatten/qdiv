Input files
*************
Example files are provided on the github pages_.

.. _pages: github.com/omvatten/qdiv/example_files

In summary, five types of files can be used used as input:

- The count table, which contains the reads counts or relative abundances associated with each OTUs/ASVs/MAGs/bins and sample. 
The first column contains the OTU/ASV/MAG/bin names. The column headings are the sample names. The count table can (optionally) also contain information about the taxonomic classification of the OTUs/ASVs/MAGs/bins. 
If so, the right-most columns should have all or some of the following headings: Domain, Kingdom, Phylum, Class, Order, Family, Genus, Species.
- The taxonomy file containing information each ASV/OTU/MAG/bin. It should be a table with the first column being the OTU/ASV/MAG/bin name, 
and the following columns having some or all all or some of the following headings: Domain, Kingdom, Phylum, Class, Order, Family, Genus, Species. 
- The fasta file, which contains the sequences of each OTU/ASV.
- The meta data file with information about the samples in the data set. The first column of the meta data contains the sample names, which must be the same as the column headings in the count table. The other columns contain information about the samples.
- A Newick tree file in case phylogenetic diversity indices are to be calculated.