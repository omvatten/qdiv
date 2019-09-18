Input files
*************
Example files are provided on the github pages_.

.. _pages: github.com/omvatten/qdiv/example_files

In summary, three types of files are used as input:

- The frequency table, which contains the reads counts associated with each OTU/SV and sample. The first column contains the OTU/SV names. The column headings are the sample names. The frequency table can (optionally) also contain information about the taxonomic classification of the OTUs/SVs. If so, the right-most columns should have the headings: Kingdom, Phylum, Class, Order, Family, Genus, Species.
- The fasta file, which contains the sequences of each OTU/SV.
- The meta data file in which the user supplies information about the samples in the data set. The first column of the meta data contains the sample names, which must be the same as the column headings in the frequency table. The other columns contain information about the samples.
