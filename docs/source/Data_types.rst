Data types
**********

qdiv uses five primary data types:

* **tab**: A table with counts or relative abundances. Features
  (e.g. species, OTUs, ASVs, MAGs, or bins) are row indices and
  samples are column headings.

* **tax**: A table with taxonomic information for the features.
  Column headings are typically Domain, Phylum, Class, Order,
  Family, Genus, and Species, although other taxonomic levels
  are also supported.

* **seq**: A table containing the sequence of each feature.
  This is typically used for amplicon sequencing data and is
  loaded from a FASTA file.

* **meta**: A table containing metadata about the samples.
  Sample names are row indices and the columns contain sample
  attributes such as treatment, location, time point, or
  environmental measurements.

* **tree**: A phylogenetic tree loaded from a Newick-formatted file.

  The tree representation consists of two components:

  * ``tree``: A pandas DataFrame containing all nodes, branches, and
    branch lengths.
  * ``leaf_order``: A list containing the names of all leaf nodes ordered internally by qdiv.

All data are stored as pandas DataFrames (and, for phylogenetic trees,
a DataFrame together with a corresponding ``leaf_order`` list) within a
:class:`qdiv.MicrobiomeData` object.

The underlying data can be accessed and exported directly. For example,
to save the abundance table:

.. code-block:: python

   obj.tab.to_csv("your_chosen_file_name.csv")
   
To save all data present in a ``MicrobiomeData`` object to the appropriate file formats:

.. code-block:: python

   obj.save(savename = "My_data")