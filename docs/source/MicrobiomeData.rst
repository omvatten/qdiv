MicrobiomeData
==============

The ``MicrobiomeData`` class is the central data container in qdiv.
It stores abundance tables, taxonomy, metadata, sequences, and
phylogenetic trees, and provides methods for importing, manipulating,
validating, and exporting microbiome datasets.

What is a data object?
----------------------

A ``MicrobiomeData`` object may contain:

* ``tab``: abundance table (features × samples)
* ``tax``: taxonomy table
* ``meta``: metadata table
* ``seq``: sequence table
* ``tree``: phylogenetic tree dataframe
* ``leaf_order``: ordering of tree leaves

Methods
-------

Creating objects
~~~~~~~~~~~~~~~~

Create a new object or construct one from existing files, example datasets,
or dictionaries.

.. autosummary::
   :nosignatures:

   ~qdiv.MicrobiomeData
   ~qdiv.MicrobiomeData.load
   ~qdiv.MicrobiomeData.load_example
   ~qdiv.MicrobiomeData.from_dict

Importing data
~~~~~~~~~~~~~~

General file formats
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~qdiv.MicrobiomeData.add_tab
   ~qdiv.MicrobiomeData.add_tax
   ~qdiv.MicrobiomeData.add_meta
   ~qdiv.MicrobiomeData.add_seq_from_fasta
   ~qdiv.MicrobiomeData.add_tree

External software formats
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~qdiv.MicrobiomeData.add_tax_from_qiime
   ~qdiv.MicrobiomeData.add_tax_from_sintax
   ~qdiv.MicrobiomeData.add_tax_from_gtdbtk
   ~qdiv.MicrobiomeData.add_tab_from_coverm
   ~qdiv.MicrobiomeData.add_ebd_tab_from_singlem

Exploring and exporting data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inspect object contents, summarize datasets, and save data to disk.

.. autosummary::
   :nosignatures:

   ~qdiv.MicrobiomeData.info
   ~qdiv.MicrobiomeData.summarize_taxa
   ~qdiv.MicrobiomeData.save
   ~qdiv.MicrobiomeData.to_dict
   ~qdiv.MicrobiomeData.copy

Subsetting and filtering
~~~~~~~~~~~~~~~~~~~~~~~~

Select samples, features, or taxa and perform rarefaction.

.. autosummary::
   :nosignatures:

   ~qdiv.MicrobiomeData.subset_samples
   ~qdiv.MicrobiomeData.subset_features
   ~qdiv.MicrobiomeData.subset_taxa
   ~qdiv.MicrobiomeData.subset_abundant
   ~qdiv.MicrobiomeData.rarefy

Sample aggregation
~~~~~~~~~~~~~~~~~~

Combine samples based on metadata variables.

.. autosummary::
   :nosignatures:

   ~qdiv.MicrobiomeData.merge_samples

Taxonomy and feature utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Utilities for renaming features and standardizing taxonomy tables.

.. autosummary::
   :nosignatures:

   ~qdiv.MicrobiomeData.rename_features
   ~qdiv.MicrobiomeData.tax_prefix
   ~qdiv.MicrobiomeData.clean_tax

Phylogenetic tree operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Manipulate and synchronize phylogenetic trees with abundance data.

.. autosummary::
   :nosignatures:

   ~qdiv.MicrobiomeData.prune_tree

Complete API reference
----------------------

.. autoclass:: qdiv.MicrobiomeData
   :members:
   :member-order: bysource
   :show-inheritance: