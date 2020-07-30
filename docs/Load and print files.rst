Load and print files
********************

Loading files
#############

.. code-block:: python

   files.load(path='', tab='None', fasta='None', meta='None', sep=',')

Loads files into an object, which is used for further analysis.

*path* is the path to the directory holding the input files, *tab* is the name of the count table file, 

*fasta* is the name of the fasta file, *meta* is the name of meta data file, *sep* is the separator used in the count table and meta data files (, or ; or \\t).

Example

.. code-block:: python

   import qdiv

   obj = qdiv.files.load(path='', tab='example_tab.csv', fasta='example_fasta.fa', meta='example_meta.csv', sep=',')

obj is now a python dictionary containing five pandas dataframes ‘tab’, ‘ra’, ‘tax’, ‘seq’, and ‘meta’.

- 'tab' holds the counts associated with each OTU/ASV (rows) and sample (columns).
- 'ra' holds the percentage relative abundance of reads per sample associated with each OTU/ASV (rows) and sample (columns).
- 'tax' holds the taxonomic classification for each OTU/ASV. The row indices are OTU/ASV names, the column names are 'Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'.
- 'seq' holds the sequences. The row indices are the OTU/ASV names, the column 'seq' holds the sequences.
- 'meta' holds the meta data. The row indices are the samples names (the same as the column headings in 'tab'). The columns are variables that define each sample as determined by the user in the meta data input file.

Save files
###########

.. code-block:: python

   files.printout(obj, path='', savename='', sep=',')

Saves the python object as files of the same type as the input files.

*obj* is the object to be printed.

*path* is the path to the directory holding the printed files.
 
*savename* is the name used for the printed files, *sep* is the separator to be used in the printed count table and meta data files (',' or ';' or '\\t').

Example

.. code-block:: python

   import qdiv

   qdiv.files.printout(obj, path='', savename='output', sep=',')
