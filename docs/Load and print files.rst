Load and print files
********************

Loading files
#############

.. code-block:: python

   files.load(tab='None', tax='None', meta='None', fasta='None', tree='None', **kwargs)

Loads files into an object, which is used for further analysis.

*tab* is the file with a table containing counts or relative abundances.

*tax* is the file with taxonomic information.

*meta* is the meta data file.

*fasta* is the fasta file.

*tree* is a Newick-formatted tree file

\**kwargs:

- *tab_sep* is separator used in tab file. Default is ',' (i.e. comma).
- *tax_sep* is separator used in tax file. Default is ',' (i.e. comma).
- *meta_sep* is separator used in meta data file. Default is ',' (i.e. comma).
- *fasta_seq_name_splitter* is a text pattern used to split the names of ASVs in the fasta file and only keep the first part. Default is *None*.
- *path* can be specified if all input files are located in the same folder. Then *path* is the location of the folder and the input files are only specified by file names. Default is ''. 
- *addTaxonPrefix* is True means that prefixes such a d__ for domain, c__ for class are added to the taxonomic names in the tax file. Default is True.
- *orderSeqs* is True means that if the ASV/OTU/MAG/bin names are ending with a number, they will be ordered in that way in the loaded dataframes. Default is True.

Example

.. code-block:: python

   import qdiv

   obj = qdiv.files.load(tab='example_tab.csv', fasta='example_fasta.fa', tree='example_tree.txt', meta='example_meta.csv')

obj is now a python dictionary containing five pandas dataframes ‘tab’, ‘tax’, ‘seq’, 'tree', and ‘meta’.

- 'tab' holds the counts associated with each OTU/ASV (rows) and sample (columns).
- 'tax' holds the taxonomic classification for each OTU/ASV. The row indices are OTU/ASV names, the column names are 'Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'.
- 'seq' holds the sequences. The row indices are the OTU/ASV names, the column 'seq' holds the sequences.
- 'tree' holds tree branches. The column 'nodes' holds the node names (end nodes are the same as OTU/ASV names), 'ASVs' holds a list of OTUs/ASVs connected to each node, and 'branchL' holds the lengths of the branches connected to each node.
- 'meta' holds the meta data. The row indices are the samples names (the same as the column headings in 'tab'). The columns are variables that define each sample as determined by the user in the meta data input file.

Save files
###########

.. code-block:: python

   files.printout(obj, path='', savename='', sep=',')

Saves the python object as files of the same type as the input files. (The Newick tree file is not returned.)

*obj* is the object to be printed.

*path* is the path to the directory holding the printed files.
 
*savename* is the name used for the printed files

*sep* is the separator to be used in the printed count table and meta data files (',' or ';' or '\\t').

Example

.. code-block:: python

   import qdiv

   qdiv.files.printout(obj, path='', savename='output', sep=',')

Load new taxonomy
#################

.. code-block:: python

   obj = files.read_sintax(obj, filename)

Reads taxonomy from a file generated using sintax in USEARCH or VSEARCH. 

*obj* is the object to which the taxonomy should be added. If the object already contains taxonomy information, it will be replaced by the new taxonomy.

*filename* is the text file with the new taxonomy.

