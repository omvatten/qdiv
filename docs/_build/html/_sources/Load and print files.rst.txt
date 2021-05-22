Load and print files
********************

Loading files
#############

.. code-block:: python

   files.load(path='', tab='None', fasta='None', tree='None', meta='None', sep=',', addPrefix=True, orderSeqs=True)

Loads files into an object, which is used for further analysis.

*path* is the path to the directory holding the input files, *tab* is the name of the count table file, 
*fasta* is the name of the fasta file, *tree* is a Newick-formatted tree file, *meta* is the name of meta data file, *sep* is the separator used in the count table and meta data files (',' or ';' or '\\t').

if *addPrefix* =True (default), a g__ will be added before genus name, f__ before family name, etc.

if *orderSeqs* =True (default), sequences will be sorted numerically if the sequence identifiers contain a number. 

Example

.. code-block:: python

   import qdiv

   obj = qdiv.files.load(path='', tab='example_tab.csv', fasta='example_fasta.fa', tree='example_tree.txt', meta='example_meta.csv', sep=',')

obj is now a python dictionary containing six pandas dataframes ‘tab’, ‘ra’, ‘tax’, ‘seq’, 'tree', and ‘meta’.

- 'tab' holds the counts associated with each OTU/ASV (rows) and sample (columns).
- 'ra' holds the percentage relative abundance of reads per sample associated with each OTU/ASV (rows) and sample (columns).
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
 
*savename* is the name used for the printed files, *sep* is the separator to be used in the printed count table and meta data files (',' or ';' or '\\t').

Example

.. code-block:: python

   import qdiv

   qdiv.files.printout(obj, path='', savename='output', sep=',')

Load new taxonomy
#################

.. code-block:: python

   files.read_rdp(obj, filename, cutoff=70)

Reads taxonomy from a file generated using the RDP classifier (https://rdp.cme.msu.edu/classifier/classifier.jsp). Go to the website, upload your fasta file, click submit. 
When the sequences have been processed, click 'show assignment detail for Root only', then click 'download allrank results'. 

*obj* is the object to which the taxonomy should be added. If the object already contains taxonomy information, it will be replaced by the new taxonomy.

*filename* is the text file with the new taxonomy.

*cutoff* is the minimum percentage needed to include a taxonomic level. 

.. code-block:: python

   files.read_sintax(obj, filename)

Reads taxonomy from a file generated using sintax in USEARCH or VSEARCH. 

*obj* is the object to which the taxonomy should be added. If the object already contains taxonomy information, it will be replaced by the new taxonomy.

*filename* is the text file with the new taxonomy.

.. code-block:: python

   files.read_sina(obj, filename, taxonomy='silva')

Reads taxonomy from a file generated using the SINA classifier (https://www.arb-silva.de/aligner/). 

*obj* is the object to which the taxonomy should be added. If the object already contains taxonomy information, this will be replaced by the new taxonomy.

*filename* is the text file with the new taxonomy.

*taxonomy* options are: 'silva', 'ltp', 'rdp', 'gtdb', 'embl_ebi_ena'
