Subset and manipulate data
**************************

Subset to specific samples
##########################

.. code-block:: python

   subset.samples(obj, var='index', slist='None', keep0=False):

Returns an object subsetted to the selected samples.

*obj* is the object generated with files.load

*var* is the column heading in the meta data used for subsetting, if var=’index’ the actual sample names will be used

*slist* is a list of samples or meta data labels to keep

if *keep0* =True, OTUs/SVs, which have zero reads associated with them after subsetting will be kept, otherwise they will be removed from the data.

Subset to specific OTUs/SVs
###########################

.. code-block:: python

   subset.sequences(obj, svlist)

Returns and object subsetted to the selected OTUs/SVs.

*obj* is the object generated with files.load

*svlist* is a list of OTU or SV names that should be kept in the data set.

Subset to the most abundant OTUs/SVs
####################################

.. code-block:: python

   subset.abundant_sequences(obj, number=25, method='sum')

Returns an object with only the most abundant OTUs/SVs.

*obj* is the object generated with files.load
 
*number* specifies the number of SVs to keep 

*method* is the method used to rank OTUs/SVs. 
If *method* ='sum', the OTUs/SVs are ranked based on the sum of the relative abundances in all samples. 
If *method* ='max', they are ranked based on the max relative abundance in a sample.

Subset based on taxonomic classification
#########################################

.. code-block:: python

   subset.text_patterns(obj, subsetLevels=[], subsetPatterns=[]):

Search for specific text patterns among the taxonomic classifications. Returns an object subsetted to OTUs/SVs matching those text patterns.

*subsetLevels* is a list taxonomic levels in which text patterns are searched for, e.g. ['Family', 'Genus']

*subsetPatterns* is a list of text patterns to search for, e.g. ['Nitrosom', 'Brochadia']

Merge samples
##############

.. code-block:: python

   subset.merge_samples(obj, var='None', slist='None', keep0=False)

Returns an object where samples belonging the same category (as defined in the meta data) have been merged.

*var* is the column heading in metadata used to merge samples, the read counts for all samples with the same text in var column will be merged

*slist* is a list of names in meta data column which specify samples to keep, if slist='None' (default), all samples are kept

if *keep0* =False, all OTUs/SVs with 0 counts after merging will be discarded from the data.


Rarefy
######

.. code-block:: python

   subset.rarefy_table(tab, depth='min', seed='None', replacement=False)
   
   subset.rarefy_object(obj, depth='min', seed='None', replacement=False):

Rarefies a frequency table to a specific number of reads per sample. The function subset.rarefy_table() operates only on the frequency table and returns only a rarefied table. 
The function subset.rarefy_object() operates on the whole object and returns a whole object. 
This means that samples and OTUs/SVs which might have been dropped from the frequency table during rarefaction
are also dropped from the 'ra', 'tax', 'seq', and 'meta' dataframes of the object.

*tab* is the frequency table to be rarefied

*object* is the object containing the frequency table to be rarefied

if *depth* ='min', the minimum number of reads in a sample is used as rarefaction depth, otherwise a number can be specified 

*seed* sets a random state for reproducible results, use an integer.

if *replacement* =False, the function is similar to rarefaction without replacement, if *replacement* =True, it does rarefaction with replacement.

Consensus table
###############

.. code-block:: python

   subset.consensus(objlist, keepObj='best', taxa='None', alreadyAligned=False, differentLengths=True)

Takes a list of objects and returns a consensus object based on SVs found in all.

*objlist* is a list of objects 

*keepObj* makes it possible to specify which object in objlist that should be kept after filtering based on common SVs, specify with integer 
(0 is the first object, 1 is the second, etc), ‘best’ means that the object which has the highest fraction of its reads mapped to the common SVs is kept; 

*taxa* makes it possible to specify with an integer the object having taxa information that should be kept 
(0 is the first object, 1 is the second, etc), if 'None', the taxa information in the kept object is used 

if *alreadyAligned* =True, the subset.align_sequences function has already been run on the objects to make sure the same sequences in different objects have the same names 

if *differentLengths* =True, it assumes that the same SV inferred with different bioinformatics pipelines could have different sequence lengths. 
