Diversity calculations
**********************

Distance between sequences
##########################

.. code-block:: python

   diversity.sequence_comparison(seq, savename='PhylDistMat')

Returns a csv file with pairwise distances between sequences in the data set. 
The distance is calculated as the Levenshtein distance dividied by the length of the longest distance in the pair.

*seq* is a dataframe holding the sequences, typically obj[‘seq’] 

*savename* specifies path and name of output file. 

Alpha diversity
###############

.. code-block:: python

   diversity.naive_alpha(tab, q=1)

Returns taxonomic (or naive) alpha diversity values (i.e. Hill numbers) for all samples.

*tab* is a frequency table holding the read counts, typically object['tab']

*q* is the diversity order for the Hill numbers.

.. code-block:: python
   
   diversity.phyl_alpha(tab, distmat, q=0):

Returns alpha diversity values that takes the pairwise distances between sequences into account. 
The values are the same as the functional diversity measures described by Chiu et al. (2014), *PLOS ONE,* 9(7), e100014.

*tab* is a frequency table holding the read counts, typically object['tab']

*distmat* is a dataframe with pairwise distances between the sequences of OTUs/SVs. The file can be generated wit the diversity.sequence_comparison() function.

*q* is the diversity order for the Hill numbers.

Beta diversity (dissimilarity)
##############################

.. code-block:: python

   naive_beta(tab, q=1, dis=True, viewpoint='local'):

Returns a dataframe of pairwise dissimilarities between samples. Taxonomic Hill-based dissimilarities of order q are calculated. 

*tab* is a frequency table holding the read counts, typically object['tab']

*q* is the diversity order.

If *dis* =True, dissimilarities constrained between 0 and 1 are calculated, 
if *dis* =False, beta diversity constrained between 1 and 2 are calculated.

*viewpoint* can ‘local’ or ‘regional’, the difference is described in Chao et al. (2014), *Annual Review of Ecology, Evolution, and Systematics,* 45, 297-324.

.. code-block:: python

   diversity.phyl_beta(tab, distmat, q=1, dis=True, viewpoint='local'):

Returns a dataframe of pairwise dissimilarities, which take sequence distances into account. The values are the functional beta diversity measures described by Chiu et al. (2014), *PLOS ONE,* 9(7), e100014.

*tab* is a frequency table holding the read counts, typically object['tab']

*distmat* is a dataframe with pairwise distances between the sequences of OTUs/SVs. The file can be generated wit the diversity.sequence_comparison() function.

*q* is the diversity order.

If *dis* =True, dissimilarities constrained between 0 and 1 are calculated, 
if *dis* =False, beta diversity constrained between 1 and 4 are calculated.

*viewpoint* can ‘local’ or ‘regional’, the difference is described in Chao et al. (2014), *Annual Review of Ecology, Evolution, and Systematics,* 45, 297-324.

.. code-block:: python

   diversity.jaccard(tab)

Returns a dataframe of pairwise Jaccard dissimilarities between samples.

*tab* is a frequency table holding the read counts, typically object['tab']

.. code-block:: python

   diversity.bray(tab)
   
Returns a dataframe of pairwise Bray-Curtis dissimilarities between samples.

*tab* is a frequency table holding the read counts, typically object['tab']

Null model analysis based on Raup-Crick
#######################################

.. code-block:: python

   diversity.rcq(obj, constrainingVar='None', randomization='abundance', weightingVar='None', weight=1, iterations=9, disIndex='Hill', distmat='None', q=1, compareVar='None', RCrange='Raup'):

The observed dissimilarities between samples are compared to a null distribution. 
The null model randomizes the frequency table to calculate a null expectation of the pairwise dissimilarities between samples. The is repeated several times (iterations) to get a null distribution.
During the randomization, the total OTU/SV count and read count for each sample are kept constant, but the distribution of reads between OTUs/SVs are randomized. 
The function returns a python dictionary with several items: 

- 'Obs' is the actually observed dissimilarity values.
- 'Nullmean' is the mean values of the null dissimilarities (i.e. the dissimilarities of the randomized tables); 
- 'Nullstd' is the standard devation; 
- 'RC' is the Raup-Crick measure (i.e. the number of times the actual dissimilarities are higher than the null expectation).

*obj* is the object. 

*constrainingVar* is a column heading in the meta data that can be used to constrain the randomizations so that read counts are only randomized with a certain category of samples. 

*randomization* specifies the randomization procedure: 

- 'abundance' means that SVs are drawn to each sample based on the total read counts in the frequency table (or part of the table defined by *constrainingVar* ) 
- 'frequency' means that SVs are drawn based on the number of samples in which they are detected. This method is the same as in Stegen et al. (2013). *ISME Journal,* 7(11), 2069-2079. 
- 'weighting' uses the abundance method but a meta data column (*weightingVar* ) can be used to categorize samples and the *weight* parameter decide the importance of the category of samples with the lowest richness. A *weight* of 0 means that the low-richness samples are not considered in the regional community used to populate the samples with read counts while a weight of 1 means that all sample groups have equal weighting.

*iterations* specifies the number of randomizations, 999 is the normal but could take several hours for large frequency tables. 

*disIndex* specifies the dissimilarity index to calculate: 'Jaccard', 'Bray', and 'Hill' are available choices, 
'Hill' refers to naive or phylogenetic dissimilarities of order q. If distmat is specified, diversity.phyl_beta are calculated.

*compareVar* is a column heading in the meta data. If compareVar is not None, 'RCmean' and 'RCstd' are returned 
and represents the mean and standard deviation of all pairwise comparison between the meta data categories specified present under compareVar 

If *RCrange* ='Raup' the range for the index will be 0 to 1, if it is 'Chase' it will -1 to 1. The names refer to Raup and Crick (1979), *J Paleontology,* 53(5), 1213-1227 and
Chase et al. (2011), *Ecosphere,* 2(2), 24.
