Diversity calculations
**********************

*Alpha diversity* refers to diversity of OTUs/ASVs within a sample. *Beta diversity* refers to differences in community composition between different sampes. 
We have three ways of measuring diversity: 

- *naive* diversity considers all OTUs/ASVs to be equally differerent from each (this is also called taxonomic diversity);
- *phylogenetic* diversity takes evolutionary distances between OTUs/ASVs in a phylogenetic tree into account;
- *functional* diversity takes pairwise distances between OTUs/ASVs in a distance matrix into account.
 
In the Hill-based framework used to calculate the naive-, phylogenetic-, and functional diversity measures, the importance of the relative abundance of OTUs/ASVs can be tuned by the diversity order (q). 

- A q of 0 means that relative abundance is not considered and OTUs/ASVs are treated as either present or absent in the samples.
- A q of 1 means that that each OTU/ASV is weighted exactly according to its relative abundance in the sample.
- A q > 1 means that more weight is given to OTUs/ASVs with high relative abundance.

Alpha diversity
###############

.. code-block:: python

   diversity.naive_alpha(tab, q=1, **kwargs)

Returns taxonomic (or naive) alpha diversity values (i.e. Hill numbers) for all samples.

*tab* is a frequency table holding the read counts, typically obj['tab']

*q* is the diversity order.

\**kwargs:

- *use_values_in_tab*. Default is *False*. If *True*, it will take values in tab without normalizing them to relative abundances. 

.. code-block:: python
   
   diversity.phyl_alpha(tab, tree, q=1, index='PD', **kwargs):

Returns alpha diversity values that takes phylogenetic tree distances between sequences into account. 

*tab* is a count table holding the read counts, typically obj['tab']

*tree* is the tree dataframe, typically obj['tree'].

*q* is the diversity order.

*index* can be: 'PD', 'D', or 'H'. 'PD' is the phylogenetic diversity, 'D' is the Hill diversity number corrected for phylogenetic relationships, and 'H' is the phylogenetic entropy.
See Chao et al. (2010). *Phil. Trans. R. Soc. B,* 365, 3599-3609 (DOI: 10.1098/rstb.2010.0272).

\**kwargs:

- *use_values_in_tab*. Default is *False*. If *True*, it will take values in tab without normalizing them to relative abundances. 

.. code-block:: python
   
   diversity.func_alpha(tab, distmat, q=0, index='FD', **kwargs):

Returns alpha diversity values that takes the pairwise distances from between sequences into account. 
The values are the same as the functional diversity measures described by Chiu et al. (2014), *PLOS ONE,* 9(7), e100014 (DOI: 10.1371/journal.pone.0100014).

*tab* is a count table holding the read counts, typically obj['tab']

*distmat* is a dataframe with pairwise distances between the sequences of OTUs/SVs. The file can be generated wit the stats.sequence_comparison() function.

*q* is the diversity order for the Hill numbers.

*index* can be: 'FD', 'MD', or 'D'. 'FD' is the functional diversity ("the effective total distance between species"), 'MD' is the mean functional diversity ("the effective sum of pairwise distances between a species and all other species"),
and 'D' is the functional Hill number ("the effective number of equally abundant and equally distinct species"). See Chiu et al. (2014), *PLOS ONE,* 9(7), e100014 (DOI: 10.1371/journal.pone.0100014), the descriptions in quotations marks are from that paper.

\**kwargs:

- *use_values_in_tab*. Default is *False*. If *True*, it will take values in tab without normalizing them to relative abundances. 

Beta diversity (dissimilarity)
##############################

The difference in community composition between different samples can be expressed as *beta diversity* or *dissimilarity*. Beta diversity takes a value between 1 (if the compared samples are identical to each) and N (which is the number of samples being compared.
For functional beta diversity it goes between 1 and NxN.
Dissimilarity takes a value between 0 (if the compared samples are identical) and 1 (if the compared samples are completely different). Dissimilarity can be calculated with a *local* or *regional* viewpoint. 
The local viewpoint quantifies the effective proportion of OTUs/ASVs in a sample that is not shared across all samples. 
The regional viewpoint quantifies the effective proportion of OTUs/ASVs in the pooled samples that is not shared across all samples.
See Chao and Chiu (2016), *Methods in Ecology and Evolution,* 7, 919-928.

.. code-block:: python

   diversity.naive_beta(tab, q=1, dis=True, viewpoint='local', **kwargs):

Returns a dataframe of pairwise dissimilarities between samples. Taxonomic (or naive) Hill-based dissimilarities of order q are calculated. 

*tab* is a frequency table holding the read counts, typically object['tab']

*q* is the diversity order.

If *dis* =True, dissimilarities constrained between 0 and 1 are calculated, 
if *dis* =False, beta diversity constrained between 1 and 2 are calculated.

*viewpoint* can ‘local’ or ‘regional’, the difference is described in Chao et al. (2014), *Annual Review of Ecology, Evolution, and Systematics,* 45, 297-324.

\**kwargs:

- *use_values_in_tab*. Default is *False*. If *True*, it will take values in tab without normalizing them to relative abundances. 

.. code-block:: python

   diversity.phyl_beta(tab, tree, q=1, dis=True, viewpoint='local', **kwargs):

Returns a dataframe of pairwise dissimilarities, which take a phylogenetic tree into account. 
The values are the phylogenetic beta diversity measures described by Chiu et al. (2014), *Ecological Monographs,* 84(1), 21-44.

*tab* is a frequency table holding the read counts, typically obj['tab']

*tree* is a dataframe with phylogenetic tree information, typically obj['tree'].

*q* is the diversity order.

If *dis* =True, dissimilarities constrained between 0 and 1 are calculated, 
if *dis* =False, beta diversity constrained between 1 and 2 are calculated.

*viewpoint* can ‘local’ or ‘regional’.

\**kwargs:

- *use_values_in_tab*. Default is *False*. If *True*, it will take values in tab without normalizing them to relative abundances. 

.. code-block:: python

   diversity.func_beta(tab, distmat, q=1, dis=True, viewpoint='local', **kwargs):

Returns a dataframe of pairwise dissimilarities, which take pairwise sequence distances into account. The values are the functional beta diversity measures described by Chiu et al. (2014), *PLOS ONE,* 9(7), e100014.

*tab* is a frequency table holding the read counts, typically obj['tab'].

*distmat* is a dataframe with pairwise distances between the sequences of OTUs/ASVs. The file can be generated with the stats.sequence_comparison() function.

*q* is the diversity order.

If *dis* =True, dissimilarities constrained between 0 and 1 are calculated, 
if *dis* =False, beta diversity constrained between 1 and 4 are calculated.

*viewpoint* can ‘local’ or ‘regional’.

\**kwargs:

- *use_values_in_tab*. Default is *False*. If *True*, it will take values in tab without normalizing them to relative abundances. 

.. code-block:: python

   diversity.jaccard(tab, **kwargs)

Returns a dataframe of pairwise Jaccard dissimilarities between samples.

*tab* is a count table holding the read counts, typically object['tab']

\**kwargs:

- *use_values_in_tab*. Default is *False*. If *True*, it will take values in tab without normalizing them to relative abundances. 

Note that Jaccard dissimilarity is the same as naive_beta(tab, q=0, dis=True, viewpoint='regional')

.. code-block:: python

   diversity.bray(tab, **kwargs)
   
Returns a dataframe of pairwise Bray-Curtis dissimilarities between samples.

*tab* is a count table holding the read counts, typically obj['tab']

\**kwargs:

- *use_values_in_tab*. Default is *False*. If *True*, it will take values in tab without normalizing them to relative abundances. 

.. code-block:: python

   diversity.naive_multi_beta(obj, var='None', q=1, **kwargs)

Returns a pandas dataframe containing taxonomic (naive) beta diversity, and local- and regional dissimilarity values for categories of samples.

*obj* is the qdiv object.

*var* is the column heading in the meta data used to categorize the samples. If a category has two or more samples, beta diversity for that category is calculated.

*q* is the diversity order. 

\**kwargs:

- *use_values_in_tab*. Default is *False*. If *True*, it will take values in tab without normalizing them to relative abundances. 

.. code-block:: python

   diversity.phyl_multi_beta(obj, var='None', q=1, **kwargs)

Returns a pandas dataframe containing phylogenetic beta diversity, and local- and regional dissimilarity values for categories of samples.

*obj* is the qdiv object.

*var* is the column heading in the meta data used to categorize the samples. If a category has two or more samples, beta diversity for that category is calculated.

*q* is the diversity order. 

\**kwargs:

- *use_values_in_tab*. Default is *False*. If *True*, it will take values in tab without normalizing them to relative abundances. 

.. code-block:: python

   diversity.func_multi_beta(obj, distmat, var='None', q=1, **kwargs)

Returns a pandas dataframe containing phylogenetic beta diversity, and local- and regional dissimilarity values for categories of samples.

*obj* is the qdiv object.

*distmat* is a distance matrix with pairwise distances between OTUs/ASVs, typically generated by the stats.sequence_comparison() function.

*var* is the column heading in the meta data used to categorize the samples. If a category has two or more samples, beta diversity for that category is calculated.

*q* is the diversity order. 

\**kwargs:

- *use_values_in_tab*. Default is *False*. If *True*, it will take values in tab without normalizing them to relative abundances. 

.. code-block:: python

   diversity.dissimilarity_contributions(obj, var='None', q=1, divType='naive', index='local', **kwargs)

Returns a pandas dataframe with information about dissimilarity, number of samples, and the percentage contribution of each OTU/ASV to the observed dissimilarity.

*obj* is the qdiv object.

*var* is the column heading in the meta data used to categorize the samples. If a category has two or more samples, dissimilarity for that category is calculated.

*q* is the diversity order. 

*divType* is the type of diversity: 'naive' or 'phyl'.

*index* is the type of dissimilarity index (either local or regional)

\**kwargs:

- *use_values_in_tab*. Default is *False*. If *True*, it will take values in tab without normalizing them to relative abundances. 

Evenness
########

.. code-block:: python

   diversity.evenness(tab, tree='None', distmat='None', q=1, divType='naive', index='local', perspective='samples', **kwargs)

Returns evenness calculated according to Chao and Ricotta (2019) *Ecology,* 100(12), e02852. If q is 1, divType is 'naive', and index is either 'CR1', 'CR2', 'local' or 'regional', the evenness is identical to Pielou's evenness. 

*tab* is a count table, typically obj['tab'] 

*tree* would typically be obj['tree']. It must be specified only if divType='phyl'.

*distmat* is a distance matrix. It must be specified only if divType='func'.

*q* is diversity order.

*divType* is 'naive', 'phyl', or 'func'.

*index* can be 'CR1', 'CR2', 'CR3', 'CR4', or 'CR5'. These refer to the indices described in Table 1 in Chao and Ricotta (2019). *index* can also be 'local', which is equal to CR2, or 'regional', which is equal to CR1.

*perspective* can be 'samples', which means an evenness value is calculated for each sample (column) in the count table.
*perspective* can also be 'taxa', which means an evenness value is calculated for each OTU/ASV (row) in the count table.

\**kwargs:

- *use_values_in_tab*. Default is *False*. If *True*, it will take values in tab without normalizing them to relative abundances. 
