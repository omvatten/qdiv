Information and statistics
**************************

Get information
###############

.. code-block:: python

   stats.print_info(obj)

Opens a dialogue box with information about the number of samples, number of reads, minimum and maximum number of reads in a samples, meta data, etc.

.. code-block:: python

   stats.taxa(obj, savename='None')

Returns a pandas dataframe with information about the number of taxa at different taxonomic levels in the data.

*obj* is an object.

if *savename* is specified, a csv file with the information is saved.

Mantel test
###########

.. code-block:: python

   stats.mantel(dis1, dis2, method='spearman', getOnlyStat=False, permutations=99)

Carries out a mantel tests and return a list containing the test statistic and the p value.

*dis1* and *dis2* are two dissimilarity matrices. 

*method* is the test statistic used. 'spearman', 'pearson', or 'absDist' can be chosen. The methods converts each dissimilarity matrix to an array of numbers and investigates the correlation between the two arrays.

- 'spearman' is a rank-based correlation coefficient.
- 'pearson' is the product-moment correlation coefficient conventionally used in the Mantel test.
- 'absDist' is the mean absolute distance between the numbers in the two arrays.

If *getOnlyStat* =True, only the test statistic will be returned, no permutations and consequently no p-value will be calculated. 

*permutations* is the number of randomizations carried out. 

Permanova
###########

.. code-block:: python

   stats.permanova(dis, meta, var, permutations=99)

Carries out a permanova tests and return a list containing the test statistic and the p value.

*dis* is a dissimilarity matrix. 

*meta* is the meta data, e.g. obj['meta'].

*var* is the column in the meta data that holds the categories of samples being compared. 

*permutations* is the number of randomizations carried out.
