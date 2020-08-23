Citations
*********

To see the citations in qdiv, use:

.. code-block:: python

   import qdiv

   qdiv.citations(f='all')

Prints a list of references used in the development of python. 

*f* is the qdiv function. For example, f='diversity.naive_alpha' prints the references that were especially relevant for developing the diversity.naive_alpha algorithm.
f='all' prints all references listed below.

qdiv package:

- Modin, O. https://github.com/omvatten/qdiv
- Modin, O., R. Liébana, S. Saheb-Alam, B.-M. Wilén, C. Suarez, M. Hermansson and F. Persson (2020). Microbiome (accepted). Preprint at Research Square DOI: 10.21203/rs.2.24335/v3.

Naive diversity (diversity.naive_alpha, diversity.naive_beta, diversity.naive_multi_beta):

- Hill, M. O. (1973). Ecology 54(2): 427-432.
- Jost, L. (2006). OIKOS 113(2): 363-375.
- Jost, L. (2007). Ecology 88(10): 2427-2439.

Phylogenetic diversity (diversity.phyl_alpha, diversity.phyl_beta, diversity.phyl_multi_beta):

- Chao, A., C.-H. Chiu and L. Jost (2010). Philosophical Transactions of the Royal Society B: Biological Sciences 365(1558): 3599-3609.
- Chiu, C. H., L. Jost and A. Chao (2014). Ecological Monographs 84(1): 21-44.

Functional diversity (diversity.func_alpha, diversity.func_beta, diversity.func_multi_beta):

- Chiu, C. H. and A. Chao (2014). PLoS One 9(7): e100014.

Evenness (diversity.evenness, diversity.dissimilarity_contributions, plot.dissimilarity_contributions):

- Chao, A. and C. Ricotta (2019). Ecology 100(12): e02852.

Raup-Crick null model (null.rcq):

- Modin, O., R. Liébana, S. Saheb-Alam, B.-M. Wilén, C. Suarez, M. Hermansson and F. Persson (2020). Microbiome (accepted). Preprint at Research Square DOI: 10.21203/rs.2.24335/v3.
- Raup, D. M. and R. E. Crick (1979). Journal of Paleontology 53(5): 1213-1227.
- Chase, J. M., N. J. B. Kraft, K. G. Smith, M. Vellend and B. D. Inouye (2011). Ecosphere 2(2): 24.
- Stegen, J. C., X. Lin, J. K. Fredrickson, X. Chen, D. W. Kennedy, C. J. Murray, M. L. Rockhold and A. Konopka (2013). ISME Journal 7(11): 2069-2079.

Phylogenetic null models (null.nriq, null.ntiq, null.beta_nriq, null.beta_ntiq):

- Webb, C. O., D. D. Ackerly, M. A. McPeek and M. J. Donoghue (2002). Annual Review of Ecology and Systematics 33(1): 475-505.
- Fine, P. V. A. and S. W. Kembel (2011). Ecography 34: 552-565.

Mantel test (stats.mantel):

- Mantel, N. (1967). Cancer Research 27(2): 209-220.

Permanova test (stats.permanova):

- Anderson, M. (2001). Austral Ecology 26(1): 32-46.