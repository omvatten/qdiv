import qdiv.files, qdiv.subset, qdiv.stats, qdiv.diversity, qdiv.plot, qdiv.null

def citations(f='all'):
    if f == 'all' or f == 'qdiv':
        print('qdiv package:')
        print('Modin, O. https://github.com/omvatten/qdiv')
        print('Modin, O., R. Liébana, S. Saheb-Alam, B.-M. Wilén, C. Suarez, M. Hermansson and F. Persson (2020). Microbiome (accepted). Preprint at Research Square DOI: 10.21203/rs.2.24335/v3.')
    if f == 'all' or f in ['diversity.naive_alpha', 'diversity.naive_beta', 'diversity.naive_multi_beta', 'naive']:
        print('Naive diversity:')
        print('Hill, M. O. (1973). Ecology 54(2): 427-432.')
        print('Jost, L. (2006). OIKOS 113(2): 363-375.')
        print('Jost, L. (2007). Ecology 88(10): 2427-2439.')
    if f == 'all' or f in ['diversity.phyl_alpha', 'diversity.phyl_beta', 'diversity.phyl_multi_beta', 'phyl']:
        print('Phylogenetic diversity:')
        print('Chao, A., C.-H. Chiu and L. Jost (2010). Philosophical Transactions of the Royal Society B: Biological Sciences 365(1558): 3599-3609.')
        print('Chiu, C. H., L. Jost and A. Chao (2014). Ecological Monographs 84(1): 21-44.')
    if f == 'all' or f in ['diversity.func_alpha', 'diversity.func_beta', 'diversity.func_multi_beta', 'func']:
        print('Functional diversity:')
        print('Chiu, C. H. and A. Chao (2014). PLoS One 9(7): e100014.')
    if f == 'all' or f in ['diversity.evenness', 'diversity.dissimilarity_contributions', 'plot.dissimilarity_contributions']:
        print('Evenness:')
        print('Chao, A. and C. Ricotta (2019). Ecology 100(12): e02852.')
    if f == 'all' or f in ['null.rcq']:
        print('RCq null model:')
        print('Modin, O., R. Liébana, S. Saheb-Alam, B.-M. Wilén, C. Suarez, M. Hermansson and F. Persson (2020). Microbiome (accepted). Preprint at Research Square DOI: 10.21203/rs.2.24335/v3.')
        print('Raup, D. M. and R. E. Crick (1979). Journal of Paleontology 53(5): 1213-1227.')
        print('Chase, J. M., N. J. B. Kraft, K. G. Smith, M. Vellend and B. D. Inouye (2011). Ecosphere 2(2): 24.')
        print('Stegen, J. C., X. Lin, J. K. Fredrickson, X. Chen, D. W. Kennedy, C. J. Murray, M. L. Rockhold and A. Konopka (2013). ISME Journal 7(11): 2069-2079.')
    if f == 'all' or f in ['null.nriq', 'null.ntiq']:
        print('NRI and NTI')
        print('Webb, C. O., D. D. Ackerly, M. A. McPeek and M. J. Donoghue (2002). Annual Review of Ecology and Systematics 33(1): 475-505.')
    if f == 'all' or f in ['null.beta_nriq', 'null.beta_ntiq']:
        print('beta NRI and beta NTI')
        print('Fine, P. V. A. and S. W. Kembel (2011). Ecography 34: 552-565.')
    if f == 'all' or f in ['stats.mantel']:
        print('Mantel test')
        print('Mantel, N. (1967). Cancer Research 27(2): 209-220.')
    if f == 'all' or f in ['stats.permanova']:
        print('Permanova test')
        print('Anderson, M. (2001). Austral Ecology 26(1): 32-46.')
