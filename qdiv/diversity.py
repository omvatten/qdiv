import pandas as pd
import numpy as np
import math
from . import subset
from . import hfunc
pd.options.mode.chained_assignment = None  # default='warn'

# ALPHA DIVERSITY

# Returns naive alpha diversity of order q for all samples
# naive = species diversity or taxonomic diversity, does not consider similarity between ASVs
def naive_alpha(tab, q=1):
    ra = tab / tab.sum()
    if isinstance(ra, pd.Series):
        ra = pd.DataFrame(ra.to_numpy(), index=ra.index, columns=[ra.name])

    if q == 1:
        raLn = ra
        raLn[ra > 0] = ra[ra > 0] * np.log(ra[ra > 0])
        Hillvalues = np.exp(-raLn.sum())
        return Hillvalues
    else:
        rapow = ra
        rapow[ra > 0] = ra[ra > 0].pow(q)
        rapow = rapow.sum()
        Hillvalues = rapow.pow(1 / (1 - q))
        return Hillvalues

# Returns phylogenetic alpha diversity of order q for all samples
# Chao et al. Phil Trans RS B, 2010.
def phyl_alpha(tab, tree, q=1, index='PD'):
    ra = tab / tab.sum()
    if isinstance(ra, pd.Series):
        ra = pd.DataFrame(ra.to_numpy(), index=ra.index, columns=[ra.name])

    #Make a tree df with ra associated with each branch
    tree2 = pd.DataFrame(0, index=tree.index, columns=ra.columns)
    for ix in tree2.index:
        asvlist = tree.loc[ix, 'ASVs']
        ra_branch = ra.reindex(asvlist).sum()
        tree2.loc[ix] = ra_branch

    #Make Tavg series
    Tavg = pd.Series(index=ra.columns)
    for smp in ra.columns:
        Tavg[smp] = tree['branchL'].mul(tree2[smp]).sum()

    #Calculate diversities
    tree_calc = tree2.copy()
    
    if q == 1:
        tree_calc[tree_calc > 0] = tree_calc[tree_calc > 0].applymap(math.log)
        tree_calc = tree2.mul(tree_calc)
        tree_calc = tree_calc.mul(tree['branchL'], axis=0) #Multiply with branch length
        tree_calc = tree_calc.div(Tavg, axis=1) #Divide by Tavg
        tree_calc = -tree_calc.sum()
        hill_div = tree_calc.apply(math.exp)
    else:
        tree_calc[tree_calc > 0] = tree_calc[tree_calc > 0].pow(q) #Take power of q
        tree_calc = tree_calc.mul(tree['branchL'], axis=0) #Multiply with branch length
        tree_calc = tree_calc.div(Tavg, axis=1) #Divide by Tavg
        tree_calc = tree_calc.sum()
        hill_div = tree_calc.pow(1/(1-q))
    
    if index == 'PD':
        return hill_div.mul(Tavg)
    elif index == 'D':
        return hill_div
    elif index == 'H':
        return tree_calc
    else:
        print('Specify index: PD or D or H')

# Returns functional alpha diversity of order q for all samples
# FD as in Chiu et al. Plos One, 2014
def func_alpha(tab, distmat, q=1, index='FD'):
    ra = tab / tab.sum()
    if isinstance(ra, pd.Series):
        ra = pd.DataFrame(ra.to_numpy(), index=ra.index, columns=[ra.name])

    outdf = pd.Series(0, index=ra.columns)
    svlist = ra.index.tolist()
    distmat = distmat.loc[svlist, svlist]
    Qframe = hfunc.rao(ra, distmat)

    if q == 1:
        for smp in ra.columns:
            ra2mat = pd.DataFrame(np.outer(ra[smp].to_numpy(), ra[smp].to_numpy()), index=ra.index, columns=ra.index)
            ra2Lnmat = ra2mat.copy()
            mask = ra2Lnmat > 0
            ra2Lnmat[mask] = ra2Lnmat[mask].applymap(math.log)
            ra2ochLn = ra2mat.mul(ra2Lnmat)
            dQmat = distmat.mul(1 / Qframe.loc[smp])
            dQ_ra2_Ln = dQmat.mul(ra2ochLn)
            Chiuvalue = math.exp(-0.5 * sum(dQ_ra2_Ln.sum()))
            outdf.loc[smp] = Chiuvalue
    else:
        for smp in ra.columns:
            ra2mat = pd.DataFrame(np.outer(ra[smp].to_numpy(), ra[smp].to_numpy()), index=ra.index, columns=ra.index)
            mask = ra2mat > 0
            ra2mat[mask] = ra2mat[mask].pow(q)
            dQmat = distmat.mul(1 / Qframe.loc[smp])
            ra2dq = (ra2mat.mul(dQmat))
            Chiuvalue = pow(sum(ra2dq.sum()), 1 / (2 * (1 - q)))
            outdf.loc[smp] = Chiuvalue
    if index == 'D':
        return outdf
    elif index == 'MD':
        MD = outdf.mul(Qframe)
        return MD
    elif index == 'FD':
        MD = outdf.mul(Qframe)
        return outdf.mul(MD)

# BETA DIVERSITY

# Returns matrix of paiwise dissimilarities of order q
# Viewpoint can be local or regional, see Chao et al. 2014
def naive_beta(tab, q=1, dis=True, viewpoint='local'):
    ra = tab / tab.sum()
    if isinstance(ra, pd.Series) or len(ra.columns) < 2:
        print('Too few samples in tab.')
        return None

    smplist = ra.columns
    outdf = pd.DataFrame(0, index=smplist, columns=smplist)
    for smp1nr in range(len(smplist) - 1):
        smp1 = smplist[smp1nr]
        for smp2nr in range(smp1nr + 1, len(smplist)):
            smp2 = smplist[smp2nr]

            if q == 1:
                mask1 = ra[smp1] > 0
                raLn1 = ra[smp1][mask1] * np.log(ra[smp1][mask1])
                raLn1 = raLn1.sum()
                mask2 = ra[smp2] != 0
                raLn2 = ra[smp2][mask2] * np.log(ra[smp2][mask2])
                raLn2 = raLn2.sum()
                alphavalue = math.exp(-0.5 * raLn1 - 0.5 * raLn2)

                ra_mean = ra[[smp1, smp2]].mean(axis=1)
                maskg = ra_mean > 0
                raLng = ra_mean[maskg] * np.log(ra_mean[maskg])
                raLng = raLng.sum()
                gammavalue = math.exp(-raLng)

                betavalue = gammavalue / alphavalue
                outdf.loc[smp1, smp2] = betavalue
                outdf.loc[smp2, smp1] = betavalue
            else:
                mask1 = ra[smp1] > 0
                ra1 = ra[smp1][mask1]
                ra1pow = ra1.pow(q)
                ra1powsum = ra1pow.sum()
                mask2 = ra[smp2] > 0
                ra2 = ra[smp2][mask2]
                ra2pow = ra2.pow(q)
                ra2powsum = ra2pow.sum()
                alphavalue = (0.5 * ra1powsum + 0.5 * ra2powsum) ** (1 / (1 - q))

                ra_mean = ra[[smp1, smp2]].mean(axis=1)
                maskg = (ra_mean != 0)
                rag = ra_mean[maskg]
                ragpow = rag.pow(q)
                ragpowsum = ragpow.sum()
                gammavalue = ragpowsum ** (1 / (1 - q))

                betavalue = gammavalue / alphavalue
                outdf.loc[smp1, smp2] = betavalue
                outdf.loc[smp2, smp1] = betavalue

    if dis:
        dist = hfunc.beta2dist(beta=outdf, q=q, N=2, divType='naive', viewpoint=viewpoint)
        return dist
    else:
        return outdf

# Returns matrix of paiwise dissimilarities of order q
# Viewpoint can be local or regional, see Chao et al. 2014
def phyl_beta(tab, tree, q=1, dis=True, viewpoint='local'):
    ra = tab / tab.sum()
    if isinstance(ra, pd.Series) or len(ra.columns) < 2:
        print('Too few samples in tab.')
        return None

    #Make a tree df with ra associated with each branch
    tree2 = pd.DataFrame(0, index=tree.index, columns=ra.columns)
    for ix in tree2.index:
        asvlist = tree.loc[ix, 'ASVs']
        ra_branch = ra.reindex(asvlist).sum()
        tree2.loc[ix] = ra_branch

    #Go through each pair of samples
    smplist = ra.columns
    outdf = pd.DataFrame(0, index=smplist, columns=smplist)
    for smp1nr in range(len(smplist) - 1):
        smp1 = smplist[smp1nr]
        for smp2nr in range(smp1nr + 1, len(smplist)):
            smp2 = smplist[smp2nr]

            subtree = tree2[[smp1, smp2]].copy()
            subtree['gamma'] = subtree.mean(axis=1)
            Tavg = subtree['gamma'].mul(tree['branchL']).sum()

            #Get gamma
            g_df = subtree['gamma'].copy()
            if q == 1:
                g_df[g_df > 0] = g_df[g_df > 0].apply(math.log)
                g_df = g_df.mul(subtree['gamma'])
                g_df = g_df.mul(tree['branchL'])
                g_df = g_df.div(Tavg)
                g_df = -g_df.sum()
                gamma_div = math.exp(g_df)
            else:
                g_df = g_df.div(Tavg)
                g_df[g_df > 0] = g_df[g_df > 0].pow(q)
                g_df = g_df.mul(tree['branchL'])
                g_df = g_df.sum()
                g_df = g_df**(1/(1-q))
                gamma_div = g_df / Tavg

            #Get alpha
            subtree[[smp1, smp2]] = subtree[[smp1, smp2]].div(2*Tavg)
            a_df = subtree[[smp1, smp2]].copy()
            if q == 1:
                a_df[a_df > 0] = a_df[a_df > 0].applymap(math.log)
                a_df = a_df.mul(subtree[[smp1, smp2]])
                a_df = a_df.mul(tree['branchL'], axis=0)
                a_df = -sum(a_df.sum()) - math.log(Tavg*2)
                alpha_div = math.exp(a_df)
            else:
                a_df[a_df > 0] = a_df[a_df > 0].pow(q)
                a_df = a_df.sum(axis=1)
                a_df = a_df.mul(tree['branchL'])
                a_df = a_df.sum()
                a_df = a_df**(1/(1-q))
                alpha_div = a_df / (Tavg * 2)

            beta_div = gamma_div / alpha_div
            outdf.loc[smp1, smp2] = beta_div
            outdf.loc[smp2, smp1] = beta_div

    if dis:
        dist = hfunc.beta2dist(beta=outdf, q=q, N=2, divType='phyl', viewpoint=viewpoint)
        return dist
    else:
        return outdf

# Returns matrix of paiwise phylogenetic dissimilarities of order q
# Based on local functional overlaps as defined in Chao et al. 2014
def func_beta(tab, distmat, q=1, dis=True, viewpoint='local'):
    ra = tab / tab.sum()
    if isinstance(ra, pd.Series) or len(ra.columns) < 2:
        print('Too few samples in tab.')
        return None

    smplist = list(ra.columns)
    outD = pd.DataFrame(0, index=smplist, columns=smplist)

    # For showing progress
    total_comp = (len(smplist)**2)/2
    show_after = total_comp / 50
    print('Progress in calculation func_beta 0', end='%.. ')
    counter = 0

    for smp1nr in range(len(smplist) - 1):
        for smp2nr in range(smp1nr + 1, len(smplist)):

            # For showing progress
            counter += 1
            if counter%show_after == 0:
                print(int(100*counter/total_comp), end='%.. ')

            smp1 = smplist[smp1nr]
            smp2 = smplist[smp2nr]

            ra12 = ra.loc[:, [smp1, smp2]]
            ra12['mean'] = ra12.mean(axis=1)
            Qvalues = hfunc.rao(ra12, distmat)
            Qpooled = Qvalues['mean']
            dqmat = distmat.mul(1 / Qpooled)

            if q == 1:
                # Get gamma
                mask = ra12['mean'] > 0
                ra2mat = pd.DataFrame(np.outer(ra12['mean'][mask], ra12['mean'][mask]), index=ra12[mask].index,
                                      columns=ra12[mask].index)
                ra2matq = ra2mat * np.log(ra2mat)
                ra2dq = ra2matq.mul(dqmat)
                Dg = math.exp(-0.5 * sum(ra2dq.sum()))

                # Get alpha
                mask1 = ra12[smp1] > 0
                ra2mat = pd.DataFrame(np.outer(ra12[mask1][smp1], ra12[mask1][smp1]), index=ra12[mask1].index,
                                      columns=ra12[mask1].index)
                ra2mat = ra2mat / 4
                ra2matq = ra2mat * np.log(ra2mat)
                ra2dq = ra2matq.mul(dqmat)
                asum1 = sum(ra2dq.sum())

                mask2 = ra12[smp2] > 0
                ra2mat = pd.DataFrame(np.outer(ra12[mask2][smp2], ra12[mask2][smp2]), index=ra12[mask2].index,
                                      columns=ra12[mask2].index)
                ra2mat = ra2mat / 4
                ra2matq = ra2mat * np.log(ra2mat)
                ra2dq = ra2matq.mul(dqmat)
                asum2 = sum(ra2dq.sum())

                ra2mat = pd.DataFrame(np.outer(ra12[mask1][smp1], ra12[mask2][smp2]), index=ra12[mask1].index,
                                      columns=ra12[mask2].index)
                ra2mat = ra2mat / 4
                ra2matq = ra2mat * np.log(ra2mat)
                ra2dq = ra2matq.mul(dqmat)
                asum12 = sum(ra2dq.sum())

                Da = 0.5 * math.exp(-0.5 * (asum1 + asum2 + 2 * asum12))

                # Calculate beta
                outD.loc[smp1, smp2] = Dg / Da;
                outD.loc[smp2, smp1] = Dg / Da

            else:
                # Get gamma
                mask = ra12['mean'] > 0
                ra2mat = pd.DataFrame(np.outer(ra12['mean'][mask], ra12['mean'][mask]), index=ra12[mask].index,
                                      columns=ra12[mask].index)
                ra2matq = ra2mat.pow(q)
                ra2dq = ra2matq.mul(dqmat)
                Dg = pow(sum(ra2dq.sum()), 1 / (2 * (1 - q)))

                # Get alpha
                mask1 = ra12[smp1] > 0
                ra2mat = pd.DataFrame(np.outer(ra12[mask1][smp1], ra12[mask1][smp1]), index=ra12[mask1].index,
                                      columns=ra12[mask1].index)
                ra2mat = ra2mat / 4
                ra2matq = ra2mat.pow(q)
                ra2dq = ra2matq.mul(dqmat)
                asum1 = sum(ra2dq.sum())

                mask2 = ra12[smp2] > 0
                ra2mat = pd.DataFrame(np.outer(ra12[mask2][smp2], ra12[mask2][smp2]), index=ra12[mask2].index,
                                      columns=ra12[mask2].index)
                ra2mat = ra2mat / 4
                ra2matq = ra2mat.pow(q)
                ra2dq = ra2matq.mul(dqmat)
                asum2 = sum(ra2dq.sum())

                ra2mat = pd.DataFrame(np.outer(ra12[mask1][smp1], ra12[mask2][smp2]), index=ra12[mask1].index,
                                      columns=ra12[mask2].index)
                ra2mat = ra2mat / 4
                ra2matq = ra2mat.pow(q)
                ra2dq = ra2matq.mul(dqmat)
                asum12 = sum(ra2dq.sum())

                Da = 0.5 * pow((asum1 + asum2 + 2 * asum12), 1 / (2 * (1 - q)))

                # Calculate beta
                outD.loc[smp1, smp2] = Dg / Da
                outD.loc[smp2, smp1] = Dg / Da

    outFD = outD.pow(2)
    print('100 %')

    if dis:
        return hfunc.beta2dist(beta=outFD, q=q, N=2, divType='func', viewpoint=viewpoint)
    else:
        return outFD

# Calculate matrix of pairwise Bray-Curtis dissimilarities
def bray(tab):
    ra = tab / tab.sum()
    if isinstance(ra, pd.Series) or len(ra.columns) < 2:
        print('Too few samples in tab.')
        return None

    smplist = list(tab.columns)
    outdf = pd.DataFrame(0, index=smplist, columns=smplist)
    for smp1nr in range(len(smplist) - 1):
        smp1 = smplist[smp1nr]
        for smp2nr in range(smp1nr + 1, len(smplist)):
            smp2 = smplist[smp2nr]
            brayvalue = 1 - (ra.loc[:, [smp1, smp2]].min(axis=1).sum())
            outdf.loc[smp1, smp2] = brayvalue
            outdf.loc[smp2, smp1] = brayvalue
    return outdf

# Calculate matrix of pairwise Jaccard dissimilarities
def jaccard(tab):
    if isinstance(tab, pd.Series) or len(tab.columns) < 2:
        print('Too few samples in tab.')
        return None

    bintab = tab.copy()
    bintab[bintab > 0] = 1
    smplist = list(bintab.columns)
    outdf = pd.DataFrame(0, index=smplist, columns=smplist)
    for smp1nr in range(len(smplist) - 1):
        smp1 = smplist[smp1nr]
        for smp2nr in range(smp1nr + 1, len(smplist)):
            smp2 = smplist[smp2nr]
            joined = bintab[[smp1, smp2]].sum(axis=1)
            shared = joined[joined == 2].count()
            total = joined[joined > 0].count()
            jacvalue = 1 - shared/total
            outdf.loc[smp1, smp2] = jacvalue
            outdf.loc[smp2, smp1] = jacvalue
    return outdf

# Calculates the beta diversity or dissimilarity between multiple samples
# obj is object containing at least meta and tab
# var is column in meta data specifying how count table should be subdivided
# if var=None, the whole count table is used 
# q is the diversity order
# returns a dataframe with the categories in var being the index
# and including the columns N (nr of samples in category), beta, local_dis, regional_dis
def naive_multi_beta(obj, var='None', q=1):
    #Make dictionary with tabs, based on var
    meta = obj['meta']
    tabdict = {}
    if var != 'None':
        catlist = []
        [catlist.append(x) for x in meta[var] if x not in catlist]
        for cat in catlist:
            tabdict[cat] = subset.samples(obj, var=var, slist=[cat])['tab']
    else:
        catlist = ['all']
        tabdict['all'] = obj['tab'].copy()
    
    #Go through each tab and calculate multi beta
    output_beta = pd.DataFrame(np.nan, index=catlist, columns=['N', 'beta', 'local_dis', 'regional_dis'])
    for cat in catlist:
        tab = tabdict[cat]
        if len(tab.columns) <= 1: #There must be at least two samples in the tab
            continue

        N_cols = len(tab.columns)
        N_rows = len(tab.index)
        df_temp = pd.DataFrame(0, index=range(N_rows*N_cols), columns=['alpha', 'gamma'])
        df_temp.loc[range(N_rows), 'gamma'] = tab.sum(axis=1).to_numpy()
        alphalist = []
        for col in tab.columns:
            alphalist = alphalist + tab[col].tolist()
        df_temp['alpha'] = alphalist
        alpha_gamma_divs = naive_alpha(df_temp, q=q)
        beta_div = alpha_gamma_divs['gamma'] / (alpha_gamma_divs['alpha'] / N_cols)
        output_beta.loc[cat, 'N'] = N_cols
        output_beta.loc[cat, 'beta'] = beta_div
        output_beta.loc[cat, 'local_dis'] = hfunc.beta2dist(beta_div, q=q, N=N_cols, divType='naive', viewpoint='local')
        output_beta.loc[cat, 'regional_dis'] = hfunc.beta2dist(beta_div, q=q, N=N_cols, divType='naive', viewpoint='regional')
    return output_beta

# Calculates the phylogenetic beta diversity or dissimilarity between multiple samples
# obj is object containing at least meta and tab
# var is column in meta data specifying how count table should be subdivided
# if var=None, the whole count table is used 
# q is the diversity order
# returns a dataframe with the categories in var being the index
# and including the columns N (nr of samples in category), beta, local_dis, regional_dis
def phyl_multi_beta(obj, var='None', q=1):
    #Make dictionary with tabs, based on var
    if 'meta' not in obj or 'tab' not in obj or 'tree' not in obj:
        print('Make sure meta, tab, and tree are included in object')
        return 0

    #Make a dictionary with all sub tables
    meta = obj['meta']
    tabdict = {}
    if var != 'None':
        catlist = []
        [catlist.append(x) for x in meta[var] if x not in catlist]
        for cat in catlist:
            tabdict[cat] = subset.samples(obj, var=var, slist=[cat])['tab']
    else:
        catlist = ['all']
        tabdict['all'] = obj['tab'].copy()

    #Tree file
    tree = obj['tree']
    
    #Go through each tab and calculate multi beta
    output_beta = pd.DataFrame(np.nan, index=catlist, columns=['N', 'beta', 'local_dis', 'regional_dis'])
    for cat in catlist:
        tab = tabdict[cat]
        if len(tab.columns) <= 1: #There must be at least two samples in the tab
            continue
        
        #Make a tree df with ra associated with each branch
        ra = tab/tab.sum()
        tree2 = pd.DataFrame(0, index=tree.index, columns=ra.columns)
        for ix in tree2.index:
            asvlist = tree.loc[ix, 'ASVs']
            ra_branch = ra.reindex(asvlist).sum()
            tree2.loc[ix] = ra_branch

        N_cols = len(ra.columns)
        mean_ra = tree2.mean(axis=1)
        Tavg = mean_ra.mul(tree['branchL']).sum()

        #Get gamma
        g_df = mean_ra.copy()
        if q == 1:
            g_df[g_df > 0] = g_df[g_df > 0].apply(math.log)
            g_df = g_df.mul(mean_ra)
            g_df = g_df.mul(tree['branchL'])
            g_df = g_df.div(Tavg)
            g_df = -g_df.sum()
            gamma_div = math.exp(g_df)
        else:
            g_df = g_df.div(Tavg)
            g_df[g_df > 0] = g_df[g_df > 0].pow(q)
            g_df = g_df.mul(tree['branchL'])
            g_df = g_df.sum()
            g_df = g_df**(1/(1-q))
            gamma_div = g_df / Tavg

        #Get alpha
        tree2 = tree2.div(N_cols * Tavg)
        a_df = tree2.copy()
        if q == 1:
            a_df[a_df > 0] = a_df[a_df > 0].applymap(math.log)
            a_df = a_df.mul(tree2)
            a_df = a_df.mul(tree['branchL'], axis=0)
            a_df = -sum(a_df.sum()) - math.log(Tavg * N_cols)
            alpha_div = math.exp(a_df)
        else:
            a_df[a_df > 0] = a_df[a_df > 0].pow(q)
            a_df = a_df.sum(axis=1)
            a_df = a_df.mul(tree['branchL'])
            a_df = a_df.sum()
            a_df = a_df**(1/(1-q))
            alpha_div = a_df / (Tavg * N_cols)

        beta_div = gamma_div / alpha_div
        dist_local = hfunc.beta2dist(beta=beta_div, q=q, N=N_cols, divType='phyl', viewpoint='local')
        dist_regional = hfunc.beta2dist(beta=beta_div, q=q, N=N_cols, divType='phyl', viewpoint='regional')

        output_beta.loc[cat, 'N'] = N_cols
        output_beta.loc[cat, 'beta'] = beta_div
        output_beta.loc[cat, 'local_dis'] = dist_local
        output_beta.loc[cat, 'regional_dis'] = dist_regional
    return output_beta

# Calculates the functional beta diversity or dissimilarity between multiple samples
# obj is object containing at least meta and tab
# distmat is a pandas dataframe with pairwise distances between ASVs
# var is column in meta data specifying how count table should be subdivided
# if var=None, the whole count table is used 
# q is the diversity order
# returns a dataframe with the categories in var being the index
# and including the columns NxN (nr of samples in category squared), beta (0 to N^2), local_dis, regional_dis
def func_multi_beta(obj, distmat='None', var='None', q=1):
    #Make dictionary with tabs, based on var
    if 'meta' not in obj or 'tab' not in obj or 'distmat' == 'None':
        print('Make sure meta, tab, and distmat are included')
        return 0

    #Make a dictionary with all sub tables
    meta = obj['meta']
    tabdict = {}
    if var != 'None':
        catlist = []
        [catlist.append(x) for x in meta[var] if x not in catlist]
        for cat in catlist:
            tabdict[cat] = subset.samples(obj, var=var, slist=[cat])['tab']
    else:
        catlist = ['all']
        tabdict['all'] = obj['tab'].copy()

    #Go through each tab and calculate multi beta
    output_beta = pd.DataFrame(np.nan, index=catlist, columns=['NxN', 'beta', 'local_dis', 'regional_dis'])
    for cat in catlist:
        tab = tabdict[cat]
        if len(tab.columns) <= 1: #There must be at least two samples in the tab
            continue

        ra = tab/tab.sum()
        smplist = ra.columns.tolist()
        N_cols = len(smplist)

        ra_mean = ra.mean(axis=1)
        Qpooled = hfunc.rao(ra_mean, distmat)
        dqmat = distmat.mul(1 / Qpooled) #dij/Qpooled

        if q == 1:
            # Get gamma
            ra2mat = pd.DataFrame(np.outer(ra_mean, ra_mean), index=ra_mean.index, columns=ra_mean.index)
            mask = ra2mat > 0
            ra2mat_ln = ra2mat.copy()
            ra2mat_ln[mask] = ra2mat_ln[mask].applymap(math.log)
            ra2mat = ra2mat.mul(ra2mat_ln)
            ra2mat = ra2mat.mul(dqmat)
            Dg = math.exp(-0.5*(sum(ra2mat.sum())))

            # Get alpha
            asum = 0
            for smp1_nr in range(len(smplist)):
                smp1 = smplist[smp1_nr]
                for smp2_nr in range(len(smplist)):
                    smp2 = smplist[smp2_nr]
                    ra12 = pd.DataFrame({'smp1': ra[smp1], 'smp2': ra[smp2]}, index=ra.index)
                    ra2mat = pd.DataFrame(np.outer(ra12['smp1'], ra12['smp2']), 
                                          index=ra12.index, columns=ra12.index)
                    ra2mat = ra2mat.div(N_cols**2)
                    mask = ra2mat > 0
                    ra2mat_ln = ra2mat.copy()
                    ra2mat_ln[mask] = ra2mat_ln[mask].applymap(math.log)
                    ra2mat = ra2mat.mul(ra2mat_ln)
                    ra2mat = ra2mat.mul(dqmat)
                    asum = asum + sum(ra2mat.sum())

            Da = (1/N_cols) * math.exp(-0.5 * asum)

        else: #if q != 1
            # Get gamma
            ra2mat = pd.DataFrame(np.outer(ra_mean, ra_mean), index=ra_mean.index, columns=ra_mean.index)
            mask = ra2mat > 0
            ra2mat[mask] = ra2mat[mask].pow(q)
            ra2mat = ra2mat.mul(dqmat)
            Dg = pow(sum(ra2mat.sum()), 1 / (2 * (1 - q)))

            # Get alpha
            asum = 0
            for smp1_nr in range(len(smplist)):
                smp1 = smplist[smp1_nr]
                for smp2_nr in range(len(smplist)):
                    smp2 = smplist[smp2_nr]
                    ra12 = pd.DataFrame({'smp1': ra[smp1], 'smp2': ra[smp2]}, index=ra.index)
                    ra2mat = pd.DataFrame(np.outer(ra12['smp1'], ra12['smp2']), 
                                          index=ra12.index, columns=ra12.index)
                    mask = ra2mat > 0
                    ra2mat = ra2mat.div(N_cols**2)
                    ra2mat[mask] = ra2mat[mask].pow(q)
                    ra2mat = ra2mat.mul(dqmat)
                    asum = asum + sum(ra2mat.sum())

            Da = (1/N_cols) * pow(asum, 1 / (2 * (1 - q)))

        beta_div = Dg / Da
        dist_local = hfunc.beta2dist(beta=beta_div, q=q, N=N_cols, divType='func', viewpoint='local')
        dist_regional = hfunc.beta2dist(beta=beta_div, q=q, N=N_cols, divType='func', viewpoint='regional')

        output_beta.loc[cat, 'NxN'] = N_cols**2
        output_beta.loc[cat, 'beta'] = beta_div
        output_beta.loc[cat, 'local_dis'] = dist_local
        output_beta.loc[cat, 'regional_dis'] = dist_regional
    return output_beta

# Calculates the evenness measures described in Chao and Ricotta (2019) Ecology 100(12), e02852
# tab is count table and q is diversity order
# index can be CR1, CR2, CR3, CR4, or CR5, as specified in Chao and Ricotta (2019)
# CR1 can also be called regional and CR2 can also be called local, because they are linked to those dissimilarity measures
# if perspective=samples, the evenness value for each sample is calculate (i.e. each column in tab)
# if perspective=taxa, the evenness value for each taxon is calculate (i.e. each row in tab)
def evenness(tab, tree='None', distmat='None', q=1, divType='naive', index='local', perspective='samples'):
    # Power to raise D and S to in evenness formula for Chao1 and Chao2
    if index == 'CR1' or index == 'regional':
        power = float(1-q)
    elif index == 'CR2' or index == 'local':
        power = float(q-1)

    # Calculate an evenness value for each sample
    if perspective == 'samples':
        S_series = tab[tab > 0].count() #S value

        # Calculate D value    
        if divType == 'naive':
            D_series = naive_alpha(tab, q=q)
        elif divType == 'phyl' and isinstance(tree, pd.DataFrame):
            D_series = phyl_alpha(tab, tree=tree, q=q, index='D')
        elif divType == 'func' and isinstance(distmat, pd.DataFrame):
            D_series = func_alpha(tab, distmat=distmat, q=q, index='D')
        else:
            print('Make sure the required input for the chosen divType is available')
            print('divType = naive --> Nothing else needed')
            print('divType = phyl --> tree must be specified')
            print('divType = func --> distmat must be specified')
            return 0

    # Calculate an evenness value for each ASV or node
    elif perspective == 'taxa':
        # Calculate D value    
        if divType == 'naive':
            tab = tab.transpose()
            S_series = tab.count() #Note all samples are counted here (even 0 reads)
            D_series = naive_alpha(tab, q=q)
        elif divType == 'phyl' and isinstance(tree, pd.DataFrame):
            #Make a tree df with ra associated with each branch
            ra = tab/tab.sum()
            tree2 = pd.DataFrame(0, index=tree.index, columns=ra.columns)
            for ix in tree2.index:
                asvlist = tree.loc[ix, 'ASVs']
                ra_branch = ra.reindex(asvlist).sum()
                tree2.loc[ix] = ra_branch
            tree2 = tree2.transpose()
            tree2 = tree2 / tree2.sum()
            S_series = tree2.count()
            D_series = naive_alpha(tree2, q)
        else:
            print('divType must be naive or phyl if perspective=taxa')
            return 0

    #Get evenness measure
    if q == 1 and index in ['CR1', 'CR2', 'regional', 'local']:
        df = pd.DataFrame({'D':D_series, 'S':S_series})
        df[df > 0] = df[df > 0].applymap(math.log)
        measure = df['D'][df['S'] > 0].div(df['S'][df['S'] > 0])
    elif index in ['CR1', 'CR2', 'regional', 'local']:
        D_series = D_series.pow(power)
        S_series = S_series.pow(power)
        measure = (1-D_series) / (1-S_series)
    elif index == 'CR3':
        measure = (D_series - 1) / (S_series - 1)
    elif index == 'CR4':
        measure = (1 - 1 / D_series) / (1 - 1 / S_series)
    elif index == 'CR5':
        measure = D_series.apply(math.log) / S_series.apply(math.log)
    else:
        print('index must be: local, regional, CR1, CR2, CR3, CR4, or CR5')
        return None

    return measure

# Calculates the contribution of individual taxa to the dissimilarity between multiple samples
# see Chao and Ricotta (2019) Ecology 100(12), e02852
# obj is object containing at least meta and tab
# var is column in meta data specifying how count table should be subdivided
# if var=None, the whole count table is used 
# q is the diversity order
# divType is the diversity type, either naive or phyl
# index can be local (or Chao2) or regional (or Chao1)
def dissimilarity_contributions(obj, var='None', q=1, divType='naive', index='local'):
    if var != 'None' and 'meta' not in obj:
        print('Meta data missing.')
        return None
    if 'meta' in obj:
        meta = obj['meta']
        
    tabdict = {}
    if var != 'None':
        catlist = []
        [catlist.append(x) for x in meta[var] if x not in catlist]
        for cat in catlist:
            tabdict[cat] = subset.samples(obj, var=var, slist=[cat])['tab']
    else:
        catlist = ['all']
        tabdict['all'] = obj['tab'].copy()

    if divType == 'naive':
        output = pd.DataFrame(np.nan, index=['dis', 'N'] + obj['tab'].index.tolist(), columns=catlist)
    elif divType == 'phyl':
        output = pd.DataFrame(np.nan, index=['dis', 'N'] + obj['tree'].index.tolist(), columns=catlist)
    
    for cat in catlist:
        tab = tabdict[cat]
        output.loc['N', cat] = len(tab.columns)
        if len(tab.columns) <= 1:
            continue

        # Naive version, calculate for each ASV
        if divType == 'naive':
            if index == 'CR1' or index == 'regional':
                w = tab.sum(axis=1).pow(q) / sum(tab.sum(axis=1).pow(q))
                ev = evenness(tab, q=q, index=index, perspective='taxa')
            elif index == 'CR2' or index == 'local':
                tabpow = tab.copy()
                tabpow[tabpow > 0] = tabpow[tabpow > 0].pow(q)
                w = tabpow.sum(axis=1) / sum(tabpow.sum())
                ev = evenness(tab, q=q, index=index, perspective='taxa')
            else:
                print('Wrong index')
                return 0
            w_mul_ev = w * (1 - ev)
            output.loc['dis', cat] = w_mul_ev.sum()
            taxa_contributions = 100 * w_mul_ev / w_mul_ev.sum()
            output.loc[taxa_contributions.index, cat] = taxa_contributions
            output[cat].fillna(0, inplace=True)

        # Phylogenetic version, calculate for each node
        elif divType == 'phyl' and 'tree' in obj:
            ra = tab/tab.sum()
            tree = obj['tree']
            tree2 = pd.DataFrame(0, index=tree.index, columns=ra.columns)
            for ix in tree2.index:
                asvlist = tree.loc[ix, 'ASVs']
                ra_branch = ra.reindex(asvlist).sum()
                tree2.loc[ix] = ra_branch

            ev = evenness(tab, tree=tree, q=q, divType='phyl', index=index, perspective='taxa')
            if index == 'CR1' or index == 'regional':
                zi_plus = tree2.sum(axis=1)
                zi_plus[zi_plus > 0] = zi_plus[zi_plus > 0].pow(q) #zi+^q
                Lz = tree['branchL'].mul(zi_plus)  #Li*zi+
                w = zi_plus.div(Lz.sum())
            elif index == 'CR2' or index == 'local':
                tree2[tree2 > 0] = tree2[tree2 > 0].pow(q)
                zv_sum = tree2.sum(axis=1)
                Lz = tree['branchL'].mul(zv_sum)  #Lv*sum(zvm^q)
                w = zv_sum.div(Lz.sum())
            w_mul_ev = w * (1 - ev)
            w_mul_ev = tree['branchL'].mul(w_mul_ev)
            output.loc['dis', cat] = w_mul_ev.sum()
            taxa_contributions = 100 * w_mul_ev / w_mul_ev.sum()
            output.loc[taxa_contributions.index, cat] = taxa_contributions

    return output
