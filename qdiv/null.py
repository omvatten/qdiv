import pandas as pd
import numpy as np
from . import diversity
pd.options.mode.chained_assignment = None  # default='warn'

# Randomizes the frequency table and compares the actual beta diversity with the null expectation
# Returns a dictionary with several items:
#    'Obs' is the actually observed dissimilarity values
#    'Nullmean' is the mean values of the null dissimilarities (i.e. the dissimilarities of the randomized tables)
#    'Nullstd' is the standard devation
#    'RC' is the Raup-Crick measure (i.e. the number of times the actual dissimilarities are higher than the null expectation)
# If RCrange = 'Raup' the range for the index will be 0 to 1, if it is 'Chase' it will -1 to 1.
# If compareVar is not None, 'RCmean' and 'RCstd' are returned and represents the mean and standard deviation of all pairwise comparison
#    within the metadata category specified by compareVar
# During randomization, the richness and read count of each sample is maintained constant.
# The randomization can be constrained based on a column heading (constrainingVar) in meta data so that counts are only
# randomized within a category.
# randomization specifies the procedure:
#    'abundance' means that SVs are drawn to each sample based on the total read counts in the table.
#    'frequency' means that SVs are drawn based on the number of samples in which they are detected (i.e. Chase 2011, Stegen 2013).
#    'weighting' uses the abundance method but a meta data column (weightingVar) can be used to categorize samples and the
#      weight parameters decide the importance of the category of samples with the lowest richness. 0 means the low-richness samples
#      are not considered in the meta community while a weight of 1 means all samples have equal weighting.
# iterations specifies the number of randomizations, 999 is normal but can take several hours for large frequency tables
# divType specifies the dissimilarity index to calculate: 'Jaccard', 'Bray', 'naive', 'phyl', and 'func' are available choices.
# For 'naive', 'phyl', and 'func', pairwise dissimilarities with viewpoint='local' are calculated
def rcq(obj, constrainingVar='None', randomization='frequency', weightingVar='None', weight=1, iterations=9,
         divType='naive', distmat='None', q=1, compareVar='None'):

    # Returns a list of randomized tables
    def randomizeTabs():

        # Print progress in calculation
        print('Step 1/2: Randomizing tabs.. ', end='')
        calc_counter = 0

        # Function to put less emphasis on less richsamples within a specific category, return a subtab
        def weighting_subtab(wtab, wmeta):
            catlist2 = []
            [catlist2.append(x) for x in wmeta[weightingVar].tolist() if x not in catlist2]
            listofcounts = []
            listofcats = []
            for cat2 in catlist2:
                samples_group = wmeta[wmeta[weightingVar] == cat2].index.tolist()
                if len(samples_group) > 1:
                    summed_column = wtab[samples_group].sum(axis=1)
                else:
                    summed_column = subtab[samples_group[0]]
                summed_column[summed_column > 0] = 1
                summed_column = summed_column.tolist()
                listofcounts.append(np.sum(summed_column))
                listofcats.append(cat2)
            if len(listofcats) > 1:  # Find category with lowest richness
                min_richness = min(listofcounts)
                max_richness = max(listofcounts)
                if min_richness < max_richness:
                    min_pos = listofcounts.index(min_richness)
                    min_cat = listofcats[min_pos]
                    min_samples = wmeta[wmeta[weightingVar] == min_cat].index.tolist()
                    wtab[min_samples] = wtab[min_samples] * weight
            return wtab.sum(axis=1)

        # Get tab from object
        tab = obj['tab'].copy()
        SVlist = tab.index.tolist()
        if 'meta' in obj.keys():
            meta = obj['meta']

        # Divide into subtabs
        if constrainingVar == 'None':
            subtablist = [tab.copy()]
        else:
            subtablist = []
            constrainCats = []
            [constrainCats.append(x) for x in meta[constrainingVar].tolist() if x not in constrainCats]
            for cat in constrainCats:
                subsamples = meta[meta[constrainingVar] == cat].index
                subtablist.append(tab[subsamples])

        # Make a list of empty tabs
        random_tabs = []
        for i in range(iterations):
            random_tabs.append(pd.DataFrame(0, index=tab.index, columns=tab.columns))

        # Go through the subtablist, for each subtab calculate abundance, frequency etc.
        for subtab in subtablist:
            # Get abundances as p for all subtabs
            if randomization == 'weighting':
                weightingtab = subtab.copy()
                abundseries = weighting_subtab(wtab=weightingtab, wmeta=meta)
            else:
                abundseries = subtab.sum(axis=1)
            abund_all_p = abundseries / abundseries.sum()

            # Get frequencies as p for all subtabs
            subtab_bin = subtab.copy()
            subtab_bin[subtab_bin > 0] = 1
            freqlist = subtab_bin.sum(axis=1).tolist()
            freqlist = freqlist / np.sum(freqlist)

            # List of samples
            smplist = subtab.columns.tolist()

            # List of richness
            richnesslist = subtab_bin.sum(axis=0).tolist()

            # List of read counts
            readslist = subtab.sum(axis=0).tolist()

            # Do iterations
            total_counts = len(subtablist) * iterations
            show_after = int(total_counts / 50) + 1
            for i in range(iterations):
                calc_counter += 1
                if calc_counter%show_after == 0:
                    print(int(100*calc_counter/total_counts), end='%.. ')

                # Go through each sample in subtab
                for cnr in range(len(smplist)):
                    smp = smplist[cnr]
                    richness = richnesslist[cnr]
                    reads = readslist[cnr]

                    if randomization in ['abundance', 'weighting']:
                        rows = np.random.choice(SVlist, size=richness, replace=False, p=abund_all_p)
                    elif randomization == 'frequency':
                        rows = np.random.choice(SVlist, size=richness, replace=False, p=freqlist)
                    random_tabs[i].loc[rows, smp] = 1

                    abund_sub_p = abundseries[rows] / abundseries[rows].sum()

                    randomchoice = np.random.choice(rows, size=reads - richness, replace=True, p=abund_sub_p)
                    uniquechoices = np.unique(randomchoice, return_counts=True)
                    random_tabs[i].loc[uniquechoices[0], smp] = random_tabs[i].loc[uniquechoices[0], smp] + \
                                                                uniquechoices[1]
        print('100%')
        return random_tabs

    # START HERE
    #Get count table
    tab = obj['tab'].copy()
    if 'meta' in obj.keys():
        meta = obj['meta']

    #Generate random tabs
    randomtabs = randomizeTabs() #Call function above

    # Keeps track of calculation progress
    print('Step 2/2: Compare beta diversity.. ', end='')
    calc_counter = 0

    # Calculate betadiv for the subtab
    if divType == 'Bray':
        betadiv = diversity.bray(tab)
    elif divType == 'Jaccard':
        betadiv = diversity.jaccard(tab)
    elif divType == 'naive':
        betadiv = diversity.naive_beta(tab, q=q)
    elif divType == 'phyl' and 'tree' in obj:
        betadiv = diversity.phyl_beta(tab, obj['tree'], q=q)
    elif divType == 'func' and isinstance(distmat, pd.DataFrame):
        betadiv = diversity.func_beta(tab, distmat, q=q)
    else:
        print('Check divType and required input')
        return None

    # The randomized output is saved in these dataframes
    random_beta_all = np.zeros((len(tab.columns), len(tab.columns), iterations))
    RC_tab = pd.DataFrame(0, index=tab.columns, columns=tab.columns)

    total_counts = iterations
    show_after = int(total_counts / 50) + 1
    for i in range(iterations):
        
        # For showing progress
        calc_counter += 1
        if calc_counter%show_after == 0:
            print(int(100*calc_counter/total_counts), end='%.. ')

        rtab = randomtabs[i]
        if divType == 'Bray':
            randombeta = diversity.bray(rtab)
        elif divType == 'Jaccard':
            randombeta = diversity.jaccard(rtab)
        elif divType == 'naive':
            randombeta = diversity.naive_beta(rtab, q=q)
        elif divType == 'phyl' and 'tree' in obj:
            randombeta = diversity.phyl_beta(rtab, obj['tree'], q=q)
        elif divType == 'func' and isinstance(distmat, pd.DataFrame):
            randombeta = diversity.func_beta(rtab, distmat, q=q)

        random_beta_all[:, :, i] = randombeta

        mask = betadiv > randombeta
        RC_tab[mask] = RC_tab[mask] + 1
        mask = betadiv == randombeta
        RC_tab[mask] = RC_tab[mask] + 0.5

    #Now we have all random values and its time to calculate the indices
    RC_tab = RC_tab.div(iterations)

    obs_beta = betadiv.to_numpy()
    null_mean = random_beta_all.mean(axis=2)
    null_std = random_beta_all.std(axis=2)
    ses_df = np.empty((len(null_mean[0,:]), len(null_mean[:,1])))
    ses_df[null_std > 0] = (null_mean[null_std > 0] - obs_beta[null_std > 0]) / null_std[null_std > 0]
    null_mean = pd.DataFrame(null_mean, index=RC_tab.index, columns=RC_tab.columns)
    null_std = pd.DataFrame(null_std, index=RC_tab.index, columns=RC_tab.columns)
    ses_df = pd.DataFrame(ses_df, index=RC_tab.index, columns=RC_tab.columns)

    out = {}
    if compareVar == 'None': #Straight forward comparison of all samples to each other
        out['divType'] = divType
        out['obs_d'] = betadiv
        out['p_index'] = RC_tab
        out['null_mean'] = null_mean
        out['null_std'] = null_std
        out['ses'] = ses_df

    else: #Average all pairwise comparisons between samples in categories specified by compareVar
        indexlist = RC_tab.index.tolist()
        metalist = meta[compareVar].tolist()
        complist = [] #Hold list of unique categories from metalist
        [complist.append(x) for x in metalist if x not in complist]
        out_RCavg = pd.DataFrame(0, index=complist, columns=complist)
        out_RCstd = pd.DataFrame(0, index=complist, columns=complist)
        out_nullavg = pd.DataFrame(0, index=complist, columns=complist)
        out_nullstd = pd.DataFrame(0, index=complist, columns=complist)
        out_sesavg = pd.DataFrame(0, index=complist, columns=complist)
        out_sesstd = pd.DataFrame(0, index=complist, columns=complist)
        out_obsavg = pd.DataFrame(0, index=complist, columns=complist)
        out_obsstd = pd.DataFrame(0, index=complist, columns=complist)
        for c1nr in range(len(complist) - 1):
            c1 = complist[c1nr] #Category 1
            s1list = meta[meta[compareVar] == c1].index #All samples in category 1
            for c2nr in range(c1nr + 1, len(complist)):
                c2 = complist[c2nr] #Category 2
                s2list = meta[meta[compareVar] == c2].index #All samples in category 2

                # Check all pairwise comparisons between c1 and c2 samples
                RC_list = []
                null_list = []
                ses_list = []
                obs_list = []
                for s1 in s1list:
                    s1pos = indexlist.index(s1)
                    for s2 in s2list: #Compare sample s1 (in cat1) to all samples in cat 2
                        s2pos = indexlist.index(s2)
                        RC_list.append(RC_tab.loc[s1, s2])
                        null_list.append(random_beta_all[s1pos, s2pos, :])
                        ses_list.append(ses_df.loc[s1, s2])
                        obs_list.append(betadiv.loc[s1, s2])
                out_RCavg.loc[c1, c2] = np.mean(RC_list)
                out_nullavg.loc[c1, c2] = np.mean(null_list)
                out_sesavg.loc[c1, c2] = np.mean(ses_list)
                out_obsavg.loc[c1, c2] = np.mean(obs_list)
                out_RCstd.loc[c1, c2] = np.std(RC_list)
                out_nullstd.loc[c1, c2] = np.std(null_list)
                out_sesstd.loc[c1, c2] = np.std(ses_list)
                out_obsstd.loc[c1, c2] = np.std(obs_list)
                
                out_RCavg.loc[c2, c1] = out_RCavg.loc[c1, c2]
                out_nullavg.loc[c2, c1] = out_nullavg.loc[c1, c2]
                out_sesavg.loc[c2, c1] = out_sesavg.loc[c1, c2]
                out_obsavg.loc[c2, c1] = out_obsavg.loc[c1, c2]
                out_RCstd.loc[c2, c1] = out_RCstd.loc[c1, c2]
                out_nullstd.loc[c2, c1] = out_nullstd.loc[c1, c2]
                out_sesstd.loc[c2, c1] = out_sesstd.loc[c1, c2]
                out_obsstd.loc[c2, c1] = out_obsstd.loc[c1, c2]

        out['divType'] = divType
        out['obs_d_mean'] = out_obsavg
        out['obs_d_std'] = out_obsstd
        out['p_index_mean'] = out_RCavg
        out['p_index_std'] = out_RCstd
        out['null_mean'] = out_nullavg
        out['null_std'] = out_nullstd
        out['ses_mean'] = out_sesavg
        out['ses_std'] = out_sesstd
    print('100%')
    return out

# Calculates net relatedness index as described in Webb et al. 2002
# Here, the index is relative abundance weighted using q
def nriq(obj, distmat, q=1, iterations=99):

    #Get q-weighted mean d
    def get_dmean(ra_series, dm):
        ras = ra_series[ra_series > 0]
        pp_df = pd.DataFrame(np.outer(ras, ras), index=ras.index, columns=ras.index)
        dm_sub = dm.loc[ras.index, ras.index]
        pp_df[pp_df > 0] = pp_df[pp_df > 0].pow(q)
        sum_denom = sum(pp_df.sum())
        pp_df = pp_df.mul(dm_sub)
        mean_d = sum(pp_df.sum()) / sum_denom
        return mean_d

    tab = obj['tab']
    ra = tab / tab.sum()
    smplist = ra.columns
    output = pd.DataFrame(np.nan, index=smplist, columns=['MPDq', 'null_mean', 'null_std', 'p_index', 'ses'])

    for smp in smplist:
        output.loc[smp, 'MPDq'] = get_dmean(ra[smp], dm=distmat)

    #Progress
    calc_counter = 0
    total_counts = iterations * len(smplist)
    show_after = int(total_counts / 50) + 1 
    print('NRIq progress: 0', end='%.. ')

    svlist = distmat.index.tolist()
    darr = np.empty((len(smplist), iterations))
    for i in range(iterations):
        np.random.shuffle(svlist)
        dm_random = pd.DataFrame(distmat.to_numpy(), index=svlist, columns=svlist)
        for j in range(len(smplist)):
            calc_counter += 1
            if calc_counter%show_after == 0:
                print(int(100 * (calc_counter / total_counts)), end='%.. ')
            smp = smplist[j]
            darr[j, i] = get_dmean(ra[smp], dm_random)
    for j in range(len(smplist)):
        smp = smplist[j]
        output.loc[smp, 'null_mean'] = darr[j, :].mean()
        output.loc[smp, 'null_std'] = darr[j, :].std()
        output.loc[smp, 'p_index'] = (len(darr[j, :][darr[j, :] < output.loc[smp, 'MPDq']]) + 0.5 * len(darr[j, :][darr[j, :] == output.loc[smp, 'MPDq']])) / len(darr[j, :])
        if output.loc[smp, 'null_std'] > 0:
            output.loc[smp, 'ses'] = (output.loc[smp, 'null_mean'] - output.loc[smp, 'MPDq']) / output.loc[smp, 'null_std']
    print('100%')
    return output

# Calculates nearest taxon index as described in Webb et al. 2002
# Here, the index is relative abundance weighted using q
def ntiq(obj, distmat, q=1, iterations=99):
    #Get q-weighted min d
    def get_dmin(ra_series, dm):
        ras = ra_series[ra_series > 0]
        dm_sub = dm.loc[ras.index, ras.index]
        dmin = dm_sub[dm_sub > 0].min()
        ras = ras.pow(q)
        sum_denom = ras.sum()
        ras = ras.mul(dmin)        
        return ras.sum() / sum_denom

    tab = obj['tab']
    ra = tab / tab.sum()
    smplist = ra.columns
    output = pd.DataFrame(np.nan, index=smplist, columns=['MNTDq', 'null_mean', 'null_std', 'p_index', 'ses'])

    for smp in smplist:
        output.loc[smp, 'MNTDq'] = get_dmin(ra[smp], dm=distmat)

    #Progress
    calc_counter = 0
    total_counts = iterations * len(smplist)
    show_after = int(total_counts / 50) + 1 
    print('NTIq progress: 0', end='%.. ')

    svlist = distmat.index.tolist()
    darr = np.empty((len(smplist), iterations))
    for i in range(iterations):
        np.random.shuffle(svlist)
        dm_random = pd.DataFrame(distmat.to_numpy(), index=svlist, columns=svlist)
        for j in range(len(smplist)):
            calc_counter += 1
            if calc_counter%show_after == 0:
                print(int(100 * (calc_counter / total_counts)), end='%.. ')
            smp = smplist[j]
            darr[j, i] = get_dmin(ra[smp], dm_random)
    for j in range(len(smplist)):
        smp = smplist[j]
        output.loc[smp, 'null_mean'] = darr[j, :].mean()
        output.loc[smp, 'null_std'] = darr[j, :].std()
        output.loc[smp, 'p_index'] = (len(darr[j, :][darr[j, :] < output.loc[smp, 'MNTDq']]) + 0.5 * len(darr[j, :][darr[j, :] == output.loc[smp, 'MNTDq']])) / len(darr[j, :])
        if output.loc[smp, 'null_std'] > 0:
            output.loc[smp, 'ses'] = (output.loc[smp, 'null_mean'] - output.loc[smp, 'MNTDq']) / output.loc[smp, 'null_mean']
    print('100%')
    return output

# Calculates pairwise beta NRI
# The index is relative abundance weighted using q
def beta_nriq(obj, distmat, q=1, iterations=99):
    #Get q-weighted min d
    def get_bdmean(ra_pair, dm):
        rap = ra_pair.copy()
        rap = rap[rap.sum(axis=1) > 0]
        smp12 = rap.columns.tolist()
        rap1 = rap[smp12[0]]
        rap2 = rap[smp12[1]]
        pp_df = pd.DataFrame(np.outer(rap1, rap2), index=rap1.index, columns=rap2.index)

        dm_sub = dm.loc[rap1.index, rap2.index]
        pp_df[pp_df > 0] = pp_df[pp_df > 0].pow(q)
        sum_denom = sum(pp_df.sum())
        pp_df = pp_df.mul(dm_sub)
        mean_d = sum(pp_df.sum()) / sum_denom
        return mean_d

    tab = obj['tab']
    ra = tab / tab.sum()
    smplist = ra.columns
    outputMPD = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputNRI = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputAvg = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputStd = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputP = pd.DataFrame(np.nan, index=smplist, columns=smplist)

    #Progress
    calc_counter = 0
    total_counts = len(smplist)**2 / 2
    print('beta_NRIq progress: 0', end='%.. ')

    for i in range(len(smplist)-1):
        smp1 = smplist[i]
        for j in range(i+1, len(smplist)):
            smp2 = smplist[j]

            calc_counter += 1
            print(int(100 * (calc_counter / total_counts)), end='%.. ')

            ra_sub = ra[[smp1, smp2]].copy()
            obs_val = get_bdmean(ra_sub, distmat)
            outputMPD.loc[smp1, smp2] = obs_val
            outputMPD.loc[smp2, smp1] = obs_val

            darr = np.empty(iterations)
            svlist = distmat.index.tolist()
            for x in range(iterations):
                np.random.shuffle(svlist)
                dm_random = pd.DataFrame(distmat.to_numpy(), index=svlist, columns=svlist)
                b_val = get_bdmean(ra_sub, dm_random)
                darr[x] = b_val

            pval = (len(darr[darr < obs_val]) + 0.5 * len(darr[darr == obs_val])) / len(darr)
            outputP.loc[smp1, smp2] = pval
            outputP.loc[smp2, smp1] = pval

            if darr.std() > 0:
                bNTI_val = (darr.mean() - obs_val) / darr.std()
                outputNRI.loc[smp1, smp2] = bNTI_val
                outputNRI.loc[smp2, smp1] = bNTI_val
            outputAvg.loc[smp1, smp2] = darr.mean()
            outputAvg.loc[smp2, smp1] = darr.mean()
            outputStd.loc[smp1, smp2] = darr.std()
            outputStd.loc[smp2, smp1] = darr.std()
    print('100%')
    out = {}
    out['beta_MPDq'] = outputMPD
    out['null_mean'] = outputAvg
    out['null_std'] = outputStd
    out['p_index'] = outputP
    out['ses'] = outputNRI
    return out

# Calculates pairwise beta NTI as described in Stegen et al. 2013
# Here, the index is relative abundance weighted using q
def beta_ntiq(obj, distmat, q=1, iterations=99):
    #Get q-weighted min d
    def bMNTDq(ra_pair, dm):
        rap = ra_pair.copy()
        rap = rap[rap.sum(axis=1) > 0]
        rap[rap > 0] = rap[rap > 0].pow(q)
        sum_denom = rap.sum().tolist()
        smp12 = rap.columns.tolist()
        ra_ser1 = rap[smp12[0]][rap[smp12[0]] > 0]
        ra_ser2 = rap[smp12[1]][rap[smp12[1]] > 0]

        dm_sub = dm.loc[ra_ser1.index, ra_ser2.index]
        ra_ser1 = ra_ser1.mul(dm_sub.min(axis=1))
        ra_ser2 = ra_ser2.mul(dm_sub.min(axis=0))
        return 0.5 * (ra_ser1.sum() / sum_denom[0] + ra_ser2.sum() / sum_denom[1])

    tab = obj['tab']
    ra = tab / tab.sum()
    smplist = ra.columns
    outputMNTD = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputNTI = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputAvg = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputStd = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputP = pd.DataFrame(np.nan, index=smplist, columns=smplist)

    #Progress
    calc_counter = 0
    total_counts = len(smplist)**2 / 2
    print('beta_NTIq progress: 0', end='%.. ')

    for i in range(len(smplist)-1):
        smp1 = smplist[i]
        for j in range(i+1, len(smplist)):
            smp2 = smplist[j]

            calc_counter += 1
            print(int(100 * (calc_counter / total_counts)), end='%.. ')

            ra_sub = ra[[smp1, smp2]].copy()
            obs_val = bMNTDq(ra_sub, distmat)
            outputMNTD.loc[smp1, smp2] = obs_val
            outputMNTD.loc[smp2, smp1] = obs_val

            darr = np.empty(iterations)
            svlist = distmat.index.tolist()
            for x in range(iterations):
                np.random.shuffle(svlist)
                dm_random = pd.DataFrame(distmat.to_numpy(), index=svlist, columns=svlist)
                b_val = bMNTDq(ra_sub, dm_random)
                darr[x] = b_val

            pval = (len(darr[darr < obs_val]) + 0.5 * len(darr[darr == obs_val])) / len(darr)
            outputP.loc[smp1, smp2] = pval
            outputP.loc[smp2, smp1] = pval

            if darr.std() > 0:
                bNTI_val = (darr.mean() - obs_val) / darr.std()
                outputNTI.loc[smp1, smp2] = bNTI_val
                outputNTI.loc[smp2, smp1] = bNTI_val
            outputAvg.loc[smp1, smp2] = darr.mean()
            outputAvg.loc[smp2, smp1] = darr.mean()
            outputStd.loc[smp1, smp2] = darr.std()
            outputStd.loc[smp2, smp1] = darr.std()
    print('100%')
    out = {}
    out['beta_MNTDq'] = outputMNTD
    out['null_mean'] = outputAvg
    out['null_std'] = outputStd
    out['p_index'] = outputP
    out['ses'] = outputNTI
    return out
