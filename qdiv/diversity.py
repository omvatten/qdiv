import pandas as pd
import numpy as np
import Levenshtein as Lv
import tkinter as tk
import math
from . import subset

pd.options.mode.chained_assignment = None  # default='warn'

# FUNCTIONS FOR ANALYSING DIVERSITY

# Returns matrix for pairwise distances between SVs based on Levenshtein/Hamming distances (uses Levenshtein package)
# Saves results as pickle file at location specified in savename
# Output is needed as input for phylogenetic diversity index calculations
def sequence_comparison(seq, savename='PhylDistMat'):
    svnames = list(seq.index)

    # For showing progress
    total_comp = (len(svnames)**2)/2
    rootPhylDistMat = tk.Tk()
    calc_progress = tk.DoubleVar(rootPhylDistMat, 0)
    counter = 0
    tk.Label(rootPhylDistMat, text='Progress in calculation (%)', width=30).pack()
    tk.Label(rootPhylDistMat, textvariable=calc_progress, width=20).pack()

    df = pd.DataFrame(0, index=svnames, columns=svnames)
    for i in range(len(svnames) - 1):
        for j in range(i + 1, len(svnames)):

            # For showing progress
            counter += 1
            if counter%100 == 0:
                calc_progress.set(round(100*counter/total_comp, 2))
                rootPhylDistMat.update()

            n1 = svnames[i]
            n2 = svnames[j]
            s1 = seq.loc[n1, 'seq']
            s2 = seq.loc[n2, 'seq']

            if len(s1) == len(s2):
                dist = Lv.hamming(s1, s2) / len(s1)
            else:
                maxlen = max(len(s1), len(s2))
                dist = Lv.distance(s1, s2) / maxlen

            df.loc[n1, n2] = dist; df.loc[n2, n1] = dist
    df.to_csv(savename+'.csv')
    rootPhylDistMat.destroy()

# Returns Rao's quadratic entropy, sum(sum(dij*pi*pj))
# Function used in Chiu's phylogenetic diversity functions
def rao(tab, distmat):
    ra = tab / tab.sum()
    outdf = pd.Series(0, index=ra.columns)
    svlist = list(ra.index)
    distmat = distmat.loc[svlist, svlist]
    for smp in ra.columns:
        ra2mat = pd.DataFrame(np.outer(ra.loc[:, smp].values, ra.loc[:, smp].values), index=ra.index, columns=ra.index)
        rao_mat = ra2mat.mul(distmat)
        Qvalue = sum(rao_mat.sum())
        outdf.loc[smp] = Qvalue
    return outdf

# Converts beta value to distances, specify q and type associated with the beta (assuming pairwise)
# Used in beta dissimilarity calculations
# The viewpoint refers to either the local or regional perspective as defined in Chao et al. 2014
def beta2dist(beta, q=1, N=2, divType='naive', viewpoint='local'):
    if isinstance(beta, pd.DataFrame):
        beta = beta.applymap(float)
        beta = beta[beta > 0]
        dist = beta.copy()

    if q == 1:
        dist = np.log(beta) / math.log(N)
    else:
        if divType == 'naive' and viewpoint == 'local':
            dist = 1 - (N**(1 - q) - beta**(1 - q)) / (N**(1 - q) - 1)
        elif divType == 'phyl' and viewpoint == 'local':
            dist = 1 - ((N**(2 * (1 - q)) - beta**(1 - q)) / (N**(2 * (1 - q)) - 1))
        elif divType == 'naive' and viewpoint == 'regional':
            dist = 1 - ((1 / beta)**(1 - q) - (1 / N)**(1 - q)) / (1 - (1 / N)**(1 - q))
        elif divType == 'phyl' and viewpoint == 'regional':
            dist = 1 - ((1 / beta)**(1 - q) - (1 / N)**(2 * (1 - q))) / (1 - (1 / N)**(2 * (1 - q)))
    return dist

# Returns taxonomic (naive) alpha diversity of order q for all samples
def naive_alpha(tab, q=1):
    ra = tab / tab.sum()

    if q == 0:
        mask = tab > 0
        Hillvalues = tab[mask].count()
        return Hillvalues
    elif q == 1:
        mask = (ra != 0)
        raLn = ra[mask] * np.log(ra[mask])
        Hillvalues = np.exp(-raLn.sum())
        return Hillvalues
    else:
        mask = (ra != 0)
        ra = ra[mask]
        rapow = ra[mask].pow(q)
        rapowsum = rapow.sum()
        Hillvalues = rapowsum.pow(1 / (1 - q))
        return Hillvalues

# Returns matrix of paiwise dissimilarities of order q
# Viewpoint can be local or regional, see Chao et al. 2014
def naive_beta(tab, q=1, dis=True, viewpoint='local'):
    ra = tab / tab.sum()

    smplist = ra.columns
    outdf = pd.DataFrame(0, index=smplist, columns=smplist)
    for smp1nr in range(len(smplist) - 1):
        smp1 = smplist[smp1nr]
        for smp2nr in range(smp1nr + 1, len(smplist)):
            smp2 = smplist[smp2nr]

            if q == 1.0:
                mask1 = ra[smp1] != 0
                raLn1 = ra[smp1][mask1] * np.log(ra[smp1][mask1])
                raLn1 = raLn1.sum()
                mask2 = ra[smp2] != 0
                raLn2 = ra[smp2][mask2] * np.log(ra[smp2][mask2])
                raLn2 = raLn2.sum()
                alphavalue = math.exp(-0.5 * raLn1 - 0.5 * raLn2)

                ra_mean = ra[[smp1, smp2]].mean(axis=1)
                maskg = (ra_mean != 0)
                raLng = ra_mean[maskg] * np.log(ra_mean[maskg])
                raLng = raLng.sum()
                gammavalue = math.exp(-raLng)

                betavalue = gammavalue / alphavalue
                outdf.loc[smp1, smp2] = betavalue
                outdf.loc[smp2, smp1] = betavalue
            else:
                mask1 = ra[smp1] != 0
                ra1 = ra[smp1][mask1]
                ra1pow = ra1.pow(q)
                ra1powsum = ra1pow.sum()
                mask2 = ra[smp2] != 0
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
        dist = beta2dist(beta=outdf, q=q, divType='naive', viewpoint=viewpoint)
        return dist
    else:
        return outdf

# Calculate matrix of pairwise Bray-Curtis dissimilarities
def bray(tab):
    ra = tab / tab.sum()
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

# Returns phylogenetic alpha diversity of order q for all samples
# FD as in Chiu et al. Plos One, 2014
def phyl_alpha(tab, distmat, q=0):
    ra = tab / tab.sum()

    outdf = pd.Series(0, index=ra.columns)
    svlist = ra.index.tolist()
    distmat = distmat.loc[svlist, svlist]
    Qframe = rao(ra, distmat)
    if q == 0:
        for smp in tab.columns:
            ra2mat = pd.DataFrame(np.outer(ra.loc[:, smp].values, ra.loc[:, smp].values), index=ra.index,
                                  columns=ra.index)
            mask = ra2mat > 0
            ra2mat[mask] = 1
            dQmat = distmat.mul(1 / Qframe.loc[smp])
            ra2dq = (ra2mat.mul(dQmat))
            Chiuvalue = pow(sum(ra2dq.sum()), 1 / (2 * (1 - q)))
            outdf.loc[smp] = Chiuvalue
    elif q == 1:
        for smp in tab.columns:
            ra2mat = pd.DataFrame(np.outer(ra.loc[:, smp].values, ra.loc[:, smp].values), index=ra.index,
                                  columns=ra.index)
            mask = (ra2mat != 0)
            ra2Lnmat = np.log(ra2mat[mask])
            ra2ochLn = ra2mat.mul(ra2Lnmat)
            dQmat = distmat.mul(1 / Qframe.loc[smp])
            dQ_ra2_Ln = dQmat.mul(ra2ochLn)
            Chiuvalue = math.exp(-0.5 * sum(dQ_ra2_Ln.sum()))
            outdf.loc[smp] = Chiuvalue
    else:
        for smp in tab.columns:
            ra2mat = pd.DataFrame(np.outer(ra.loc[:, smp].values, ra.loc[:, smp].values), index=ra.index,
                                  columns=ra.index)
            ra2matq = ra2mat.pow(q)
            dQmat = distmat.mul(1 / Qframe.loc[smp])
            ra2dq = (ra2matq.mul(dQmat))
            Chiuvalue = pow(sum(ra2dq.sum()), 1 / (2 * (1 - q)))
            outdf.loc[smp] = Chiuvalue
    MD = outdf.mul(Qframe)
    return outdf.mul(MD)

# Returns matrix of paiwise phylogenetic dissimilarities of order q
# Based on local functional overlaps as defined in Chao et al. 2014
def phyl_beta(tab, distmat, q=1, dis=True, viewpoint='local'):
    ra = tab / tab.sum()

    smplist = list(ra.columns)
    outD = pd.DataFrame(0, index=smplist, columns=smplist)

    # For showing progress
    total_comp = (len(smplist)**2)/2
    rootPhylDivBeta = tk.Tk()
    calc_progress = tk.DoubleVar(rootPhylDivBeta, 0)
    counter = 0
    tk.Label(rootPhylDivBeta, text='Progress in calculation (%)', width=30).pack()
    tk.Label(rootPhylDivBeta, textvariable=calc_progress, width=20).pack()

    for smp1nr in range(len(smplist) - 1):
        for smp2nr in range(smp1nr + 1, len(smplist)):

            # For showing progress
            counter += 1
            if counter%10 == 0:
                calc_progress.set(round(100*counter/total_comp, 2))
                rootPhylDivBeta.update()

            smp1 = smplist[smp1nr]
            smp2 = smplist[smp2nr]

            ra12 = ra.loc[:, [smp1, smp2]]
            ra12['mean'] = ra12.mean(axis=1)
            Qvalues = rao(ra12, distmat)
            Qpooled = Qvalues['mean']
            dqmat = distmat.mul(1 / Qpooled)

            if q != 1:
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

            else:
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
    outFD = outD.pow(2)

    calc_progress.set(100)
    rootPhylDivBeta.update()
    rootPhylDivBeta.destroy()

    if dis:
        return beta2dist(beta=outFD, q=q, divType='phyl', viewpoint=viewpoint)
    else:
        return outFD

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
# disIndex specifies the dissimilarity index to calculate: 'Jaccard', 'Bray', and 'Hill' are available choices.
#    'Hill' refers to naive or phylogenetic dissimilarities of order q. If distmat is specified, phylogenetic are calculated.
def rcq(obj, constrainingVar='None', randomization='frequency', weightingVar='None', weight=1, iterations=9,
         disIndex='Hill', distmat='None', q=1, compareVar='None', RCrange='Raup'):

    # Returns a list of randomized tables
    def randomizeTabs():
        # Make tkinter object that keeps track of calculation progress
        rootRT = tk.Tk()
        rootRT.title('Step 1/2: randomizeTabs')
        calc_progress = tk.DoubleVar(rootRT, 0)
        calc_counter = 0
        tk.Label(rootRT, text='Progress in calculation (%)', width=30).pack()
        tk.Label(rootRT, textvariable=calc_progress, width=20).pack()

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
            for i in range(iterations):
                # For showing progress
                calc_counter += 1
                calc_progress.set(round(100 * calc_counter / (len(subtablist) * iterations), 1))
                rootRT.update()

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

        rootRT.destroy()  # Close "show progress" window
        return random_tabs

    #Get frequency table
    tab = obj['tab'].copy()
    if 'meta' in obj.keys():
        meta = obj['meta']

    #Generate random tabs
    randomtabs = randomizeTabs()

    # Make tkinter object that keeps track of calculationsprogress
    rootRCq = tk.Tk()
    rootRCq.title('Step 2/2: Compare beta diversity')
    calc_progress = tk.DoubleVar(rootRCq, 0)
    calc_counter = 0
    tk.Label(rootRCq, text='Progress in calculation (%)', width=30).pack()
    tk.Label(rootRCq, textvariable=calc_progress, width=20).pack()

    # Calculate betadiv for the subtab
    if disIndex == 'Bray':
        betadiv = bray(tab)
    elif disIndex == 'Jaccard':
        betadiv = jaccard(tab)
    elif disIndex == 'Hill' and isinstance(distmat, str):
        betadiv = naive_beta(tab, q=q)
    elif disIndex == 'Hill':
        betadiv = phyl_beta(tab, distmat=distmat, q=q)

    # The output is saved in these dataframes
    RC_tab = pd.DataFrame(0, index=tab.columns, columns=tab.columns)
    random_beta_all = np.zeros((len(tab.columns), len(tab.columns), iterations))

    for i in range(iterations):
        # For showing progress
        calc_counter += 1
        calc_progress.set(round(100 * calc_counter / iterations, 1))
        rootRCq.update()

        rtab = randomtabs[i]
        if disIndex == 'Bray':
            randombeta = bray(rtab)
        elif disIndex == 'Jaccard':
            randombeta = jaccard(rtab)
        elif disIndex == 'Hill' and isinstance(distmat, str):
            randombeta = naive_beta(rtab, q=q)
        elif disIndex == 'Hill':
            randombeta = phyl_beta(rtab, distmat=distmat, q=q)

        mask = betadiv > randombeta
        RC_tab[mask] = RC_tab[mask] + 1
        mask = betadiv == randombeta
        RC_tab[mask] = RC_tab[mask] + 0.5
        random_beta_all[:, :, i] = randombeta

    RC_tab = RC_tab / iterations #Calculate RC values
    if RCrange == 'Chase':
        RC_tab = (RC_tab - 0.5) * 2

    rootRCq.destroy() #Close "show progress" window

    out = {}
    if compareVar == 'None': #Straight forward comparison of all samples to each other
        out['Obs'] = betadiv
        out['RC'] = RC_tab
        medel = np.mean(random_beta_all, axis=2)
        stdav = np.std(random_beta_all, axis=2)
        out['Nullmean'] = pd.DataFrame(medel, index=RC_tab.index, columns=RC_tab.columns)
        out['Nullstd'] = pd.DataFrame(stdav, index=RC_tab.index, columns=RC_tab.columns)

    else: #Average all pairwise comparisons between samples in categories specified by compareVar
        indexlist = RC_tab.index.tolist()
        metalist = meta[compareVar].tolist()
        complist = [] #Hold list of unique categories from metalist
        [complist.append(x) for x in metalist if x not in complist]
        out_RCavg = pd.DataFrame(0, index=complist, columns=complist)
        out_RCstd = pd.DataFrame(0, index=complist, columns=complist)
        out_nullavg = pd.DataFrame(0, index=complist, columns=complist)
        out_nullstd = pd.DataFrame(0, index=complist, columns=complist)
        for c1nr in range(len(complist) - 1):
            c1 = complist[c1nr] #Category 1
            s1list = meta[meta[compareVar] == c1].index #All samples in category 1
            for c2nr in range(c1nr + 1, len(complist)):
                c2 = complist[c2nr] #Category 2
                s2list = meta[meta[compareVar] == c2].index #All samples in category 2

                # Check all pairwise comparisons between c1 and c2 samples
                RC_list = []
                null_list = []
                for s1 in s1list:
                    s1pos = indexlist.index(s1)
                    for s2 in s2list: #Compare sample s1 (in cat1) to all samples in cat 2
                        s2pos = indexlist.index(s2)
                        RC_list.append(RC_tab.loc[s1, s2])
                        null_list.append(random_beta_all[s1pos, s2pos, :])
                out_RCavg.loc[c1, c2] = np.mean(RC_list)
                out_nullavg.loc[c1, c2] = np.mean(null_list)
                out_RCstd.loc[c1, c2] = np.std(RC_list)
                out_nullstd.loc[c1, c2] = np.std(null_list)

                out_RCavg.loc[c2, c1] = out_RCavg.loc[c1, c2]
                out_nullavg.loc[c2, c1] = out_nullavg.loc[c1, c2]
                out_RCstd.loc[c2, c1] = out_RCstd.loc[c1, c2]
                out_nullstd.loc[c2, c1] = out_nullstd.loc[c1, c2]
        out['Obs'] = betadiv
        out['RCmean'] = out_RCavg
        out['RCstd'] = out_RCstd
        out['Nullmean'] = out_nullavg
        out['Nullstd'] = out_nullstd
    return out

# Calculates the evenness measures described in Chao and Ricotta (2019) Ecology 100(12), e02852
# tab is count table and q is diversity order
# index can be Chao1, Chao2, Chao3, Chao4, or Chao5, as specified in Chao and Ricotta (2019)
# Chao1 can also be called regional and Chao2 can also be called local
# if perspective=samples, the evenness value for each sample is calculate (i.e. each column in tab)
# if perspective=taxa, the evenness value for each taxon is calculate (i.e. each row in tab)
def evenness(tab, q=1, index='local', perspective='samples'):
    if perspective == 'samples':
        ra = tab / tab.sum()
        S_series = tab[tab > 0].count()
    elif perspective == 'taxa':
        tab = tab.transpose()
        ra = tab / tab.sum()
        S_series = tab.count() #Note all samples are counted here (even 0 reads)
    
    if index == 'Chao1' or index == 'regional':
        power = float(1-q)
    elif index == 'Chao2' or index == 'local':
        power = float(q-1)

    if q==1 and index in ['Chao1', 'Chao2', 'regional', 'local']:
        raLn = ra[ra > 0].applymap(math.log)
        Shannon = ra[ra > 0] * raLn[ra > 0]
        Shannon = (-1) * Shannon.sum()
        measure = Shannon / S_series.apply(math.log)
    elif index in ['Chao1', 'Chao2', 'regional', 'local']:
        D_series = naive_alpha(tab, q=q)
        D_series = D_series.pow(power)
        S_series = S_series.pow(power)
        measure = (1-D_series) / (1-S_series)
    elif index == 'Chao3':
        D_series = naive_alpha(tab, q=q)
        measure = (D_series - 1) / (S_series - 1)
    elif index == 'Chao4':
        D_series = naive_alpha(tab, q=q)
        measure = (1 - 1 / D_series) / (1 - 1 / S_series)
    elif index == 'Chao5':
        D_series = naive_alpha(tab, q=q)
        measure = D_series.apply(math.log) / S_series.apply(math.log)
    return measure

# Calculates the contribution of individual taxa to the dissimilarity between multiple samples
# see Chao and Ricotta (2019) Ecology 100(12), e02852
# obj is object containing at least meta and tab
# var is column in meta data specifying how count table should be subdivided
# if var=None, the whole count table is used 
# q is the diversity order
# index can be local (or Chao2) or regional (or Chao1)
def naive_dissimilarity_contributions(obj, var='None', q=1, index='local'):
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
    
    output = pd.DataFrame(np.nan, index=['dis', 'N'] + obj['tab'].index.tolist(), columns=catlist)
    for cat in catlist:
        tab = tabdict[cat]
        output.loc['N', cat] = len(tab.columns)
        if len(tab.columns) <= 1:
            continue

        if index == 'Chao1' or index == 'regional':
            w = tab.sum(axis=1).pow(q) / sum(tab.sum(axis=1).pow(q))
            ev = evenness(tab, q=q, index=index, perspective='taxa')
        elif index == 'Chao2' or index == 'local':
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
    return output

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
        df_temp.loc[range(N_rows), 'gamma'] = tab.sum(axis=1).values
        alphalist = []
        for col in tab.columns:
            alphalist = alphalist + tab[col].tolist()
        df_temp['alpha'] = alphalist
        alpha_gamma_divs = naive_alpha(df_temp, q=q)
        beta_div = alpha_gamma_divs['gamma'] / (alpha_gamma_divs['alpha'] / N_cols)
        output_beta.loc[cat, 'N'] = N_cols
        output_beta.loc[cat, 'beta'] = beta_div
        output_beta.loc[cat, 'local_dis'] = beta2dist(beta_div, q=q, N=N_cols, divType='naive', viewpoint='local')
        output_beta.loc[cat, 'regional_dis'] = beta2dist(beta_div, q=q, N=N_cols, divType='naive', viewpoint='regional')
    return output_beta
