import pandas as pd
import numpy as np
import random

# Returns some information about an object, e.g. number of samples, reads, headings in meta data etc.
def print_info(obj):
    print('Dataframes in object:', list(obj.keys()))
    if 'tab' in obj.keys():
        tab = obj['tab']
        print('Total number of samples= ', len(tab.columns))
        print('Total number of ASVs= ', len(tab.index))
        print('Total reads= ', sum(tab.sum()))
        print('Minimum number of reads in a sample= ', min(tab.sum()))
    if 'tax' in obj.keys():
        print('Taxonomic levels:', obj['tax'].columns.tolist())
    if 'tree' in obj.keys():
        print('Branches in tree:', len(obj['tree']))
    if 'meta' in obj.keys():
        print('Column headings in meta data:')
        print(list(obj['meta'].columns))
        print('First row in meta data:')
        print(list(obj['meta'].iloc[0, :]))
    return None

# Prints file with number of different taxa at different taxonomic levels
def taxa(obj, savename='None'):
    if 'tax' in obj.keys():
        tax = obj['tax'].copy()
        taxlevels = tax.columns.tolist() #List of taxonomic levels
    else:
        taxlevels = []

    tab = obj['tab'].copy()
    samples = tab.columns.tolist() #List of samples in freq table

    # Make output file
    output = [['Sample', 'ASVs', 'Reads'] + taxlevels]
    for smp in ['Total'] + samples:
        templist = [smp]
        if smp == 'Total':  # For the whole freq table
            templist.append(len(tab.index))
            templist.append(sum(tab.sum()))
        else:
            templist.append(len(tab[tab[smp] > 0].index))
            templist.append(tab[smp].sum())

        if 'tax' in obj.keys():
            for tlev in taxlevels:
                if smp == 'Total': #For the whole freq table
                    nrofdifferent = len(tax[tax[tlev].notnull()].groupby(by=tlev).first().index)
                    templist.append(nrofdifferent)
                else: # For each sample
                    taxsmp = tax[tab[smp] != 0]
                    nrofdifferent = len(taxsmp[taxsmp[tlev].notnull()].groupby(by=tlev).first().index)
                    templist.append(nrofdifferent)
        output.append(templist)
    output = pd.DataFrame(output[1:], columns=output[0])
    output = output.set_index('Sample')
    if savename != 'None':
        output.to_csv(savename + '.csv')
    return output

# dis1 and dis2 are two dissimilarity matrices
# the mantel test check the correlation between the two
# 'spearman', 'pearson', on 'absDist' can be chosen as method
# a p-value for the correlation is obtained by permutation of one of the matrices
# the number of times the random matrices have higher correlation (or lower absDist) are counted
# if getOnlyStat is true, only the statistic is returned, no permutations are carried out
# returns list [statistic, p_value]
def mantel(dis1, dis2, method='spearman', getOnlyStat=False, permutations=99):

    # Function to calculate a statistic as a dissimilarity, i.e. 1-r or 1-rho
    def get_stat(mat1, mat2):
        help_df = np.tril(np.ones(mat1.shape), k=-1).astype(np.bool)
        vect1 = mat1.where(help_df).stack().values
        vect2 = mat2.where(help_df).stack().values

        if method in ['spearman', 'pearson']:
            dfcorr = pd.DataFrame({'mat1': vect1, 'mat2': vect2})
            return 1 - dfcorr.corr(method=method).loc['mat1', 'mat2']
        elif method == 'absDist':
            subtr = np.subtract(vect1, vect2)
            subtr = np.absolute(subtr)
            return np.sum(subtr) / len(subtr)

    # Sort matrices
    smplist = dis1.columns.tolist()
    smplist = sorted(smplist)
    dis1 = dis1.reindex(smplist, axis=1)
    dis1 = dis1.reindex(smplist, axis=0)
    dis2 = dis2.reindex(smplist, axis=1)
    dis2 = dis2.reindex(smplist, axis=0)

    # Calculate real stat
    real_stat = get_stat(dis1, dis2)
    if getOnlyStat:
        return real_stat
    else: #Go through permutations
        null_stats = np.empty(permutations)
        for i in range(permutations):
            random_smplist = smplist.copy()
            random.shuffle(random_smplist)
            random_dis1 = pd.DataFrame(dis1.values, index=random_smplist, columns=random_smplist)
            random_dis1 = random_dis1.reindex(smplist, axis=1)
            random_dis1 = random_dis1.reindex(smplist, axis=0)
            null_stats[i] = get_stat(random_dis1, dis2)

        p_val = (1 + len(null_stats[null_stats < real_stat] + 0.5 * len(null_stats[null_stats == real_stat]))) / (1 + len(null_stats))
        return [real_stat, p_val]

# dis is dissimilarity matrix
# meta is the metadata
# var is the heading in the meta data having categories which group the samples
# returns list [F_stat, p_value]
def permanova(dis, meta, var, permutations=99):

    def get_F(dis, cats_list):
        # Get SStotal
        help_df = np.tril(np.ones(dis.shape), k=-1).astype(np.bool)
        vectDis = dis.where(help_df).stack().values
        SStotal = sum(vectDis**2) / len(dis.index)
        # Get SSwithin
        SSw = 0
        for cat in cats_list:
            subsmplist = meta[var][meta[var] == cat].index #Take all samples within the same group
            disw = dis.loc[subsmplist, subsmplist]
            help_df = np.tril(np.ones(disw.shape), k=-1).astype(np.bool)
            vectDisw = disw.where(help_df).stack().values
            SSw = SSw + sum(vectDisw**2) / len(subsmplist)

        # Calculate Fstat
        SSa = SStotal - SSw
        Fstat = (SSa / (len(cats_list) - 1)) / (SSw / (len(dis.index) - len(cats_list)))
        return Fstat

    # Get unique categories in meta[var]
    cats_list = np.unique(meta[var])
    real_F = get_F(dis, cats_list)

    null_F = []
    smplist = dis.index.tolist()
    for i in range(permutations):
        random_smplist = smplist.copy()
        random.shuffle(random_smplist)
        random_dis = pd.DataFrame(dis.values, index=random_smplist, columns=random_smplist)
        null_F.append(get_F(random_dis, cats_list))

    p_val = 0
    for nF in null_F:
        if nF >= real_F:
            p_val += 1
    p_val = (p_val + 1) / (len(null_F) + 1)
    return [real_F, p_val]

# Returns matrix for pairwise distances between ASVs.
# The inputType can either be seq or tree.
# If the input is seq, pairwise Levenshtein/Hamming distances (uses Levenshtein package) are calculated.
# If the input is tree, the distances between end nodes in the phylogenetic tree are calculated.
# Results are saved as csv file at location specified in savename.
# The output distance matrix can be used as input for functional diversity index calculations (func_alpha, func_beta)
# or for phylogenetic-based null models (nriq, ntiq, beta_nriq, beta_ntiq)
def sequence_comparison(obj, inputType='seq', savename='DistMat'):

    if inputType == 'seq': #Sequences as input
        seq = obj['seq']

        svnames = list(seq.index)
        # For showing progress
        total_comp = (len(svnames)**2)/2
        show_after = int(total_comp / 50) + 1
        counter = 0
        print('Progress in sequence_comparison.. 0%.. ')
    
        df = pd.DataFrame(0, index=svnames, columns=svnames)
        for i in range(len(svnames) - 1):
            n1 = svnames[i]
            s1 = seq.loc[n1, 'seq']

            for j in range(i + 1, len(svnames)):
                counter += 1 #Showing progress
                if counter%show_after == 0:
                    print(int(100*counter/total_comp), end='%.. ')
   
                n2 = svnames[j]
                s2 = seq.loc[n2, 'seq']

                matrix = np.empty((len(s1)+1, len(s2)+1))
                matrix[:, 0] = range(len(s1)+1)
                matrix[0, :] = range(len(s2)+1)
                for pos1 in range(1, len(s1)+1):
                    lowerbound = max(1, pos1-12)
                    upperbound = min(pos1+12, len(s2)+1)
                    matrix[pos1, :lowerbound] = matrix[pos1-1, :lowerbound]
                    matrix[pos1, upperbound:] = matrix[pos1-1, upperbound:]
                    for pos2 in range(lowerbound, upperbound):
                        if s1[pos1-1] == s2[pos2-1]:
                            matrix[pos1, pos2] = matrix[pos1-1, pos2-1]
                        else:
                            matrix[pos1, pos2] = min(matrix[pos1-1, pos2]+1, matrix[pos1, pos2-1]+1, matrix[pos1-1, pos2-1]+1)
                df.loc[n1, n2] = matrix[-1, -1] / max(len(s1), len(s2))
                df.loc[n2, n1] = df.loc[n1, n2]
        print('100%')
        df.to_csv(savename+'.csv')
        return df

    elif inputType == 'tree': #Tree as input
        tree = obj['tree'].copy()

        #Sort out end nodes and internal nodes
        tree['asv_count'] = np.nan
        for ix in tree.index:
            tree.loc[ix, 'asv_count'] = len(tree.loc[ix, 'ASVs'])
        tree_endN = tree[tree['asv_count'] == 1]
        
        #Get list of asvs
        svnames = sorted(tree_endN['nodes'].tolist())
        df = pd.DataFrame(0, index=svnames, columns=svnames)

        # For showing progress
        total_comp = (len(svnames)**2 / 2)
        show_after = int(total_comp / 50) + 1
        counter = 0
        print('Progress in sequence_comparison.. 0%.. ', end='')
        
        #Go through branchL and add total branch length for each, df will thus give branchL to root
        for ix in tree.index:
            BL = tree.loc[ix, 'branchL']
            asvlist = tree.loc[ix, 'ASVs']
            df.loc[asvlist, asvlist] = df.loc[asvlist, asvlist] + BL

        df_dist = pd.DataFrame(0, index=svnames, columns=svnames)
        for i in range(len(svnames)-1):
            sv1 = svnames[i]
            sv1_toroot = df.loc[sv1, sv1]
            for j in range(i+1, len(svnames)):
                counter += 1
                if counter%show_after == 0:
                    print(int(100*counter/total_comp), end='%.. ')

                sv2 = svnames[j]
                sv2_toroot = df.loc[sv2, sv2]
                total_dist = sv1_toroot + sv2_toroot
                shared_dist = df.loc[sv1, sv2]
                df_dist.loc[sv1, sv2] = total_dist - 2 * shared_dist
                df_dist.loc[sv2, sv1] = total_dist - 2 * shared_dist
        
        print('100%')
        df_dist.to_csv(savename+'.csv')
        return df_dist
