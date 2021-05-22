import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import math

# Reads a newick tree file and returns a pandas dataframe
# Each column in the df represents a node in the tree: 
# nodes has the node names, ASVs have lists of tip nodes, branchL has the branch length.
def parse_newick(filename):
    #Read Newick tree file into one string
    newick = ''
    with open(filename, 'r') as f:
        for line in f:
            newick = newick + line.strip()
    if ':' not in newick:
        print('Error, no branch lengths in file')
        return 0
    if newick[0] == '(':
        newick = newick[1:-2]

    #Parsing: get all nodes, the end nodes the contain, and the branch lengths
    intnodenames = []
    intnodecounter = 0
    temp_endnodes = []
    nodelist = [] #All nodes
    asvlist = [] #ASV names connected to node
    BLlist = [] #Branch length list
    for i, char in enumerate(newick):
        
        #Find positions of parentheses
        if char == '(':
            temp_endnodes.append([])
            intnodecounter += 1
            intnodenames.append('i'+str(intnodecounter))

        #Find nodes
        elif char == ':':
            #Get node name
            for j in range(i, -2, -1):
                if newick[j] in [',', '('] or j == -1: #It is end node
                    endnodename = newick[j+1:i]
                    nodelist.append(endnodename)
                    asvlist.append([endnodename])
                    if len(temp_endnodes) > 0: #Append it to all open internal nodes
                        for sub_list in temp_endnodes:
                            sub_list.append(endnodename)
                    break
                elif newick[j] == ')': #It is internal node
                    endnodename = intnodenames[-1]
                    nodelist.append(endnodename)
                    asvlist.append(temp_endnodes[-1])
                    intnodenames.pop(-1)
                    temp_endnodes.pop(-1)
                    break
            #Get branch length of node
            for j in range(i+1, len(newick)):
                if newick[j] in [',', ')']:
                    branchlen = float(newick[i+1:j])
                    break
                elif j == len(newick)-1:
                    branchlen = float(newick[i+1:j+1])
            BLlist.append(branchlen)
    df = pd.DataFrame({'nodes': nodelist, 'ASVs': asvlist, 'branchL': BLlist})
    return df

# Returns color- or marker lists to use in figures
# type is 'colors' or 'markers'. If plot=True a figure with available options will be shown else a list is returned
def get_colors_markers(type='colors', plot=False):
    # Sort colors by hue, saturation, value and name.
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name) for name, color in colors.items())
    color_names = [name for hsv, name in by_hsv]

    if type == 'colors' and not plot:
        numberlist = [128, 24, 38, 79, 146, 49, 152, 117, 58, 80, 119, 20, 97, 57, 138, 120, 153, 60, 16]
        outputlist = []
        for i in numberlist:
            outputlist.append(color_names[i])
        return outputlist

    elif type == 'colors' and plot:
        n = len(color_names)
        ncols = 4
        nrows = n // ncols

        fig, ax = plt.subplots(figsize=(12, 10))

        # Get height and width
        X, Y = fig.get_dpi() * fig.get_size_inches()
        h = Y / (nrows + 1)
        w = X / ncols

        for i, name in enumerate(color_names):
            row = i % nrows
            col = i // nrows
            y = Y - (row * h) - h

            xi_line = w * (col + 0.05)
            xf_line = w * (col + 0.25)
            xi_text = w * (col + 0.3)

            ax.text(xi_text, y, str(i)+':'+name, fontsize=(h * 0.6),
                    horizontalalignment='left',
                    verticalalignment='center')
            ax.hlines(y + h * 0.1, xi_line, xf_line,
                      color=colors[name], linewidth=(h * 0.8))

        ax.set_xlim(0, X)
        ax.set_ylim(0, Y)
        ax.set_axis_off()

        fig.subplots_adjust(left=0, right=1,
                            top=1, bottom=0,
                            hspace=0, wspace=0)
        plt.show()

    elif type == 'markers' and not plot:
        return ['o', 's', 'v', 'X', '.', '*', 'P', 'D', '<', ',', '^', '>', '1', '2', '3', '4', '8', 'h', 'H', '+']

    elif type == 'markers' and plot:
        mlist = ['o', 's', 'v', 'X', '*', 'P', 'D', '<', '1', '^', '2', '>', '3', '4', '.']
        for i in range(len(mlist)):
            plt.scatter(i+1, i+1, marker=mlist[i], s=30)
            plt.text(i+1, i+1.5, mlist[i])
        plt.show()

# Groups SVs based on taxa, returns object with grouped sequences
# levels specifies taxonomic levels to use in the grouping
# nameType specifies the abbreviation to be used for unclassified sequences
# if nameType='merge', all unclassified sequences will be merged
def groupbytaxa(obj, levels=['Phylum', 'Genus'], merge=False):
    # Clean up tax data frame
    taxM = obj['tax'].copy()
    taxS = obj['tax'].copy()

    taxlevels = taxM.columns.tolist()
    group_level = levels[-1]
    if group_level not in taxlevels:
        print('Error in groupbytaxa, level not found')
        return None

    pos = taxlevels.index(group_level)
    taxM[taxlevels[0]][taxM[taxlevels[0]].isna()] = 'Unclassified'
    if pos != 0:
        for i in range(1, pos+1):
            t0 = taxlevels[i-1]
            t1 = taxlevels[i]
            taxM[t1][taxM[t1].isna()] = taxM[t0][taxM[t1].isna()]

    taxS['Name'] = ''
    if merge and len(levels) == 1:
        taxS['Name'] = taxM[group_level]
    elif merge and len(levels) > 1:
        taxS['Name'] = taxM[levels[0]]
        for i in range(1, len(levels)):
            t0 = levels[i-1]
            t1 = levels[i]
            taxS['Name'][taxM[t0] != taxM[t1]] = taxS['Name'][taxM[t0] != taxM[t1]] + '; ' + taxM[t1][taxM[t0] != taxM[t1]]
    if merge:
        mask = (taxS[group_level].isna()) & (taxM[group_level] != 'Unclassified')
        taxS['Name'][mask] = taxS['Name'][mask] + ':unclassified'
        
    elif not merge and len(levels) == 1:
        taxS['Name'][taxS[group_level].notna()] = taxS[group_level][taxS[group_level].notna()]
        taxS['Name'][taxS[group_level].isna()] = taxM[group_level][taxS[group_level].isna()] + ':' + list(map(str, taxM[group_level][taxS[group_level].isna()].index))
    elif not merge and len(levels) > 1:
        taxS['Name'] = taxM[levels[0]]
        for i in range(1, len(levels)):
            t0 = levels[i-1]
            t1 = levels[i]
            if t1 != group_level:
                taxS['Name'][taxM[t0] != taxM[t1]] = taxS['Name'][taxM[t0] != taxM[t1]] + '; ' + taxM[t1][taxM[t0] != taxM[t1]]
            else:
                mask = (taxS[t1].isna()) & (taxM[t0] != taxM[t1])
                taxS['Name'][mask] = taxS['Name'][mask] + '; ' + taxM[t1][mask] + ':' + list(map(str, taxM[t1][mask].index))
                mask = (taxS[t1].notna()) & (taxM[t0] != taxM[t1])
                taxS['Name'][mask] = taxS['Name'][mask] + '; ' + taxM[t1][mask]
                mask = (taxS[t1].isna()) & (taxM[t0] == taxM[t1])
                taxS['Name'][mask] = taxS['Name'][mask] + ':' + list(map(str, taxM[t1][mask].index))

    #Grouby Name and return object
    out = {}
    if 'tab' in obj:
        tab = obj['tab'].copy()
        tab['Name'] = taxS['Name']
        tab = tab.set_index(['Name'])
        tab = tab.groupby(tab.index).sum()
        out['tab'] = tab
    if 'ra' in obj:
        ra = obj['ra'].copy()
        ra['Name'] = taxS['Name']
        ra = ra.set_index('Name')
        ra = ra.groupby(ra.index).sum()
        out['ra'] = ra
    if 'meta' in obj:
        out['meta'] = obj['meta'].copy()
    if 'tax' in obj:
        newtax = obj['tax'].copy()
        newtax['Name'] = taxS['Name']
        newtax = newtax.set_index(['Name'])
        newtax = newtax.groupby(newtax.index).first()
        out['tax'] = newtax
    return out

# Converts beta value to distances, specify q and type associated with the beta (assuming pairwise)
# Used in beta dissimilarity calculations
# The viewpoint refers to either the local or regional perspective as defined in Chao et al. 2014
def beta2dist(beta, q=1, N=2, divType='naive', viewpoint='local'):
    if isinstance(beta, pd.DataFrame): #if beta is a pd.DataFrame
        beta = beta.applymap(float)
        mask = beta > 0
        dist = beta.copy()

        if q == 1:
            if divType in ['naive', 'phyl']:
                dist[mask] = np.log(beta[mask]) / math.log(N)
            elif divType == 'func':
                dist[mask] = np.log(beta[mask]) / (2 * math.log(N))
        else:
            if divType in ['naive', 'phyl'] and viewpoint == 'local':
                dist[mask] = 1 - (N**(1 - q) - beta[mask]**(1 - q)) / (N**(1 - q) - 1)
            elif divType == 'func' and viewpoint == 'local':
                dist[mask] = 1 - ((N**(2 * (1 - q)) - beta[mask]**(1 - q)) / (N**(2 * (1 - q)) - 1))
            elif divType in ['naive', 'phyl'] and viewpoint == 'regional':
                dist[mask] = 1 - ((1 / beta[mask])**(1 - q) - (1 / N)**(1 - q)) / (1 - (1 / N)**(1 - q))
            elif divType == 'func' and viewpoint == 'regional':
                dist[mask] = 1 - ((1 / beta[mask])**(1 - q) - (1 / N)**(2 * (1 - q))) / (1 - (1 / N)**(2 * (1 - q)))
    else: #if beta is just a number
        if q == 1:
            if divType in ['naive', 'phyl']:
                dist = math.log(beta) / math.log(N)
            elif divType == 'func':
                dist = math.log(beta) / (2 * math.log(N))
        else:
            if divType in ['naive', 'phyl'] and viewpoint == 'local':
                dist = 1 - (N**(1 - q) - beta**(1 - q)) / (N**(1 - q) - 1)
            elif divType == 'func' and viewpoint == 'local':
                dist = 1 - ((N**(2 * (1 - q)) - beta**(1 - q)) / (N**(2 * (1 - q)) - 1))
            elif divType in ['naive', 'phyl'] and viewpoint == 'regional':
                dist = 1 - ((1 / beta)**(1 - q) - (1 / N)**(1 - q)) / (1 - (1 / N)**(1 - q))
            elif divType == 'func' and viewpoint == 'regional':
                dist = 1 - ((1 / beta)**(1 - q) - (1 / N)**(2 * (1 - q))) / (1 - (1 / N)**(2 * (1 - q)))
    return dist

# Returns Rao's quadratic entropy, sum(sum(dij*pi*pj))
# Function used in Chiu's functional diversity functions
def rao(tab, distmat):
    ra = tab / tab.sum()
    svlist = list(ra.index)
    distmat = distmat.loc[svlist, svlist]
    if isinstance(tab, pd.DataFrame):
        outdf = pd.Series(0, index=ra.columns)
        for smp in ra.columns:
            ra2mat = pd.DataFrame(np.outer(ra.loc[:, smp].values, ra.loc[:, smp].values), index=ra.index, columns=ra.index)
            rao_mat = ra2mat.mul(distmat)
            Qvalue = sum(rao_mat.sum())
            outdf.loc[smp] = Qvalue
        return outdf
    elif isinstance(tab, pd.Series):
        ra2mat = pd.DataFrame(np.outer(ra.values, ra.values), index=ra.index, columns=ra.index)
        rao_mat = ra2mat.mul(distmat)
        Qvalue = sum(rao_mat.sum())
        return Qvalue
        
# Return confidence ellipse, used in plot.pcoa
def pcoa_ellipse(x, y, n_std=2): #Method from https://matplotlib.org/3.1.0/gallery/statistics/confidence_ellipse.html
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    #Get extreme values
    if len(x) != len(y):
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    if cov[0, 0] * cov[1, 1] != 0:
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    elif cov[0, 0] == 0:
        pearson = -1
    else:
        pearson = 1

    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    return [ell_radius_x, ell_radius_y, scale_x, scale_y, mean_x, mean_y]

