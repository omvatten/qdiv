import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

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
        return None
    if newick[0] == '(':
        newick = newick[1:-2]

    #Parsing: get all nodes, the end nodes they contain, and the branch lengths
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
def groupbytaxa(obj, levels=['Phylum', 'Genus'], includeIndex=False):
    if 'tax' not in obj:
        print('Error: tax not in obj.')
        return None
    if 'tab' not in obj:
        print('Error: tab not in obj.')
        return None

    levdict = {'superkingdom':'sk__','clade':'cl__', 'kingdom':'k__','domain':'d__','realm':'r__','phylum':'p__','class':'c__','order':'o__','family':'f__','subfamily':'sf__', 'genus':'g__','species':'s__'}

    # Simplify tax data frame and fill with names in all empty fields
    taxM = obj['tax'].copy()
    taxlevels = taxM.columns.tolist()
    taxlevels = [x.lower() for x in taxlevels]
    taxM.columns = taxlevels

    group_level = levels[-1].lower()
    if group_level not in taxlevels:
        print('Error in hfunc.groupbytaxa, level not found')
        return None
    pos = taxlevels.index(group_level)
    taxlevels = taxlevels[:pos+1]
    taxM = taxM[taxlevels]

    highest = taxlevels[0]
    if highest not in levdict.keys():
        print('Error in hfunc.groupbytaxa, highest level not recognized')
        print(highest)
        return None

    taxM.loc[taxM[highest].isna(), highest] = levdict[highest]+'Unclassified'
    if len(taxlevels) > 1:
        for i in range(1, len(taxlevels)):
            t0 = taxlevels[i-1]; t1 = taxlevels[i]
            taxM.loc[taxM[t1].isna(), t1] = taxM.loc[taxM[t1].isna(), t0]

    #Accumulate names from highest to lowest taxonomic level
    taxAcc = taxM.copy()

    if includeIndex: #Include index name at lowest level
        taxAcc[group_level] = taxAcc[group_level]+':'+taxAcc.index
        taxAcc['index'] = taxAcc.index

    if len(taxlevels) > 1:
        for i in range(1, len(taxlevels)):
            t0 = taxlevels[i-1]; t1 = taxlevels[i]
            taxAcc[t1] = taxAcc[t0] + taxAcc[t1]

    #Set groupname for all relevant dataframe and groupby
    taxAcc['gName'] = taxAcc[group_level]
    taxM['gName'] = taxAcc['gName']
    tab = obj['tab'].copy()
    tab['gName'] = taxAcc['gName']
    
    if 'seq' in obj:
        seq = obj['seq'].copy()
        seq['gName'] = taxAcc['gName']
        seq = seq.groupby('gName').first()
    taxAcc = taxAcc.groupby('gName').first()
    taxM = taxM.groupby('gName').first()
    tab = tab.groupby('gName').sum()
    
    #Fix italics in taxM
    for c in taxM.columns:
        candidatus = taxM[(taxM[c].notna())&((taxM[c].str.contains('Candidatus'))|(taxM[c].str.contains('Ca.', regex=False)))].index.tolist()
        for asv in candidatus:
            taxM.loc[asv, c] = taxM.loc[asv, c].replace('Ca.', '$\it{Ca.}$').replace('Candidatus', '$\it{Ca.}$')
        unclassified = taxM.loc[taxM[c].str.contains('unclassified', case=False), c].index.tolist()
        underscores = taxM[(taxM[c].notna())&(taxM[c].str.contains('__'))].index.tolist()
        underscores = list(set(underscores)-set(candidatus)-set(unclassified))
        taxM.loc[underscores, c] = taxM.loc[underscores, c].str.split('__').str[0]+'__$\it{'+taxM.loc[underscores, c].str.split('__').str[1].str.replace(' ','\ ').str.replace('_','\ ')+'}$'
        rest = taxM[taxM[c].notna()].index.tolist()
        rest = list(set(rest)-set(candidatus)-set(underscores)-set(unclassified))
        taxM.loc[rest, c] = '$\it{'+taxM.loc[rest, c].str.replace(' ','\ ').str.replace('_','\ ')+'}$'

    #Make new index name based on levels
    if len(levels) == 1:
        taxM['Name'] = taxM[group_level]
    elif len(levels) > 1:
        taxM['Name'] = taxM[levels[0]]
        for i in range(1, len(levels)):
            t0 = levels[i-1].lower(); t1 = levels[i].lower()
            taxM.loc[taxM[t0] != taxM[t1], 'Name'] = taxM.loc[taxM[t0] != taxM[t1], 'Name'] + '; ' + taxM.loc[taxM[t0] != taxM[t1], t1]

    if includeIndex: #Include index name at lowest level
        taxM['Name'] = taxM['Name']+':'+taxAcc['index']

    #Grouby Name and return object
    out = {}
    tab['Name'] = taxM['Name']
    out['tab'] = tab.groupby('Name').sum()
    
    if 'seq' in obj:
        seq['Name'] = taxM['Name']
        out['seq'] = seq.groupby('Name').first()

    out['tax'] = taxM.groupby('Name').first()
    
    if 'meta' in obj:
        out['meta'] = obj['meta'].copy()

    return out

# Converts beta value to distances, specify q and type associated with the beta (assuming pairwise)
# Used in beta dissimilarity calculations
# The viewpoint refers to either the local or regional perspective as defined in Chao et al. 2014
def beta2dist(beta, q=1, N=2, divType='naive', viewpoint='local'):
    if isinstance(beta, pd.DataFrame): #if beta is a pd.DataFrame
        beta = beta.map(float)
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
        outdf = pd.Series(0.0, index=ra.columns)
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

def pcoa_calculation(dist):
    #Function for centering and eigen-decomposition of distance matrix
    def get_eig(d): 
        dist2 = -0.5*(d**2)
        col_mean = dist2.mean(axis=0)
        row_mean = dist2.mean(axis=1)
        tot_mean = np.array(dist2).flatten().mean()
        dist2_cent = dist2.subtract(col_mean, axis=1)
        dist2_cent = dist2_cent.subtract(row_mean, axis=0)
        dist2_cent = dist2_cent.add(tot_mean)
        vals, vects = np.linalg.eig(dist2_cent)
        return [vals, vects]

    #Get eigenvalues and eigenvectors from the dissimilarity matrix
    dist.fillna(0, inplace=True)
    ev_ev = get_eig(dist)
    #Correction method for negative eigenvalues
    if min(ev_ev[0]) < 0: #From Legendre 1998, method 1 (derived from Lingoes 1971) chapter 9, page 502
        mask = np.eye(N=len(dist.index))
        mask = mask == 0
        mask = pd.DataFrame(mask, index=dist.index, columns=dist.columns)
        d2 = dist.copy()
        d2[mask] = d2[mask].pow(2) + 2*abs(min(ev_ev[0]))
        d2[mask] = d2[mask].pow(0.5)
        d2 = d2.fillna(0)
        ev_ev = get_eig(d2)
        ev_ev[0] = np.real(ev_ev[0])
        ev_ev[1] = np.real(ev_ev[1])

    #Get proportions and coordinates
    vv = pd.DataFrame(pd.NA, index=range(len(ev_ev[0])), columns=dist.columns)
    for i in range(len(ev_ev[0])):
        e_val = ev_ev[0][i]
        if e_val > 0:
            vv.iloc[i, :] = (e_val**0.5)*ev_ev[1][:, i]
        else:
            vv.iloc[i, :] = 0*ev_ev[1][:, i]
    e_val_frac = 100*ev_ev[0]/sum(ev_ev[0])
    vv['eig_vals'] = ev_ev[0]
    vv['frac'] = e_val_frac
    vv['frac'] = vv['frac'].apply(lambda x: round(x, 2))
    vv = vv.sort_values('frac', ascending=False)
    vv['PCo'] = np.arange(len(e_val_frac)) + 1
    vv['PCo'] = 'PCo'+vv['PCo'].apply(str)
    vv['PCo'] = vv['PCo'] + ' ('+vv['frac'].apply(str) + '%)'
    vv = vv.set_index('PCo')
    vv = vv.drop('frac', axis=1)
    eig_vals = vv['eig_vals']
    vv = vv.drop('eig_vals', axis=1)
    vv = vv.transpose()
    return vv, eig_vals

#Order the index of a dataframe based on trailing numbers
def orderSeqs(df):
    df['letterstartX'] = df.index.str.rstrip('0123456789').tolist()
    df['letterstartLEN'] = df['letterstartX'].str.len()
    df['numberendX'] = [x[i:] for x, i in zip(df.index, df['letterstartLEN'])]
    df.loc[df['numberendX'] == '', 'numberendX'] = 0
    df['numberendX'] = df['numberendX'].apply(int)
    df = df.sort_values(by='numberendX', ascending=True)
    df = df.drop('letterstartX', axis=1)
    df = df.drop('letterstartLEN', axis=1)
    df = df.drop('numberendX', axis=1)
    return df
    
