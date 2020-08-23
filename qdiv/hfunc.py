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
def groupbytaxa(obj, levels=['Phylum', 'Genus'], nameType='ASV'):
    # Clean up tax data frame
    tax = obj['tax'].copy()
    tax = tax.fillna('0')
    taxSV = tax.copy() #df to hold nameTypes in undefined
    taxNames = tax.copy() #df to hold lowest known taxaname in undefined

    # Check which OTU or SV name is used in the index
    indexlist = tax.index.tolist()
    if indexlist[0][:3] in ['Otu', 'OTU', 'ASV', 'ESV']:
        currentname = indexlist[0][:3]
        startpos = 3
    elif indexlist[0][:2] in ['SV']:
        currentname = indexlist[0][:2]
        startpos = 2
    else:
        print('Error in groupbyTaxa, ASV/OTU name not known')
        return 0

    # If incorrect name is in tax, make column with correct name
    if nameType != currentname:
        newnames = []
        for i in range(len(indexlist)):
            newnames.append(nameType+indexlist[i][startpos:])
        indexlist = newnames

    # Put the ASV/OTU name in all empty spots in taxSV
    for col in range(len(taxSV.columns)):
        for row in range(len(taxSV.index)):
            if taxSV.iloc[row, col] == '0':
                taxSV.iloc[row, col] = indexlist[row]
    taxSV[nameType] = indexlist

    # Change all 0 in tax to lowest determined taxa level in taxNames
    for ix_nr in range(len(taxNames.index)):
        if taxNames.iloc[ix_nr, 0] == '0':
            taxNames.iloc[ix_nr, 0] = taxSV.loc[taxNames.index[ix_nr], nameType]

    taxanameslist = taxNames.columns.tolist() #List with Kingdom, Phylum .. SV
    for s_nr in range(1, len(taxanameslist)):
        s0 = taxanameslist[s_nr-1]
        s1 = taxanameslist[s_nr]
        taxNames[s1][tax[s1] == '0'] = taxNames[s0][tax[s1] == '0']

    # Create names to use in output
    if len(levels) == 1:
        tax['Name'] = taxSV[levels[0]]
        for ix in tax.index:
            if tax.loc[ix, levels[0]] == '0':
                tax.loc[ix, 'Name'] = taxNames.loc[ix, levels[0]] + '; ' + tax.loc[ix, 'Name']
    elif len(levels) == 2:
        tax['Name'] = taxNames[levels[0]]+'; '+taxSV[levels[1]]
    else:
        print('Error in GroupbyTaxa, levels should be a list with 1 or 2 items')
        return 0

    #Grouby Name and return object
    out = {}
    if 'tab' in obj:
        tab = obj['tab'].copy()
        tab['Name'] = tax['Name']
        tab = tab.set_index(['Name'])
        tab = tab.groupby(tab.index).sum()
        out['tab'] = tab
    if 'ra' in obj:
        ra = obj['ra'].copy()
        ra['Name'] = tax['Name']
        ra = ra.set_index('Name')
        ra = ra.groupby(ra.index).sum()
        out['ra'] = ra
    if 'meta' in obj:
        out['meta'] = obj['meta'].copy()
    if 'tax' in obj:
        newtax = obj['tax'].copy()
        newtax['Name'] = tax['Name']
        newtax = newtax.set_index(['Name'])
        newtax = newtax.groupby(newtax.index).first()
        if len(levels) == 1:
            lowest_level = levels[0]
        else:
            lowest_level = levels[-1]
        position_nr = tax.columns.tolist().index(lowest_level)
        for c_nr in range(len(newtax.columns)):
            if c_nr > position_nr:
                newtax.iloc[:, c_nr] = np.nan
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
