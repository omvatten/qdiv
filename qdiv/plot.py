import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pickle
import math
from . import subset
from . import diversity
from . import null
from . import hfunc
pd.options.mode.chained_assignment = None  # default='warn'

# PLOT FUNCTIONS

## Plots heatmap
    # xAxis specifies heading in meta data used to merge samples
    # levels specifies taxonomic levels used in y axis
    # subsetLevels and subsetPatterns refer to subsetTextPatters function which can be used to filter results
    # order refers to heading in meta data used to order samples
    # numberToPlot refers to the number of taxa with highest abundance to include in the heatmap
    # method refers to the method used to define the taxa with highest abundance, 'max_sample' is max relative abundance in a sample,
    # 'mean_all' is the mean relative abundance across all samples
    # nameType specifies the abbreviation to be used for unclassified sequences
    # figsize is the width and height of the figure
    # fontsize is refers to the axis text
    # sepCol is a list of column numbers between which to include a separator, i.e. to clarify grouping of samples
    # if labels=True, include relative abundance values in heatmap, if False they are not included
    # labelsize is the font size of the relative abundance lables in the heatmap
    # cThreshold is the relative abundance % at which the label color switches from black to white (for clarity)
    # cMap is the color map used in the heatmap
    # cLinear is a parameter determining how the color change with relative abundance, a value of 1 means the change is linear
    # cBar is a list of tick marks to use if a color bar is included as legend
    # savename is the name (also include path) of the saved png file, if 'None' no figure is saved
def heatmap(obj, xAxis='None', levels=['Phylum', 'Genus'], levelsShown='None', subsetLevels='None', subsetPatterns='None',
                order='None', numberToPlot=20, method='max_sample', nameType='ASV',
                 figsize=(14, 10), fontsize=15, sepCol = [],
                labels=True, labelsize=10, cThreshold=8,
                cMap='Reds', cLinear=0.5, cBar=[], savename='None'):

    #Merge samples based on xAxis
    if xAxis != 'None':
        merged_obj = subset.merge_samples(obj, var=xAxis)
    else:
        merged_obj = obj.copy()

    #Calculate relative abundances and store in df ra
    tab = merged_obj['tab']
    ra = 100*tab/tab.sum()
    merged_obj['ra'] = ra

    ## Make sure samples are in the right order in meta data
    if order != 'None':
        md = merged_obj['meta']
        md[order] = md[order].astype(float)
        md = md.sort_values(by=order)
        logiclist = []
        if xAxis != 'None':
            [logiclist.append(item) for item in md[xAxis] if item not in logiclist]
        else:
            [logiclist.append(item) for item in md.index if item not in logiclist]
        merged_obj['meta'] = md

    ## Subset based on pattern
    if subsetLevels != 'None' and isinstance(subsetLevels, list) and isinstance(subsetPatterns, list):
        subset_obj = subset.text_patterns(merged_obj, subsetLevels, subsetPatterns)
        merged_obj = subset_obj

    ## Groupby taxa
    taxa_obj = hfunc.groupbytaxa(merged_obj, levels=levels, nameType=nameType)
    ra = taxa_obj['ra']
    table = ra.copy()

    # Subset for top taxa
    if method == 'max_sample':
        ra['max'] = ra.max(axis=1)
        ra = ra.sort_values(by=['max'], ascending=False)
        retain = ra.index.tolist()[:numberToPlot]

    elif method == 'mean_all':
        ra['mean'] = ra.mean(axis=1)
        ra = ra.sort_values(by=['mean'], ascending=False)
        retain = ra.index.tolist()[:numberToPlot]

    table = table.loc[retain]

    if order != 'None':
        table = table.loc[:, logiclist]

    # Change to italics in table labels
    taxa_list = table.index.tolist()
    new_taxa_list = []
    for n in taxa_list:
        if ';' in n and '__' in n: #Check if there are two taxa names
            splitname = n.split(';')
            splitname1 = splitname[0].split('__')
            newname1 = splitname1[0]+'__'+'$\it{'+splitname1[1].replace('_',' ')+'}$'
            if '__' in splitname[1]:
                splitname2 = splitname[1].split('__')
                newname2 = splitname2[0]+'__'+'$\it{'+splitname2[1].replace('_',' ')+'}$'
            else:
                newname2 = splitname[1].replace('_',' ')
            newname = newname1+'; '+newname2
        elif ';' in n and '__' not in n:
            splitname = n.split(';')
            newname = splitname[0].replace('_', ' ')
        else: #If there is only one taxa name
            if '__' in n:
                splitname = n.split('__')
                newname = splitname[0]+'__'+'$\it{'+splitname[1].replace('_', ' ')+'}$'
            else:
                newname = n.replace('_', ' ')
        new_taxa_list.append(newname)
    table = pd.DataFrame(table.to_numpy(), index=new_taxa_list, columns=table.columns)
    
    # Print heatmap
    table['avg'] = table.mean(axis=1)
    table = table.sort_values(by=['avg'], ascending=True)
    table = table.drop(['avg'], axis=1)

    if levelsShown == 'Number':
        numbered_list = []
        for i in range(len(table.index), 0, -1):
            numbered_list.append(i)
        table = pd.DataFrame(table.to_numpy(), index=numbered_list, columns=table.columns)

    #Fix datalabels
    if labels:
        labelvalues = table.copy()
        for r in table.index:
            for c in table.columns:
                value = float(table.loc[r, c])
                if value < 0.1 and value > 0:
                    labelvalues.loc[r, c] = '<0.1'
                elif value < 10 and value >= 0.1:
                    labelvalues.loc[r, c] = str(round(value, 1))
                elif value > 99:
                    labelvalues.loc[r, c] = '99'
                elif value >= 10:
                    labelvalues.loc[r, c] = str(int(round(value, 0)))
                else:
                    labelvalues.loc[r, c] = '0'

    # Include empty columns in table to separate samples
    #print(table.head())
    if len(sepCol) > 0:
        for i in range(len(sepCol)):
            table.insert(loc=sepCol[i]+i, column=' '*(i+1), value=0)
            if labels:
                labelvalues.insert(loc=sepCol[i]+i, column=' '*(i+1), value='')

    #Plot
    plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(table, cmap=cMap, norm=mcolors.PowerNorm(gamma=cLinear), aspect='auto')
    if len(cBar) > 0:
        fig.colorbar(im, ticks=cBar)

    # Fix axes
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_xticklabels(table.columns.tolist(), rotation=90)
    ax.set_yticklabels(table.index.tolist(), rotation=0)

    # Fix grid lines
    ax.set_xticks(np.arange(-0.5, len(table.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(table.index), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

    if len(sepCol) > 0:
        for i in range(len(sepCol)):
            for j in range(6):
                ax.axvline(sepCol[i]+i-0.5+j/5, 0, len(table.index), linestyle='-', lw=1, color='white')

    # Fix labels
    if labels:
        for r in range(len(table.index)):
            for c in range(len(table.columns)):
                if table.iloc[r, c] > cThreshold:
                    textcolor = 'white'
                else:
                    textcolor = 'black'
                ax.text(c, r, labelvalues.iloc[r, c], fontsize=labelsize, ha='center', va='center', color=textcolor)
    fig.tight_layout()
    if savename != 'None':
        plt.savefig(savename+'.pdf', format='pdf')
        plt.savefig(savename)

# Visualizes how alpha diversity depends on diversity order
# If distmat is specified phylogenetic alpha diversity is calculated, else naive
# var refers to column heading in meta data used to color code samples
# slist is a list of samples from the var column to include (default is all)
# order refers to column heading in meta data used to order the samples
# If ylog=True, the y-axis of the plot will be logarithmic
def alpha_diversity(obj, distmat='None', divType='naive', var='None', slist='None', order='None', ylog=False, 
                    figsize=(10, 6), fontsize=18, colorlist='None', savename='None'):
    #Pick out samples to include based on var and slist
    meta = obj['meta']
    if order != 'None':
        meta = meta.sort_values(order)

    if var == 'None':
        smplist = meta.index.tolist()
    elif slist == 'None':
        smplist = meta[var].index.tolist()
    else:
        smplist = meta[meta[var].isin(slist)].index.tolist()

    #Dataframe for holding results
    xvalues = np.arange(0, 2.01, 0.05)
    df = pd.DataFrame(0, index=xvalues, columns=smplist)

    #Put data in dataframe
    tab = obj['tab'][smplist]
    for x in xvalues:
        if divType == 'naive':
            alphadiv = diversity.naive_alpha(tab, q=x)
        elif divType == 'phyl':
            alphadiv = diversity.phyl_alpha(tab, tree=obj['tree'], q=x)
        elif divType == 'func' and isinstance(distmat, pd.DataFrame):
            alphadiv = diversity.func_alpha(tab, distmat=distmat, q=x)
        else:
            print('Check divType and required input')
            return None
        df.loc[x, smplist] = alphadiv

    #Plot data
    plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(figsize=figsize)

    if colorlist == 'None':
        colorlist = hfunc.get_colors_markers('colors')
    else:
        colorlist = colorlist
    colorcheck = []

    for s in df.columns:
        if var != 'None':
            cv = meta.loc[s, var]
        else:
            cv = s

        if cv not in colorcheck:
            colorcheck.append(cv)
            lab = cv
        else:
            lab = '_nolegend_'

        colnr = colorcheck.index(cv)
        col = colorlist[colnr % len(colorlist)]

        if ylog:
            ax.semilogy(df.index, df[s], lw=1, color=col, label=lab)
        else:
            ax.plot(df.index, df[s], lw=1, color=col, label=lab)

    if divType == 'naive':
        ax.set_ylabel('Diversity number ($^q$D)')
    elif divType == 'phyl':
        ax.set_ylabel('Diversity number ($^q$PD)')
    elif divType == 'func':
        ax.set_ylabel('Diversity number ($^q$FD)')

    ax.set_xlabel('Diversity order (q)')
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlim(0, 2)

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    if savename != 'None':
        plt.savefig(savename+'.pdf', format='pdf')
        plt.savefig(savename)
    plt.show()

# Visualizes dissimilarities in PCoA plot
# dist is distance matrix and meta is meta data
# biplot is list with meta data column headings used in biplot, the columns must contain numeric data
# var1 is heading in meta used to color code, var2 is heading in meta used to code by marker type
# var1_title and var_2 title are the titles used in the legend
# whitePad sets the space between the outermost points and the plot limits (1.0=no space)
# rightSpace is the space for the legend on the right
# var2pos is the vertical position of the var2 legend
# tag is heading in meta used to add labels to each point in figure
# order is heading in meta used to order samples
# title is title of the entire figure
# if connectPoints is a metadata column header, it will use that data to connect the points
# colorlist specifies colorlist to use for var1; same for markerlist and var2
# savename is path and name to save png figure output
def pcoa(dist, meta, var1='None', var2='None', var1_title='', var2_title='', biplot=[], 
             whitePad=1.1, var2pos=0.4, tag='None', order='None', title='', connectPoints='None',
             figsize=(10, 14), fontsize=18, markersize=100, markerscale=1.1,
             hideAxisValues=False, showLegend=True,
             colorlist='None', markerlist='None', savename='None'):
    def get_eig(d): #Function for centering and eigen-decomposition of distance matrix
        dist2 = -0.5*(d**2)
        col_mean = dist2.mean(axis=0)
        row_mean = dist2.mean(axis=1)
        tot_mean = np.array(dist2).flatten().mean()
        dist2_cent = dist2.subtract(col_mean, axis=1)
        dist2_cent = dist2_cent.subtract(row_mean, axis=0)
        dist2_cent = dist2_cent.add(tot_mean)
        vals, vects = np.linalg.eig(dist2_cent)
        return [vals, vects]

    dist.fillna(0, inplace=True)
    ev_ev = get_eig(dist)
    #Correction method for negative eigenvalues
    if min(ev_ev[0]) < 0: #From Legendre 1998, method 1 (derived from Lingoes 1971) chapter 9, page 502
        d2 = (dist[dist > 0].pow(2) + 2*abs(min(ev_ev[0])))**0.5
        d2 = d2.fillna(0)
        ev_ev = get_eig(d2)

    #Get proportions and coordinates
    vals = ev_ev[0].copy()
    vects = ev_ev[1]
    prop = [] #Fraction proportion of eigvalues for axis labels
    coords = [] #Two arrays with coordinates
    U_vectors = [] #Main vectors used for biplot
    Eig_vals = [] #Main vectors used for biplot
    for i in range(2):
        maxpos = np.argmax(vals)
        prop.append(vals[maxpos]/sum(ev_ev[0]))
        coords.append((vals[maxpos]**0.5)*vects[:, maxpos])
        Eig_vals.append(vals[maxpos])
        vals[maxpos] = 0
    #Check if prop is imaginary
    for i in range(2):
        if '+' in str(prop[i]):
            realpart = str(prop[i]).split('+')[0]
            prop[i] = float(realpart[1:])

    xaxislims = [min(coords[0])*whitePad, max(coords[0])*whitePad]
    yaxislims = [min(coords[1])*whitePad, max(coords[1])*whitePad]

    #Check biplot
    if len(biplot) > 0:
        #Standardize U (eigenvectors)
        U_vectors = pd.DataFrame(coords, columns=dist.columns).transpose()
        U_vectors[0] = (U_vectors[0]-U_vectors[0].mean())/U_vectors[0].std()
        U_vectors[1] = (U_vectors[1]-U_vectors[1].mean())/U_vectors[1].std()

        #Standardize Y
        Y = pd.DataFrame(index=dist.columns)
        for mh in biplot:
            Y[mh] = meta[mh]
            Y[mh] = (Y[mh]-Y[mh].mean())/Y[mh].std()
        Y_cent = Y.transpose()
        Spc =(1/(len(dist.columns)-1))*np.matmul(Y_cent, U_vectors)
        biglambda = np.array([[Eig_vals[0]**-0.5, 0], [0, Eig_vals[1]**-0.5]])
        Uproj = ((len(dist.columns)-1)**0.5)*np.matmul(Spc, biglambda)

        #Scale to the plot
        Uscalefactors = []
        Uscalefactors.append(max(coords[0])/max(Uproj[:, 0]))
        Uscalefactors.append(min(coords[0])/min(Uproj[:, 0]))
        Uscalefactors.append(max(coords[1])/max(Uproj[:, 1]))
        Uscalefactors.append(min(coords[1])/min(Uproj[:, 1]))
        Uscale = 1
        for i in Uscalefactors:
            if i < Uscale and i > 0:
                Uscale = i
        Uproj = Uproj*Uscale

    # Do the plotting

    #Set axis names and make dataframe for plotting
    pc1_perc = round(100 * prop[0], 1)
    xn = 'PC1 (' + str(pc1_perc) + '%)'
    if '+' in xn:
        xn = xn[:4] + xn[-1]
    pc2_perc = round(100 * prop[1], 1)
    yn = 'PC2 (' + str(pc2_perc) + '%)'
    if '+' in yn:
        yn = yn[:4] + yn[-1]

    smplist = dist.index
    pcoadf = pd.DataFrame({xn: coords[0], yn: coords[1]}, index=smplist)

    # Combine pcoa results with meta data
    meta = meta.copy()
    meta[xn] = pcoadf[xn]
    meta[yn] = pcoadf[yn]
    metaPlot = meta[meta[xn].notnull()]
    if order != 'None':
        meta = meta.sort_values(by=[order])

    if var1 == 'None':
        print('Error, var1 is not specified')
        return None
    if var1 != 'None': #List of names used for different colors in legend
        smpcats1 = []
        [smpcats1.append(item) for item in meta[var1] if item not in smpcats1]
    if var2 != 'None': #List of names used for different marker types in legend
        smpcats2 = []
        [smpcats2.append(item) for item in meta[var2] if item not in smpcats2]
    if tag != 'None': #List of labels placed next to points
        tagcats = []
        [tagcats.append(item) for item in meta[tag] if item not in tagcats]

    # Create figure
    if colorlist == 'None':
        colorlist = hfunc.get_colors_markers('colors')
    if markerlist == 'None':
        markerlist = hfunc.get_colors_markers('markers')

    plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(figsize=figsize)

    linesColor = [[], []]
    linesShape = [[], []]
    shapeTracker = []

    for i in range(len(smpcats1)):
        metaPlot_i = metaPlot[metaPlot[var1] == smpcats1[i]] #Subset metaPlot based on var1 in smpcats1

        if connectPoints != 'None':
            metaPlot_i[connectPoints] = metaPlot_i[connectPoints].apply(int)
            metaPlot_i = metaPlot_i.sort_values(connectPoints)
            ax.plot(metaPlot_i[xn], metaPlot_i[yn], color=colorlist[i])

        if var2 != 'None':
            linesColor[0].append(ax.scatter([], [], label=str(smpcats1[i]), color=colorlist[i]))
            linesColor[1].append(smpcats1[i])

            jcounter = 0
            for j in range(len(smpcats2)):
                if smpcats2[j] in list(metaPlot_i[var2]):
                    metaPlot_ij = metaPlot_i[metaPlot_i[var2] == smpcats2[j]]
                    xlist = metaPlot_ij[xn]
                    ylist = metaPlot_ij[yn]
                    ax.scatter(xlist, ylist, label=None, color=colorlist[i], marker=markerlist[jcounter], s=markersize)

                    if jcounter not in shapeTracker:
                        linesShape[0].append(ax.scatter([], [], label=str(smpcats2[j]), color='black', marker=markerlist[jcounter]))
                        linesShape[1].append(smpcats2[jcounter])
                        shapeTracker.append(jcounter)
                jcounter += 1

            # Here set both legends for color and marker
            if showLegend:
                ax.legend(linesColor[0], linesColor[1], ncol=1, bbox_to_anchor=(1, 1), title=var1_title, frameon=False, markerscale=markerscale, fontsize=fontsize, loc=2)
                from matplotlib.legend import Legend
                leg = Legend(ax, linesShape[0], linesShape[1], ncol=1, bbox_to_anchor=(1, var2pos), title=var2_title, frameon=False, markerscale=markerscale, fontsize=fontsize, loc=2)
                ax.add_artist(leg)

        else: #If there is no var2, change both color and marker with each category in var1
            linesColor[0].append(ax.scatter([], [], label=str(smpcats1[i]), color=colorlist[i], marker=markerlist[i]))
            linesColor[1].append(smpcats1[i])

            xlist = metaPlot_i[xn]
            ylist = metaPlot_i[yn]
            ax.scatter(xlist, ylist, label=None, color=colorlist[i], marker=markerlist[i], s=markersize)
            if showLegend:
                ax.legend(linesColor[0], linesColor[1], bbox_to_anchor=(1, 1), title=var1_title, frameon=False, markerscale=markerscale, fontsize=fontsize, loc='upper left')

    ##Put tags at each point
    if tag != 'None':
        for ix in metaPlot.index.tolist():
            tagtext = metaPlot.loc[ix, tag]
            tagx = metaPlot.loc[ix, xn]
            tagy = metaPlot.loc[ix, yn]
            ax.annotate(tagtext, (tagx, tagy))

    ##Input arrows
    if len(biplot) > 0:
        for mh_nr in range(len(biplot)):
            ax.arrow(0, 0, Uproj[mh_nr, 0], Uproj[mh_nr, 1])
            ax.annotate(biplot[mh_nr], (Uproj[mh_nr, 0], Uproj[mh_nr, 1]))
        ax.axhline(0, 0, 1, linestyle='--', color='grey', lw=0.5)
        ax.axvline(0, 0, 1, linestyle='--', color='grey', lw=0.5)

    ax.set_xlabel(xn)
    ax.set_ylabel(yn)
    ax.set_xlim(xaxislims)
    ax.set_ylim(yaxislims)
    if hideAxisValues:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.title(title)
    plt.tight_layout()
    if savename != 'None':
        plt.savefig(savename+'.pdf', format='pdf')
        plt.savefig(savename)
    plt.show()

# Plots dissimilarity between pairs of samples types
def pairwise_beta(obj, divType='naive', distmat='None', compareVar='None', spairs=[],
                nullModel=True, randomization='frequency', weight=0, iterations=10,
                qrange=[0, 2, 0.5], colorlist='None',
                    onlyPlotData='None', skipJB=False, onlyReturnData=False,
                    savename='None'):

    ##Function for plotting data
    def plot_pairwiseDB(dd, colorlist=colorlist, savename=savename):
        if colorlist == 'None':
            colorlist = hfunc.get_colors_markers()

        df = dd['Hill']
        dn = dd['Hill_Null']
        dr = dd['Hill_RC']
        for c_nr in range(len(df.columns)):
            plt.rcParams.update({'font.size': 10})
            fig, ax = plt.subplots(figsize=(11/2.54, 5/2.54))

            c = df.columns[c_nr]

            #Plot Hill lines
            ylow = df[c] - dd['Hillstd'][c]
            yhigh = df[c] + dd['Hillstd'][c]
            ax.fill_between(df.index.tolist(), ylow.tolist(), yhigh.tolist(), alpha=0.3, color=colorlist[0])
            ax.plot(df.index, df[c], lw=2, label='Obs. Hill', color=colorlist[0])

            #Plot Null Hill lines
            ylow = dn[c] - dd['Hill_Nullstd'][c]
            yhigh = dn[c] + dd['Hill_Nullstd'][c]
            ax.fill_between(dn.index.tolist(), ylow.tolist(), yhigh.tolist(), alpha=0.3, color=colorlist[1])
            ax.plot(dn.index, dn[c], lw=2, label='Null Hill', linestyle='--', color=colorlist[1])

            #Plot RC Hill lines
            ylow = dr[c] - dd['Hill_RCstd'][c]
            yhigh = dr[c] + dd['Hill_RCstd'][c]
            ax.fill_between(dr.index.tolist(), ylow.tolist(), yhigh.tolist(), alpha=0.3, color=colorlist[2])
            ax.plot(dr.index, dr[c], lw=2, label='RC Hill', linestyle='--', color=colorlist[2])

            if not skipJB:
                #Plot Jaccard
                ax.scatter([0], [dd['BJ'].loc['Jac', c]], label='Obs. Jaccard', marker='s', s=50, color=colorlist[0])
                ax.errorbar([0], [dd['BJ'].loc['Jac', c]], yerr=[dd['BJstd'].loc['Jac', c]], color=colorlist[0])
                ax.scatter([0.01], [dd['BJ_Null'].loc['Jac', c]], label='Null Jaccard', marker='s', s=50, color=colorlist[1])
                ax.errorbar([0.01], [dd['BJ_Null'].loc['Jac', c]], yerr=[dd['BJ_Nullstd'].loc['Jac', c]], color=colorlist[1])
                ax.scatter([0.02], [dd['BJ_RC'].loc['Jac', c]], label='RC Jaccard', marker='s', s=50, color=colorlist[2])
                ax.errorbar([0.02], [dd['BJ_RC'].loc['Jac', c]], yerr=[dd['BJ_RCstd'].loc['Jac', c]], color=colorlist[2])

                #Plot Bray
                ax.scatter([1], [dd['BJ'].loc['Bray', c]], label='Obs. Bray', marker='o', s=50, color=colorlist[0])
                ax.errorbar([1], [dd['BJ'].loc['Bray', c]], yerr=[dd['BJstd'].loc['Bray', c]], color=colorlist[0])
                ax.scatter([0.99], [dd['BJ_Null'].loc['Bray', c]], label='Null Bray', marker='o', s=50, color=colorlist[1])
                ax.errorbar([0.99], [dd['BJ_Null'].loc['Bray', c]], yerr=[dd['BJ_Nullstd'].loc['Bray', c]], color=colorlist[1])
                ax.scatter([1.01], [dd['BJ_RC'].loc['Bray', c]], label='RC Bray', marker='o', s=50, color=colorlist[2])
                ax.errorbar([1.01], [dd['BJ_RC'].loc['Bray', c]], yerr=[dd['BJ_RCstd'].loc['Bray', c]], color=colorlist[2])

            ax.set_ylabel('Dissimilarity ($^q$d)')
            ax.set_xlabel('Diversity order (q)')
            ax.set_xticks(np.arange(qrange[0], qrange[1] + 0.01, qrange[2]))
            ax.set_xlim(qrange[0] - 0.05, qrange[1] + 0.05)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0, 0, 1, color='grey', alpha=0.5, lw=0.5)
            ax.axhline(1, 0, 1, color='grey', alpha=0.5, lw=0.5)
            plt.legend(bbox_to_anchor=(1, 1), fontsize=8, loc='upper left', frameon=False)
            plt.tight_layout()
            if savename != 'None':
                #plt.savefig(savename + c + '.pdf', format='pdf')
                plt.savefig(savename + c)
            plt.show()

    def plot_withoutNull(dd, colorlist=colorlist, savename=savename):
        if colorlist == 'None':
            colorlist = hfunc.get_colors_markers()

        plt.rcParams.update({'font.size': 10})
        fig, ax = plt.subplots(figsize=(10/2.54, 5/2.54))

        df = dd['Hill']
        for c_nr in range(len(df.columns)):

            c = df.columns[c_nr]

            #Plot Hill lines
            ylow = df[c] - dd['Hillstd'][c]
            yhigh = df[c] + dd['Hillstd'][c]
            ax.fill_between(df.index.tolist(), ylow.tolist(), yhigh.tolist(), alpha=0.3, color=colorlist[c_nr])
            ax.plot(df.index, df[c], lw=1, label=c, color=colorlist[c_nr])

            if not skipJB:
                #Plot Jaccard
                ax.scatter([0], [dd['BJ'].loc['Jac', c]], label='_nolegend_', marker='s', s=25, color=colorlist[c_nr])
                ax.errorbar([0], [dd['BJ'].loc['Jac', c]], yerr=[dd['BJstd'].loc['Jac', c]], color=colorlist[c_nr])

                #Plot Bray
                ax.scatter([1], [dd['BJ'].loc['Bray', c]], label='_nolegend_', marker='o', s=25, color=colorlist[c_nr])
                ax.errorbar([1], [dd['BJ'].loc['Bray', c]], yerr=[dd['BJstd'].loc['Bray', c]], color=colorlist[c_nr])

        if not skipJB:
            ax.scatter([], [], label='Jaccard', marker='s', s=25, color='black')
            ax.scatter([], [], label='Bray', marker='o', s=25, color='black')

        ax.set_ylabel('Dissimilarity ($^q$d)')
        ax.set_xlabel('Diversity order (q)')
        ax.set_xticks(np.arange(qrange[0], qrange[1] + 0.01, qrange[2]))
        ax.set_xlim(qrange[0] - 0.05, qrange[1] + 0.05)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, 0, 1, color='grey', alpha=0.5, lw=0.5)
        ax.axhline(1, 0, 1, color='grey', alpha=0.5, lw=0.5)
        plt.legend(bbox_to_anchor=(1, 1), fontsize=10, loc='upper left', frameon=False)
        plt.tight_layout()
        if savename != 'None':
            #plt.savefig(savename + 'Nonull' + '.pdf', format='pdf')
            plt.savefig(savename + 'Nonull')
        plt.show()

    ##############
    if onlyPlotData != 'None':
        if nullModel:
            plot_pairwiseDB(onlyPlotData, colorlist, savename)
            plot_withoutNull(onlyPlotData, colorlist, savename)
        else:
            plot_withoutNull(onlyPlotData, colorlist, savename)

    else:
        meta = obj['meta']

        # Make dataframe for holding results
        qvalues = np.arange(qrange[0], qrange[1] + 0.01, 0.1)
        qvaluesnew = []
        for v in qvalues:
            qvaluesnew.append(round(v, 2))
        qvalues = qvaluesnew

        pairnames = []
        for pair in spairs:
            pairnames.append(pair[0] + '-' + pair[1])
        df_Hill = pd.DataFrame(0, index=qvalues, columns=pairnames)
        df_Hillstd = pd.DataFrame(0, index=qvalues, columns=pairnames)
        df_Hill_Null = pd.DataFrame(0, index=qvalues, columns=pairnames)
        df_Hill_Nullstd = pd.DataFrame(0, index=qvalues, columns=pairnames)
        df_Hill_RC = pd.DataFrame(0, index=qvalues, columns=pairnames)
        df_Hill_RCstd = pd.DataFrame(0, index=qvalues, columns=pairnames)
        df_BJ = pd.DataFrame(0, index=['Bray', 'Jac'], columns=pairnames)
        df_BJstd = pd.DataFrame(0, index=['Bray', 'Jac'], columns=pairnames)
        df_BJ_Null = pd.DataFrame(0, index=['Bray', 'Jac'], columns=pairnames)
        df_BJ_Nullstd = pd.DataFrame(0, index=['Bray', 'Jac'], columns=pairnames)
        df_BJ_RC = pd.DataFrame(0, index=['Bray', 'Jac'], columns=pairnames)
        df_BJ_RCstd = pd.DataFrame(0, index=['Bray', 'Jac'], columns=pairnames)

        #Iterate through each pair
        for pair_nr in range(len(pairnames)):

            pair0 = spairs[pair_nr][0]
            pair1 = spairs[pair_nr][1]
            obj2 = subset.samples(obj, var=compareVar, slist=spairs[pair_nr]) #Subset obj to pair
            tab = obj2['tab'].copy()

            #Calculate Bray-Curtis and Jaccard
            print('Processing ' + pairnames[pair_nr] + ' Jaccard & Bray')

            betabray = diversity.bray(tab)
            betajac = diversity.jaccard(tab)
            if compareVar != 'None':
                templistB = []
                templistJ = []
                slist0 = meta[meta[compareVar] == pair0].index.tolist()
                slist1 = meta[meta[compareVar] == pair1].index.tolist()
                for s0 in slist0:
                    for s1 in slist1:
                        templistB.append(betabray.loc[s0, s1])
                        templistJ.append(betajac.loc[s0, s1])
                df_BJ.loc['Bray', pairnames[pair_nr]] = np.mean(templistB)
                df_BJstd.loc['Bray', pairnames[pair_nr]] = np.std(templistB)
                df_BJ.loc['Jac', pairnames[pair_nr]] = np.mean(templistJ)
                df_BJstd.loc['Jac', pairnames[pair_nr]] = np.std(templistJ)
            else:
                df_BJ.loc['Bray', pairnames[pair_nr]] = betabray.loc[pair0, pair1]
                df_BJ.loc['Jac', pairnames[pair_nr]] = betajac.loc[pair0, pair1]

            if nullModel:
                braynull = null.rcq(obj2, randomization=randomization, weightingVar=compareVar, weight=weight,
                                    compareVar=compareVar, iterations=iterations, divType='Bray')
                jacnull = null.rcq(obj2, randomization=randomization, weightingVar=compareVar, weight=weight,
                                   compareVar=compareVar, iterations=iterations, divType='Jaccard')

                df_BJ_Null.loc['Bray', pairnames[pair_nr]] = braynull['null_mean'].loc[pair0, pair1]
                df_BJ_Nullstd.loc['Bray', pairnames[pair_nr]] = braynull['null_std'].loc[pair0, pair1]
                df_BJ_Null.loc['Jac', pairnames[pair_nr]] = jacnull['null_mean'].loc[pair0, pair1]
                df_BJ_Nullstd.loc['Jac', pairnames[pair_nr]] = jacnull['null_std'].loc[pair0, pair1]
                if compareVar == 'None':
                    df_BJ_RC.loc['Bray', pairnames[pair_nr]] = braynull['p_index'].loc[pair0, pair1]
                    df_BJ_RC.loc['Jac', pairnames[pair_nr]] = jacnull['p_index'].loc[pair0, pair1]
                else:
                    df_BJ_RC.loc['Bray', pairnames[pair_nr]] = braynull['p_index_mean'].loc[pair0, pair1]
                    df_BJ_RCstd.loc['Bray', pairnames[pair_nr]] = braynull['p_index_std'].loc[pair0, pair1]
                    df_BJ_RC.loc['Jac', pairnames[pair_nr]] = jacnull['p_index_mean'].loc[pair0, pair1]
                    df_BJ_RCstd.loc['Jac', pairnames[pair_nr]] = jacnull['p_index_std'].loc[pair0, pair1]

            #Calculate Hill, Iterate for different diversity orders, q
            for q in qvalues:
                print('Processing ' + pairnames[pair_nr] + ' q=' + str(q))

                #Calculate dis for real tab
                if divType == 'naive':
                    betadiv = diversity.naive_beta(tab, q=q)
                elif divType == 'phyl' and 'tree' in obj:
                    betadiv = diversity.phyl_beta(tab, tree=obj['tree'], q=q)
                elif divType == 'func' and isinstance(distmat, pd.DataFrame):
                    betadiv = diversity.func_beta(tab, distmat=distmat, q=q)
                else:
                    print('Check divType and required input')
                    return None

                if compareVar != 'None':
                    templistH = []
                    slist0 = meta[meta[compareVar] == spairs[pair_nr][0]].index.tolist()
                    slist1 = meta[meta[compareVar] == spairs[pair_nr][1]].index.tolist()
                    for s0 in slist0:
                        for s1 in slist1:
                            templistH.append(betadiv.loc[s0, s1])
                    df_Hill.loc[q, pairnames[pair_nr]] = np.mean(templistH)
                    df_Hillstd.loc[q, pairnames[pair_nr]] = np.std(templistH)
                else:
                    df_Hill.loc[q, pairnames[pair_nr]] = betadiv.loc[pair0, pair1]

                if nullModel:
                    Hillnull = null.rcq(obj2, distmat=distmat, randomization=randomization, weightingVar=compareVar, weight=weight, q=q,
                                        compareVar=compareVar, divType=divType, iterations=iterations)

                    df_Hill_Null.loc[q, pairnames[pair_nr]] = Hillnull['null_mean'].loc[pair0, pair1]
                    df_Hill_Nullstd.loc[q, pairnames[pair_nr]] = Hillnull['null_std'].loc[pair0, pair1]
                    if compareVar == 'None':
                        df_Hill_RC.loc[q, pairnames[pair_nr]] = Hillnull['p_index'].loc[pair0, pair1]
                    else:
                        df_Hill_RC.loc[q, pairnames[pair_nr]] = Hillnull['p_index_mean'].loc[pair0, pair1]
                        df_Hill_RCstd.loc[q, pairnames[pair_nr]] = Hillnull['p_index_std'].loc[pair0, pair1]

        # Make dictionary holding all data frames
        output = {}
        output['Settings'] = {'randomization': randomization, 'weight': weight, 'iterations': iterations, 'nullModel': nullModel}
        output['Hill'] = df_Hill
        output['Hillstd'] = df_Hillstd
        output['Hill_Null'] = df_Hill_Null
        output['Hill_Nullstd'] = df_Hill_Nullstd
        output['Hill_RC'] = df_Hill_RC
        output['Hill_RCstd'] = df_Hill_RCstd
        output['BJ'] = df_BJ
        output['BJstd'] = df_BJstd
        output['BJ_Null'] = df_BJ_Null
        output['BJ_Nullstd'] = df_BJ_Nullstd
        output['BJ_RC'] = df_BJ_RC
        output['BJ_RCstd'] = df_BJ_RCstd

        if savename != 'None':
            with open(savename+'.pickle', 'wb') as f:
                pickle.dump(output, f)
        if onlyReturnData:
            return output
        elif nullModel:
            plot_pairwiseDB(output, colorlist, savename)
            plot_withoutNull(output, colorlist, savename)
        else:
            plot_withoutNull(output, colorlist, savename)

# Calculate and plot rarefaction curve
# step is the step size used during subsampling, if 'flexible' the total reads are divided by 20
# figsize is with ahd height of figure in inches
# fontsize is size of text in figure
# var is column in meta data used to color code lines in plot
# order is column in meta data used to order sample
# tag is column in meta data used to name lines in plot, if 'index', sample names are used
# colorlist is colors to be used in plot, if 'None' qdiv default is used
# if onlyReturnData=True, function will return a dictionary with data
# if onlyPlotData is a dictionary with data, it will be plotted and no calculations will be carried out
# is savename is specified, plots will be saved and data will be saved as pickle file
def rarefactioncurve(obj, step='flexible', divType='naive', q=0, distmat='None',
                     figsize=(14, 10), fontsize=18, 
                     var='None', order='None', tag='None', colorlist='None',
                     onlyReturnData=False, onlyPlotData='None', savename='None'):
    # Function for plotting
    def plot_rarefactioncurve(meta, rd):
        if order != 'None':
            meta = meta.sort_values(by=[order])
        nonlocal colorlist
        if colorlist == 'None':
            colorlist = hfunc.get_colors_markers('colors')
        if var != 'None': #List of names used for different colors in legend
            smpcats = []
            [smpcats.append(item) for item in meta[var] if item not in smpcats]

        # Create figure
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=figsize)

        # If special colorcoding (var) plot like this
        if var != 'None':
            for cat_nr in range(len(smpcats)):
                cat = smpcats[cat_nr]
                ax.plot([], [], label=cat, color=colorlist[cat_nr])
                smplist = meta[meta[var] == cat].index.tolist()
                for smp in smplist:
                    ax.plot(rd[smp][0], rd[smp][1], label='_nolegend_', color=colorlist[cat_nr])
        else: #Else plot like this
            for smp_nr in range(len(meta.index)):
                smp = meta.index.tolist()[smp_nr]
                ax.plot(rd[smp][0], rd[smp][1], label='_nolegend_', color=colorlist[smp_nr])
        
        #Adding tags if not 'None'
        if tag == 'index':
            for smp in rd.keys():
                ax.annotate(smp, (rd[smp][0][-1], rd[smp][1][-1]), color='black')
        elif tag != 'None':
            for smp in rd.keys():
                antext = meta.loc[smp, tag]
                ax.annotate(antext, (rd[smp][0][-1], rd[smp][1][-1]), color='black')
        
        #Legend added if we have color coding
        if var != 'None':
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
        ax.set_xlabel('Reads')
        ax.set_ylabel('$^{'+str(q)+'}$D')
        plt.tight_layout()
        if savename != 'None':
            plt.savefig(savename)
            plt.savefig(savename + '.pdf', format='pdf')
        plt.show()

    # Getting the data
    if onlyPlotData == 'None':
        tab = obj['tab']
        res_di = {} #Dictionary holding x and y for each samples   
        print('Working on rarefaction curve for sample: ', end='')
        for smp in tab.columns: #Start going through the samples
            print(smp, end='.. ')
            smp_series = tab[smp][tab[smp] > 0]
            totalreads = smp_series.sum()
            
            #Make shuffled list of sv names
            name_arr = smp_series.index.tolist()
            counts_arr = smp_series.to_numpy()
            cumreads2 = np.cumsum(counts_arr)
            cumreads1 = cumreads2 - counts_arr
            ind_reads_arr = np.empty(totalreads, dtype=object)
            for i, (v1, v2) in enumerate(zip(cumreads1, cumreads2)):
                ind_reads_arr[v1:v2] = name_arr[i]
            np.random.shuffle(ind_reads_arr) #Shuffle the SVs
    
            #Make x- and y values for rarefaction curve    
            if step == 'flexible':
                step = int(totalreads/20)
            xvals = np.arange(step, totalreads, step)
            yvals = np.zeros(len(xvals))
            for i, depth in enumerate(xvals): #Go through each step
                bin_counts = np.unique(ind_reads_arr[:depth], return_counts=True)
                temp_tab = pd.DataFrame(bin_counts[1], index=bin_counts[0], columns=[smp])
                if divType == 'naive':
                    div_val = diversity.naive_alpha(temp_tab, q=q)
                elif divType == 'phyl':
                    div_val = diversity.phyl_alpha(temp_tab, obj['tree'], q=q)
                elif divType == 'func':
                    div_val = diversity.func_alpha(temp_tab, distmat, q=q)
                yvals[i] = div_val[smp]
            #Add true value to the end and 0 to beginning
            if divType == 'naive':
                div_val = diversity.naive_alpha(tab[smp], q=q)
            elif divType == 'phyl':
                div_val = diversity.phyl_alpha(tab[smp], obj['tree'], q=q)
            elif divType == 'func':
                div_val = diversity.func_alpha(tab[smp], distmat, q=q)
            xvals = np.append(xvals, totalreads)
            yvals = np.append(yvals, div_val[smp])
            xvals = np.insert(xvals, 0, [0, 1])        
            yvals = np.insert(yvals, 0, [0, 1])        
            res_di[smp] = [xvals, yvals]
        print('Done') #Done going through the samples

        if savename != 'None':
            with open(savename + '.pickle', 'wb') as f:
                pickle.dump(res_di, f)

    elif onlyPlotData != 'None':
        plot_rarefactioncurve(obj['meta'], onlyPlotData)

    if not onlyReturnData:
        plot_rarefactioncurve(obj['meta'], res_di)
    else:
        return res_di

# Octave plot according to Edgar and Flyvbjerg, DOI: https://doi.org/10.1101/38983
# var is the column heading in metadata used to select samples to include. The counts for all samples with the same text in var column will be merged.
# slist is a list of names in meta data column which specify samples to keep. If slist='None' (default), the whole meta data column is used
# nrows and ncols are the number of rows and columns in the plot. 
# nrows*ncols must be equal to or more than the number of samples plotted
# if xlabels=True, k is shown for the bins on the x-axis
# if ylabels=True, ASV counts are shown on the y-axis
# if title=True, sample name is shown as title for each panel
# color determines color of bars
# savename is path and name of file
def octave(obj, var='None', slist='None', nrows=2, ncols=2, fontsize=11, figsize=(10, 6), 
           xlabels=True, ylabels=True, title=True, color='blue', savename='None'):
    if var == 'None':
        tab = obj['tab'].copy()
        smplist = tab.columns.tolist()
    else:
        merged_obj = subset.merge_samples(obj, var=var, slist=slist)
        tab = merged_obj['tab'].copy()
        smplist = tab.columns.tolist()

    if len(smplist) > nrows*ncols:
        print('Too few panels, ', len(smplist),' are needed')
        return 0

    max_read = max(tab.max())
    max_k = math.floor(math.log(max_read, 2))
    k_index = np.arange(max_k+1)
    df = pd.DataFrame(0, index=k_index, columns=['k', 'min_count', 'max_count']+smplist)
    df['k'] = k_index
    df['min_count'] = 2**k_index
    df['max_count'] = 2**(k_index+1)-1

    plt.rcParams.update({'font.size': fontsize})
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(nrows, ncols)
    gs.update(wspace=0, hspace=0)

    for smp_nr in range(len(smplist)):
        row = math.floor(smp_nr/ncols)
        col = smp_nr%ncols
        ax = fig.add_subplot(gs[row, col], frame_on=True)
        smp = smplist[smp_nr]
        for k in df.index:
            bin_min = df.loc[k, 'min_count']
            bin_max = df.loc[k, 'max_count']
            temp = tab[smp][tab[smp] >= bin_min]
            temp = temp[temp <= bin_max]
            df.loc[k, smp] = len(temp)
            
        ax.bar(df['k'], df[smp], color=color)
        ax.set_xticks(range(0, len(df.index), 2))
        if xlabels and row == nrows-1:
            ax.set_xticklabels(range(0, len(df.index), 2))
            ax.set_xlabel('k (bin [2$^k$..2$^{k+1}$-1])')
        elif xlabels:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xticklabels([])
        if ylabels and col == 0:
            ax.set_ylabel('ASV count')
        elif ylabels:
            ax.set_ylabel('')
        else:
            ax.set_yticklabels([])
        if title:
            ax.text(0.98*ax.get_xlim()[1], 0.98*ax.get_ylim()[1], str(smp), verticalalignment='top', horizontalalignment='right')

    if savename != 'None':
        plt.savefig(savename)
        plt.savefig(savename+'.pdf', format='pdf')
        df.to_csv(savename + '.csv', index=False)
        
# Plot contribution of taxa to observed naive dissimilarity
# var is the column heading in metadata used to categorize samples. Dissimilarity within each category is calculated.
# q is diversity order and index is local or regional
# numberToPlot is the number of taxa to include
# levels are taxonomic levels to include on y-axis
# fromFile could be that path to a file generated with the output from diversity.naive_dissimilarity_contributions()
# savename is path and name of files to be saved
def dissimilarity_contributions(obj, var='None', q=1, divType='naive', index='local', 
                                numberToPlot=20, levels=['Genus'], fromFile='None',
                                figsize=(18/2.54, 14/2.54), fontsize=10, savename='None'):
    if not isinstance(levels, list):
        print('levels must be a list [], either empty or containing taxonomic levels to include in the plot.')
        return None

    #Get dissimilarity file
    if fromFile == 'None': #Generate the data
        dis_data = diversity.dissimilarity_contributions(obj, var=var, q=q, divType=divType, index=index)
    else: #Read the data from file
        dis_data = pd.read_csv(fromFile, index_col=0)
    
    df = dis_data.drop(['N', 'dis'], axis=0)
    df['avg'] = df.mean(axis=1)
    df = df.sort_values(by='avg', ascending=False).iloc[:numberToPlot]

    #Plot
    catlist = dis_data.columns.tolist()
    ylist = range(len(df.index))
    taxlist = df.index.tolist()          
    if len(levels) > 0 and 'tax' in obj and divType == 'naive': #Fix taxonomy names
        tax = obj['tax']
        tax_df = tax.loc[df.index]
        tax_df.fillna('', inplace=True)
        for i, asv in enumerate(df.index.tolist()):
            taxname = ''
            for taxlevel in levels:
                tn = tax_df.loc[asv, taxlevel]
                if len(tn) > 3:
                    taxname = taxname + tn + '; '
            taxlist[i] = taxname + taxlist[i]

    elif len(levels) > 0 and 'tax' in obj and 'tree' in obj and divType == 'phyl': #Fix taxonomy names
        tree = obj['tree']
        tree_df = tree.loc[df.index]
        tax = obj['tax'].copy()
        tax.fillna('', inplace=True)
        for i, ix in enumerate(tree_df.index):
            asvlist = tree_df.loc[ix, 'ASVs']
            if len(asvlist) == 1:
                taxname = ''
                for taxlevel in levels:
                    tn = tax.loc[asvlist[0], taxlevel]
                    if len(tn) > 3:
                        taxname = taxname + tn + '; '
                taxlist[i] = taxname + asvlist[0]
            elif len(asvlist) == 2:
                taxname = '(' + '-'.join(asvlist) + ')'
                tn = ''
                for taxlevel in tax.columns:
                    col_list = tax.loc[asvlist, taxlevel].to_numpy()
                    if len(col_list[0]) > 3 and (col_list[0] == col_list).all():
                        tn = col_list[0]
                if len(tn) > 3:
                    taxname = tn + '; ' + taxname
                taxlist[i] = taxname
            else:
                taxname = tree_df.loc[ix, 'nodes']
                tn = ''
                for taxlevel in tax.columns:
                    col_list = tax.loc[asvlist, taxlevel].to_numpy()
                    if len(col_list[0]) > 3 and (col_list[0] == col_list).all():
                        tn = col_list[0]
                if len(tn) > 3:
                    taxname = tn + '; ' + taxname
                taxlist[i] = taxname

    elif len(levels) == 0 and divType == 'phyl': #Fix taxonomy names
        tree = obj['tree']
        tree_df = tree.loc[df.index]
        for i, ix in enumerate(tree_df.index):
            taxlist[i] = tree.loc[ix, 'nodes']

    plt.rcParams.update({'font.size': fontsize})
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, len(catlist))
    gs.update(wspace=0, hspace=0)

    for cat_nr in range(len(catlist)):
        cat = catlist[cat_nr]
        ax = fig.add_subplot(gs[0, cat_nr], frame_on=True)
        ax.barh(ylist, df[cat])
        ax.set_yticks(range(0, len(df.index)))
        if cat_nr == 0:
            ax.set_yticklabels(taxlist)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('%')

        if cat == 'all':
            ax.set_title('N=' + str(int(dis_data.loc['N', cat])) + 
                     '\n$^{'+str(q)+'}$d='+str(round(dis_data.loc['dis', cat], 2)))
        else:
            ax.set_title(cat + '\nN=' + str(int(dis_data.loc['N', cat])) + 
                     '\n$^{'+str(q)+'}$d='+str(round(dis_data.loc['dis', cat], 2)))
    if savename != 'None':
        plt.savefig(savename)
        plt.savefig(savename+'.pdf', format='pdf')
 
#Plot phylogram from tree dataframe
#width is the width of the plot (height is set automatically)
#if nameInternalNodes=True, labels are put on the internal nodes
#if abundanceInfo='index' or a column heading the meta data, a bar chart with relative abundance info for each ASV is plotted to the right of the tree
#if xlog=True, the relative abundance bar chart has a log axis
#savename is path and name of the file to be saved
def phyl_tree(obj, width=12, nameInternalNodes=False, abundanceInfo='None', xlog=False, savename='None'):
    if 'tree' not in obj:
        print('Tree not in object')
        return None
    df = obj['tree'].copy()
    
    #Sort out end nodes and internal nodes
    df['asv_count'] = np.nan
    for ix in df.index:
        df.loc[ix, 'asv_count'] = len(df.loc[ix, 'ASVs'])
    df_endN = df[df['asv_count'] == 1]
    df_intN = df[df['asv_count'] > 1]

    #Set starting xpos and ypos for endnodes
    df_endN = df_endN.set_index('nodes')
    df_endN['ypos'] = range(len(df_endN.index))
    df_endN['xpos'] = df_endN['branchL']
    for ix in df_intN.index:
        asvlist = df_intN.loc[ix, 'ASVs']
        BL = df_intN.loc[ix, 'branchL']
        df_endN.loc[asvlist, 'xpos'] += BL
    
    #Sort internal nodes based on number of connected endnodes
    df_intN = df_intN.sort_values('asv_count', ascending=True)

    #If abundanceInfo
    if abundanceInfo != 'None' and 'tab' in obj and 'meta' in obj:
        catlist = []
        if abundanceInfo != 'index':
            metalist = obj['meta'][abundanceInfo].tolist()
            [catlist.append(item) for item in metalist if item not in catlist]
        elif abundanceInfo == 'index':
            catlist = obj['meta'].index.tolist()
        for cat in catlist:
            df_endN['ra:'+cat] = 0
            temp_obj = subset.samples(obj, var=abundanceInfo, slist=[cat])
            ra = temp_obj['ra'].mean(axis=1)
            df_endN.loc[ra.index, 'ra:'+cat] = ra

    #Plot
    textspacing = df_endN['xpos'].max() / 50
    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(width/2.54, 0.7*len(df_endN.index)/2.54), constrained_layout=True)
    gs = fig.add_gridspec(1, 10)
    gs.update(wspace=0, hspace=0)

    if abundanceInfo != 'None':
        ax = fig.add_subplot(gs[0, :9], frame_on=True)
    else:
        ax = fig.add_subplot(gs[0, :10], frame_on=True)

    # First, plot endnodes and their end branches    
    for node in df_endN.index:
        ypos = df_endN.loc[node, 'ypos']
        xpos = df_endN.loc[node, 'xpos']    
        ax.text(xpos+textspacing, ypos, node, verticalalignment='center', color='red')
        node_BL = df_endN.loc[node, 'branchL']
        ax.plot([xpos-node_BL, xpos], [ypos, ypos], lw=1, color='black')
        df_endN.loc[node, 'xpos'] = xpos - node_BL

    # Then go through the internal nodes
    for intN in df_intN.index:
        asvlist = df_intN.loc[intN, 'ASVs']
        xpos = df_endN.loc[asvlist, 'xpos'].mean()
        ymax = df_endN.loc[asvlist, 'ypos'].max()
        ymin = df_endN.loc[asvlist, 'ypos'].min()
        ax.plot([xpos, xpos], [ymin, ymax], lw=1, color='black')
        ymean = (ymax + ymin) / 2
        xmin = xpos - df_intN.loc[intN, 'branchL']
        ax.plot([xmin, xpos], [ymean, ymean], lw=1, color='black')
        df_endN.loc[asvlist, 'ypos'] = ymean      
        df_endN.loc[asvlist, 'xpos'] = xmin
        if nameInternalNodes:
            ax.text(xpos, ymean, df_intN.loc[intN, 'nodes'], verticalalignment='center', color='red')

    ax.plot([0, 0], [df_endN['ypos'].min(), df_endN['ypos'].max()], lw=1, color='black')
    ax.set_ylim(-1, len(df_endN.index))
    ax.axis('off')

    if abundanceInfo != 'None':
        bars_leg1 = []
        bars_leg2 = []
        orig_ypos = range(len(df_endN.index))
        ax2 = fig.add_subplot(gs[0, 9], frame_on=True)
        for cat_nr in range(len(catlist)):
            bar_thickness = 0.8 / len(catlist)
            bar_yoffset = bar_thickness * (cat_nr - (len(catlist)-1)/2)

            cat = catlist[cat_nr]
            ylist = np.array(orig_ypos) + bar_yoffset
            xlist = df_endN['ra:'+cat]
            bl = ax2.barh(ylist, xlist, height=0.95*bar_thickness, label=cat)
            bars_leg1.append(bl)
            bars_leg2.append(cat)

        if xlog:
            ax2.set_xscale('log')
        ax2.set_ylim(-1, len(df_endN.index))
        ax2.set_xticks([])
        ax2.set_yticks(range(len(df_endN.index)))
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax.legend(bars_leg1, bars_leg2, loc='lower right', bbox_to_anchor=(1,1), ncol=4, frameon=False)

    if savename != 'None':
        plt.savefig(savename)
        plt.savefig(savename + '.pdf', format='pdf')
