import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import colors as mcolors
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pickle
import math
from . import subset
from . import diversity
from . import null
from . import hfunc

# PLOT FUNCTIONS

## Plots heatmap
def heatmap(obj, xAxis='None', levels=['Phylum', 'Genus'], includeIndex=False, levelsShown='None', subsetLevels='None', subsetPatterns='None',
                order='None', numberToPlot=20, asvlist='None',
                 figsize=(14, 10), fontsize=15, sepCol='None', sepLine='None',
                labels=True, labelsize=10, cThreshold=8,
                cMap='Reds', cLinear=0.5, cBar=[], savename='None', **kwargs):

    defaultKwargs = {'use_values_in_tab':False, 'value_aggregation':'sum', 'method':'max',
                     'sorting':'abundance'}
    kwargs = {**defaultKwargs, **kwargs}

    """ FUNCTION FOR PLOTTING A HEATMAP
    xAxis specifies heading in meta data used to merge samples
    levels specifies taxonomic levels used in y axis
    if includeIndex=True, the index, e.g. ASV name is included in the lowest tax level name
    if levelsShown='number', numbers instead of taxonomic levels are shown on the y-axis; 
    subsetLevels and subsetPatterns refer to subsetTextPatters function which can be used to filter results
    order refers to heading in meta data used to order samples
    numberToPlot refers to the number of taxa with highest abundance to include in the heatmap
    asvlist is a list of ASVs that should be included in the plot (this means numberToPlot is disregarded)
    figsize is the width and height of the figure
    fontsize is refers to the axis text
    sepCol is a list of column numbers between which to include a separator, i.e. to clarify grouping of samples
    sepLine is a list of column numbers after which a line separator is drawn
    if labels=True, include relative abundance values in heatmap, if False they are not included
    labelsize is the font size of the relative abundance lables in the heatmap
    cThreshold is the relative abundance % at which the label color switches from black to white (for clarity)
    cMap is the color map used in the heatmap
    cLinear is a parameter determining how the color change with relative abundance, a value of 1 means the change is linear
    cBar is a list of tick marks to use if a color bar is included as legend
    savename is the name (also include path) of the saved png file, if 'None' no figure is saved

    **kwargs: use_values_in_tab, value_aggregation, sorting
    if use_values_in_tab is True, no normalization to 100% will be done.
    value_aggregation can be 'sum' or 'mean'. It described the method used to calculate values for each group of samples merged by the xAxis parameter.
    method can 'max' or 'mean'. It is used to rank taxa based on their abundance, either max in a sample or mean across all samples.
    sorting can 'abundance' or 'tax'. If it is abundance, the taxa are ordered based on abundance. If tax, they are ordered alphabetically.
    """

    if 'tab' not in obj:
        print('Error: tab missing in obj.')
        return None
    
    #Merge samples based on xAxis
    if xAxis != 'None':
        merged_obj = subset.merge_samples(obj, var=xAxis, method=kwargs['value_aggregation'])
    else:
        merged_obj = obj.copy()

    #Calculate relative abundances and store in df ra
    if not kwargs['use_values_in_tab']:
        merged_obj['tab'] = 100*merged_obj['tab']/merged_obj['tab'].sum()

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

    ## Subset based on pattern or asvlist
    if isinstance(asvlist, list):
        merged_obj = subset.sequences(merged_obj, asvlist)
    elif subsetLevels != 'None' and isinstance(subsetLevels, list) and isinstance(subsetPatterns, list):
        merged_obj = subset.text_patterns(merged_obj, subsetLevels, subsetPatterns)

    ## Groupby taxa
    taxa_obj = hfunc.groupbytaxa(merged_obj, levels=levels, includeIndex=includeIndex)
    ra = taxa_obj['tab']
    table = ra.copy()

    # Subset for top taxa or list
    if kwargs['method'] == 'max' and not isinstance(asvlist, list):
        ra['max'] = ra.max(axis=1)
        ra = ra.sort_values(by=['max'], ascending=False)
        retain = ra.index.tolist()[:numberToPlot]
        table = table.loc[retain]
    elif kwargs['method'] == 'mean' and not isinstance(asvlist, list):
        ra['mean'] = ra.mean(axis=1)
        ra = ra.sort_values(by=['mean'], ascending=False)
        retain = ra.index.tolist()[:numberToPlot]
        table = table.loc[retain]

    #Sort order on the y-axis
    if kwargs['sorting'] == 'abundance':
        table['avg'] = table.mean(axis=1)
        table = table.sort_values(by=['avg'], ascending=True)
        table = table.drop(['avg'], axis=1)
    elif kwargs['sorting'] == 'tax':
        tax = taxa_obj['tax'].loc[table.index]
        for c in tax.columns:
            tax.loc[tax[c].isna(), c] = 'zzz'
        tax = tax.sort_values(tax.columns.tolist())
        table = table.loc[tax.index]

    #Sort order on the x-axis
    if order != 'None':
        table = table.loc[:, logiclist]
    
    # Print heatmap
    if levelsShown == 'number':
        numbered_list = []
        for i in range(len(table.index), 0, -1):
            numbered_list.append(i)
        table = pd.DataFrame(table.to_numpy(), index=numbered_list, columns=table.columns)
    
    #Fix datalabels
    if labels:
        labelvalues = table.copy()
        labelvalues = labelvalues.map(str)
        for r in table.index:
            for c in table.columns:
                value = float(table.loc[r, c])
                if value < 0.1 and value > 0:
                    labelvalues.loc[r, c] = '<0.1'
                elif value < 9.95 and value >= 0.1:
                    labelvalues.loc[r, c] = str(round(value, 1))
                elif value > 99:
                    labelvalues.loc[r, c] = '99'
                elif value >= 9.95:
                    labelvalues.loc[r, c] = str(int(round(value, 0)))
                else:
                    labelvalues.loc[r, c] = '0'

    # Include empty columns in table to separate samples
    if isinstance(sepCol, list):
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

    if isinstance(sepCol, list):
        for i in range(len(sepCol)):
            for j in range(6):
                ax.axvline(sepCol[i]+i-0.5+j/5, 0, len(table.index), linestyle='-', lw=1, color='white')

    if isinstance(sepLine, list):
        for i in range(len(sepLine)):
            ax.axvline(sepLine[i]-0.5, 0, len(table.index), linestyle='-', lw=1, color='black')

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
                    figsize=(10, 6), fontsize=18, colorlist='None', savename='None', **kwargs):
    defaultKwargs = {'use_values_in_tab':False}
    kwargs = {**defaultKwargs, **kwargs}

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
    df = pd.DataFrame(0.0, index=xvalues, columns=smplist)

    #Put data in dataframe
    tab = obj['tab'][smplist]
    for x in xvalues:
        if divType == 'naive':
            alphadiv = diversity.naive_alpha(tab, q=x, use_values_in_tab=kwargs['use_values_in_tab'])
        elif divType == 'phyl':
            alphadiv = diversity.phyl_alpha(tab, tree=obj['tree'], q=x, use_values_in_tab=kwargs['use_values_in_tab'])
        elif divType == 'func' and isinstance(distmat, pd.DataFrame):
            alphadiv = diversity.func_alpha(tab, distmat=distmat, q=x, use_values_in_tab=kwargs['use_values_in_tab'])
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
def pcoa(dist, meta, var1='None', var2='None', var1_title='', var2_title='', biplot=[], arrow_width=0.001,
             whitePad=1.1, var2pos=0.4, tag='None', order='None', title='', connectPoints='None',
             figsize=(9, 6), fontsize=12, markersize=50, markerscale=1.1, lw=1,
             hideAxisValues=False, showLegend=True, 
             ellipse='None', n_std=2, ellipse_tag=False, ellipse_connect='None',
             flipx=False, flipy=False, returnData=False, ax='None',
             colorlist='None', markerlist='None', savename='None'):
    """
    PLOT PCoA
    dist is distance matrix and meta is meta data
    var1 is heading in meta used to color code, var2 is heading in meta used to code by marker type
    var1_title and var_2 title are the titles used in the legend
    biplot is list with meta data column headings used in biplot, the columns must contain numeric data
    arrow_width is width of arrows in biplot
    whitePad sets the space between the outermost points and the plot limits (1.0=no space)
    var2pos is the vertical position of the var2 legend
    tag is heading in meta used to add labels to each point in figure
    order is heading in meta used to order samples (should be numbers)
    title is title of the entire figure
    if connectPoints is a metadata column header, it will use that data to connect the points
    figsize is figure dimension
    fontsize is font size
    markersize is size of markers in the figures
    markerscale sets the size of the markers in the legend
    lw is linewidth of lines in the plot
    if hideAxisValues=True, no numbers are shown
    if showLegend=False, the legend is removed
    ellipse is metadata column header with categories of samples that should be grouped with confidence ellipses
    n_std is number of standard deviations of confidence ellipses
    ellipse_tag is True, labels will be printed in each ellipse
    ellipse_connect is metadata column with data to connect centers of ellipses with lines 
    if flipx or flipy is True, invert x or y axis
    if returnData=True, the metadata and coordinates in PCoA is returned as a pandas dataframe
    colorlist specifies colorlist to use for var1; same for markerlist and var2
    savename is path and name to save png and pdf output
    """
    
    coords, ev = hfunc.pcoa_calculation(dist)
    coords = coords.iloc[:, :2]
    ev = ev.iloc[:2]

    xaxislims = [min(coords.iloc[:, 0])*whitePad, max(coords.iloc[:, 0])*whitePad]
    yaxislims = [min(coords.iloc[:, 1])*whitePad, max(coords.iloc[:, 1])*whitePad]

    #Check biplot
    if len(biplot) > 0:
        #Standardize U (eigenvectors)
        U_vectors = coords.copy()
        U_vectors.iloc[:, 0] = (U_vectors.iloc[:, 0]-U_vectors.iloc[:, 0].mean())/U_vectors.iloc[:, 0].std()
        U_vectors.iloc[:, 1] = (U_vectors.iloc[:, 1]-U_vectors.iloc[:, 1].mean())/U_vectors.iloc[:, 1].std()

        #Standardize Y
        Y = pd.DataFrame(index=dist.columns)
        for mh in biplot:
            Y[mh] = meta[mh]
            Y[mh] = (Y[mh]-Y[mh].mean())/Y[mh].std()
        Y_cent = Y.transpose()

        Spc =(1/(len(dist.columns)-1))*np.matmul(Y_cent, U_vectors.to_numpy())
        biglambda = np.array([[ev.iloc[0]**-0.5, 0], [0, ev.iloc[1]**-0.5]])
        Uproj = ((len(dist.columns)-1)**0.5)*np.matmul(Spc, biglambda)

        #Scale to the plot
        Uscalefactors = []
        Uscalefactors.append(max(coords.iloc[:, 0])/Uproj[0].max())
        Uscalefactors.append(min(coords.iloc[:, 0])/Uproj[0].min())
        Uscalefactors.append(max(coords.iloc[:, 1])/Uproj[1].max())
        Uscalefactors.append(min(coords.iloc[:, 1])/Uproj[1].min())
        Uscale = 1
        for i in Uscalefactors:
            if i < Uscale and i > 0:
                Uscale = i
        Uproj = Uproj*Uscale

    # Do the plotting
    xn = coords.columns.tolist()[0]
    yn = coords.columns.tolist()[1]
    metaPlot = pd.concat([meta, coords], axis=1, join='outer')

    if order != 'None' and isinstance(meta, pd.DataFrame):
        metaPlot[order] = meta[order]
        metaPlot = metaPlot.sort_values(order, ascending=True)

    # Check data for ellipses if those are to be plotted        
    if ellipse != 'None' and isinstance(meta, pd.DataFrame): #https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
        metaPlot[ellipse] = meta[ellipse]
        ellcats = metaPlot[ellipse].unique().tolist()
        ell_df = pd.DataFrame(pd.NA, index=ellcats, columns=['ell_radius_x', 'ell_radius_y', 'scale_x', 'scale_y', 'mean_x', 'mean_y', 'xmid', 'ymid'])
        for cat in ellcats:
            xs = metaPlot[xn][metaPlot[ellipse] == cat].tolist()
            ys = metaPlot[yn][metaPlot[ellipse] == cat].tolist()
            ell_df.loc[cat, 'xmid'] = np.mean(xs)
            ell_df.loc[cat, 'ymid'] = np.mean(ys)
            if len(xs) == len(ys) and len(xs) > 2:
                ell_df.loc[cat, ['ell_radius_x', 'ell_radius_y', 'scale_x', 'scale_y', 'mean_x', 'mean_y']] = hfunc.pcoa_ellipse(xs, ys, n_std)

    if var1 == 'None':
        metaPlot['None'] = 'all'
        smpcats1 = ['all']
    if var1 != 'None': #List of names used for different colors in legend
        metaPlot[var1] = meta[var1]
        smpcats1 = []
        [smpcats1.append(item) for item in metaPlot[var1] if item not in smpcats1]
    if var2 != 'None' and var1 != 'None': #List of names used for different marker types in legend
        metaPlot[var2] = meta[var2]
        smpcats2 = []
        [smpcats2.append(item) for item in metaPlot[var2] if item not in smpcats2]

    # Create figure
    if colorlist == 'None':
        colorlist = hfunc.get_colors_markers('colors')
    if markerlist == 'None':
        markerlist = hfunc.get_colors_markers('markers')

    if ax == 'None':
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=figsize)

    linesColor = [[], []]
    linesShape = [[], []]
    shapeTracker = []

    for i in range(len(smpcats1)):
        metaPlot_i = metaPlot[metaPlot[var1] == smpcats1[i]] #Subset metaPlot based on var1 in smpcats1

        # Connect midpoints between ellipses
        if ellipse_connect != 'None':
            metaPlot_i['xmid_ell'] = np.nan
            metaPlot_i['ymid_ell'] = np.nan
            for ell_cat in ell_df.index:
                metaPlot_i['xmid_ell'][metaPlot_i[ellipse] == ell_cat] = ell_df.loc[ell_cat, 'xmid']
                metaPlot_i['ymid_ell'][metaPlot_i[ellipse] == ell_cat] = ell_df.loc[ell_cat, 'ymid']
            metaPlot_i[ellipse_connect] = metaPlot_i[ellipse_connect].apply(float)
            metaPlot_i = metaPlot_i.sort_values(ellipse_connect)
            ax.plot(metaPlot_i['xmid_ell'], metaPlot_i['ymid_ell'], color=colorlist[i], lw=lw)

        # Connect points between normal points
        elif connectPoints != 'None':
            metaPlot_i[connectPoints] = metaPlot_i[connectPoints].apply(float)
            metaPlot_i = metaPlot_i.sort_values(connectPoints)
            ax.plot(metaPlot_i[xn], metaPlot_i[yn], color=colorlist[i], lw=lw)

        # Draw ellipse
        if ellipse != 'None':
            ell_df_index = []
            [ell_df_index.append(item) for item in metaPlot_i[ellipse] if item not in ell_df_index]
            for ix in ell_df_index:
                #ell_radius_x, ell_radius_y, scale_x, scale_y, mean_x, mean_y
                ellipse_ax = Ellipse((0, 0), width=ell_df.loc[ix, 'ell_radius_x']*2, height=ell_df.loc[ix, 'ell_radius_y']*2, facecolor='none', edgecolor=colorlist[i])
                transf_ax = transforms.Affine2D().rotate_deg(45).scale(ell_df.loc[ix, 'scale_x'], ell_df.loc[ix, 'scale_y']).translate(ell_df.loc[ix, 'mean_x'], ell_df.loc[ix, 'mean_y'])
                ellipse_ax.set_transform(transf_ax + ax.transData)
                ax.add_patch(ellipse_ax)

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
            if showLegend and var1 != 'None':
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
            if showLegend and var1 != 'None':
                ax.legend(linesColor[0], linesColor[1], bbox_to_anchor=(1, 1), title=var1_title, frameon=False, markerscale=markerscale, fontsize=fontsize, loc='upper left')

    ##Put tags at each point or ellipse, or both
    if ellipse_tag:
        for ix in ell_df.index:
            tagx = ell_df.loc[ix, 'xmid']
            tagy = ell_df.loc[ix, 'ymid']
            ax.annotate(ix, (tagx, tagy))
    elif tag == 'index':
        for ix in metaPlot.index:
            tagx = metaPlot.loc[ix, xn]
            tagy = metaPlot.loc[ix, yn]
            ax.annotate(str(ix), (tagx, tagy))
    elif tag in meta.columns.tolist():
        for ix in metaPlot.index:
            tagtext = str(metaPlot.loc[ix, tag])
            tagx = metaPlot.loc[ix, xn]
            tagy = metaPlot.loc[ix, yn]
            ax.annotate(tagtext, (tagx, tagy))

    ##Input arrows
    if len(biplot) > 0:
        for mh_nr in range(len(biplot)):
            var_name = biplot[mh_nr]
            ha = 'center'
            if Uproj.loc[var_name, 0] > 0:
                ha = 'left'
            elif Uproj.loc[var_name, 0] < 0:
                ha = 'right'
            va = 'center'
            if Uproj.loc[var_name, 1] > 0:
                va = 'bottom'
            elif Uproj.loc[var_name, 1] < 0:
                va = 'top'
            ax.arrow(0, 0, Uproj.loc[var_name, 0], Uproj.loc[var_name, 1], color='black', width=arrow_width)
            ax.annotate(var_name, (1.03*Uproj.loc[var_name, 0], 1.03*Uproj.loc[var_name, 1]), horizontalalignment=ha, verticalalignment=va)
        ax.axhline(0, 0, 1, linestyle='--', color='grey', lw=0.5)
        ax.axvline(0, 0, 1, linestyle='--', color='grey', lw=0.5)

    ax.set_xlabel(xn)
    ax.set_ylabel(yn)
    ax.set_xlim(xaxislims)
    ax.set_ylim(yaxislims)
    if flipx:
        ax.invert_xaxis()
    if flipy:
        ax.invert_yaxis()
    
    if hideAxisValues:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.title(title)
    plt.tight_layout()
    if savename != 'None':
        plt.savefig(savename+'.pdf', format='pdf')
        plt.savefig(savename)
    if returnData and len(biplot) > 0:
        return metaPlot, Uproj
    elif returnData:
        return metaPlot
    if ax != 'None':
        return ax

# Calculate and plot rarefaction curve
def rarefactioncurve(obj, step='flexible', divType='naive', q=0, distmat='None',
                     figsize=(14, 10), fontsize=18, 
                     var='None', order='None', tag='None', colorlist='None',
                     onlyReturnData=False, onlyPlotData='None', savename='None'):

    """
    Calculate and plot rarefaction curve
    step is the step size used during subsampling, if 'flexible' the total reads are divided by 20
    figsize is with ahd height of figure in inches
    fontsize is size of text in figure
    var is column in meta data used to color code lines in plot
    order is column in meta data used to order sample
    tag is column in meta data used to name lines in plot, if 'index', sample names are used
    colorlist is colors to be used in plot, if 'None' qdiv default is used
    if onlyReturnData=True, function will return a dictionary with data
    if onlyPlotData is a dictionary with data, it will be plotted and no calculations will be carried out
    is savename is specified, plots will be saved and data will be saved as pickle file
    """
    
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
            totalreads = int(smp_series.sum())
            
            #Make shuffled list of sv names
            name_arr = smp_series.index.tolist()
            counts_arr = smp_series.to_numpy()
            cumreads2 = np.cumsum(counts_arr)
            cumreads1 = cumreads2 - counts_arr
            ind_reads_arr = np.empty(totalreads, dtype=object)
            for i, (v1, v2) in enumerate(zip(cumreads1, cumreads2)):
                ind_reads_arr[int(v1):int(v2)] = name_arr[i]
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
                div_val = diversity.naive_alpha(tab[[smp]], q=q)
            elif divType == 'phyl':
                div_val = diversity.phyl_alpha(tab[[smp]], obj['tree'], q=q)
            elif divType == 'func':
                div_val = diversity.func_alpha(tab[[smp]], distmat, q=q)
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
def octave(obj, var='None', slist='None', nrows=2, ncols=2, fontsize=11, figsize=(10, 6), 
           xlabels=True, ylabels=True, title=True, color='blue', savename='None'):

    """
    # var is the column heading in metadata used to select samples to include. The counts for all samples with the same text in var column will be merged.
    # slist is a list of names in meta data column which specify samples to keep. If slist='None' (default), the whole meta data column is used
    # nrows and ncols are the number of rows and columns in the plot. 
    # nrows*ncols must be equal to or more than the number of samples plotted
    # if xlabels=True, k is shown for the bins on the x-axis
    # if ylabels=True, ASV counts are shown on the y-axis
    # if title=True, sample name is shown as title for each panel
    # color determines color of bars
    # savename is path and name of file
    """

    if var == 'None':
        tab = obj['tab'].copy()
        smplist = tab.columns.tolist()
    else:
        merged_obj = subset.merge_samples(obj, var=var, slist=slist)
        tab = merged_obj['tab'].copy()
        smplist = tab.columns.tolist()

    if len(smplist) > nrows*ncols:
        print('Too few panels, ', len(smplist),' are needed')
        return None

    max_read = max(tab.max())
    if max_read >= 1:
        max_k = math.floor(math.log(max_read, 2))
    else:
        max_k = math.ceil(math.log(max_read, 2))

    min_read = min(tab[tab>0].min())
    if min_read >= 1:
        min_k = math.floor(math.log(min_read, 2))
    else:
        min_k = math.ceil(math.log(min_read, 2))
    min_k = min(min_k, 0)

    k_index = np.arange(min_k, max_k+1)
    df = pd.DataFrame(0, index=k_index, columns=['k', 'min_count', 'max_count']+smplist)
    df['k'] = k_index
    df['min_count'] = 2.0**k_index
    df['max_count'] = 2.0**(k_index+1)

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
            temp = tab.loc[(tab[smp] >= bin_min)&(tab[smp] < bin_max), smp]
            df.loc[k, smp] = len(temp)
            
        ax.bar(df['k'], df[smp], color=color)
        ax.set_xticks(k_index[::2])
        if xlabels and row == nrows-1:
            ax.set_xticklabels(k_index[::2])
            ax.set_xlabel(u'k (bin [$\u2265$2$^k$ and <2$^{k+1}$])')
        elif xlabels:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xticklabels([])
        if ylabels and col == 0:
            ax.set_ylabel('Count')
        elif ylabels:
            ax.set_ylabel('')
        else:
            ax.set_yticklabels([])
        if title:
            ax.text(0.97*ax.get_xlim()[1], 0.97*ax.get_ylim()[1], str(smp), verticalalignment='top', horizontalalignment='right')

    if savename != 'None':
        plt.savefig(savename)
        plt.savefig(savename+'.pdf', format='pdf')
        df.to_csv(savename + '.csv', index=False)
        
# Plot contribution of taxa to observed naive dissimilarity
def dissimilarity_contributions(obj, var='None', q=1, divType='naive', index='local', 
                                numberToPlot=20, levels=['Genus'], fromFile='None',
                                figsize=(18/2.54, 14/2.54), fontsize=10, savename='None'):

    """
    # var is the column heading in metadata used to categorize samples. Dissimilarity within each category is calculated.
    # q is diversity order and index is local or regional
    # numberToPlot is the number of taxa to include
    # levels are taxonomic levels to include on y-axis
    # fromFile could be that path to a file generated with the output from diversity.naive_dissimilarity_contributions()
    # savename is path and name of files to be saved
    """

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
        tax_df = tax_df.map(str)
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
    df_endN['ypos'] = range(len(df_endN.index)); df_endN['ypos'].apply(float)
    df_endN['xpos'] = df_endN['branchL'].apply(float)
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
            df_endN['ra:'+cat] = 0.0
            temp_obj = subset.samples(obj, var=abundanceInfo, slist=[cat])
            temp_obj['ra'] = temp_obj['tab'] / temp_obj['tab'].sum()
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
        df_endN.loc[asvlist, 'ypos'] = float(ymean)      
        df_endN.loc[asvlist, 'xpos'] = float(xmin)
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

#Plot Harvey balls based on information in metadata
#var is the column heading in the metadata with categories that will be used as row names in the figure
#columns is a list of column headings in the metadata that contains percentages (0-100) that will be plotted 
def harvey_balls(meta, var='index', columns=[], var_colors='None', columns_colorlist=[], var_width=4, figsize=(18/2.54, 14/2.54), fontsize=10, savename='None'):
    if var != 'index' and var not in meta.columns:
        print('var not in metadata column headings')
        return None
    if len(columns) == 0 or len(list(set(columns).intersection(meta.columns.tolist()))) != len(columns):
        print('columns list not in metadata column headings')
        return None

    if len(columns_colorlist) == 0:
        columns_colorlist = ['black']*len(columns)
    elif len(columns_colorlist) < len(columns):
        for i in range(len(columns_colorlist), len(columns)):
            columns_colorlist.append('black')

    plt.rcParams.update({'font.size':fontsize})
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(len(meta)+1, var_width+len(columns), figure=fig)

    ax01 = plt.subplot(gs[0, :var_width])
    ax01.text(0, 0, var, va='center')
    ax01.set_ylim(-1, 1)
    ax01.axis('off')

    for i, g in enumerate(columns):
        ax = plt.subplot(gs[0, var_width+i])
        ax.text(0, 0, g, color=columns_colorlist[i], ha='center', va='center')
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1,1)
        ax.axis('off')

    if var=='index':
        rowlist = meta.index.tolist()
    else:
        rowlist = meta[var].tolist()
    if var_colors == 'None':
        rowcolors = ['black']*len(rowlist)
    else:
        rowcolors = meta[var_colors].tolist()

    for j, r in enumerate(rowlist):
        ax1 = plt.subplot(gs[j+1, :var_width])
        ax1.text(0, 0, rowlist[j], color=rowcolors[j], va='center')
        ax1.set_ylim(-1, 1)
        ax1.axis('off')
        for i, g in enumerate(columns):
            ax = plt.subplot(gs[j+1, var_width+i])
            black = meta[g].tolist()[j]
            white = 100-black
            if white == 100:
                ax.pie([white], colors=['white'], startangle=90, wedgeprops={'linewidth':1, 'edgecolor':'black'})
            else:
                ax.pie([white, black], colors=['white', 'black'], startangle=90, wedgeprops={'linewidth':1, 'edgecolor':'black'})
    if savename != 'None':
        plt.savefig(savename, dpi=240)
        plt.savefig(savename+'.pdf', format='pdf')
    
    


