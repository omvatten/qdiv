import tkinter as tk
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename
from qDiv_functions import *
import pandas as pd

def Subsetting(obj, path):

    # Create GUI window
    master = tk.Toplevel()
    master.title('Subsetting')
    master.geometry('500x600')

    # Create scrollbar, root is the frame the all widgets are later placed on
    def onFrameConfigure(canvas):
        ###Reset the scroll region to encompass the inner frame
        canvas.configure(scrollregion=canvas.bbox("all"))
    canvas = tk.Canvas(master, borderwidth=0, background="#bebebe")
    root = tk.Frame(canvas, background="#bebebe")
    vsb = tk.Scrollbar(master, orient="vertical", command=canvas.yview)
    hsb = tk.Scrollbar(master, orient="horizontal", command=canvas.xview)

    canvas.configure(yscrollcommand=vsb.set)
    canvas.configure(xscrollcommand=hsb.set)

    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")

    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((8, 20), window=root, anchor="nw")

    root.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
    ######

    # Rarefy
    rarecount = tk.StringVar()
    rarecount.set('min')
    tk.Label(root, text='Rarefy').pack(anchor=tk.W)
    tk.Label(root, text='Write count (if "min", the frequency table will be rarefied to the smallest sample').pack(anchor=tk.W)
    tk.Entry(root, textvariable=rarecount).pack(anchor=tk.W)

    def rarefycounts():
        if rarecount.get() != 'min':
            rcount = int(rarecount.get())
        else:
            rcount = rarecount.get()
        rtab = rarefy1(obj['tab'], depth=rcount)
        rtab = rtab[rtab.sum(axis=1) > 0]
        obj['tab'] = rtab
        obj_sub = subsetSVs(obj, rtab.index.tolist())
        returnFiles(obj_sub, path=path, sep=',', savename='Rarefied')
    tk.Button(root, text='Rarefy and save files', command=rarefycounts).pack(anchor=tk.W)
    tk.Label(root, text='-'*100).pack(anchor=tk.W)

    # Top SVs
    top_svs = tk.IntVar()
    tk.Label(root, text='Subset to most abundant sequences').pack(anchor=tk.W)
    tk.Label(root, text='Write number of sequences to keep').pack(anchor=tk.W)
    tk.Entry(root, textvariable=top_svs).pack(anchor=tk.W)

    def subset_top_svs():
        obj_sub = subsetTopSVs(obj, top_svs.get())
        returnFiles(obj_sub, path=path, sep='\t')
    tk.Button(root, text='Subset and save files', command=subset_top_svs).pack(anchor=tk.W)
    tk.Label(root, text='-'*100).pack(anchor=tk.W)

    # Merge samples
    meta_h = tk.StringVar()
    tk.Label(root, text='Merge sample based on metadata column heading').pack(anchor=tk.W)
    tk.Label(root, text='Write column heading').pack(anchor=tk.W)
    tk.Entry(root, textvariable=meta_h).pack(anchor=tk.W)

    def merge_smps():
        obj_sub = mergeSamples(obj, var=meta_h.get())
        returnFiles(obj_sub, path=path, sep='\t')
    tk.Button(root, text='Merge samples and save files', command=merge_smps).pack(anchor=tk.W)

    def quit():
        master.destroy()
    tk.Button(root, text='Quit', command=quit).pack(anchor=tk.W)

def Calc_phyl_dist(obj, path):
    # Create GUI window
    master = tk.Toplevel()
    master.title('Phylogenetic distances')
    master.geometry('500x400')

    # Create scrollbar, root is the frame the all widgets are later placed on
    def onFrameConfigure(canvas):
        ###Reset the scroll region to encompass the inner frame
        canvas.configure(scrollregion=canvas.bbox("all"))
    canvas = tk.Canvas(master, borderwidth=0, background="#bebebe")
    root = tk.Frame(canvas, background="#bebebe")
    vsb = tk.Scrollbar(master, orient="vertical", command=canvas.yview)
    hsb = tk.Scrollbar(master, orient="horizontal", command=canvas.xview)

    canvas.configure(yscrollcommand=vsb.set)
    canvas.configure(xscrollcommand=hsb.set)

    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")

    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((8, 20), window=root, anchor="nw")

    root.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
    ###########

    # Name of distmat
    distmat_name = tk.StringVar()
    distmat_name.set('phyl_dist')
    tk.Label(root, text='Specify name of the file to be saved (e.g. phyl_dist').pack(anchor=tk.W)
    tk.Label(root, text='(No need to add .csv, it will be added automatically)').pack(anchor=tk.W)
    tk.Entry(root, textvariable=distmat_name).pack(anchor=tk.W)

    def save_func():
        savename = distmat_name.get()
        phylDistMat(obj['seq'], savename=path+savename)

    tk.Label(root, text='The calculation may take quite long time').pack(anchor=tk.W)
    tk.Button(root, text='Calculate and save file', command=save_func).pack(anchor=tk.W)
    tk.Label(root, text='-'*70).pack(anchor=tk.W)

    def quit():
        master.destroy()
    tk.Button(root, text='Quit', command=quit).pack(anchor=tk.W)

def Heatmap(obj, path):
    def run():
        levels = []
        for i in range(len(v)):
            if v[i].get() == 1:
                levels.append(options[i])

        if useLabels.get() == 'Yes':
            labs = True
        else:
            labs = False

        if len(usecbar.get()) > 0 and usecbar.get() != 'None':
            cbar = usecbar.get()
            cbar = cbar.replace(' ', '')
            cbar = cbar.split(',')
            for i in range(len(cbar)):
                cbar[i] = float(cbar[i])
        else:
            cbar = []

        if len(sepCol.get()) > 0 and sepCol.get() != 'None':
            sepc = sepCol.get()
            sepc = sepc.replace(' ', '')
            sepc = sepc.split(',')
            for i in range(len(sepc)):
                sepc[i] = int(sepc[i])
        else:
            sepc = []

        stringlevels = []
        for i in range(len(stringlev)):
            if stringlev[i].get() == 1:
                stringlevels.append(options[i])
        if len(stringlevels) <1:
            stringlevels = 'None'

        if stringpattern.get() != 'None':
            stringpatterns = stringpattern.get().replace(' ', '')
            stringpatterns = stringpatterns.split(',')
        else:
            stringpatterns = 'None'

        plotHeatmap(obj, xAxis=var.get(), levels=levels, subsetLevels=stringlevels, subsetPatterns=stringpatterns,
                        order=order.get(), numberToPlot=number.get(), method='max_sample', nameType=nametype.get(),
                        figSize=(figSizeW.get(), figSizeH.get()), fontSize=fontSize.get(), sepCol=sepc,
                        labels=labs, labelSize=labelSize.get(), cThreshold=ctresh.get(),
                        cMap=colmap.get(), cLinear=colgamma.get(), cBar=cbar, savename=path+'Heatmap')

    def quit():
        master.destroy()

    # Create GUI window
    master = tk.Toplevel()
    master.title('Heatmap')
    master.geometry('600x700')

    # Create scrollbar, root is the frame the all widgets are later placed on
    def onFrameConfigure(canvas):
        ###Reset the scroll region to encompass the inner frame
        canvas.configure(scrollregion=canvas.bbox("all"))
    canvas = tk.Canvas(master, borderwidth=0, background="#bebebe")
    root = tk.Frame(canvas, background="#bebebe")
    vsb = tk.Scrollbar(master, orient="vertical", command=canvas.yview)
    hsb = tk.Scrollbar(master, orient="horizontal", command=canvas.xview)

    canvas.configure(yscrollcommand=vsb.set)
    canvas.configure(xscrollcommand=hsb.set)

    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")

    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((8, 20), window=root, anchor="nw")

    root.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))

    ####################
    tk.Label(root, text='Various input options for heatmap').grid(row=0, columnspan=3, sticky=tk.W)
    tk.Label(root, text='-'*120).grid(row=1, columnspan=3, sticky=tk.W)

    # Input taxonomic levels
    options = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    v = []
    for i in range(len(options)):
        v.append(tk.IntVar())
    tk.Label(root, text='Choose one or two taxonomic levels to include on the y-axis').grid(row=5, columnspan=3, sticky=tk.W)
    tk.Label(root, text='Sequences are grouped based on the lowest taxonomic level').grid(row=6, columnspan=3, sticky=tk.W)
    for val, opt in enumerate(options):
        tk.Checkbutton(root, text=opt, variable=v[val]).grid(row=7+val, sticky=tk.W)
    tk.Label(root, text='-'*120).grid(row=14, columnspan=3, sticky=tk.W)

    # xAxis
    var = tk.StringVar(root, 'None')
    tk.Label(root, text='Enter metadata column for x-axis labels').grid(row=15, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var).grid(row=16, sticky=tk.W)

    #Order
    order = tk.StringVar(root, 'None')
    tk.Label(root, text='Specify metadata column used to order the samples on the x-axis').grid(row=20, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=order).grid(row=21, sticky=tk.W)

    #Number to plot
    number = tk.IntVar(root, 20)
    tk.Label(root, text='Specify number of taxa to include in heatmap').grid(row=23, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=number, width=10).grid(row=24, sticky=tk.W)

    #nameType
    nametype = tk.StringVar()
    nametype.set('SV')
    tk.Label(root, text='Specify how unclassified taxa should be named (e.g. SV or OTU)?').grid(row=26, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=nametype, width=10).grid(row=27, sticky=tk.W)

    #Figure dimensions
    tk.Label(root, text='-'*120).grid(row=30, columnspan=3, sticky=tk.W)

    #figSize
    tk.Label(root, text='Specify figure dimensions and text size').grid(row=31, columnspan=3, sticky=tk.W)
    figSizeW = tk.IntVar(root, 14)
    figSizeH = tk.IntVar(root, 10)
    tk.Label(root, text='Width').grid(row=32, sticky=tk.E)
    tk.Entry(root, textvariable=figSizeW).grid(row=32, column=1, sticky=tk.W)
    tk.Label(root, text='Height').grid(row=33, sticky=tk.E)
    tk.Entry(root, textvariable=figSizeH).grid(row=33, column=1, sticky=tk.W)

    #FontSize
    fontSize = tk.IntVar(root, 15)
    tk.Label(root, text='Axis text font size').grid(row=35, sticky=tk.E)
    tk.Entry(root, textvariable=fontSize).grid(row=35, column=1, sticky=tk.W)

    #sepCol
    tk.Label(root, text='-'*120).grid(row=36, columnspan=3, sticky=tk.W)
    sepCol = tk.StringVar()
    tk.Label(root, text='Group samples. Insert numbers of samples after which a blank column should be inserted').grid(row=37, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=sepCol).grid(row=38, sticky=tk.W)
    tk.Label(root, text='(separate values by commas)').grid(row=38, column=1, sticky=tk.W)

    #Data labels
    tk.Label(root, text='-'*120).grid(row=40, columnspan=3, sticky=tk.W)
    tk.Label(root, text='Information about data labels').grid(row=41, sticky=tk.W)

    tk.Label(root, text='Do you want to include data labels in heatmap').grid(row=42, columnspan=3, sticky=tk.W)
    labeloptions = ['Yes', 'No']
    useLabels = tk.StringVar(root, 'Yes')
    for val, opt in enumerate(labeloptions):
        tk.Radiobutton(root, text=opt, variable=useLabels, value=opt).grid(row=43, column=val, sticky=tk.W)

    labelSize = tk.IntVar(root, 12)
    tk.Label(root, text='Label font size').grid(row=45, sticky=tk.W)
    tk.Entry(root, textvariable=labelSize, width=10).grid(row=46, sticky=tk.W)

    ctresh = tk.IntVar(root, 10)
    tk.Label(root, text='Percent relative abundance at which the label text shifts from black to white').grid(row=48, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=ctresh, width=10).grid(row=49, sticky=tk.W)

    #Coloring
    tk.Label(root, text='-'*120).grid(row=50, columnspan=3, sticky=tk.W)
    tk.Label(root, text='Color of heatmap').grid(row=51, columnspan=3, sticky=tk.W)
    colmap = tk.StringVar(root, 'Reds')
    tk.Label(root, text='Colormap').grid(row=52, sticky=tk.E)
    tk.Entry(root, textvariable=colmap).grid(row=52, column=1, sticky=tk.W)
    tk.Label(root, text='(see available colormaps in python)').grid(row=52, column=2, sticky=tk.W)

    colgamma = tk.DoubleVar(root, 0.5)
    tk.Label(root, text='Linearity of colormap').grid(row=55, sticky=tk.E)
    tk.Entry(root, textvariable=colgamma).grid(row=55, column=1, sticky=tk.W)
    tk.Label(root, text='(1=linear change in color)').grid(row=55, column=2, sticky=tk.W)

    tk.Label(root, text='If you want a colorbar showing the scale, specify tick marks on the bar').grid(row=58, columnspan=3, sticky=tk.W)
    usecbar = tk.StringVar(root, 'None')
    tk.Entry(root, textvariable=usecbar).grid(row=59, sticky=tk.W)
    tk.Label(root, text='(the values should be separated by comma)').grid(row=59, column=1, columnspan=2, sticky=tk.W)

    # Input subset based on string patterns
    stringlev = []
    for i in range(len(options)):
        stringlev.append(tk.IntVar())
    tk.Label(root, text='-'*120).grid(row=60, columnspan=3, sticky=tk.W)
    tk.Label(root, text='Subset data based on text patterns').grid(row=61, columnspan=3, sticky=tk.W)
    tk.Label(root, text='Choose which taxonomic levels to search for text').grid(row=62, columnspan=3, sticky=tk.W)
    for val, opt in enumerate(options):
        tk.Checkbutton(root, text=opt, variable=stringlev[val]).grid(row=63+val, sticky=tk.W)

    stringpattern = tk.StringVar()
    tk.Label(root, text='Enter words to search for, separate by comma').grid(row=70, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=stringpattern, width=50).grid(row=71, columnspan=2, sticky=tk.W)
    tk.Label(root, text='-'*120).grid(row=72, columnspan=3, sticky=tk.W)

    # Buttons to run functions
    tk.Button(root, text='Plot heatmap', command=run).grid(row=80)
    tk.Button(root, text='Quit', command=quit).grid(row=80, column=1)
    tk.Label(root, text='-'*120).grid(row=82, columnspan=3, sticky=tk.W)

    root.mainloop()

def Alpha_div(obj, path):
    # Create GUI window
    master = tk.Toplevel()
    master.title('Alpha diversity')
    master.geometry('500x700')

    # Create scrollbar, root is the frame the all widgets are later placed on
    def onFrameConfigure(canvas):
        ###Reset the scroll region to encompass the inner frame
        canvas.configure(scrollregion=canvas.bbox("all"))
    canvas = tk.Canvas(master, borderwidth=0, background="#bebebe")
    root = tk.Frame(canvas, background="#bebebe")
    vsb = tk.Scrollbar(master, orient="vertical", command=canvas.yview)
    hsb = tk.Scrollbar(master, orient="horizontal", command=canvas.xview)

    canvas.configure(yscrollcommand=vsb.set)
    canvas.configure(xscrollcommand=hsb.set)

    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")

    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((8, 20), window=root, anchor="nw")

    root.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
    ########

    tk.Label(root, text='Show alpha diversity in plots or save as data files').pack(anchor=tk.W)
    tk.Label(root, text='-'*100).pack(anchor=tk.W)

    # Input distmat
    tk.Label(root, text='Are you working with phylogenetic diversity?').pack(anchor=tk.W)
    tk.Label(root, text='If so, select distance matrix file').pack(anchor=tk.W)
    distmat = tk.StringVar(root, 'Select')
    def openDistmat():
        distmat.set(askopenfilename())
    tk.Button(root, textvariable=distmat, command=openDistmat).pack(anchor=tk.W)
    def resetNone():
        distmat.set('Select')
    tk.Button(root, text='Reset selection', command=resetNone).pack(anchor=tk.W)

    #Plotting
    def run_plot():
        if y_v.get() == 'Yes':
            ylog = True
        else:
            ylog = False

        if distmat.get() != 'Select':
            fulldistmat = pd.read_csv(distmat.get(), index_col=0)
            name2save = 'Phyl_alpha_div_fig'
        else:
            fulldistmat = 'None'
            name2save = 'Naive_alpha_div_fig'

        plotDivAlpha(obj, distmat=fulldistmat, rarefy='None', var=var_col.get(), slist='All', order=order.get(), ylog=ylog,
                     colorlist='None', savename=path + name2save)

    tk.Label(root, text='-'*100).pack(anchor=tk.W)
    tk.Label(root, text='The following input is used for plotting figure...').pack(anchor=tk.W)
    tk.Label(root, text='. '*50).pack(anchor=tk.W)

    # var to use for color coding
    var_col = tk.StringVar()
    var_col.set('None')
    tk.Label(root, text='Specify metadata column heading to use for color coding').pack(anchor=tk.W)
    tk.Entry(root, textvariable=var_col).pack(anchor=tk.W)

    #Order
    order = tk.StringVar()
    order.set('None')
    tk.Label(root, text='Specify metadata column used to order the samples on the x-axis').pack(anchor=tk.W)
    tk.Entry(root, textvariable=order).pack(anchor=tk.W)

    #Semi log y-axis
    options = ['Yes', 'No']
    tk.Label(root, text='Use logarithmic y-axis?').pack(anchor=tk.W)
    y_v = tk.StringVar()
    for opt in options:
        tk.Radiobutton(root, text=opt, variable=y_v, value=opt).pack(anchor=tk.W)

    # Buttons to plot
    tk.Button(root, text='Plot alpha diversity', command=run_plot).pack(anchor=tk.W)

    ## Printing
    def run_print():
        qlist = qvalues.get().replace(' ', '')
        qlist = qlist.split(',')
        qnumbers = []
        for q in qlist:
            qnumbers.append(float(q))

        output = pd.DataFrame(0, index=obj['tab'].columns, columns=qnumbers)
        for q in qnumbers:
            if distmat.get() == 'Select':
                alfa = naiveDivAlpha(obj['tab'], q=q, rarefy='None')
                output[q] = alfa
            else:
                dist = pd.read_csv(distmat.get(), index_col=0)
                alfa = phylDivAlpha(obj['tab'], distmat=dist, q=q, rarefy='None')
                output[q] = alfa
        output.to_csv(path + sname.get() + '.csv')

    tk.Label(root, text='-'*100).pack(anchor=tk.W)
    tk.Label(root, text='The following input is used to save a csv file with data').pack(anchor=tk.W)
    tk.Label(root, text='. '*50).pack(anchor=tk.W)

    tk.Label(root, text='Specify diversity orders to calculate, use comma to separate numbers').pack(anchor=tk.W)
    qvalues = tk.StringVar()
    tk.Entry(root, textvariable=qvalues).pack(anchor=tk.W)

    tk.Label(root, text='Specify name of saved file (do not include .csv)').pack(anchor=tk.W)
    sname = tk.StringVar()
    tk.Entry(root, textvariable=sname).pack(anchor=tk.W)

    # Buttons to save
    tk.Button(root, text='Save alpha diversity data as file', command=run_print).pack(anchor=tk.W)
    tk.Label(root, text='-'*100).pack(anchor=tk.W)

    def quit():
        master.destroy()
    tk.Button(root, text='Quit', command=quit).pack(anchor=tk.W)

def Beta_div(obj, path):

    # Create GUI window
    master = tk.Toplevel()
    master.title('Beta diversity')
    master.geometry('500x500')

    # Create scrollbar, root is the frame the all widgets are later placed on
    def onFrameConfigure(canvas):
        ###Reset the scroll region to encompass the inner frame
        canvas.configure(scrollregion=canvas.bbox("all"))
    canvas = tk.Canvas(master, borderwidth=0, background="#bebebe")
    root = tk.Frame(canvas, background="#bebebe")
    vsb = tk.Scrollbar(master, orient="vertical", command=canvas.yview)
    hsb = tk.Scrollbar(master, orient="horizontal", command=canvas.xview)

    canvas.configure(yscrollcommand=vsb.set)
    canvas.configure(xscrollcommand=hsb.set)

    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")

    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((8, 20), window=root, anchor="nw")

    root.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
    ######

    # Heading
    tk.Label(root, text='Calculate dissimilarities and plot PCoA').grid(row=0, columnspan=3, sticky=tk.W)
    tk.Label(root, text='-'*100).grid(row=1, columnspan=3, sticky=tk.W)

    # Input distmat
    tk.Label(root, text='Are you working with phylogenetic diversity?').grid(row=5, columnspan=3, sticky=tk.W)
    tk.Label(root, text='If so, select distance matrix file').grid(row=6, columnspan=3, sticky=tk.W)
    distmat = tk.StringVar(root, 'Select')
    def openDistmat():
        distmat.set(askopenfilename())
    tk.Button(root, textvariable=distmat, command=openDistmat).grid(row=7, columnspan=3, sticky=tk.W)
    def resetNone():
        distmat.set('Select')
    tk.Button(root, text='Reset selection', command=resetNone).grid(row=8, columnspan=2, sticky=tk.W)

    # Calculate dis matrix
    tk.Label(root, text='-'*100).grid(row=10, columnspan=3, sticky=tk.W)
    tk.Label(root, text='Calculate dissimilarity matrix').grid(row=11, columnspan=3, sticky=tk.W)

    tk.Label(root, text='Choose index').grid(row=15, columnspan=3, sticky=tk.W)
    dis_index = tk.StringVar()
    dis_options = ['Hill', 'Bray-Curtis', 'Jaccard']
    for val, opt in enumerate(dis_options):
        tk.Radiobutton(root, text=opt, variable=dis_index, value=opt).grid(row=16+val, sticky=tk.W)
    qval = tk.DoubleVar()
    qval.set(1)
    tk.Label(root, text='Diversity order (q)').grid(row=16, column=1, sticky=tk.W)
    tk.Entry(root, textvariable=qval).grid(row=16, column=2, sticky=tk.W)

    savename_matrix = tk.StringVar()
    tk.Label(root, text='Write name for distance matrix file (do not include .csv)').grid(row=20, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=savename_matrix, width=50).grid(row=21, columnspan=3, sticky=tk.W)

    def calc_dis_mat():
        if dis_index.get() == 'Bray-Curtis':
            df = bray(obj['tab'])
        elif dis_index.get() == 'Jaccard':
            df = jaccard(obj['tab'])
        elif dis_index.get() == 'Hill' and distmat.get() == 'Select':
            df = naiveDivBeta(obj['tab'], q=qval.get())
        elif dis_index.get() == 'Hill' and distmat.get() != 'Select':
            distmat_df = pd.read_csv(distmat.get(), index_col=0)
            df = phylDivBeta(obj['tab'], distmat=distmat_df, q=qval.get())
        df.to_csv(path + savename_matrix.get() + '.csv')

    tk.Button(root, text='Calculate dissimilarities', command=calc_dis_mat).grid(row=29, columnspan=2, sticky=tk.W)

    def quit():
        master.destroy()
    tk.Button(root, text='Quit', command=quit).grid(row=29, column=2, sticky=tk.W)

def Plot_PCoA(obj, path):

    # Create GUI window
    master = tk.Toplevel()
    master.title('Plot PCoA')
    master.geometry('500x700')

    # Create scrollbar, root is the frame the all widgets are later placed on
    def onFrameConfigure(canvas):
        ###Reset the scroll region to encompass the inner frame
        canvas.configure(scrollregion=canvas.bbox("all"))
    canvas = tk.Canvas(master, borderwidth=0, background="#bebebe")
    root = tk.Frame(canvas, background="#bebebe")
    vsb = tk.Scrollbar(master, orient="vertical", command=canvas.yview)
    hsb = tk.Scrollbar(master, orient="horizontal", command=canvas.xview)

    canvas.configure(yscrollcommand=vsb.set)
    canvas.configure(xscrollcommand=hsb.set)

    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")

    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((8, 20), window=root, anchor="nw")

    root.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
    ######

    # Heading
    tk.Label(root, text='Plot PCoA').grid(row=0, columnspan=3, sticky=tk.W)
    tk.Label(root, text='-'*100).grid(row=1, columnspan=3, sticky=tk.W)

    # Get dist
    tk.Label(root, text='Select dissimilarity matrix file (comma separated)').grid(row=31, columnspan=3, sticky=tk.W)
    dis_mat_name = tk.StringVar(root, 'None')
    def openDisMat():
        dis_mat_name.set(askopenfilename())
    tk.Button(root, text='Dissimilarity matrix', command=openDisMat).grid(row=35, sticky=tk.W)
    tk.Label(root, textvariable=dis_mat_name).grid(row=35, column=1, sticky=tk.W)

    # Get var1 and var2
    var_col = tk.StringVar()
    var_col.set('')
    tk.Label(root, text='Set metadata column heading for color coding of points (required)').grid(row=40, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var_col).grid(row=41, sticky=tk.W)

    var1_title = tk.StringVar()
    var1_title.set('')
    tk.Label(root, text='Set title for color legend').grid(row=42, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var1_title).grid(row=43, sticky=tk.W)

    var_marker = tk.StringVar()
    var_marker.set('None')
    tk.Label(root, text='Set metadata column heading for marker type of points (not required)').grid(row=44, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var_marker).grid(row=45, sticky=tk.W)

    var2_title = tk.StringVar()
    var2_title.set('')
    tk.Label(root, text='Set title for marker legend').grid(row=46, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var2_title).grid(row=47, sticky=tk.W)

    var2_pos = tk.DoubleVar()
    var2_pos.set(0.4)
    tk.Label(root, text='Specify position of marker legend (typically 0.2-0.5)').grid(row=48, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var2_pos).grid(row=49, sticky=tk.W)

    # Right space for legend
    right_space = tk.DoubleVar()
    right_space.set(0.10)
    tk.Label(root, text='Specify fraction of the figure to be used for legend (typically 0-0.2)').grid(row=50, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=right_space).grid(row=51, sticky=tk.W)

    # Logic order
    order = tk.StringVar()
    order.set('None')
    tk.Label(root, text='Specify metadata column used to order the samples in the legend').grid(row=55, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=order).grid(row=56, sticky=tk.W)
    tk.Label(root, text='-'*100).grid(row=57, columnspan=3, sticky=tk.W)

    #figSize
    tk.Label(root, text='Specify figure dimensions and text size').grid(row=60, columnspan=3, sticky=tk.W)
    figSizeW = tk.IntVar(root, 14)
    figSizeH = tk.IntVar(root, 10)
    tk.Label(root, text='Width').grid(row=61, sticky=tk.E)
    tk.Entry(root, textvariable=figSizeW).grid(row=61, column=1, sticky=tk.W)
    tk.Label(root, text='Height').grid(row=63, sticky=tk.E)
    tk.Entry(root, textvariable=figSizeH).grid(row=63, column=1, sticky=tk.W)

    #FontSize
    fontSize = tk.IntVar(root, 15)
    tk.Label(root, text='Axis text font size').grid(row=65, sticky=tk.E)
    tk.Entry(root, textvariable=fontSize).grid(row=65, column=1, sticky=tk.W)

    # savename
    savename_pcoa = tk.StringVar()
    tk.Label(root, text='Write name for PCoA file').grid(row=70, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=savename_pcoa, width=50).grid(row=71, columnspan=2, sticky=tk.W)
    tk.Label(root, text='-'*100).grid(row=72, columnspan=3, sticky=tk.W)

    def plot_pcoa():
        dist_file = pd.read_csv(dis_mat_name.get(), index_col=0)
        plotPCoA(dist_file, obj['meta'], biplot=[], var1=var_col.get(), var2=var_marker.get(),
                 var1_title=var1_title.get(), var2_title=var2_title.get(),
                 whitePad=1.1, rightSpace=right_space.get(), var2pos=0.4, tag='None', order=order.get(), title='',
                 figSize=(figSizeW.get(), figSizeH.get()), fontSize=fontSize.get(), colorlist='None', markerlist='None',
                 savename=path + savename_pcoa.get())

    tk.Button(root, text='Plot PCoA', command=plot_pcoa).grid(row=75, sticky=tk.W)

    def quit():
        master.destroy()
    tk.Button(root, text='Quit', command=quit).grid(row=75, column=1, sticky=tk.W)

def Null_model(obj, path):
    # Create GUI window
    master = tk.Toplevel()
    master.title('Null model')
    master.geometry('500x700')

    # Create scrollbar, root is the frame the all widgets are later placed on
    def onFrameConfigure(canvas):
        ###Reset the scroll region to encompass the inner frame
        canvas.configure(scrollregion=canvas.bbox("all"))
    canvas = tk.Canvas(master, borderwidth=0, background="#bebebe")
    root = tk.Frame(canvas, background="#bebebe")
    vsb = tk.Scrollbar(master, orient="vertical", command=canvas.yview)
    hsb = tk.Scrollbar(master, orient="horizontal", command=canvas.xview)

    canvas.configure(yscrollcommand=vsb.set)
    canvas.configure(xscrollcommand=hsb.set)

    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")

    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((8, 20), window=root, anchor="nw")

    root.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
    ########

    # First line introducing what can be done
    tk.Label(root, text='Run null model and return matrices of Raup-Crick and actual null values').grid(row=0, columnspan=3, sticky=tk.W)
    tk.Label(root, text='-'*100).grid(row=1, columnspan=3, sticky=tk.W)

    # Input distmat
    tk.Label(root, text='Are you working with phylogenetic diversity?').grid(row=5, columnspan=3, sticky=tk.W)
    tk.Label(root, text='If so, select distance matrix file').grid(row=6, columnspan=3, sticky=tk.W)
    distmat = tk.StringVar(root, 'Select')
    def openDistmat():
        distmat.set(askopenfilename())
    tk.Button(root, textvariable=distmat, command=openDistmat).grid(row=7, columnspan=3, sticky=tk.W)
    def resetNone():
        distmat.set('Select')
    tk.Button(root, text='Reset selection', command=resetNone).grid(row=8, columnspan=3, sticky=tk.W)

    tk.Label(root, text='-'*100).grid(row=10, columnspan=3, sticky=tk.W)

    # iterations
    var_iter = tk.IntVar()
    var_iter.set(99)
    tk.Label(root, text='Number of randomization').grid(row=15, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var_iter, width=10).grid(row=16, sticky=tk.W)

    # dis index
    tk.Label(root, text='Choose dissimilarity index').grid(row=20, columnspan=3, sticky=tk.W)
    dis_index = tk.StringVar()
    dis_options = ['Hill', 'Bray-Curtis', 'Jaccard']
    for val, opt in enumerate(dis_options):
        tk.Radiobutton(root, text=opt, variable=dis_index, value=opt).grid(row=22+val, sticky=tk.W)
    qval = tk.DoubleVar()
    qval.set(1)
    tk.Label(root, text='Diversity order (q)').grid(row=22, column=1, sticky=tk.W)
    tk.Entry(root, textvariable=qval, width=10).grid(row=22, column=2, sticky=tk.W)

    tk.Label(root, text='-'*100).grid(row=25, columnspan=3, sticky=tk.W)

    # randomization procedure
    tk.Label(root, text='Randomization procedure').grid(row=29, columnspan=3, sticky=tk.W)
    rand_proc = tk.StringVar()
    rand_options = ['abundance', 'frequency', 'weighting']
    for val, opt in enumerate(rand_options):
        tk.Radiobutton(root, text=opt, variable=rand_proc, value=opt).grid(row=30+val, column=0, sticky=tk.W)

    # weighting variable
    var_wt = tk.StringVar()
    var_wt.set('None')
    tk.Label(root, text='Weighting variable').grid(row=32, column=1, sticky=tk.W)
    tk.Entry(root, textvariable=var_wt).grid(row=33, column=1, sticky=tk.W)

    # weight
    wt = tk.DoubleVar()
    wt.set(1)
    tk.Label(root, text='Weight').grid(row=32, column=2, sticky=tk.W)
    tk.Entry(root, textvariable=wt, width=10).grid(row=33, column=2, sticky=tk.W)

    # constraining variable
    tk.Label(root, text='-'*100).grid(row=35, columnspan=3, sticky=tk.W)
    var_con = tk.StringVar()
    var_con.set('None')
    tk.Label(root, text='Constraining variable (optional)').grid(row=36, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var_con).grid(row=37, columnspan=2, sticky=tk.W)

    # return variable
    var_ret = tk.StringVar()
    var_ret.set('None')
    tk.Label(root, text='Variable specifying how to group samples (optional)').grid(row=40, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var_ret).grid(row=41, columnspan=2, sticky=tk.W)

    # range
    tk.Label(root, text='Choose range for RC index').grid(row=50, columnspan=3, sticky=tk.W)
    RC_range = tk.StringVar(root, '0 to 1')
    range_options = ['0 to 1', '-1 to 1']
    for val, opt in enumerate(range_options):
        tk.Radiobutton(root, text=opt, variable=RC_range, value=opt).grid(row=51+val, sticky=tk.W)

    tk.Label(root, text='-'*100).grid(row=55, columnspan=3, sticky=tk.W)
    def run_null():
        if distmat.get() == 'Select':
            dm = 'None'
        else:
            dm = distmat.get()

        if RC_range.get() == '0 to 1':
            RCrange = 'Raup'
        else:
            RCrange = 'Chase'

        if dis_index.get() == 'Bray-Curtis':
            namn = 'Bray'
        else:
            namn = dis_index.get()

        rcq = RCq(obj, constrainingVar=var_con.get(), randomization=rand_proc.get(), weightingVar=var_wt.get(), weight=wt.get(),
            iterations=var_iter.get(), disIndex=namn, distmat=dm, q=qval.get(), compareVar=var_ret.get(),
            RCrange=RCrange)

        if namn == 'Hill':
            namn2 = namn + str(qval.get())
        else:
            namn2 = namn

        rcq['Nullmean'].to_csv(path + 'Null_mean_' + namn2 + '.csv')
        rcq['Nullstd'].to_csv(path + 'Null_std_' + namn2 + '.csv')
        rcq['Obs'].to_csv(path + 'Obs_' + namn2 + '.csv')
        if var_ret.get() == 'None':
            rcq['RC'].to_csv(path + 'RC_' + namn2 + '.csv')
        else:
            rcq['RCmean'].to_csv(path + 'RC_mean_' + namn2 + '.csv')
            rcq['RCstd'].to_csv(path + 'RC_std_' + namn2 + '.csv')
        return 0

    # Buttons to run model or quit
    tk.Button(root, text='Run null model', command=run_null).grid(row=60, sticky=tk.W)

    def quit():
        master.destroy()
    tk.Button(root, text='Quit', command=quit).grid(row=60, column=1, sticky=tk.W)

def Consensus_object(path):
    # Start GUI window
    master = tk.Toplevel()
    master.title('Make consensus')
    master.geometry('500x700')

    # Create scrollbar, root is the frame the all widgets are later placed on
    def onFrameConfigure(canvas):
        ###Reset the scroll region to encompass the inner frame
        canvas.configure(scrollregion=canvas.bbox("all"))
    canvas = tk.Canvas(master, borderwidth=0, background="#bebebe")
    root = tk.Frame(canvas, background="#bebebe")
    vsb = tk.Scrollbar(master, orient="vertical", command=canvas.yview)
    hsb = tk.Scrollbar(master, orient="horizontal", command=canvas.xview)

    canvas.configure(yscrollcommand=vsb.set)
    canvas.configure(xscrollcommand=hsb.set)

    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")

    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((8, 20), window=root, anchor="nw")

    root.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))

    ##########
    tk.Label(root, text='Select input files').grid(row=1, sticky=tk.W)

    meta_name = tk.StringVar(root, 'None')
    def openMeta():
        meta_name.set(askopenfilename())
    tk.Button(root, text='Meta data', command=openMeta).grid(row=4, sticky=tk.W)
    tk.Label(root, textvariable=meta_name).grid(row=5, column=0, columnspan=2, sticky=tk.W)

    tk.Label(root, text='-'*100).grid(row=6, columnspan=2)

    table_name1 = tk.StringVar(root, 'None')
    def openTable1():
        table_name1.set(askopenfilename())
    tk.Button(root, text='Frequency table 1', command=openTable1).grid(row=10, column=0, sticky=tk.W)
    tk.Label(root, textvariable=table_name1).grid(row=11, column=0, columnspan=2, sticky=tk.W)

    fasta_name1 = tk.StringVar(root, 'None')
    def openFasta1():
        fasta_name1.set(askopenfilename())
    tk.Button(root, text='Fasta file 1', command=openFasta1).grid(row=10, column=1, sticky=tk.W)
    tk.Label(root, textvariable=fasta_name1).grid(row=12, column=1, columnspan=2, sticky=tk.W)

    tk.Label(root, text='-'*100).grid(row=13, columnspan=2)

    table_name2 = tk.StringVar(root, 'None')
    def openTable2():
        table_name2.set(askopenfilename())
    tk.Button(root, text='Frequency table 2', command=openTable2).grid(row=15, column=0, sticky=tk.W)
    tk.Label(root, textvariable=table_name2).grid(row=16, column=0, columnspan=2, sticky=tk.W)

    fasta_name2 = tk.StringVar(root, 'None')
    def openFasta2():
        fasta_name2.set(askopenfilename())
    tk.Button(root, text='Fasta file 2', command=openFasta2).grid(row=15, column=1, sticky=tk.W)
    tk.Label(root, textvariable=fasta_name2).grid(row=17, column=1, columnspan=2, sticky=tk.W)

    tk.Label(root, text='-'*100).grid(row=19, columnspan=2)

    table_name3 = tk.StringVar(root, 'None')
    def openTable3():
        table_name3.set(askopenfilename())
    tk.Button(root, text='Frequency table 3', command=openTable3).grid(row=20, column=0, sticky=tk.W)
    tk.Label(root, textvariable=table_name3).grid(row=21, column=0, columnspan=2, sticky=tk.W)

    fasta_name3 = tk.StringVar(root, 'None')
    def openFasta3():
        fasta_name3.set(askopenfilename())
    tk.Button(root, text='Fasta file 3', command=openFasta3).grid(row=20, column=1, sticky=tk.W)
    tk.Label(root, textvariable=fasta_name3).grid(row=22, column=1, columnspan=2, sticky=tk.W)

    tk.Label(root, text='-'*100).grid(row=23, columnspan=2)

    table_name4 = tk.StringVar(root, 'None')
    def openTable4():
        table_name4.set(askopenfilename())
    tk.Button(root, text='Frequency table 4', command=openTable4).grid(row=25, column=0, sticky=tk.W)
    tk.Label(root, textvariable=table_name4).grid(row=26, column=0, columnspan=2, sticky=tk.W)

    fasta_name4 = tk.StringVar(root, 'None')
    def openFasta4():
        fasta_name4.set(askopenfilename())
    tk.Button(root, text='Fasta file 4', command=openFasta4).grid(row=25, column=1, sticky=tk.W)
    tk.Label(root, textvariable=fasta_name4).grid(row=27, column=1, columnspan=2, sticky=tk.W)

    tk.Label(root, text='-'*100).grid(row=28, columnspan=2)

    table_name5 = tk.StringVar(root, 'None')
    def openTable5():
        table_name5.set(askopenfilename())
    tk.Button(root, text='Frequency table 5', command=openTable5).grid(row=30, column=0, sticky=tk.W)
    tk.Label(root, textvariable=table_name5).grid(row=31, column=0, columnspan=2, sticky=tk.W)

    fasta_name5 = tk.StringVar(root, 'None')
    def openFasta5():
        fasta_name5.set(askopenfilename())
    tk.Button(root, text='Fasta file 5', command=openFasta5).grid(row=30, column=1, sticky=tk.W)
    tk.Label(root, textvariable=fasta_name5).grid(row=32, column=1, columnspan=2, sticky=tk.W)

    tk.Label(root, text='-'*100).grid(row=33, columnspan=2)

    tk.Label(root, text='Which type of separator was used in the table and meta files?').grid(row=35, columnspan=2, sticky=tk.W)
    sep_name = tk.StringVar(root, ',')
    optionsSep = [',', ';', 'tab']
    for val, opt in enumerate(optionsSep):
        tk.Radiobutton(root, text=opt, variable=sep_name, value=opt).grid(row=36+val, sticky=tk.W)

    #Specify path for output files
    path_name_out = tk.StringVar(root, path)
    def openFolder():
        path_name_out.set(askdirectory())
    tk.Button(root, text='Folder for output files', command=openFolder).grid(row=40, column=0, sticky=tk.W)
    tk.Label(root, textvariable=path_name_out).grid(row=41, column=0, sticky=tk.W)

    tk.Label(root, text='-'*100).grid(row=48, columnspan=2)

    # Make consensus object
    def consensusObj():
        tablist = [table_name1.get(), table_name2.get(), table_name3.get(), table_name4.get(), table_name5.get()]
        fastalist = [fasta_name1.get(), fasta_name2.get(), fasta_name3.get(), fasta_name4.get(), fasta_name5.get()]
        if sep_name.get() == 'tab':
            sep = '\t'
        else:
            sep = sep_name.get()

        objlist = []
        for nr in range(len(tablist)):
            t = tablist[nr]
            f = fastalist[nr]
            if t != 'None' and f != 'None':
                objlist.append(loadFiles(tab=t, fasta=f, meta=meta_name.get(), sep=sep))
        cons = makeConsensusObject(objlist)
        returnFiles(cons, path=path_name_out.get() + '/', savename='Consensus', sep=sep)

    tk.Button(root, text='Make consensus', command=consensusObj).grid(row=50, column=0, sticky=tk.W)

    def quit():
        master.destroy()
    tk.Button(root, text='Quit', command=quit).grid(row=50, column=1, sticky=tk.W)

def startwindow():

    # Functions that specify what to do with input choices
    def choose():
        if sep_name.get() == 'tab':
            separator = '\t'
        else:
            separator = sep_name.get()

        obj = loadFiles(path='', tab=table_name.get(), fasta=seq_name.get(), meta=meta_name.get(), sep=separator)

        if v.get() == 'Subset_data':
            Subsetting(obj, path_name.get()+'/')
        elif v.get() == 'Calculate_phyl_dist':
            Calc_phyl_dist(obj, path_name.get() + '/')
        elif v.get() == 'Heatmap':
            Heatmap(obj, path_name.get()+'/')
        elif v.get() == 'Alpha_div':
            Alpha_div(obj, path_name.get()+'/')
        elif v.get() == 'Beta_div':
            Beta_div(obj, path_name.get()+'/')
        elif v.get() == 'PCoA':
            Plot_PCoA(obj, path_name.get() + '/')
        elif v.get() == 'Null_model':
            Null_model(obj, path_name.get() + '/')
        elif v.get() == 'Make_consensus':
            Consensus_object(path_name.get() + '/')


    def quit():
        master.destroy()

    # Start GUI window
    master = tk.Tk()
    master.title('Start window')
    master.geometry('500x600')

    # Create scrollbar, root is the frame the all widgets are later placed on
    def onFrameConfigure(canvas):
        ###Reset the scroll region to encompass the inner frame
        canvas.configure(scrollregion=canvas.bbox("all"))
    canvas = tk.Canvas(master, borderwidth=0, background="#bebebe")
    root = tk.Frame(canvas, background="#bebebe")
    vsb = tk.Scrollbar(master, orient="vertical", command=canvas.yview)
    hsb = tk.Scrollbar(master, orient="horizontal", command=canvas.xview)

    canvas.configure(yscrollcommand=vsb.set)
    canvas.configure(xscrollcommand=hsb.set)

    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")

    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((8, 20), window=root, anchor="nw")

    root.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))

    ##########
    tk.Label(root, text='Select input files').grid(row=1, sticky=tk.W)

    table_name = tk.StringVar(root, 'None')
    def openTable():
        table_name.set(askopenfilename())
    tk.Button(root, text='Frequency table', command=openTable).grid(row=2, sticky=tk.W)
    tk.Label(root, textvariable=table_name).grid(row=2, column=1, sticky=tk.W)

    seq_name = tk.StringVar(root, 'None')
    def openSeq():
        seq_name.set(askopenfilename())
    tk.Button(root, text='Fasta file', command=openSeq).grid(row=3, sticky=tk.W)
    tk.Label(root, textvariable=seq_name).grid(row=3, column=1, sticky=tk.W)

    meta_name = tk.StringVar(root, 'None')
    def openMeta():
        meta_name.set(askopenfilename())
    tk.Button(root, text='Meta data', command=openMeta).grid(row=4, sticky=tk.W)
    tk.Label(root, textvariable=meta_name).grid(row=4, column=1, sticky=tk.W)

    tk.Label(root, text='Which type of separator was used in the table and meta files?').grid(row=5, columnspan=2, sticky=tk.W)
    sep_name = tk.StringVar(root, ',')
    optionsSep = [',', ';', 'tab']
    for val, opt in enumerate(optionsSep):
        tk.Radiobutton(root, text=opt, variable=sep_name, value=opt).grid(row=10+val, sticky=tk.W)

    #Specify path for output files
    path_name = tk.StringVar(root, ' ')
    def openFolder():
        path_name.set(askdirectory())
    tk.Button(root, text='Folder for output files', command=openFolder).grid(row=16, sticky=tk.W)
    tk.Label(root, textvariable=path_name).grid(row=16, column=1, sticky=tk.W)

    #Print statistics about the data
    def stats():
        if sep_name.get() == 'tab':
            separator = '\t'
        else:
            separator = sep_name.get()
        obj = loadFiles(path='', tab=table_name.get(), fasta=seq_name.get(), meta=meta_name.get(), sep=separator)
        tab = obj['tab']
        totalreads = sum(tab.sum())
        totalsvs = len(tab.index)
        totalsmps = len(tab.columns)
        minreads = min(tab.sum())
        maxreads = max(tab.sum())

        rootStats = tk.Tk()
        rootStats.title('Stats')
        tk.Label(rootStats, text='Total number of samples').grid(row=1, column=0, sticky=tk.W)
        tk.Label(rootStats, text=str(totalsmps)).grid(row=1, column=1, sticky=tk.W)
        tk.Label(rootStats, text='Total number of SVs or OTUs').grid(row=3, column=0, sticky=tk.W)
        tk.Label(rootStats, text=str(totalsvs)).grid(row=3, column=1, sticky=tk.W)
        tk.Label(rootStats, text='Total number of reads').grid(row=5, column=0, sticky=tk.W)
        tk.Label(rootStats, text=str(totalreads)).grid(row=5, column=1, sticky=tk.W)
        tk.Label(rootStats, text='---').grid(row=7, column=0, sticky=tk.W)

        tk.Label(rootStats, text='Minimum number of reads in a sample').grid(row=10, column=0, sticky=tk.W)
        tk.Label(rootStats, text=str(minreads)).grid(row=10, column=1, sticky=tk.W)
        tk.Label(rootStats, text='Maximum number of reads in a sample').grid(row=12, column=0, sticky=tk.W)
        tk.Label(rootStats, text=str(maxreads)).grid(row=12, column=1, sticky=tk.W)
        tk.Label(rootStats, text='---').grid(row=15, column=0, sticky=tk.W)

        if 'meta' in obj.keys():
            meta = obj['meta']
            headings = meta.columns.tolist()
            tk.Label(rootStats, text='Headings in metadata').grid(row=20, column=0, sticky=tk.W)
            for val, hd in enumerate(headings):
                tk.Label(rootStats, text=hd).grid(row=21+val, column=0, sticky=tk.W)

    tk.Label(root, text='-'*100).grid(row=18, columnspan=2, sticky=tk.W)
    tk.Label(root, text='Print statistics about the data').grid(row=19, columnspan=2, sticky=tk.W)
    tk.Button(root, text='Press for stats', command=stats).grid(row=20, columnspan=2, sticky=tk.W)

    # Choices of analysis
    tk.Label(root, text='-'*100).grid(row=30, columnspan=2, sticky=tk.W)
    tk.Label(root, text='Choose a task').grid(row=31, columnspan=2, sticky=tk.W)

    v = tk.StringVar()
    options = ['Subset_data', 'Calculate_phyl_dist', 'Heatmap', 'Alpha_div', 'Beta_div', 'PCoA', 'Null_model', 'Make_consensus']
    for val, opt in enumerate(options):
        tk.Radiobutton(root, text=opt, variable=v, value=opt).grid(row=40+val, sticky=tk.W)

    # Buttons that connects to functions
    tk.Label(root, text='-'*100).grid(row=51, columnspan=2, sticky=tk.W)
    tk.Button(root, text='Choose', command=choose).grid(row=52, sticky=tk.W)
    tk.Button(root, text='Quit', command=quit).grid(row=52, column=1, sticky=tk.W)

    root.mainloop()

startwindow()


