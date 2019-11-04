import tkinter as tk
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename
import pandas as pd

#Set background color for all windows
bgcol = '#d8d8d8'

def Consensus_object(path):
    # Start GUI window
    master = tk.Toplevel()
    master.title('Make consensus')
    master.geometry('500x700')
    # Create scrollbar, root is the frame the all widgets are later placed on
    def onFrameConfigure(canvas):
        ###Reset the scroll region to encompass the inner frame
        canvas.configure(scrollregion=canvas.bbox("all"))
    canvas = tk.Canvas(master, borderwidth=0, background=bgcol)
    root = tk.Frame(canvas, background=bgcol)
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
    tk.Label(root, text='Select input files (up to four different tables, meta data should be the same for all)', bg=bgcol).grid(row=1, columnspan=3, sticky=tk.W)

    meta_name = tk.StringVar(root, 'None')
    def openMeta():
        meta_name.set(askopenfilename())
    tk.Button(root, text='Meta data', command=openMeta).grid(row=4, column=0, sticky=tk.W)
    tk.Label(root, textvariable=meta_name, bg=bgcol).grid(row=4, column=1, columnspan=2, sticky=tk.W)
    tk.Label(root, text='-'*80, bg=bgcol).grid(row=6, columnspan=3)
    #------------
    diffL = tk.IntVar()
    tk.Label(root, text='Check this box if the same SV could have different lengths', bg=bgcol).grid(row=7, columnspan=3, sticky=tk.W)
    tk.Checkbutton(root, text='Different lengths', var=diffL, bg=bgcol).grid(row=8, columnspan=2, sticky=tk.W)
    tk.Label(root, text='-'*80, bg=bgcol).grid(row=9, columnspan=3)
    #------------
    table_name1 = tk.StringVar(root, 'None')
    def openTable1():
        table_name1.set(askopenfilename())
        root.lift()
    tk.Button(root, text='Frequency table 1', command=openTable1).grid(row=10, column=0, sticky=tk.W)
    tk.Label(root, textvariable=table_name1, bg=bgcol).grid(row=11, column=0, columnspan=2, sticky=tk.W)

    fasta_name1 = tk.StringVar(root, 'None')
    def openFasta1():
        fasta_name1.set(askopenfilename())
    tk.Button(root, text='Fasta file 1', command=openFasta1).grid(row=10, column=1, sticky=tk.W)
    tk.Label(root, textvariable=fasta_name1, bg=bgcol).grid(row=12, column=1, columnspan=2, sticky=tk.W)

    tk.Label(root, text='-'*80, bg=bgcol).grid(row=13, columnspan=2)

    table_name2 = tk.StringVar(root, 'None')
    def openTable2():
        table_name2.set(askopenfilename())
    tk.Button(root, text='Frequency table 2', command=openTable2).grid(row=15, column=0, sticky=tk.W)
    tk.Label(root, textvariable=table_name2, bg=bgcol).grid(row=16, column=0, columnspan=2, sticky=tk.W)

    fasta_name2 = tk.StringVar(root, 'None')
    def openFasta2():
        fasta_name2.set(askopenfilename())
    tk.Button(root, text='Fasta file 2', command=openFasta2).grid(row=15, column=1, sticky=tk.W)
    tk.Label(root, textvariable=fasta_name2, bg=bgcol).grid(row=17, column=1, columnspan=2, sticky=tk.W)

    tk.Label(root, text='-'*80, bg=bgcol).grid(row=19, columnspan=2)

    table_name3 = tk.StringVar(root, 'None')
    def openTable3():
        table_name3.set(askopenfilename())
    tk.Button(root, text='Frequency table 3', command=openTable3).grid(row=20, column=0, sticky=tk.W)
    tk.Label(root, textvariable=table_name3, bg=bgcol).grid(row=21, column=0, columnspan=2, sticky=tk.W)

    fasta_name3 = tk.StringVar(root, 'None')
    def openFasta3():
        fasta_name3.set(askopenfilename())
    tk.Button(root, text='Fasta file 3', command=openFasta3).grid(row=20, column=1, sticky=tk.W)
    tk.Label(root, textvariable=fasta_name3, bg=bgcol).grid(row=22, column=1, columnspan=2, sticky=tk.W)

    tk.Label(root, text='-'*80, bg=bgcol).grid(row=23, columnspan=2)

    table_name4 = tk.StringVar(root, 'None')
    def openTable4():
        table_name4.set(askopenfilename())
    tk.Button(root, text='Frequency table 4', command=openTable4).grid(row=25, column=0, sticky=tk.W)
    tk.Label(root, textvariable=table_name4, bg=bgcol).grid(row=26, column=0, columnspan=2, sticky=tk.W)

    fasta_name4 = tk.StringVar(root, 'None')
    def openFasta4():
        fasta_name4.set(askopenfilename())
    tk.Button(root, text='Fasta file 4', command=openFasta4).grid(row=25, column=1, sticky=tk.W)
    tk.Label(root, textvariable=fasta_name4, bg=bgcol).grid(row=27, column=1, columnspan=2, sticky=tk.W)

    tk.Label(root, text='-'*80, bg=bgcol).grid(row=28, columnspan=2)

    tk.Label(root, text='Which type of separator was used in the table and meta files?', bg=bgcol).grid(row=35, columnspan=3, sticky=tk.W)
    sep_name = tk.StringVar(root, ',')
    optionsSep = [',', ';', 'tab']
    for val, opt in enumerate(optionsSep):
        tk.Radiobutton(root, text=opt, variable=sep_name, value=opt, bg=bgcol).grid(row=36+val, sticky=tk.W)

    #Specify path for output files
    path_name_out = tk.StringVar(root, path)
    def openFolder():
        path_name_out.set(askdirectory())
    tk.Button(root, text='Change folder for output files?', command=openFolder).grid(row=40, column=0, sticky=tk.W)
    tk.Label(root, textvariable=path_name_out, bg=bgcol).grid(row=40, column=1, sticky=tk.W)

    tk.Label(root, text='-'*80, bg=bgcol).grid(row=48, columnspan=3)

    # Make consensus object
    def consensusObj():
        from . import files
        from . import subset

        tablist = [table_name1.get(), table_name2.get(), table_name3.get(), table_name4.get()]
        fastalist = [fasta_name1.get(), fasta_name2.get(), fasta_name3.get(), fasta_name4.get()]
        if sep_name.get() == 'tab':
            sep = '\t'
        else:
            sep = sep_name.get()

        if diffL.get() == 1:
            differentLengths = True
        else:
            differentLengths = False

        objlist = []
        for nr in range(len(tablist)):
            t = tablist[nr]
            f = fastalist[nr]
            if t != 'None' and f != 'None':
                objlist.append(files.load(tab=t, fasta=f, meta=meta_name.get(), sep=sep))
        cons, info = subset.consensus(objlist, differentLengths=differentLengths)
        files.printout(cons, path=path_name_out.get() + '/', savename='Consensus', sep=sep)

        # Window to show info on consensus object
        tk.Label(root, text='-'*80, bg=bgcol).grid(row=59, columnspan=2)
        tk.Label(root, text='Information about consensus object', bg=bgcol).grid(row=60, columnspan=3, sticky=tk.W)
        tk.Label(root, text='Total relative abundance of in common SVs in each freq. table:', bg=bgcol).grid(row=61, columnspan=3, sticky=tk.W)
        listbox1 = tk.Listbox(root, width=20, height=4)
        listbox1.grid(row=62, sticky=tk.W)
        for item in info['ra_in_tab']:
            listbox1.insert(tk.END, str(round(item, 2)) + ' %')
        tk.Label(root, text='Maximum relative abundance of not in common SVs in a sample in each freq. table:', bg=bgcol).grid(row=67, columnspan=3, sticky=tk.W)
        listbox2 = tk.Listbox(root, width=20, height=4)
        listbox2.grid(row=71, sticky=tk.W)
        for item in info['ra_sample_max']:
            listbox2.insert(tk.END, str(round(item, 2)) + ' %')
        tk.Label(root, text='The freq. table with the highest relative abundance of reads associated with the in common SVs', bg=bgcol).grid(row=75, columnspan=3, sticky=tk.W)
        tk.Label(root, text='was subsetted to the in common SVs and returned as consensus table.', bg=bgcol).grid(row=76, columnspan=3, sticky=tk.W)
        tk.Label(root, text='If taxonomic information was available in any of the freq. tables, that information was returned as well', bg=bgcol).grid(row=77, columnspan=3, sticky=tk.W)

    tk.Button(root, text='Make consensus', command=consensusObj).grid(row=50, column=0, sticky=tk.W)

    def quit():
        master.destroy()
    tk.Button(root, text='Quit', command=quit).grid(row=50, column=1, sticky=tk.W)

def Subsetting(obj, path):

    # Create GUI window
    master = tk.Toplevel()
    master.title('Subsetting')
    master.geometry('500x600')

    # Create scrollbar, root is the frame the all widgets are later placed on
    def onFrameConfigure(canvas):
        ###Reset the scroll region to encompass the inner frame
        canvas.configure(scrollregion=canvas.bbox("all"))
    canvas = tk.Canvas(master, borderwidth=0, background=bgcol)
    root = tk.Frame(canvas, background=bgcol)
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
    tk.Label(root, text='Rarefy', bg=bgcol).pack(anchor=tk.W)
    tk.Label(root, text='Write count (if "min", the frequency table will be rarefied to the smallest sample', bg=bgcol).pack(anchor=tk.W)
    tk.Entry(root, textvariable=rarecount).pack(anchor=tk.W)

    def rarefycounts():
        from . import files
        from . import subset

        if rarecount.get() != 'min':
            rcount = int(rarecount.get())
        else:
            rcount = rarecount.get()
        rtab = subset.rarefy_table(obj['tab'], depth=rcount)
        rtab = rtab[rtab.sum(axis=1) > 0]
        obj['tab'] = rtab
        obj_sub = subset.sequences(obj, rtab.index.tolist())
        files.printout(obj_sub, path=path, sep=',', savename='Rarefied')
    tk.Button(root, text='Rarefy and save files', command=rarefycounts).pack(anchor=tk.W)
    tk.Label(root, text='-'*100, bg=bgcol).pack(anchor=tk.W)

    # Top SVs
    top_svs = tk.IntVar()
    tk.Label(root, text='Subset to most abundant sequences', bg=bgcol).pack(anchor=tk.W)
    tk.Label(root, text='Write number of sequences to keep', bg=bgcol).pack(anchor=tk.W)
    tk.Entry(root, textvariable=top_svs).pack(anchor=tk.W)

    def subset_top_svs():
        from . import files
        from . import subset

        obj_sub = subset.abundant_sequences(obj, top_svs.get())
        files.printout(obj_sub, path=path, sep=',', savename='TopSeqs')
    tk.Button(root, text='Subset and save files', command=subset_top_svs).pack(anchor=tk.W)
    tk.Label(root, text='-'*100, bg=bgcol).pack(anchor=tk.W)

    # Merge samples
    meta_h = tk.StringVar()
    tk.Label(root, text='Merge sample based on metadata column heading', bg=bgcol).pack(anchor=tk.W)
    tk.Label(root, text='Write column heading', bg=bgcol).pack(anchor=tk.W)
    tk.Entry(root, textvariable=meta_h).pack(anchor=tk.W)

    def merge_smps():
        from . import files
        from . import subset

        obj_sub = subset.merge_samples(obj, var=meta_h.get())
        files.printout(obj_sub, path=path, sep=',', savename='Merged')
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
    canvas = tk.Canvas(master, borderwidth=0, background=bgcol)
    root = tk.Frame(canvas, background=bgcol)
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
    tk.Label(root, text='Specify name of the file to be saved (e.g. phyl_dist', bg=bgcol).pack(anchor=tk.W)
    tk.Label(root, text='(No need to add .csv, it will be added automatically)', bg=bgcol).pack(anchor=tk.W)
    tk.Entry(root, textvariable=distmat_name).pack(anchor=tk.W)

    def save_func():
        from . import diversity

        savename = distmat_name.get()
        diversity.sequence_comparison(obj['seq'], savename=path+savename)

    tk.Label(root, text='The calculation may take quite long time', bg=bgcol).pack(anchor=tk.W)
    tk.Button(root, text='Calculate and save file', command=save_func).pack(anchor=tk.W)
    tk.Label(root, text='-'*70, bg=bgcol).pack(anchor=tk.W)

    def quit():
        master.destroy()
    tk.Button(root, text='Quit', command=quit).pack(anchor=tk.W)

def Heatmap(obj, path):
    def run():
        from . import plot

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

        plot.heatmap(obj, xAxis=var.get(), levels=levels, subsetLevels=stringlevels, subsetPatterns=stringpatterns,
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
    canvas = tk.Canvas(master, borderwidth=0, background=bgcol)
    root = tk.Frame(canvas, background=bgcol)
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

    tk.Label(root, text='Various input options for heatmap', bg=bgcol).grid(row=0, columnspan=4, sticky=tk.W)
    tk.Label(root, text='-'*90, bg=bgcol).grid(row=1, columnspan=3, sticky=tk.W)

    # Input taxonomic levels
    options = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    v = []
    for i in range(len(options)):
        v.append(tk.IntVar())
    tk.Label(root, text='Choose one or two taxonomic levels to include on the y-axis', bg=bgcol).grid(row=5, columnspan=4, sticky=tk.W)
    tk.Label(root, text='Sequences are grouped based on the lowest taxonomic level', bg=bgcol).grid(row=6, columnspan=4, sticky=tk.W)
    for val, opt in enumerate(options):
        if val < 4:
            colnr = 0
            rownr = val
        else:
            colnr = 1
            rownr = val-4
        tk.Checkbutton(root, text=opt, variable=v[val], bg=bgcol).grid(row=7+rownr, column=colnr, sticky=tk.W)
    tk.Label(root, text='-'*90, bg=bgcol).grid(row=14, columnspan=3, sticky=tk.W)

    # xAxis
    var = tk.StringVar(root, 'None')
    tk.Label(root, text='Enter metadata column for x-axis labels', bg=bgcol).grid(row=15, columnspan=4, sticky=tk.W)
    tk.Entry(root, textvariable=var).grid(row=16, sticky=tk.W)

    #Order
    order = tk.StringVar(root, 'None')
    tk.Label(root, text='Specify metadata column used to order the samples on the x-axis', bg=bgcol).grid(row=20, columnspan=4, sticky=tk.W)
    tk.Entry(root, textvariable=order).grid(row=21, sticky=tk.W)

    #Number to plot
    number = tk.IntVar(root, 20)
    tk.Label(root, text='Specify number of taxa to include in heatmap', bg=bgcol).grid(row=23, columnspan=4, sticky=tk.W)
    tk.Entry(root, textvariable=number, width=10).grid(row=24, sticky=tk.W)

    #nameType
    nametype = tk.StringVar()
    nametype.set('SV')
    tk.Label(root, text='Specify how unclassified taxa should be named (e.g. SV or OTU)?', bg=bgcol).grid(row=26, columnspan=4, sticky=tk.W)
    tk.Entry(root, textvariable=nametype, width=10).grid(row=27, sticky=tk.W)

    #Figure dimensions
    tk.Label(root, text='-'*90, bg=bgcol).grid(row=30, columnspan=3, sticky=tk.W)

    #figSize
    tk.Label(root, text='Specify figure dimensions and text size', bg=bgcol).grid(row=31, columnspan=4, sticky=tk.W)
    figSizeW = tk.IntVar(root, 14)
    figSizeH = tk.IntVar(root, 10)
    tk.Label(root, text='Width', bg=bgcol).grid(row=32, sticky=tk.E)
    tk.Entry(root, textvariable=figSizeW).grid(row=32, column=1, sticky=tk.W)
    tk.Label(root, text='Height', bg=bgcol).grid(row=33, sticky=tk.E)
    tk.Entry(root, textvariable=figSizeH).grid(row=33, column=1, sticky=tk.W)

    #FontSize
    fontSize = tk.IntVar(root, 15)
    tk.Label(root, text='Axis text font size', bg=bgcol).grid(row=35, sticky=tk.E)
    tk.Entry(root, textvariable=fontSize).grid(row=35, column=1, sticky=tk.W)

    #sepCol
    tk.Label(root, text='-'*90, bg=bgcol).grid(row=36, columnspan=4, sticky=tk.W)
    sepCol = tk.StringVar()
    tk.Label(root, text='Group samples. Insert numbers of samples after which a blank column should be inserted.', bg=bgcol).grid(row=37, columnspan=4, sticky=tk.W)
    tk.Entry(root, textvariable=sepCol).grid(row=38, sticky=tk.W)
    tk.Label(root, text='(separate values by commas)', bg=bgcol).grid(row=38, column=1, sticky=tk.W)

    #Data labels
    tk.Label(root, text='-'*90, bg=bgcol).grid(row=40, columnspan=4, sticky=tk.W)
    tk.Label(root, text='Information about data labels', bg=bgcol).grid(row=41, sticky=tk.W)

    tk.Label(root, text='Do you want to include data labels in heatmap', bg=bgcol).grid(row=42, columnspan=4, sticky=tk.W)
    labeloptions = ['Yes', 'No']
    useLabels = tk.StringVar(root, 'Yes')
    for val, opt in enumerate(labeloptions):
        tk.Radiobutton(root, text=opt, variable=useLabels, value=opt, bg=bgcol).grid(row=43, column=val, sticky=tk.W)

    labelSize = tk.IntVar(root, 12)
    tk.Label(root, text='Label font size', bg=bgcol).grid(row=45, sticky=tk.W)
    tk.Entry(root, textvariable=labelSize, width=10).grid(row=46, sticky=tk.W)

    ctresh = tk.IntVar(root, 10)
    tk.Label(root, text='Percent relative abundance at which the label text shifts from black to white', bg=bgcol).grid(row=48, columnspan=4, sticky=tk.W)
    tk.Entry(root, textvariable=ctresh, width=10).grid(row=49, sticky=tk.W)
    tk.Label(root, text='%', bg=bgcol).grid(row=49, column=1, sticky=tk.W)

    #Coloring
    tk.Label(root, text='-'*90, bg=bgcol).grid(row=50, columnspan=3, sticky=tk.W)
    tk.Label(root, text='Color of heatmap', bg=bgcol).grid(row=51, columnspan=4, sticky=tk.W)
    colmap = tk.StringVar(root, 'Reds')
    tk.Label(root, text='Colormap', bg=bgcol).grid(row=52, sticky=tk.E)
    tk.Entry(root, textvariable=colmap).grid(row=52, column=1, sticky=tk.W)
    tk.Label(root, text='(see available colormaps in python)', bg=bgcol).grid(row=52, column=2, columnspan=2, sticky=tk.W)

    colgamma = tk.DoubleVar(root, 0.5)
    tk.Label(root, text='Linearity of colormap', bg=bgcol).grid(row=55, sticky=tk.E)
    tk.Entry(root, textvariable=colgamma).grid(row=55, column=1, sticky=tk.W)
    tk.Label(root, text='(1=linear change in color)', bg=bgcol).grid(row=55, column=2, columnspan=2, sticky=tk.W)

    tk.Label(root, text='If you want a colorbar showing the scale, specify tick marks on the bar', bg=bgcol).grid(row=58, columnspan=4, sticky=tk.W)
    usecbar = tk.StringVar(root, 'None')
    tk.Entry(root, textvariable=usecbar).grid(row=59, sticky=tk.W)
    tk.Label(root, text='(the values should be separated by comma)', bg=bgcol).grid(row=59, column=1, columnspan=3, sticky=tk.W)

    # Input subset based on string patterns
    stringlev = []
    for i in range(len(options)):
        stringlev.append(tk.IntVar())
    tk.Label(root, text='-'*90, bg=bgcol).grid(row=60, columnspan=3, sticky=tk.W)
    tk.Label(root, text='Subset data based on text patterns', bg=bgcol).grid(row=61, columnspan=4, sticky=tk.W)
    tk.Label(root, text='Choose which taxonomic levels to search for text', bg=bgcol).grid(row=62, columnspan=4, sticky=tk.W)
    for val, opt in enumerate(options):
        if val < 4:
            colnr = 0
            rownr = val
        else:
            colnr = 1
            rownr = val-4
        tk.Checkbutton(root, text=opt, variable=stringlev[val], bg=bgcol).grid(row=63+rownr, column=colnr, sticky=tk.W)

    stringpattern = tk.StringVar()
    tk.Label(root, text='Enter words to search for, separate by comma', bg=bgcol).grid(row=70, columnspan=4, sticky=tk.W)
    tk.Entry(root, textvariable=stringpattern, width=80).grid(row=71, columnspan=4, sticky=tk.W)
    tk.Label(root, text='-'*90, bg=bgcol).grid(row=72, columnspan=4, sticky=tk.W)

    # Buttons to run functions
    tk.Button(root, text='Plot heatmap', command=run).grid(row=80)
    tk.Button(root, text='Quit', command=quit).grid(row=80, column=1)
    tk.Label(root, text='-'*90, bg=bgcol).grid(row=82, columnspan=4, sticky=tk.W)

    root.mainloop()

def Rarefaction_curve(obj, path):
    def run():
        from . import plot

        if step.get() != 'flexible':
            stepin = int(step.get())
        else:
            stepin = 'flexible'

        plot.rarefactioncurve(obj, step=stepin, figSize=(figSizeW.get(), figSizeH.get()), fontSize=fontSize.get(), 
                             var=var.get(), order=order.get(), tag=tag.get(), colorlist='None',
                             onlyReturnData=False, onlyPlotData='None', savename=path + 'Rarefaction_curve')

    def quit():
        master.destroy()

    # Create GUI window
    master = tk.Toplevel()
    master.title('Rarefaction_curve')
    master.geometry('600x700')
    # Create scrollbar, root is the frame the all widgets are later placed on
    def onFrameConfigure(canvas):
        ###Reset the scroll region to encompass the inner frame
        canvas.configure(scrollregion=canvas.bbox("all"))
    canvas = tk.Canvas(master, borderwidth=0, background=bgcol)
    root = tk.Frame(canvas, background=bgcol)
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

    tk.Label(root, text='Various input options for rarefaction curve', bg=bgcol).grid(row=0, columnspan=4, sticky=tk.W)
    tk.Label(root, text='-'*70, bg=bgcol).grid(row=1, columnspan=3, sticky=tk.W)

    # Input step size
    step = tk.StringVar(root, 'flexible')
    tk.Label(root, text='Enter step size', bg=bgcol).grid(row=5, sticky=tk.W)
    tk.Entry(root, textvariable=step).grid(row=5, column=1, sticky=tk.W)
    tk.Label(root, text='-'*70, bg=bgcol).grid(row=7, columnspan=3, sticky=tk.W)

    #figSize
    tk.Label(root, text='Specify figure dimensions and text size', bg=bgcol).grid(row=10, columnspan=3, sticky=tk.W)
    figSizeW = tk.IntVar(root, 14)
    figSizeH = tk.IntVar(root, 10)
    tk.Label(root, text='Width', bg=bgcol).grid(row=11, sticky=tk.E)
    tk.Entry(root, textvariable=figSizeW).grid(row=11, column=1, sticky=tk.W)
    tk.Label(root, text='Height', bg=bgcol).grid(row=12, sticky=tk.E)
    tk.Entry(root, textvariable=figSizeH).grid(row=12, column=1, sticky=tk.W)

    #FontSize
    fontSize = tk.IntVar(root, 15)
    tk.Label(root, text='Font size', bg=bgcol).grid(row=15, sticky=tk.E)
    tk.Entry(root, textvariable=fontSize).grid(row=15, column=1, sticky=tk.W)
    tk.Label(root, text='-'*70, bg=bgcol).grid(row=17, columnspan=3, sticky=tk.W)

    # var
    var = tk.StringVar(root, 'None')
    tk.Label(root, text='Enter metadata column for color labels', bg=bgcol).grid(row=20, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var).grid(row=21, sticky=tk.W)

    # Logic order
    order = tk.StringVar(root, 'None')
    tk.Label(root, text='Meta data column used to order the samples in the legend', bg=bgcol).grid(row=25, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=order).grid(row=26, sticky=tk.W)

    # tag
    tag = tk.StringVar(root, 'None')
    tk.Label(root, text='Meta data column used to label lines in plot', bg=bgcol).grid(row=30, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=tag).grid(row=31, sticky=tk.W)
    tk.Label(root, text='-'*70, bg=bgcol).grid(row=35, columnspan=3, sticky=tk.W)

    # Buttons to run functions
    tk.Button(root, text='Plot rarefaction curve', command=run).grid(row=80)
    tk.Button(root, text='Quit', command=quit).grid(row=80, column=1)

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
    canvas = tk.Canvas(master, borderwidth=0, background=bgcol)
    root = tk.Frame(canvas, background=bgcol)
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

    tk.Label(root, text='Show alpha diversity in plots or save as data files', bg=bgcol).pack(anchor=tk.W)
    tk.Label(root, text='-'*70, bg=bgcol).pack(anchor=tk.W)

    # Input distmat
    tk.Label(root, text='Are you working with phylogenetic diversity?', bg=bgcol).pack(anchor=tk.W)
    tk.Label(root, text='If so, select distance matrix file', bg=bgcol).pack(anchor=tk.W)
    distmat = tk.StringVar(root, 'Select')
    def openDistmat():
        distmat.set(askopenfilename())
    tk.Button(root, textvariable=distmat, command=openDistmat).pack(anchor=tk.W)
    def resetNone():
        distmat.set('Select')
    tk.Button(root, text='Reset selection', command=resetNone).pack(anchor=tk.W)

    #Plotting
    def run_plot():
        from . import plot

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

        plot.alpha_diversity(obj, distmat=fulldistmat, var=var_col.get(), slist='All', order=order.get(), ylog=ylog,
                     colorlist='None', savename=path + name2save)

    tk.Label(root, text='-'*70, bg=bgcol).pack(anchor=tk.W)
    tk.Label(root, text='The following input is used for plotting figure...', bg=bgcol).pack(anchor=tk.W)
    tk.Label(root, text='. '*50, bg=bgcol).pack(anchor=tk.W)

    # var to use for color coding
    var_col = tk.StringVar()
    var_col.set('None')
    tk.Label(root, text='Specify metadata column heading to use for color coding', bg=bgcol).pack(anchor=tk.W)
    tk.Entry(root, textvariable=var_col).pack(anchor=tk.W)

    #Order
    order = tk.StringVar()
    order.set('None')
    tk.Label(root, text='Specify metadata column used to order the samples on the x-axis', bg=bgcol).pack(anchor=tk.W)
    tk.Entry(root, textvariable=order).pack(anchor=tk.W)

    #Semi log y-axis
    options = ['Yes', 'No']
    tk.Label(root, text='Use logarithmic y-axis?', bg=bgcol).pack(anchor=tk.W)
    y_v = tk.StringVar()
    for opt in options:
        tk.Radiobutton(root, text=opt, variable=y_v, value=opt, bg=bgcol).pack(anchor=tk.W)

    # Buttons to plot
    tk.Button(root, text='Plot alpha diversity', command=run_plot).pack(anchor=tk.W)

    ## Printing
    def run_print():
        from . import diversity

        qlist = qvalues.get().replace(' ', '')
        qlist = qlist.split(',')
        qnumbers = []
        for q in qlist:
            qnumbers.append(float(q))

        output = pd.DataFrame(0, index=obj['tab'].columns, columns=qnumbers)
        for q in qnumbers:
            if distmat.get() == 'Select':
                alfa = diversity.naive_alpha(obj['tab'], q=q)
                output[q] = alfa
            else:
                dist = pd.read_csv(distmat.get(), index_col=0)
                alfa = diversity.phyl_alpha(obj['tab'], distmat=dist, q=q)
                output[q] = alfa
        output.to_csv(path + sname.get() + '.csv')

    tk.Label(root, text='-'*70, bg=bgcol).pack(anchor=tk.W)
    tk.Label(root, text='The following input is used to save a csv file with data', bg=bgcol).pack(anchor=tk.W)
    tk.Label(root, text='. '*50, bg=bgcol).pack(anchor=tk.W)

    tk.Label(root, text='Specify diversity orders to calculate, use comma to separate numbers', bg=bgcol).pack(anchor=tk.W)
    qvalues = tk.StringVar()
    tk.Entry(root, textvariable=qvalues).pack(anchor=tk.W)

    tk.Label(root, text='Specify name of saved file (do not include .csv)', bg=bgcol).pack(anchor=tk.W)
    sname = tk.StringVar()
    tk.Entry(root, textvariable=sname).pack(anchor=tk.W)

    # Buttons to save
    tk.Button(root, text='Save alpha diversity data as file', command=run_print).pack(anchor=tk.W)
    tk.Label(root, text='-'*70, bg=bgcol).pack(anchor=tk.W)

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
    canvas = tk.Canvas(master, borderwidth=0, background=bgcol)
    root = tk.Frame(canvas, background=bgcol)
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
    tk.Label(root, text='Calculate pairwise dissimilarity matrix', bg=bgcol).grid(row=0, columnspan=3, sticky=tk.W)
    tk.Label(root, text='-'*50, bg=bgcol).grid(row=1, columnspan=3, sticky=tk.W)

    # Input distmat
    tk.Label(root, text='Are you working with phylogenetic diversity?', bg=bgcol).grid(row=5, columnspan=3, sticky=tk.W)
    tk.Label(root, text='If so, select distance matrix file', bg=bgcol).grid(row=6, columnspan=3, sticky=tk.W)
    distmat = tk.StringVar(root, 'Select')
    def openDistmat():
        distmat.set(askopenfilename())
    tk.Button(root, textvariable=distmat, command=openDistmat).grid(row=7, columnspan=3, sticky=tk.W)
    def resetNone():
        distmat.set('Select')
    tk.Button(root, text='Reset selection', command=resetNone).grid(row=8, columnspan=2, sticky=tk.W)

    # Calculate dis matrix
    tk.Label(root, text='-'*50, bg=bgcol).grid(row=10, columnspan=3, sticky=tk.W)
    tk.Label(root, text='Calculate dissimilarity matrix', bg=bgcol).grid(row=11, columnspan=3, sticky=tk.W)

    tk.Label(root, text='Choose index', bg=bgcol).grid(row=15, columnspan=3, sticky=tk.W)
    dis_index = tk.StringVar()
    dis_options = ['Hill', 'Bray-Curtis', 'Jaccard']
    for val, opt in enumerate(dis_options):
        tk.Radiobutton(root, text=opt, variable=dis_index, value=opt, bg=bgcol).grid(row=16+val, sticky=tk.W)
    qval = tk.DoubleVar()
    qval.set(1)
    tk.Label(root, text='Diversity order (q)', bg=bgcol).grid(row=16, column=1, sticky=tk.W)
    tk.Entry(root, textvariable=qval, width=5).grid(row=16, column=2, sticky=tk.W)

    savename_matrix = tk.StringVar()
    tk.Label(root, text='Write name for distance matrix file (do not include .csv)', bg=bgcol).grid(row=20, columnspan=4, sticky=tk.W)
    tk.Entry(root, textvariable=savename_matrix, width=50).grid(row=21, columnspan=4, sticky=tk.W)

    def calc_dis_mat():
        from . import diversity

        if dis_index.get() == 'Bray-Curtis':
            df = diversity.bray(obj['tab'])
        elif dis_index.get() == 'Jaccard':
            df = diversity.jaccard(obj['tab'])
        elif dis_index.get() == 'Hill' and distmat.get() == 'Select':
            df = diversity.naive_beta(obj['tab'], q=qval.get())
        elif dis_index.get() == 'Hill' and distmat.get() != 'Select':
            distmat_df = pd.read_csv(distmat.get(), index_col=0)
            df = diversity.phyl_beta(obj['tab'], distmat=distmat_df, q=qval.get())
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
    canvas = tk.Canvas(master, borderwidth=0, background=bgcol)
    root = tk.Frame(canvas, background=bgcol)
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
    tk.Label(root, text='Plot PCoA', bg=bgcol).grid(row=0, columnspan=3, sticky=tk.W)
    tk.Label(root, text='-'*70, bg=bgcol).grid(row=1, columnspan=3, sticky=tk.W)

    # Get dist
    tk.Label(root, text='Select dissimilarity matrix file (comma separated)', bg=bgcol).grid(row=31, columnspan=3, sticky=tk.W)
    dis_mat_name = tk.StringVar(root, 'None')
    def openDisMat():
        dis_mat_name.set(askopenfilename())
    tk.Button(root, text='Dissimilarity matrix', command=openDisMat).grid(row=35, sticky=tk.W)
    tk.Label(root, textvariable=dis_mat_name, bg=bgcol).grid(row=35, column=1, sticky=tk.W)

    # Get var1 and var2
    var_col = tk.StringVar()
    var_col.set('')
    tk.Label(root, text='Set metadata column heading for color coding of points (required)', bg=bgcol).grid(row=40, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var_col).grid(row=41, sticky=tk.W)

    var1_title = tk.StringVar()
    var1_title.set('')
    tk.Label(root, text='Set title for color legend', bg=bgcol).grid(row=42, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var1_title).grid(row=43, sticky=tk.W)

    var_marker = tk.StringVar()
    var_marker.set('None')
    tk.Label(root, text='Set metadata column heading for marker type of points (not required)', bg=bgcol).grid(row=44, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var_marker).grid(row=45, sticky=tk.W)

    var2_title = tk.StringVar()
    var2_title.set('')
    tk.Label(root, text='Set title for marker legend', bg=bgcol).grid(row=46, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var2_title).grid(row=47, sticky=tk.W)

    var2_pos = tk.DoubleVar()
    var2_pos.set(0.4)
    tk.Label(root, text='Specify position of marker legend (typically 0.2-0.5)', bg=bgcol).grid(row=48, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var2_pos).grid(row=49, sticky=tk.W)

    # Tag
    tag = tk.StringVar(root, 'None')
    tk.Label(root, text='Meta data column used to label points', bg=bgcol).grid(row=52, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=tag).grid(row=53, sticky=tk.W)

    # Connect points
    connect = tk.StringVar(root, 'None')
    tk.Label(root, text='Meta data column used to connect points', bg=bgcol).grid(row=54, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=connect).grid(row=55, sticky=tk.W)

    # Logic order
    order = tk.StringVar(root, 'None')
    tk.Label(root, text='Meta data column used to order the samples in the legend', bg=bgcol).grid(row=56, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=order).grid(row=57, sticky=tk.W)
    tk.Label(root, text='-'*70, bg=bgcol).grid(row=58, columnspan=3, sticky=tk.W)

    #figSize
    tk.Label(root, text='Specify figure dimensions and text size', bg=bgcol).grid(row=60, columnspan=3, sticky=tk.W)
    figSizeW = tk.IntVar(root, 14)
    figSizeH = tk.IntVar(root, 10)
    tk.Label(root, text='Width', bg=bgcol).grid(row=61, sticky=tk.E)
    tk.Entry(root, textvariable=figSizeW).grid(row=61, column=1, sticky=tk.W)
    tk.Label(root, text='Height', bg=bgcol).grid(row=63, sticky=tk.E)
    tk.Entry(root, textvariable=figSizeH).grid(row=63, column=1, sticky=tk.W)

    #FontSize
    fontSize = tk.IntVar(root, 15)
    tk.Label(root, text='Axis text font size', bg=bgcol).grid(row=65, sticky=tk.E)
    tk.Entry(root, textvariable=fontSize).grid(row=65, column=1, sticky=tk.W)

    # savename
    savename_pcoa = tk.StringVar()
    tk.Label(root, text='Write name for PCoA file', bg=bgcol).grid(row=70, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=savename_pcoa, width=50).grid(row=71, columnspan=2, sticky=tk.W)
    tk.Label(root, text='-'*70, bg=bgcol).grid(row=72, columnspan=3, sticky=tk.W)

    def plot_pcoa():
        from . import plot

        dist_file = pd.read_csv(dis_mat_name.get(), index_col=0)
        plot.pcoa(dist_file, obj['meta'], biplot=[], var1=var_col.get(), var2=var_marker.get(),
                 var1_title=var1_title.get(), var2_title=var2_title.get(),
                 whitePad=1.1, var2pos=0.4, tag=tag.get(), order=order.get(), title='',
                 connectPoints=connect.get(),
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
    canvas = tk.Canvas(master, borderwidth=0, background=bgcol)
    root = tk.Frame(canvas, background=bgcol)
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
    tk.Label(root, text='Run null model and return matrices of Raup-Crick and actual null values', bg=bgcol).grid(row=0, columnspan=3, sticky=tk.W)
    tk.Label(root, text='-'*100, bg=bgcol).grid(row=1, columnspan=3, sticky=tk.W)

    # Input distmat
    tk.Label(root, text='Are you working with phylogenetic diversity?', bg=bgcol).grid(row=5, columnspan=3, sticky=tk.W)
    tk.Label(root, text='If so, select distance matrix file', bg=bgcol).grid(row=6, columnspan=3, sticky=tk.W)
    distmat = tk.StringVar(root, 'Select')
    def openDistmat():
        distmat.set(askopenfilename())
    tk.Button(root, textvariable=distmat, command=openDistmat).grid(row=7, columnspan=3, sticky=tk.W)
    def resetNone():
        distmat.set('Select')
    tk.Button(root, text='Reset selection', command=resetNone).grid(row=8, columnspan=3, sticky=tk.W)

    tk.Label(root, text='-'*100, bg=bgcol).grid(row=10, columnspan=3, sticky=tk.W)

    # iterations
    var_iter = tk.IntVar()
    var_iter.set(99)
    tk.Label(root, text='Number of randomization', bg=bgcol).grid(row=15, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var_iter, width=10).grid(row=16, sticky=tk.W)

    # dis index
    tk.Label(root, text='Choose dissimilarity index', bg=bgcol).grid(row=20, columnspan=3, sticky=tk.W)
    dis_index = tk.StringVar()
    dis_options = ['Hill', 'Bray-Curtis', 'Jaccard']
    for val, opt in enumerate(dis_options):
        tk.Radiobutton(root, text=opt, variable=dis_index, value=opt, bg=bgcol).grid(row=22+val, sticky=tk.W)
    qval = tk.DoubleVar()
    qval.set(1)
    tk.Label(root, text='Diversity order (q)', bg=bgcol).grid(row=22, column=1, sticky=tk.W)
    tk.Entry(root, textvariable=qval, width=10).grid(row=22, column=2, sticky=tk.W)

    tk.Label(root, text='-'*100, bg=bgcol).grid(row=25, columnspan=3, sticky=tk.W)

    # randomization procedure
    tk.Label(root, text='Randomization procedure', bg=bgcol).grid(row=29, columnspan=3, sticky=tk.W)
    rand_proc = tk.StringVar()
    rand_options = ['abundance', 'frequency', 'weighting']
    for val, opt in enumerate(rand_options):
        tk.Radiobutton(root, text=opt, variable=rand_proc, value=opt, bg=bgcol).grid(row=30+val, column=0, sticky=tk.W)

    # weighting variable
    var_wt = tk.StringVar()
    var_wt.set('None')
    tk.Label(root, text='Weighting variable', bg=bgcol).grid(row=32, column=1, sticky=tk.W)
    tk.Entry(root, textvariable=var_wt).grid(row=33, column=1, sticky=tk.W)

    # weight
    wt = tk.DoubleVar()
    wt.set(1)
    tk.Label(root, text='Weight', bg=bgcol).grid(row=32, column=2, sticky=tk.W)
    tk.Entry(root, textvariable=wt, width=10).grid(row=33, column=2, sticky=tk.W)

    # constraining variable
    tk.Label(root, text='-'*100, bg=bgcol).grid(row=35, columnspan=3, sticky=tk.W)
    var_con = tk.StringVar()
    var_con.set('None')
    tk.Label(root, text='Constraining variable (optional)', bg=bgcol).grid(row=36, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var_con).grid(row=37, columnspan=2, sticky=tk.W)

    # return variable
    var_ret = tk.StringVar()
    var_ret.set('None')
    tk.Label(root, text='Variable specifying how to group samples (optional)', bg=bgcol).grid(row=40, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var_ret).grid(row=41, columnspan=2, sticky=tk.W)

    # range
    tk.Label(root, text='Choose range for RC index', bg=bgcol).grid(row=50, columnspan=3, sticky=tk.W)
    RC_range = tk.StringVar(root, '0 to 1')
    range_options = ['0 to 1', '-1 to 1']
    for val, opt in enumerate(range_options):
        tk.Radiobutton(root, text=opt, variable=RC_range, value=opt, bg=bgcol).grid(row=51+val, sticky=tk.W)

    tk.Label(root, text='-'*100, bg=bgcol).grid(row=55, columnspan=3, sticky=tk.W)
    def run_null():
        from . import diversity

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

        rcq = diversity.rcq(obj, constrainingVar=var_con.get(), randomization=rand_proc.get(), weightingVar=var_wt.get(), weight=wt.get(),
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

def Mantel_test():
    # Create GUI window
    master = tk.Toplevel()
    master.title('Mantel')
    master.geometry('500x700')
    # Create scrollbar, root is the frame the all widgets are later placed on
    def onFrameConfigure(canvas):
        ###Reset the scroll region to encompass the inner frame
        canvas.configure(scrollregion=canvas.bbox("all"))
    canvas = tk.Canvas(master, borderwidth=0, background=bgcol)
    root = tk.Frame(canvas, background=bgcol)
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

    #Specify input dissimilarity matrices
    tk.Label(root, text='Select dissimilarity matrices', bg=bgcol).grid(row=1, sticky=tk.W)
    file1_name = tk.StringVar(root, 'None')
    def openFile1():
        file1_name.set(askopenfilename())
    tk.Button(root, text='File 1', command=openFile1).grid(row=3, sticky=tk.W)
    tk.Label(root, textvariable=file1_name, bg=bgcol).grid(row=3, column=1, sticky=tk.W)

    file2_name = tk.StringVar(root, 'None')
    def openFile2():
        file2_name.set(askopenfilename())
    tk.Button(root, text='File 2', command=openFile2).grid(row=5, sticky=tk.W)
    tk.Label(root, textvariable=file2_name, bg=bgcol).grid(row=5, column=1, sticky=tk.W)

    # Specify method and permutations
    tk.Label(root, text='Choose test statistic', bg=bgcol).grid(row=8, columnspan=2, sticky=tk.W)
    cor_met = tk.StringVar(root, 'spearman')
    met_options = ['1-Spearmans rho', '1-Pearsons r', 'mean absolute distance']
    for val, opt in enumerate(met_options):
        tk.Radiobutton(root, text=opt, variable=cor_met, value=opt, bg=bgcol).grid(row=10+val, columnspan=2, sticky=tk.W)

    perm_nr = tk.IntVar(root, 999)
    tk.Label(root, text='Number of permutations', bg=bgcol).grid(row=15, sticky=tk.W)
    tk.Entry(root, textvariable=perm_nr, width=10).grid(row=15, column=1, sticky=tk.W)

    def run_mantel():
        from . import stats

        dis1 = pd.read_csv(file1_name.get(), index_col=0)
        dis2 = pd.read_csv(file2_name.get(), index_col=0)
        methodname = cor_met.get()
        if methodname == '1-Spearmans rho':
            methodname = 'spearman'
        if methodname == '1-Pearsons r':
            methodname = 'pearson'
        if methodname == 'mean absolute distance':
            methodname = 'absDist'
        res = stats.mantel(dis1, dis2, method=methodname, permutations=perm_nr.get())

        rootRes = tk.Tk()
        rootRes.title('Mantel results')
        tk.Label(rootRes, text='Test statistic=').grid(row=1, column=0, sticky=tk.W)
        tk.Label(rootRes, text=str(res[0])).grid(row=1, column=1, sticky=tk.W)
        tk.Label(rootRes, text='p value=').grid(row=2, column=0, sticky=tk.W)
        tk.Label(rootRes, text=str(res[1])).grid(row=2, column=1, sticky=tk.W)

    # Buttons to run model or quit
    tk.Button(root, text='Run Mantel test', command=run_mantel).grid(row=60, sticky=tk.W)

    def quit():
        master.destroy()
    tk.Button(root, text='Quit', command=quit).grid(row=60, column=1, sticky=tk.W)

def Permanova(obj, path):
    # Create GUI window
    master = tk.Toplevel()
    master.title('Permanova')
    master.geometry('500x700')
    # Create scrollbar, root is the frame the all widgets are later placed on
    def onFrameConfigure(canvas):
        ###Reset the scroll region to encompass the inner frame
        canvas.configure(scrollregion=canvas.bbox("all"))
    canvas = tk.Canvas(master, borderwidth=0, background=bgcol)
    root = tk.Frame(canvas, background=bgcol)
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

    #Specify input dissimilarity matrix
    tk.Label(root, text='Select dissimilarity matrix', bg=bgcol).grid(row=1, sticky=tk.W)
    file1_name = tk.StringVar(root, 'None')
    def openFile1():
        file1_name.set(askopenfilename())
    tk.Button(root, text='Dissimilarity matrix', command=openFile1).grid(row=3, sticky=tk.W)
    tk.Label(root, textvariable=file1_name, bg=bgcol).grid(row=3, column=1, sticky=tk.W)

    # category
    var = tk.StringVar(root, 'None')
    tk.Label(root, text='Meta data column specifying categories', bg=bgcol).grid(row=5, columnspan=3, sticky=tk.W)
    tk.Entry(root, textvariable=var).grid(row=6, columnspan=2, sticky=tk.W)

    perm_nr = tk.IntVar(root, 999)
    tk.Label(root, text='Number of permutations', bg=bgcol).grid(row=15, sticky=tk.W)
    tk.Entry(root, textvariable=perm_nr, width=10).grid(row=15, column=1, sticky=tk.W)

    def run_permanova():
        from . import stats

        dis1 = pd.read_csv(file1_name.get(), index_col=0)
        res = stats.permanova(dis1, meta=obj['meta'], var=var.get(), permutations=perm_nr.get())

        rootRes = tk.Tk()
        rootRes.title('Permanova results')
        tk.Label(rootRes, text='Test statistic=').grid(row=1, column=0, sticky=tk.W)
        tk.Label(rootRes, text=str(res[0])).grid(row=1, column=1, sticky=tk.W)
        tk.Label(rootRes, text='p value=').grid(row=2, column=0, sticky=tk.W)
        tk.Label(rootRes, text=str(res[1])).grid(row=2, column=1, sticky=tk.W)

    # Buttons to run model or quit
    tk.Button(root, text='Run permanova', command=run_permanova).grid(row=20, sticky=tk.W)

    def quit():
        master.destroy()
    tk.Button(root, text='Quit', command=quit).grid(row=20, column=1, sticky=tk.W)

def run():

    # Functions that specify what to do with input choices
    def choose():
        from . import files

        if sep_name.get() == 'tab':
            separator = '\t'
        else:
            separator = sep_name.get()

        if table_name.get() != 'None':
            obj = files.load(path='', tab=table_name.get(), fasta=seq_name.get(), meta=meta_name.get(), sep=separator)


        if v.get() == 'Make_consensus':
            Consensus_object(path_name.get() + '/')
        elif v.get() == 'Subset_data':
            Subsetting(obj, path_name.get()+'/')
        elif v.get() == 'Calculate_phyl_dist':
            Calc_phyl_dist(obj, path_name.get() + '/')
        elif v.get() == 'Heatmap':
            Heatmap(obj, path_name.get()+'/')
        elif v.get() == 'Rarefaction_curve':
            Rarefaction_curve(obj, path_name.get()+'/')
        elif v.get() == 'Alpha_div':
            Alpha_div(obj, path_name.get()+'/')
        elif v.get() == 'Beta_div':
            Beta_div(obj, path_name.get()+'/')
        elif v.get() == 'PCoA':
            Plot_PCoA(obj, path_name.get() + '/')
        elif v.get() == 'Null_model':
            Null_model(obj, path_name.get() + '/')
        elif v.get() == 'Mantel_test':
            Mantel_test()
        elif v.get() == 'Permanova':
            Permanova(obj, path_name.get() + '/')

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
    canvas = tk.Canvas(master, borderwidth=1, background=bgcol)
    root = tk.Frame(canvas, background=bgcol)
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

    #Specify path for output files
    tk.Label(root, text='Specify folder for output files', bg=bgcol).grid(row=1, columnspan=2, sticky=tk.W)
    path_name = tk.StringVar(root, 'None')
    def openFolder():
        path_name.set(askdirectory())
    tk.Button(root, text='Folder for output files', command=openFolder).grid(row=2, sticky=tk.W)
    tk.Label(root, textvariable=path_name, bg=bgcol).grid(row=2, column=1, sticky=tk.W)
    tk.Label(root, text='-'*70, bg=bgcol).grid(row=3, columnspan=2, sticky=tk.W)

    #Specify input files
    tk.Label(root, text='Select input files', bg=bgcol).grid(row=5, sticky=tk.W)
    table_name = tk.StringVar(root, 'None')
    def openTable():
        table_name.set(askopenfilename())
    tk.Button(root, text='Frequency table', command=openTable).grid(row=6, sticky=tk.W)
    tk.Label(root, textvariable=table_name, bg=bgcol).grid(row=6, column=1, sticky=tk.W)

    seq_name = tk.StringVar(root, 'None')
    def openSeq():
        seq_name.set(askopenfilename())
    tk.Button(root, text='Fasta file', command=openSeq).grid(row=8, sticky=tk.W)
    tk.Label(root, textvariable=seq_name, bg=bgcol).grid(row=8, column=1, sticky=tk.W)

    meta_name = tk.StringVar(root, 'None')
    def openMeta():
        meta_name.set(askopenfilename())
    tk.Button(root, text='Meta data', command=openMeta).grid(row=10, sticky=tk.W)
    tk.Label(root, textvariable=meta_name, bg=bgcol).grid(row=10, column=1, sticky=tk.W)

    tk.Label(root, text='Which type of separator was used in the table and meta files?', bg=bgcol).grid(row=12, columnspan=2, sticky=tk.W)
    sep_name = tk.StringVar(root, ',')
    optionsSep = [',', ';', 'tab']
    for val, opt in enumerate(optionsSep):
        tk.Radiobutton(root, text=opt, variable=sep_name, value=opt, bg=bgcol).grid(row=13+val, sticky=tk.W)

    #Print statistics about the data
    def statsinfo():
        from . import files

        if sep_name.get() == 'tab':
            separator = '\t'
        else:
            separator = sep_name.get()
        obj = files.load(path='', tab=table_name.get(), fasta=seq_name.get(), meta=meta_name.get(), sep=separator)
        tab = obj['tab']
        totalreads = sum(tab.sum())
        totalsvs = len(tab.index)
        totalsmps = len(tab.columns)
        minreads = min(tab.sum())
        maxreads = max(tab.sum())

        rootStats = tk.Tk()
        rootStats.configure(background=bgcol)
        rootStats.title('Stats')
        tk.Label(rootStats, text='Total number of samples', bg=bgcol).grid(row=1, column=0, sticky=tk.W)
        tk.Label(rootStats, text=str(totalsmps), bg=bgcol).grid(row=1, column=1, sticky=tk.W)
        tk.Label(rootStats, text='Total number of SVs or OTUs', bg=bgcol).grid(row=3, column=0, sticky=tk.W)
        tk.Label(rootStats, text=str(totalsvs), bg=bgcol).grid(row=3, column=1, sticky=tk.W)
        tk.Label(rootStats, text='Total number of reads', bg=bgcol).grid(row=5, column=0, sticky=tk.W)
        tk.Label(rootStats, text=str(totalreads), bg=bgcol).grid(row=5, column=1, sticky=tk.W)
        tk.Label(rootStats, text='---', bg=bgcol).grid(row=7, column=0, sticky=tk.W)

        tk.Label(rootStats, text='Minimum number of reads in a sample', bg=bgcol).grid(row=10, column=0, sticky=tk.W)
        tk.Label(rootStats, text=str(minreads), bg=bgcol).grid(row=10, column=1, sticky=tk.W)
        tk.Label(rootStats, text='Maximum number of reads in a sample', bg=bgcol).grid(row=12, column=0, sticky=tk.W)
        tk.Label(rootStats, text=str(maxreads), bg=bgcol).grid(row=12, column=1, sticky=tk.W)
        tk.Label(rootStats, text='---', bg=bgcol).grid(row=15, column=0, sticky=tk.W)

        if 'meta' in obj.keys():
            meta = obj['meta']
            headings = meta.columns.tolist()
            tk.Label(rootStats, text='Headings in metadata', bg=bgcol).grid(row=20, column=0, sticky=tk.W)
            for val, hd in enumerate(headings):
                tk.Label(rootStats, text=hd, bg=bgcol).grid(row=21+val, column=0, sticky=tk.W)

    def taxastats():
        from . import files
        from . import stats

        if sep_name.get() == 'tab':
            separator = '\t'
        else:
            separator = sep_name.get()

        obj = files.load(path='', tab=table_name.get(), fasta=seq_name.get(), meta=meta_name.get(), sep=separator)
        ts = stats.taxa(obj)
        ts.to_csv(path_name.get() + '/' + 'Taxa_stats.csv')

    #-------------
    tk.Label(root, text='-'*70, bg=bgcol).grid(row=18, columnspan=2, sticky=tk.W)
    tk.Label(root, text='Print statistics about the data', bg=bgcol).grid(row=19, columnspan=2, sticky=tk.W)
    tk.Button(root, text='Press for info', command=statsinfo).grid(row=20, column=0, sticky=tk.W)
    tk.Button(root, text='Print stats on taxa', command=taxastats).grid(row=20, column=1, sticky=tk.W)

    # Choices of analysis
    tk.Label(root, text='-'*70, bg=bgcol).grid(row=30, columnspan=2, sticky=tk.W)
    tk.Label(root, text='Choose a task', bg=bgcol).grid(row=31, columnspan=2, sticky=tk.W)

    v = tk.StringVar()
    options = ['Make_consensus', 'Subset_data', 'Calculate_phyl_dist', 'Heatmap', 'Rarefaction_curve', 'Alpha_div', 'Beta_div', 'PCoA', 'Null_model', 'Mantel_test', 'Permanova']
    for val, opt in enumerate(options):
        if val < 5:
            colnr = 0
            rownr = val
        else:
            colnr = 1
            rownr = val-5
        tk.Radiobutton(root, text=opt, variable=v, value=opt, bg=bgcol).grid(row=40+rownr, column=colnr, sticky=tk.W)

    # Buttons that connects to functions
    tk.Label(root, text=' '*70, bg=bgcol).grid(row=51, columnspan=2, sticky=tk.W)
    tk.Button(root, text='Choose', command=choose).grid(row=52, sticky=tk.W)
    tk.Button(root, text='Quit', command=quit).grid(row=52, column=1, sticky=tk.W)
    tk.Label(root, text='-'*70, bg=bgcol).grid(row=55, columnspan=2, sticky=tk.W)

    root.mainloop()
