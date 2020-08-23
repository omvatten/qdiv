import pandas as pd
import numpy as np
import copy
pd.options.mode.chained_assignment = None  # default='warn'

# FUNCTIONS FOR SUBSETTING DATA

# Subsets samples based on metadata
# var is the column heading in metadata used to subset samples, if var=='index' slist are the index names of meta data
# slist is a list of names in meta data column (or index) which specify samples to keep
# if keep0 is false, all SVs with 0 counts after the subsetting will be discarded from the data
def samples(obj, var='index', slist='None', keep0=False):
    if not isinstance(slist, list):
        print('slist must be specified as a [list]')
        return None

    #Correct meta
    if 'meta' in obj and var != 'index':
        meta = obj['meta']
        if var in meta.columns.tolist():
            meta = meta[meta[var].isin(slist)]
        else:
            print('var not found')
            return 0
    elif 'meta' in obj and var == 'index':
        meta = obj['meta']
        meta = meta.loc[slist, :]

    if 'tab' in obj:
        tab = obj['tab']
        if 'meta' in obj:
            tab = tab[meta.index]
        else:
            tab = tab[slist]
    if 'ra' in obj:
        ra = obj['ra']
        if 'meta' in obj:
            ra = ra[meta.index]
        else:
            ra = ra[slist]
    if 'seq' in obj:
        seq = obj['seq']
    if 'tax' in obj:
        tax = obj['tax']

    out = {}  # Dictionary object to return, dataframes and dictionaries

    # Remove SV with zero count
    if not keep0 and 'tab' in obj:
        tab['sum'] = tab.sum(axis=1)
        tab2 = tab[tab['sum'] > 0]
        keepSVs = tab2.index
        tab2 = tab2.drop('sum', axis=1)
        out['tab'] = tab2
        if 'ra' in obj:
            ra2 = ra.loc[keepSVs, :]
            out['ra'] = ra2
        if 'seq' in obj:
            seq2 = seq.loc[keepSVs, :]
            out['seq'] = seq2
        if 'tax' in obj:
            tax2 = tax.loc[keepSVs, :]
            out['tax'] = tax2
        if 'tree' in obj:
            out['tree'] = obj['tree']
        if 'meta' in obj:
            out['meta'] = meta
    else:
        if 'tab' in obj:
            out['tab'] = tab
        if 'ra' in obj:
            out['ra'] = ra
        if 'seq' in obj:
            out['seq'] = seq
        if 'tax' in obj:
            out['tax'] = tax
        if 'tree' in obj:
            out['tree'] = obj['tree']
        if 'meta' in obj:
            out['meta'] = meta
    return out

# Subsets object based on list of OTUs/ASVs to keep
def sequences(obj, svlist):
    out = {}
    tab = obj['tab']
    tab = tab.loc[svlist, :]
    out['tab'] = tab
    if 'ra' in obj:
        ra = obj['ra']
        ra = ra.loc[svlist, :]
        out['ra'] = ra
    if 'tax' in obj:
        tax = obj['tax']
        tax = tax.loc[svlist, :]
        out['tax'] = tax
    if 'seq' in obj:
        seq = obj['seq']
        seq = seq.loc[svlist, :]
        out['seq'] = seq
    if 'tree' in obj:
        out['tree'] = obj['tree']
    if 'meta' in obj:
        out['meta'] = obj['meta']
    return out

# Subsets object to the most abundant OTUs/SVs 
# number specifies the number of SVs to keep
# if method='sum' or method='mean', the OTUs/SVs are ranked based on the sum of the relative abundances in all samples
# if method='max', they are ranked based on the max relative abundance in a sample
def abundant_sequences(obj, number=25, method='sum'):
    out = {}
    tab = obj['tab']
    ra = tab/tab.sum()

    if method in ['sum', 'mean']:
        ra['rank'] = ra.sum(axis=1)
        ra = ra.sort_values(by='rank', ascending=False)
    elif method=='max':
        ra['rank'] = ra.max(axis=1)
        ra = ra.sort_values(by='rank', ascending=False)
        
    svlist = ra.index[:number]
    tab2 = tab.loc[svlist, :]
    out['tab'] = tab2
    if 'ra' in obj:
        ra2 = obj['ra']
        ra2 = ra2.loc[svlist, :]
        out['ra'] = ra2
    if 'tax' in obj:
        tax = obj['tax']
        tax = tax.loc[svlist, :]
        out['tax'] = tax
    if 'seq' in obj:
        seq = obj['seq']
        seq = seq.loc[svlist, :]
        out['seq'] = seq
    if 'tree' in obj:
        out['tree'] = obj['tree']
    if 'meta' in obj:
        out['meta'] = obj['meta']
    return out

# Subset object based on text in taxonomic names
# subsetLevels is list taxonomic levels searched for text patterns, e.g. ['Family', 'Genus']
# subsetPatterns is list of text to search for, e.g. ['Nitrosom', 'Brochadia']
def text_patterns(obj, subsetLevels=[], subsetPatterns=[]):
    if len(subsetLevels) == 0 or len(subsetPatterns) == 0:
        print('No taxlevels or pattern')
        return None
    else:
        taxPatternCheck = obj['tax'].applymap(str)
        keepIX = []
        for col in subsetLevels:
            for ix in taxPatternCheck.index:
                for ptrn in subsetPatterns:
                    if ptrn in taxPatternCheck.loc[ix, col] and ix not in keepIX:
                        keepIX.append(ix)
    out = {}
    if 'tab' in obj:
        out['tab'] = obj['tab'].loc[keepIX]
    if 'ra' in obj:
        out['ra'] = obj['ra'].loc[keepIX]
    if 'tax' in obj:
        out['tax'] = obj['tax'].loc[keepIX]
    if 'seq' in obj:
        out['seq'] = obj['seq'].loc[keepIX]
    if 'tree' in obj:
        out['tree'] = obj['tree']
    if 'meta' in obj:
        out['meta'] = obj['meta']
    return out

# Merges samples based on information in meta data
# var is the column heading in metadata used to merge samples. The counts for all samples with the same text in var column will be merged.
# slist is a list of names in meta data column which specify samples to keep. If slist='None' (default), the whole meta data column is used
# if keep0 is false, all SVs with 0 counts after the subsetting will be discarded from the data
def merge_samples(obj, var='None', slist='None', keep0=False):
    if 'meta' not in obj.keys():
        print('Meta data missing')
        return None
    if var != 'None' and slist == 'None':
        slist = obj['meta'][var]

    tabdi = {}
    radi = {}  # Temp dict that holds sum for each type in slist
    for smp in slist:
        tempobj = samples(obj, var, [smp], keep0=True)
        tab = tempobj['tab']
        tab['sum'] = tab.sum(axis=1)
        tab['ra'] = 100 * tab['sum'] / tab['sum'].sum()
        tabdi[smp] = tab.loc[:, 'sum']
        radi[smp] = tab.loc[:, 'ra']
    temptab = pd.DataFrame(tabdi, index=obj['tab'].index)
    tempra = pd.DataFrame(radi, index=obj['tab'].index)

    out = {}
    if keep0 == False:  ## Remove SV with zero count
        temptab['sum'] = temptab.sum(axis=1)
        tab2 = temptab[temptab['sum'] > 0]
        keepSVs = list(tab2.index)
        tab2 = tab2.drop('sum', axis=1)
        ra2 = tempra.loc[keepSVs, :]
        if 'seq' in obj.keys():
            seq = obj['seq']
            seq2 = seq.loc[keepSVs, :]
            out['seq'] = seq2
        if 'tax' in obj.keys():
            tax = obj['tax']
            tax2 = tax.loc[keepSVs, :]
            out['tax'] = tax2
    else:
        tab2 = temptab
        ra2 = tempra
        if 'seq' in obj.keys():
            out['seq'] = obj['seq']
        if 'tax' in obj.keys():
            out['tax'] = obj['tax']
    out['tab'] = tab2
    out['ra'] = ra2
    if 'tree' in obj:
        out['tree'] = obj['tree']

    meta = obj['meta']
    meta2 = meta.groupby(var).first()
    meta2[var] = meta2.index
    out['meta'] = meta2
    return out

# Rarefies frequency table to a specific number of reads per sample
# if depth = 'min', the minimum number of reads in a sample is used
# seed sets a random state for reproducible results
def rarefy_table(tab, depth='min', seed='None', replacement=False):
    #Make sure table elements are all integers
    tab = tab.fillna(0)
    tab = tab.applymap(int) 

    ## Set read depth
    if depth == 'min':
        depth = int(np.min(tab.sum()))
    else:
        depth = int(depth)

    #Method rarefy with replacment
    if replacement:
        if seed != 'None':
            prng = np.random.RandomState(seed) # reproducible results
        nvar = len(tab.index) # number of SVs
   
        rtab = tab.copy()
        for s in tab.columns: # for each sample
            if tab[s].sum() < depth:
                rtab = rtab.drop(s, axis=1)
                continue
            else:
                p = tab[s]/tab[s].sum()
                if seed != 'None':
                    choice = prng.choice(nvar, depth, p=p)
                else:
                    choice = np.random.choice(nvar, depth, p=p)
                rtab[s] = np.bincount(choice, minlength=nvar)

    #Rarefy without replacment
    else:
        rtab = pd.DataFrame(0, index=tab.index, columns=tab.columns)
        for smp in tab.columns:
            totalreads = tab[smp].sum()
            # Remove sample if sum of reads less than read depth
            if totalreads < depth:
                rtab = rtab.drop(smp, axis=1)
                continue

            smp_series = tab[smp][tab[smp] > 0]
            name_arr = smp_series.index.tolist()
            counts_arr = smp_series.to_numpy()
            cumreads2 = np.cumsum(counts_arr)
            cumreads1 = cumreads2 - counts_arr

            ind_reads_arr = np.empty(totalreads, dtype=object)
            for i, (v1, v2) in enumerate(zip(cumreads1, cumreads2)):
                ind_reads_arr[v1:v2] = name_arr[i]
            if seed != 'None':
                np.random.seed(seed)
            np.random.shuffle(ind_reads_arr)
            bins_counts = np.unique(ind_reads_arr[:depth], return_counts=True)
            rtab.loc[bins_counts[0], smp] = bins_counts[1]
    #Return rarefied table
    return rtab

# Rarefies the tab in an object, then all SVs with 0 count are removed from the object
def rarefy_object(obj, depth='min', seed='None', replacement=False):
    robj = obj.copy()
    rtab = rarefy_table(robj['tab'], depth=depth, seed=seed, replacement=replacement)
    keepSVs = rtab[rtab.sum(axis=1) > 0].index.tolist()
    robj['tab'] = rtab
    robj = sequences(robj, keepSVs)
    robj = samples(robj, slist=rtab.columns.tolist())
    return robj

# ANALYSING AND COMBINING OBJECTS

# Function that makes sure different objects have the same SV names. Returns a list of objects with aligned names
# If differentLengths=True, it assumes that the same SV inferred with different bioinformatics pipelines could have different sequence length
# For example, Deblur sets a specific read length while Dada2 allows different lengths. Comparing SVs from these two pipelines is thus impossible unless differentLengths=True
def align_sequences(objectlist, differentLengths=False):
    objlist = copy.deepcopy(objectlist)

    # Make sure no duplicate sequences within objects
    for i in range(len(objlist)):
        obj = objlist[i]
        seq = obj['seq']
        seq['sequence'] = seq['seq']
        seq['Newname'] = seq.index
        if 'tab' in objlist[i].keys():
            tab = obj['tab']
            tab['sequence'] = seq['sequence']
            ra = obj['ra']
            ra['sequence'] = seq['sequence']
        if 'tax' in objlist[i].keys():
            tax = obj['tax']
            tax['sequence'] = seq['sequence']

        seq = seq.groupby(by='sequence').first()
        if 'tab' in objlist[i].keys():
            tab = tab.groupby(by='sequence').sum()
            tab['Newname'] = seq['Newname']
            tab = tab.set_index('Newname')
            tab = tab.sort_index()
            objlist[i]['tab'] = tab
            ra = ra.groupby(by='sequence').sum()
            ra['Newname'] = seq['Newname']
            ra = ra.set_index('Newname')
            ra = ra.sort_index()
            objlist[i]['ra'] = ra
        if 'tax' in objlist[i].keys():
            tax = tax.groupby(by='sequence').first()
            tax['Newname'] = seq['Newname']
            tax = tax.set_index('Newname')
            tax = tax.sort_index()
            objlist[i]['tax'] = tax
        seq = seq.set_index('Newname')
        seq = seq.sort_index()
        objlist[i]['seq'] = seq

    # Find all unique SVs in all objects and give them an id
    svdict = {}
    seqdict = {}
    counter = 0
    s_list = objlist[0]['seq']['seq'].tolist()
    for s_check in s_list:
        counter += 1
        SVname = 'SV' + str(counter)
        svdict[s_check] = SVname
        seqdict[SVname] = s_check

    # Go through the rest of the objects
    print('Aligning ASVs in ' + str(len(objectlist)) + ' objects: 1.. ', end='')
    for i in range(1, len(objlist)):
        print(str(i+1), end='.. ')
        this_obj_s_list = objlist[i]['seq']['seq'].tolist()
        temp_s_list = []
        for s_check in this_obj_s_list:

            if s_check not in s_list:
                in_dict = 'No'

                if differentLengths: #Check if it is part of the sequence of one sv in the dictionary
                    for s_in_list in s_list:
                        if s_check in s_in_list or s_in_list in s_check: #Then give same name as previously found one
                            SVname = svdict[s_in_list]
                            svdict[s_check] = SVname
                            temp_s_list.append(s_check)
                            in_dict = 'Yes'
                            if len(s_check) > len(seqdict[SVname]):
                                seqdict[SVname] = s_check
                            break

                if in_dict == 'No': # Then give it new name
                    counter += 1
                    SVname = 'SV' + str(counter)
                    svdict[s_check] = SVname
                    seqdict[SVname] = s_check
                    temp_s_list.append(s_check)

        s_list = s_list + temp_s_list

    # Change the name of all SVs
    print('\nChanging ASV names in ' + str(len(objlist)) + ' objects: ', end='')
    for i in range(len(objlist)):
        print(str(i+1), end='.. ')

        seq = objlist[i]['seq']
        seq['newSV'] = np.nan
        if 'tab' in objlist[i].keys():
            tab = objlist[i]['tab']
            ra = objlist[i]['ra']
            tab['newSV'] = np.nan
            ra['newSV'] = np.nan
        if 'tax' in objlist[i].keys():
            tax = objlist[i]['tax']
            tax['newSV'] = np.nan

        for n in seq.index:
            newSVname = svdict[seq.loc[n, 'seq']]
            newSVseq = seqdict[newSVname]
            seq.loc[n, 'newSV'] = newSVname
            seq.loc[n, 'seq'] = newSVseq

            if 'tab' in objlist[i].keys():
                tab.loc[n, 'newSV'] = newSVname
                ra.loc[n, 'newSV'] = newSVname
            if 'tax' in objlist[i].keys():
                tax.loc[n, 'newSV'] = newSVname


        seq = seq.groupby('newSV').first()
        if 'tab' in objlist[i].keys():
            tab = tab.groupby('newSV').sum()
            ra = ra.groupby('newSV').sum()
        if 'tax' in objlist[i].keys():
            tax = tax.groupby('newSV').first()

        # Add updated ones to objectlist
        objlist[i]['seq'] = seq
        if 'tab' in objlist[i].keys():
            objlist[i]['tab'] = tab
            objlist[i]['ra'] = ra
        if 'tax' in objlist[i].keys():
            objlist[i]['tax'] = tax
    print('\nDone with subset.align_sequences')
    return objlist

# Function that takes a list of objects and return a consensus object based on ASVs found in all
# Returns two items: consensus object, information about the object
# If alreadyAligned=True, the alignSVsInObjects function has already been run on the objects
# differentLengths is input for alignSVsInObjects function (see above)
# keepObj makes it possible to specify which object in objlist that should be kept after filtering based on common SVs. Specify with integer (0 is the first object, 1 is the second, etc)
# if keepObj='best', the frequency table having the largest fraction of its reads mapped to the common SVs is kept
# taxa makes it possible to specify with an integer the object having taxa information that should be kept (0 is the first object, 1 is the second, etc). If 'None', the taxa information in the kept Obj is used
# if onlyReturnSeqs=True, only a dataframe with the shared ASVs is returned
def consensus(objlist, keepObj='best', taxa='None', alreadyAligned=False, differentLengths=False, onlyReturnSeqs=False):
    print('Running subset.consensus..')
    if alreadyAligned:
        aligned_objects = objlist.copy()
    else:
        aligned_objects = align_sequences(objlist, differentLengths=differentLengths)

    #Make a list with SVs in common.
    incommonSVs = aligned_objects[0]['tab'].index.tolist()
    for i in range(1, len(aligned_objects)):
        obj = aligned_objects[i]
        incommonSVs = list(set(incommonSVs).intersection(obj['tab'].index.tolist()))

    #If there are no tabs in objects, return object with only seq
    if onlyReturnSeqs:
        seq = aligned_objects[0]['seq']
        seq = seq.loc[incommonSVs]
        info = 'Only seq returned'
        return seq, info

    #Calculate relative abundance of incommon SVs in each tab
    ra_in_tab = []
    ra_sample_max = []
    for i in range(len(aligned_objects)):
        tab_all = aligned_objects[i]['tab']
        tab_incommon = aligned_objects[i]['tab'].loc[incommonSVs]
        ra_in_tab.append(100 * sum(tab_incommon.sum()) / sum(tab_all.sum()))

        tab_notincommon = aligned_objects[i]['tab'].loc[~tab_all.index.isin(incommonSVs)]
        ra_of_notincommon = 100 * tab_notincommon / tab_all.sum()
        ra_sample_max.append(max(ra_of_notincommon.sum()))

    #Get the number of the object with the highest ra associated with incommon SVs
    maxvalue = max(ra_in_tab)
    if keepObj == 'best':
        ra_max_pos = ra_in_tab.index(maxvalue)
    else:
        ra_max_pos = keepObj

    # Make consensus object based on the table having most fraction of reads belonging to incommon SVs
    cons_obj = {}
    cons_obj['tab'] = aligned_objects[ra_max_pos]['tab'].loc[incommonSVs, :]
    cons_obj['ra'] = cons_obj['tab']/cons_obj['tab'].sum()
    cons_obj['seq'] = aligned_objects[ra_max_pos]['seq'].loc[incommonSVs, :]
    cons_obj['meta'] = aligned_objects[ra_max_pos]['meta']

    #Check if tax is in that object, if not get taxa info from on the other
    if 'tax' in aligned_objects[ra_max_pos].keys() and taxa == 'None':
        cons_obj['tax'] = aligned_objects[ra_max_pos]['tax'].loc[incommonSVs, :]
    elif 'tax' not in aligned_objects[ra_max_pos].keys() and taxa == 'None':
        for a_obj in aligned_objects:
            if 'tax' in a_obj.keys():
                cons_obj['tax'] = a_obj['tax'].loc[incommonSVs, :]
    elif taxa != 'None':
        cons_obj['tax'] = aligned_objects[taxa]['tax'].loc[incommonSVs, :]
    
    #Change ASV names in consensus object
    sort_df = cons_obj['ra'].copy()
    sort_df['avg'] = sort_df.mean(axis=1)
    sort_df = sort_df.sort_values(by='avg', ascending=False)
    correct_order_svlist = sort_df.index.tolist()
    cons_obj['tab'] = cons_obj['tab'].loc[correct_order_svlist]
    cons_obj['ra'] = cons_obj['ra'].loc[correct_order_svlist]
    cons_obj['seq'] = cons_obj['seq'].loc[correct_order_svlist]
    if 'tax' in cons_obj.keys():
        cons_obj['tax'] = cons_obj['tax'].loc[correct_order_svlist]
    
    newindex_dict = {}
    for i in range(len(correct_order_svlist)):
        newindex_dict[correct_order_svlist[i]]='ASV' + str(i+1)
    cons_obj['tab'].rename(index=newindex_dict, inplace=True)
    cons_obj['ra'].rename(index=newindex_dict, inplace=True)
    cons_obj['seq'].rename(index=newindex_dict, inplace=True)
    if 'tax' in cons_obj.keys():
        cons_obj['tax'].rename(index=newindex_dict, inplace=True)

    info = {'Kept obj pos': ra_max_pos, 'Relative abundance (%) of reads associated with retained ASVs': ra_in_tab, 
            'Maximum relative abundance (%) of lost reads in a sample': ra_sample_max}
    print('Done with subset.consensus (note that this function does not keep tree in the object).')
    return cons_obj, info
