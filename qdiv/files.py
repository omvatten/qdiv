import pandas as pd
import numpy as np
from . import hfunc
pd.options.mode.chained_assignment = None  # default='warn'

# FUNCTIONS FOR LOADING AND SAVING DATA FILES

# Returns dictionary object containing data as pandas dataframes ('tab', 'ra', 'tax', 'seq', and 'meta')
# path is path to input files
# tab is frequency table is frequency table. OTU/ASV names should be in first column, taxa should be in the final columns and start with Kingdom or Domain
# fasta is fasta file with sequences. OTU/ASV names should correspond to those in tab
# meta is meta data
# sep specifies separator used in input files e.g. ',' or '\t'
def load(path='', tab='None', fasta='None', meta='None', tree='None', sep=','):  # Import file and convert them to suitable format
    print('Running files.load .. ', end='')    
    sv_name_lists = {} #To check and compare ASVs in the different input files
    
    #Prepare tab and tax
    if tab != 'None':
        # Read count table with taxa information
        readtab = pd.read_csv(path + tab, sep=sep, header=0, index_col=0)

        #Check if taxa information is in table
        taxaavailable = 0
        taxpos = len(readtab.columns)
        for level in ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Speces']:
            if level in readtab.columns:
                pos = readtab.columns.get_loc(level)
                if pos < taxpos:
                    taxpos = pos
                    taxaavailable = 1

        readtab.index.name = 'ASV'
        ctab = readtab.iloc[:, :taxpos]
        ratab = 100 * ctab / ctab.sum()

        ctab = ctab.sort_index()
        ratab = ratab.sort_index()
        sv_name_lists['tab'] = ctab.index.tolist()

        # Prepare taxa dataframe
        if taxaavailable == 1:
            taxtab = readtab.iloc[:, taxpos:]
            taxtab = taxtab.sort_index()
            taxtab = taxtab.applymap(str)
            for name in taxtab.columns: #Remove items containing only one letter
                taxtab[name][taxtab[name].str.len() == 1] = np.nan

            #Check if __ is in taxa names
            prefixdict = {'D':'d__', 'K':'k__', 'P':'p__','C':'c__', 'O':'o__',
                          'F':'f__', 'G':'g__', 'S':'s__'}
            for c in taxtab.columns.tolist():
                mask = taxtab[c][taxtab[c].notna()]
                if len(mask) > 0:
                    mask2 = mask[~mask.str.contains('__', na=False)]
                    if len(mask2) > 0:
                        prefix = prefixdict[c[0]]
                        taxtab.loc[mask2.index, c] = prefix + taxtab.loc[mask2.index, c]

    # Read fasta file with ASV sequences
    if fasta != 'None':
        fastalist = [['ASV', 'seq']]
        with open(path + fasta, 'r') as f:
            for line in f:
                if line[0] == '>':
                    fastalist.append([line[1:].strip().split(';')[0], ''])
                else:
                    fastalist[-1][1] = fastalist[-1][1] + line.strip()
        seqtab = pd.DataFrame(fastalist[1:], columns=fastalist[0])
        seqtab = seqtab.set_index('ASV')
        seqtab = seqtab.sort_index()
        sv_name_lists['fasta'] = seqtab.index.tolist()

    # Read Newick tree file
    if tree != 'None':
        branch_df = hfunc.parse_newick(path + tree)
        asvlist = []
        for ix in branch_df.index:
            asvlist = asvlist + branch_df.loc[ix, 'ASVs']
        asvlist = sorted(list(set(asvlist)))
        sv_name_lists['tree'] = asvlist
    
    # Compare number of ASVs in tab, fasta and tree
    adjustment_necessary = False
    if len(sv_name_lists) > 1:
        sv_name_lists_keys = list(sv_name_lists.keys())
        key0 = sv_name_lists_keys[0]
        set0 = set(sv_name_lists[key0])
        for i in range(1, len(sv_name_lists_keys)):
            key1 = sv_name_lists_keys[i]
            set1 = set(sv_name_lists[key1])
            if len(set1) != len(set0):
                print('Comparing number of OTUs/ASVs: ', end='')
                print(key0, len(sv_name_lists[key0]), key1, len(sv_name_lists[key1]))
                set0 = set1.intersection(set0)
                key0 = key1
                adjustment_necessary = True
    if adjustment_necessary:
        keepSVs = sorted(list(set0))
        if tab != 'None':
            ctab = ctab.loc[keepSVs]
            ratab = ratab.loc[keepSVs]
        if fasta != 'None':
            seqtab = seqtab.loc[keepSVs]
        print('The numbers of OTUs/ASVs were unequal in the tab, fasta, or tree files.')
        print('The object was subset to common OTUs/ASVs (but the tree is intact).')

    # Read meta data
    if meta != 'None':
        readmeta = pd.read_csv(path + meta, sep=sep, header=0, index_col=0)
    # Go through metadata and remove lines not in tab
    if meta != 'None' and tab != 'None':
        for ix in readmeta.index:
            if ix not in ctab.columns:
                readmeta = readmeta.drop([ix])

        # Sort samples in tab in same order as in meta data
        metalist_samples = readmeta.index.tolist()
        ctab = ctab[metalist_samples]
        ratab = ratab[metalist_samples]

    # Return dictionary object with all dataframes
    out = {}
    if tab != 'None':
        ctab = ctab.applymap(int)
        out['tab'] = ctab
        out['ra'] = ratab
    if tab != 'None' and taxaavailable == 1:
        out['tax'] = taxtab
    if fasta != 'None':
        out['seq'] = seqtab
    if tree != 'None':
        out['tree'] = branch_df
    if meta != 'None':
        out['meta'] = readmeta
    print('Done!')
    return out

# Outputs frequency table, fasta file and meta data from an object
# obj is object to be returned, path is path to folder where files are to be saved
# savename is optional
def printout(obj, path='', savename='output', sep=','):  # Saves files in the same format as they were loaded
    # Return taxa-count table
    if 'tab' in obj and 'tax' in obj:
        tab = obj['tab']
        tax = obj['tax']
        tab_tax = pd.concat([tab, tax], axis=1)
        tab_tax.to_csv(path + savename + '_table.csv', sep=sep)
    elif 'tab' in obj:
        tab = obj['tab']
        tab.to_csv(path + savename + '_table.csv', sep=sep)
        print('No tax')
    else:
        print('No tab and tax')

    if 'meta' in obj:
        meta = obj['meta']
        meta.to_csv(path + savename + '_meta.csv', sep=sep)
    else:
        print('No meta')

    if 'seq' in obj:
        seq = obj['seq']
        fasta = []
        for s in seq.index:
            fasta.append('>' + s + '\n')
            fasta.append(seq.loc[s, 'seq'] + '\n')
        with open(path + savename + '_seq.fa', 'w') as f:
            for i in fasta:
                f.write(i)
    else:
        print('No seq')

    if 'tree' in obj:
        print('Tree file cannot be saved.')

    print('Files saved')

# Adds rdp taxonomy to object
# obj is the qdiv object,
# filename is a text file containing rdp taxonomy, generated here: https://rdp.cme.msu.edu/classifier/classifier.jsp
# at the website, click show assignment detail for Root only, then click download allrank results.
# cutoff is the minimum % cutoff to include a taxonomic level in the classification
def read_rdp(obj, filename, cutoff=70):
    read_in_lines = []
    with open(filename, 'r') as f:
        for line in f:
            if '+' in line and ';' in line:
                read_in_lines.append(line.strip())
    
    headings = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    prefixes = ['d__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
    df_dict = {}
    for i in range(len(read_in_lines)):
        lines = read_in_lines[i].replace('"','').split(';+')
        if len(lines) == 2:       
            asvname = lines[0]
            taxnames = lines[1].split('%;')
        else:
            asvname = 'None'
            taxnames = []
        
        if len(taxnames) > 1:
            templist = []
            for j in range(1, len(taxnames)):
                tax_perc = taxnames[j].split(';')
                perc = tax_perc[1].replace('%','')
                if float(perc) >= cutoff and j <= len(prefixes):
                    templist.append(prefixes[j-1] + tax_perc[0])
            if len(templist) < 7:
                templist = templist + [np.nan] * (7 - len(templist))
        
        if asvname != 'None':
            df_dict[asvname] = templist

    df = pd.DataFrame(np.nan, index=df_dict.keys(), columns=headings)
    for ix in df.index:
        df.loc[ix, :] = df_dict[ix]
    df.sort_index(axis=0, inplace=True)

    tax_svlist = df.index.tolist()
    if 'seq' in obj:
        seq_svlist = obj['seq'].index.tolist()
        common = list(set(tax_svlist).intersection(set(seq_svlist)))
        print(len(common))
        if len(common) != len(seq_svlist):
            print('Warning: the sequences in tax and seq are different')
            print('tax:', len(tax_svlist), 'seq:', len(seq_svlist), 'in common:', len(common))
    if 'tab' in obj:
        tab_svlist = obj['tab'].index.tolist()
        common = list(set(tax_svlist).intersection(set(tab_svlist)))
        if len(common) != len(tab_svlist):
            print('Warning: the sequences in tax and tab are different')
            print('tax:', len(tax_svlist), 'tab:', len(seq_svlist), 'in common:', len(common))

    obj['tax'] = df
    return obj

# Adds sintax generated taxonomy to object
# obj is the qdiv object,
# filename is a text file containing the sintax output file
def read_sintax(obj, filename):
    read_in_lines = []
    with open(filename, 'r') as f:
        for line in f:
            read_in_lines.append(line.strip())

    df_dict = {}
    for i in range(len(read_in_lines)):
        lines = read_in_lines[i].replace('"','').replace(':','__')
        if i%2 == 0:
            asvname = read_in_lines[i]
        else:
            taxlist = []
            if '+' in lines:
                lines = lines.split('+')
                taxlist = lines[-1].strip()
                taxlist = taxlist.split(',')
            df_dict[asvname] = taxlist

    headings = ['Kingdom', 'Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    heading_dict = {'k':'Kingdom', 'd':'Domain', 'p':'Phylum', 'c':'Class', 'o':'Order',
                    'f':'Family', 'g':'Genus', 's':'Species'}
    df = pd.DataFrame(np.nan, index=df_dict.keys(), columns=headings)
    for ix in df_dict.keys():
        taxlist = df_dict[ix]
        for tax in taxlist:
            firstletter = tax[0]
            df.loc[ix, heading_dict[firstletter]] = tax
    df.dropna(axis=1, how='all', inplace=True)
    df.sort_index(axis=0, inplace=True)

    tax_svlist = df.index.tolist()
    if 'seq' in obj:
        seq_svlist = obj['seq'].index.tolist()
        common = list(set(tax_svlist).intersection(set(seq_svlist)))
        if len(common) != len(seq_svlist):
            print('Warning: the sequences in tax and seq are different')
            print('tax:', len(tax_svlist), 'seq:', len(seq_svlist), 'in common:', len(common))
    if 'tab' in obj:
        tab_svlist = obj['tab'].index.tolist()
        common = list(set(tax_svlist).intersection(set(tab_svlist)))
        if len(common) != len(tab_svlist):
            print('Warning: the sequences in tax and tab are different')
            print('tax:', len(tax_svlist), 'tab:', len(seq_svlist), 'in common:', len(common))
        
    obj['tax'] = df
    return obj

# Adds SINA generated taxonomy to object
# obj is the qdiv object,
# filename is a text file containing output from the SINA classifier, generated here: https://www.arb-silva.de/aligner/
# taxonomy is the taxonomy database used with the SINA classifier. Options are: silva, ltp, rdp, gtdb, embl_ebi_ena
def read_sina(obj, filename, taxonomy='silva'):
    read_in_lines = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.replace('""""', '')
            line = line.replace('"";""', 'SPLITTER')
            line = line.replace('"', '')
            read_in_lines.append(line.strip().split('SPLITTER'))
    
    df = pd.DataFrame(read_in_lines[1:], columns=read_in_lines[0])

    headings = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    prefixes = ['d__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
    output = pd.DataFrame(np.nan, index=df['sequence_identifier'], columns=headings)
    returntext = ''

    for ix in df.index:
        asvname = df.loc[ix, 'sequence_identifier']
        taxlist = 'None'
        if taxonomy == 'silva':
            taxlist = df.loc[ix, 'lca_tax_slv'].split(';')
        elif taxonomy == 'rdp':
            taxlist = df.loc[ix, 'lca_tax_rdp'].split(';')
        elif taxonomy == 'ltp':
            taxlist = df.loc[ix, 'lca_tax_ltp'].split(';')
        elif taxonomy == 'gtdb':
            taxlist = df.loc[ix, 'lca_tax_gtdb'].split(';')
        elif taxonomy == 'embl_ebi_ena':
            taxlist = df.loc[ix, 'lca_tax_embl_ebi_ena'].split(';')

        if isinstance(taxlist, list) and len(taxlist) > len(headings):
            returntext = returntext + 'Line ' + str(ix+2) + ': '
            for tax in taxlist:
                returntext = returntext + tax + '; '
            returntext += '\n'
        elif isinstance(taxlist, list):
            for j in range(len(taxlist)):
                tax = taxlist[j]
                if len(tax) > 2:
                    output.loc[asvname, headings[j]] = prefixes[j] + tax

    if len(returntext) > 1:
        print('There are too many taxonomic levels on some of the lines in the input file.', end='')
        print('They should conform to:', 'Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species.', 'Check the following lines:')
        print(returntext)
    else:
        output.dropna(axis=1, how='all', inplace=True)
        output.sort_index(axis=0, inplace=True)
        obj['tax'] = output
        return obj
