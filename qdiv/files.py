import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

# FUNCTIONS FOR LOADING AND SAVING DATA FILES

# Returns dictionary object containing data as pandas dataframes ('tab', 'ra', 'tax', 'seq', and 'meta')
# path is path to input files
# tab is frequency table is frequency table. SV names should be in first column, taxa should be in the final columns and start with Kingdom or Domain
# fasta is fasta file with sequences. SV names should correspond to those in tab
# meta is meta data
# sep specifies separator used in input files e.g. ',' or '\t'

def load(path='', tab='None', fasta='None', meta='None', sep=','):  # Import file and convert them to suitable format
    #Prepare tab and tax
    if tab != 'None':
        # Read count table with taxa information
        readtab = pd.read_csv(path + tab, sep=sep, header=0, index_col=0)

        #Check if taxa information is in table
        taxaavailable = 1
        if 'Kingdom' in readtab.columns:
            taxpos = readtab.columns.get_loc('Kingdom')
        elif 'Domain' in readtab.columns:
            taxpos = readtab.columns.get_loc('Domain')
        else:
            taxpos = len(readtab.columns)
            taxaavailable = 0

        readtab.index.name = 'SV'
        ctab = readtab.iloc[:, :taxpos]
        ratab = 100 * ctab / ctab.sum()

        ctab = ctab.sort_index()
        ratab = ratab.sort_index()

        # Prepare taxa dataframe
        if taxaavailable == 1:
            taxtab = readtab.iloc[:, taxpos:]
            taxtab = taxtab.sort_index()
            for name in taxtab.columns.tolist()[:-1]: #Remove items containing unknowns or only one letter
                taxtab[name][taxtab[name].str.contains('nknown', na=False)] = np.nan
                taxtab[name][taxtab[name].str.contains('uncult', na=False)] = np.nan
                taxtab[name][taxtab[name].str.len() == 1] = np.nan

            #Check if __ is in taxa names
            correct_taxa_format = 0
            checktax = taxtab.iloc[:, 0][taxtab.iloc[:, 0].notna()]
            for i in range(len(checktax.index)):
                if '__' in checktax.iloc[i]:
                    correct_taxa_format = 1
                    break
            if correct_taxa_format == 0:
                tax_prefix = ['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
                taxatab_col_len = len(taxtab.columns)
                for nr, bokstav in enumerate(tax_prefix[:taxatab_col_len]):
                    checktax = taxtab.iloc[:, nr][taxtab.iloc[:, nr].notna()]
                    for ix in checktax.index:
                        checktax.loc[ix] = bokstav + checktax.loc[ix]
                    taxtab.loc[checktax.index, taxtab.columns[nr]] = checktax

    ##Read fasta file with SV sequences
    if fasta != 'None':
        fastalist = [['SV', 'seq']]
        with open(path + fasta, 'r') as f:
            for line in f:
                if line[0] == '>':
                    fastalist.append([line[1:].strip().split(';')[0], ''])
                else:
                    fastalist[-1][1] = fastalist[-1][1] + line.strip()

        # Correct fasta list based on SVs actually in count table (some might not be represented)
        if tab != 'None':
            tabSVs = list(ctab.index)
            corrfastalist = [fastalist[0]]
            for i in range(1, len(fastalist)):
                if fastalist[i][0] in tabSVs:
                    corrfastalist.append(fastalist[i])
            seqtab = pd.DataFrame(corrfastalist[1:], columns=corrfastalist[0])
        else:
            seqtab = pd.DataFrame(fastalist[1:], columns=fastalist[0])

        seqtab = seqtab.set_index('SV')
        seqtab = seqtab.sort_index()

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
        out['tab'] = ctab
        out['ra'] = ratab
    if tab != 'None' and taxaavailable == 1:
        out['tax'] = taxtab
    if fasta != 'None':
        out['seq'] = seqtab
    if meta != 'None':
        out['meta'] = readmeta
    return out

# Outputs frequency table, fasta file and meta data from an object
# obj is object to be returned, path is path to folder where files are to be saved
# savename is optional
def printout(obj, path='', savename='', sep=','):  # Saves files in the same format as they were loaded
    # Return taxa-count table
    if 'tab' in obj and 'tax' in obj:
        tab = obj['tab']
        tax = obj['tax']
        tab_tax = pd.concat([tab, tax], axis=1)
        tab_tax.to_csv(path + 'output_table_' + savename + '.csv', sep=sep)
    elif 'tab' in obj:
        tab = obj['tab']
        tab.to_csv(path + 'output_table_' + savename + '.csv', sep=sep)
    else:
        print('No tab and tax')

    if 'meta' in obj:
        meta = obj['meta']
        meta.to_csv(path + 'output_meta_' + savename + '.csv', sep=sep)
    else:
        print('No meta')

    if 'seq' in obj:
        seq = obj['seq']
        fasta = []
        for s in seq.index:
            fasta.append('>' + s + '\n')
            fasta.append(seq.loc[s, 'seq'] + '\n')
        with open(path + 'output_seqs_' + savename + '.fa', 'w') as f:
            for i in fasta:
                f.write(i)
    else:
        print('No seq')
    print('Files saved')
