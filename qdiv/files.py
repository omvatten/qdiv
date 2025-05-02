import pandas as pd
import numpy as np
from . import hfunc

# FUNCTIONS FOR LOADING AND SAVING DATA FILES

# Returns dictionary object containing data as pandas dataframes ('tab', 'tax', 'seq', and 'meta')
# tab is frequency table is frequency table. OTU/ASV names should be in first column, taxa should be in the final columns and start with Kingdom or Domain
# fasta is fasta file with sequences. OTU/ASV names should correspond to those in tab
# tax is table with taxonomic information. OTU/ASV names should correspond to those in tab
# meta is meta data
# if addTaxonPrefix is True (default), g__ is added before genera, f__ before families, etc.
# if orderSeqs is True (default), sequences names are sorted numerically (if they contain numbers in the end of the names)

def load(tab=None, tax=None, meta=None, fasta=None, tree=None, **kwargs):  # Import file and convert them to suitable format
    print('Running files.load .. ', end='')
    defaultKwargs = {'tab_sep':',', 'meta_sep':',', 'tax_sep':',', 'fasta_seq_name_splitter':None, 'path':'',
                     'addTaxonPrefix':True, 'orderSeqs':True}
    kwargs = {**defaultKwargs, **kwargs}

    readtab = None
    readtax = None
    readmeta = None
    seqtab = None
    branch_df = None
    
    if tab != None:
        try:
            readtab = pd.read_csv(kwargs['path']+tab, sep=kwargs['tab_sep'], header=0, index_col=0)
        except:
            print('ERROR: Cannot read tab file. Is the separator incorrectly specified? (e.g. tab_sep="," or tab_sep="\t"). Or is the path to the file incorrectly specified?')
            return None

        if kwargs['orderSeqs']:
            readtab = hfunc.orderSeqs(readtab)

        if tax == None: #Check if tax in tab
            taxlevels = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'strain', 'realm', 'superkingdom']
            taxdict = {}
            for c in readtab.columns:
                if str(c).lower() in taxlevels:
                    taxdict[c] = readtab[c].tolist()
                    readtab = readtab.drop(c, axis=1)
            if len(taxdict.keys())>0:
                readtax = pd.DataFrame(taxdict, index=readtab.index)

        try:
            readtab = readtab.map(float)
        except:
            print('ERROR: All values in the tab file must be numeric. Taxonomic information should be placed in a separate file.')
            return None

    # Read meta data
    if meta != None:
        try:
            readmeta = pd.read_csv(kwargs['path']+meta, sep=kwargs['meta_sep'], header=0, index_col=0)
        except:
            print('ERROR: Cannot read meta file. Is the separator correctly specified? (e.g. meta_sep="," or meta_sep="\t").')
            return None

        # Go through metadata and remove lines not in tab
        if tab != None:
            for ix in readmeta.index:
                if ix not in readtab.columns:
                    readmeta = readmeta.drop([ix])
    
            # Sort samples in tab in same order as in meta data
            readtab = readtab[readmeta.index]

    if tax != None:
        try:
            readtax = pd.read_csv(kwargs['path']+tax, sep=kwargs['tax_sep'], header=0, index_col=0)
        except:
            print('Cannot read tax file. Is the separator correctly specified? (e.g. tax_sep="," or tax_sep="\t").')

        # Prepare taxa dataframe if it exists
        if isinstance(readtax, pd.DataFrame):
            readtax.dropna(axis=1, how='all', inplace=True)
            readtax = readtax.map(str)
            for name in readtax.columns: #Remove items containing only one letter
                readtax[name][readtax[name].str.len() == 1] = pd.NA
                readtax[name][readtax[name] == 'nan'] = pd.NA
    
            #Check if __ is in taxa names
            if kwargs['addTaxonPrefix']:
                prefixdict = {'domain':'d__', 'kingdom':'k__', 'phylum':'p__','class':'c__', 'order':'o__',
                              'family':'f__', 'genus':'g__', 'species':'s__', 'realm':'r__', 'superkingdom':'z__', 'strain':'x__'}
                for c in readtax.columns.tolist():
                    if c.lower() in prefixdict.keys():
                        prefix = prefixdict[c.lower()]
                    elif 'sub' in c.lower() and len(c) > 4:
                        prefix = c[:4]+'__'
                    else:
                        prefix = ''
                    mask = readtax[c][readtax[c].notna()]
                    if len(mask) > 0:
                        mask2 = mask[~mask.str.contains('__', na=False)]
                        if len(mask2) > 0:
                            readtax.loc[mask2.index, c] = prefix + readtax.loc[mask2.index, c]
    
                #Sanity check by comparing to tab
                if tab != None:
                    if sorted(readtab.index.tolist()) != sorted(readtax.index.tolist()):
                        print('Warning, different index names in tab and tax')
                    else:
                        readtax = readtax.loc[readtab.index]
                elif kwargs['orderSeqs']:
                    readtax = hfunc.orderSeqs(readtax)

    # Read fasta file with ASV sequences
    if fasta != None:
        fastalist = [['taxon', 'seq']]
        with open(kwargs['path']+fasta, 'r') as f:
            for line in f:
                if line[0] == '>':
                    name = line[1:].strip()
                    if kwargs['fasta_seq_name_splitter'] != None:
                        name = name.split(kwargs['fasta_seq_name_splitter'])[0]
                    fastalist.append([name, ''])
                else:
                    fastalist[-1][1] = fastalist[-1][1] + line.strip()
        seqtab = pd.DataFrame(fastalist[1:], columns=fastalist[0])
        seqtab = seqtab.set_index('taxon')

        #Sanity check by comparing to tab
        if tab != None:
            if sorted(readtab.index.tolist()) != sorted(seqtab.index.tolist()):
                print('Warning, different index names in tab and seq')
            else:
                seqtab = seqtab.loc[readtab.index]
        elif kwargs['orderSeqs']:
            seqtab = hfunc.orderSeqs(seqtab)
        if tax != None:
            if sorted(readtax.index.tolist()) != sorted(seqtab.index.tolist()):
                print('Warning, different index names in tax and seq')

    # Read Newick tree file
    if tree != None:
        branch_df = hfunc.parse_newick(kwargs['path']+tree)

        #Sanity check by comparing to tab
        if tab != None:
            asvlist = []
            for ix in branch_df.index:
                asvlist = asvlist + branch_df.loc[ix, 'ASVs']
            asvlist = sorted(list(set(asvlist)))
            if sorted(readtab.index.tolist()) != sorted(asvlist):
                print('Warning, different index names in tab and tree')

    # Return dictionary object with all dataframes
    out = {}
    if isinstance(readtab, pd.DataFrame):
        out['tab'] = readtab
    if isinstance(readtax, pd.DataFrame):
        out['tax'] = readtax
    if isinstance(seqtab, pd.DataFrame):
        out['seq'] = seqtab
    if isinstance(branch_df, pd.DataFrame):
        out['tree'] = branch_df
    if isinstance(readmeta, pd.DataFrame):
        out['meta'] = readmeta
    print('Done!')
    return out

# Outputs frequency table, fasta file and meta data from an object
# obj is object to be returned, path is path to folder where files are to be saved
# savename is optional
def printout(obj, path='', savename='output', sep=','):  # Saves files in the same format as they were loaded
    # Return taxa-count table
    if 'tab' in obj:
        tab = obj['tab']
        tab.to_csv(path + savename + '_tab.csv', sep=sep)
    if 'tax' in obj:
        tax = obj['tax']
        tax.to_csv(path + savename + '_tax.csv', sep=sep)
    if 'meta' in obj:
        meta = obj['meta']
        meta.to_csv(path + savename + '_meta.csv', sep=sep)
    if 'seq' in obj:
        seq = obj['seq']
        fasta = []
        for s in seq.index:
            fasta.append('>' + s + '\n')
            fasta.append(seq.loc[s, 'seq'] + '\n')
        with open(path + savename + '_seq.fa', 'w') as f:
            for i in fasta:
                f.write(i)
    if 'tree' in obj:
        print('Tree file cannot be saved.')
    print('Files saved')

# Adds sintax generated taxonomy to object
# obj is the qdiv object,
# filename is a text file containing the sintax output file
def read_sintax(obj, filename):

    headings = ['Kingdom', 'Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    heading_dict = {'k':'Kingdom', 'd':'Domain', 'p':'Phylum', 'c':'Class', 'o':'Order',
                    'f':'Family', 'g':'Genus', 's':'Species'}

    read_in_lines = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                linelist = line.strip().split('\t')
                asv = linelist[0]
                if len(linelist) == 4:
                    tax = linelist[-1].replace('"','').replace(':','__').split(',')
                read_in_lines[asv] = tax
    
        df = pd.DataFrame(pd.NA, index=read_in_lines.keys(), columns=headings)
        for ix in df.index:
            taxlist = read_in_lines[ix]
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
    except:
        print('Error in read_sintax. Cannot read input file.')
