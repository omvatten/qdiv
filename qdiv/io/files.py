from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Mapping, List, Union
import re
import os
import gzip
import pandas as pd
import numpy as np
from ..utils import parse_newick, to_newick

# -----------------------------------------------------------------------------
#  Add tab
# -----------------------------------------------------------------------------
def add_tab(
    tab: str | None = None,
    *,
    path: str = "",
    sep: Optional[str] = None,
    taxonomy_levels: Optional[list[str]] = None
) -> dict[str, pd.DataFrame]:
    """
    Load a count or relative-abundance data table into a dictionary of pandas DataFrames.
    Handles standard CSV/TSV and BIOM-like tables with a comment header.

    Parameters
    ----------
    tab : str, default None
        File name of the frequency table (.csv/.tsv, optionally gzipped).
    path : str, default ""
        Directory path containing `tab`.
    sep : str or None, default None
        Column separator. If None, auto-detect based on first non-comment line.
    taxonomy_levels : list of str, optional
        Case-insensitive taxonomy column names to extract.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys:
        - 'tab': numeric-only abundance table
        - 'tax': taxonomy table (if taxonomy columns were present)
    """
    if tab is None:
        raise ValueError("'tab' must be provided.")

    file_path = Path(path) / tab
    if not file_path.exists():
        raise ValueError(f"Tab file not found: {file_path}")

    # Default taxonomy levels
    if taxonomy_levels is None:
        taxonomy_levels = [
            "domain", "kingdom", "phylum", "class", "order", "family",
            "genus", "species", "strain", "realm", "superkingdom",
            "clade", "subfamily",
        ]

    # --- Detect BIOM-like header and separator ---
    skiprows = 0
    detected_sep = sep
    lines_to_check = []
    linecounter = 0
    with open(file_path, "rt") as f:
        for line in f:
            lines_to_check.append(line.strip())
            linecounter += 1
            if linecounter > 4:
                break

        # If first line starts with BIOM marker, skip it
        if lines_to_check[0].startswith("\ufeff"):
            lines_to_check[0] = lines_to_check[0].lstrip("\ufeff")
        if lines_to_check[0].startswith("#") and "from biom" in lines_to_check[0].lower():
            skiprows = 1

        # Auto-detect separator if not provided
        if detected_sep is None:
            counts = {"\t": 0, ",": 0, ";": 0}
            for ln in lines_to_check[skiprows:]:
                counts["\t"] += ln.count("\t")
                counts[","] += ln.count(",")
                counts[";"] += ln.count(";")

            detected_sep = max(counts, key=lambda k: (counts[k], 1 if k == "\t" else 0))
            if counts[detected_sep] == 0:
                detected_sep = "\t"

    # --- Read table ---
    read_csv_kwargs: Mapping[str, object] = dict(
        sep=detected_sep,
        header=0,
        index_col=0,
        skiprows=skiprows,
        engine="c",
        compression="infer",
        dtype=None,
    )

    try:
        df = pd.read_csv(file_path, **read_csv_kwargs)
    except Exception as e:
        raise ValueError(f"Cannot read tab file '{file_path}': {e}") from e

    if df.empty:
        raise ValueError(f"Table is empty: {file_path}")

    if df.index.name is None:
        df.index.name = "feature"

    # Identify taxonomy columns (case-insensitive)
    lower_to_original = {str(c).lower(): c for c in df.columns}
    tax_set = {lvl.lower() for lvl in taxonomy_levels}
    tax_cols = [lower_to_original[c] for c in lower_to_original.keys() if c in tax_set]

    # Extract taxonomy
    readtax: Optional[pd.DataFrame] = None
    if tax_cols:
        readtax = df.loc[:, tax_cols].copy()
        df = df.drop(columns=tax_cols)

    # Coerce remaining columns to float
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        except Exception:
            raise ValueError(f"Non-numeric values found in column '{col}'.")

    result: dict[str, pd.DataFrame] = {"tab": df}
    if readtax is not None and not readtax.empty:
        result["tax"] = readtax

    return result

# -----------------------------------------------------------------------------
#  Add tax
# -----------------------------------------------------------------------------
def add_tax(
    tax: Optional[str] = None,
    *,
    path: str = "",
    sep: Optional[str] = None,
    add_taxon_prefix: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load a taxonomy data table into a dictionary with a pandas DataFrame.

    Parameters
    ----------
    tax : str, default None
        File name of the taxonomy (.csv/.tsv, optionally gzipped, e.g. .csv.gz).
        Feature names (OTU/ASV/bin/MAG) should be in the first column (index).
    path : str, default ""
        Directory path (absolute or relative) containing `tax`. Can be "" for CWD.
    sep : str or None, default ","
        Column separator. If None, pandas will attempt to auto-detect (engine='python').
    add_taxon_prefix : bool, default True
        If True, add letters and two underscores before taxon names to indicate taxonomic level.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys:
        - 'tax': taxonomy table

    Raises
    ------
    ValueError
        - If `tax` is not provided or the file cannot be read
        - If the table is empty or the index is missing

    Notes
    -----
    - Supports gzipped files automatically via compression='infer' (e.g., '.csv.gz', '.tsv.gz').
    - Index is read from the first column of the file.
    - Taxonomy detection is case-insensitive and preserves the original column order.

    Examples
    --------
    >>> data = add_tax(tax="taxonomy_table.csv", sep=",")
    >>> data["tax"].head()
    """
    if tax is None:
        raise ValueError("'tax' must be provided.")

    file_path = Path(path) / tax
    if not file_path.exists():
        raise ValueError(f"Tax file not found: {file_path}")

    if sep is None and (".tsv" in tax or ".txt" in tax):
        sep = "\t"
    elif sep is None and ".csv" in tax:
        sep = ","

    # Read CSV/TSV with robust options and gzip support
    read_csv_kwargs: Mapping[str, object] = dict(
        sep=sep,
        header=0,
        index_col=0,
        engine="python" if sep is None else "c",
        compression="infer",   # enables transparent .gz handling
        dtype=None,
    )
    try:
        df = pd.read_csv(file_path, **read_csv_kwargs)
    except Exception as e:
        raise ValueError(
            f"Cannot read tax file '{file_path}': {e}. "
            "Check path, separator, compression, and file encoding."
        ) from e

    if df.empty:
        raise ValueError(f"Tax is empty: {file_path}")

    if df.index.name is None:
        df.index.name = "feature"

    # Optional taxonomy post-processing (prefixes) if requested
    if add_taxon_prefix:
        levdict = {
            "superkingdom": "sk__", "clade": "cl__", "kingdom": "k__", "domain": "d__", "realm": "r__",
            "phylum": "p__", "class": "c__", "order": "o__", "family": "f__", "subfamily": "sf__",
            "genus": "g__", "species": "s__"
        }
        for col in df.columns:
            # Ensure string type for safe .str operations
            df[col] = df[col].astype(str)
            # Clean single-letter and "nan" entries
            df.loc[df[col].str.len() == 1, col] = pd.NA
            df.loc[df[col] == "nan", col] = pd.NA
            # Add prefix if not already present
            prefix = levdict.get(col.lower(), col[0] + "__")
            mask = (df[col].notna()) & (~df[col].str.contains("__", na=False))
            df.loc[mask, col] = prefix + df.loc[mask, col]

    return {"tax": df}

# -----------------------------------------------------------------------------
#  Add fasta
# -----------------------------------------------------------------------------
def add_seq_from_fasta(
    fasta: str,
    *,
    path: str = "",
    name_splitter: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load sequences from a FASTA file into a dictionary with a pandas DataFrame.

    Parameters
    ----------
    fasta : str
        Name of the FASTA file with sequences of OTUs or ASVs (.fa, .fasta, optionally gzipped).
    path : str, default ""
        Directory path (absolute or relative) containing `fasta`. Can be "" for CWD.
    name_splitter : str, optional
        If provided, splits sequence names on this delimiter and keeps the first part.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys:
        - 'seq': DataFrame with index = sequence names, column = 'seq'

    Raises
    ------
    ValueError
        - If `fasta` is missing or file cannot be read
        - If no sequences are found

    Notes
    -----
    - Supports gzipped files automatically (.fa.gz, .fasta.gz).
    - Multi-line sequences are concatenated.

    Examples
    --------
    >>> data = add_seq_from_fasta(fasta="sequences.fa.gz")
    >>> data["seq"].head()
    """
    file_path = Path(path) / fasta
    if not file_path.exists():
        raise ValueError(f"FASTA file not found: {file_path}")

    # Choose open method based on extension
    open_func = gzip.open if file_path.suffix.endswith("gz") else open

    sequences = []
    current_name = None
    current_seq = []

    try:
        with open_func(file_path, "rt") as f:  # "rt" for text mode
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    # Save previous sequence if exists
                    if current_name:
                        sequences.append([current_name, "".join(current_seq)])
                    # Start new sequence
                    current_name = line[1:].strip()
                    if name_splitter:
                        current_name = current_name.split(name_splitter)[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            # Append last sequence
            if current_name:
                sequences.append([current_name, "".join(current_seq)])
    except Exception as e:
        raise ValueError(f"Error reading FASTA file '{file_path}': {e}") from e

    if not sequences:
        raise ValueError(f"No sequences found in FASTA file: {file_path}")

    # Build DataFrame
    df = pd.DataFrame(sequences, columns=["taxon", "seq"]).set_index("taxon")

    return {"seq": df}

# -----------------------------------------------------------------------------
#  Add tree
# -----------------------------------------------------------------------------
def add_tree(
    tree: str,
    *,
    path: str = ""
) -> Dict[str, pd.DataFrame]:
    """
    Load tree from a newick file into a dictionary with a pandas DataFrame.

    Parameters
    ----------
    tree : str
        Name of the newick file with the tree.
    path : str, default ""
        Directory path (absolute or relative) containing `tree`. Can be "" for CWD.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys:
        - 'tree': DataFrame with nodes and branch lengths

    Raises
    ------
    ValueError
        - If `tree` is missing or file cannot be read
        - If no nodes are found

    Examples
    --------
    >>> data = add_tree(tree="tree.nwk")
    >>> data["tree"].head()
    """
    file_path = Path(path) / tree
    if not file_path.exists():
        raise ValueError(f"FASTA file not found: {file_path}")

    # --- Read tree file ---
    branch_df = parse_newick(file_path)
    return {'tree': branch_df}

# -----------------------------------------------------------------------------
#  Add meta
# -----------------------------------------------------------------------------
def add_meta(
    meta: str,
    *,
    path: str = "",
    sep: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load meta data into a dictionary with a pandas DataFrame.

    Parameters
    ----------
    meta : str
        Name of the meta data file.
    path : str, default ""
        Directory path (absolute or relative) containing `meta`. Can be "" for CWD.
    sep : str or None, default ","
        Column separator. If None, pandas will attempt to auto-detect (engine='python').

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys:
        - 'meta': DataFrame with meta data.

    Raises
    ------
    ValueError
        - If `meta` is missing or file cannot be read
        - If no samples are found

    Examples
    --------
    >>> data = add_meta(meta="metadata.csv")
    >>> data["meta"].head()
    """
    if meta is None:
        raise ValueError("'meta' must be provided.")

    file_path = Path(path) / meta
    if not file_path.exists():
        raise ValueError(f"Meta file not found: {file_path}")

    if sep is None and (".tsv" in meta or ".txt" in meta):
        sep = "\t"
    elif sep is None and ".csv" in meta:
        sep = ","

    # Read CSV/TSV with robust options and gzip support
    read_csv_kwargs: Mapping[str, object] = dict(
        sep=sep,
        header=0,
        index_col=0,
        engine="python" if sep is None else "c",
        compression="infer",   # <— enables transparent .gz handling
        dtype=None,
    )
    try:
        df = pd.read_csv(file_path, **read_csv_kwargs)
    except Exception as e:
        raise ValueError(
            f"Cannot read tab file '{file_path}': {e}. "
            "Check path, separator, compression, and file encoding."
        ) from e

    if df.empty:
        raise ValueError(f"Meta is empty: {file_path}")

    if df.index.name is None:
        df.index.name = "sample"

    result: Dict[str, pd.DataFrame] = {"meta": df}
    return result

# -----------------------------------------------------------------------------
#  Main loader
# -----------------------------------------------------------------------------
def load(
    tab: Optional[str] = None,
    tax: Optional[str] = None,
    meta: Optional[str] = None,
    fasta: Optional[str] = None,
    tree: Optional[str] = None,
    *,
    path: str = "",
    tab_sep: str = None,
    tax_sep: str = None,
    meta_sep: str = None,
    fasta_seq_name_splitter: Optional[str] = None,
    add_taxon_prefix: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load microbiome-related data files into a dictionary of DataFrames.
    Uses specialized loader functions for each file type.
    """
    out: Dict[str, pd.DataFrame] = {}

    if tab:
        out.update(add_tab(tab=tab, path=path, sep=tab_sep))
    if tax:
        out.update(add_tax(tax=tax, path=path, sep=tax_sep, add_taxon_prefix=add_taxon_prefix))
    if meta:
        out.update(add_meta(meta=meta, path=path, sep=meta_sep))
    if fasta:
        out.update(add_seq_from_fasta(fasta=fasta, path=path, name_splitter=fasta_seq_name_splitter))
    if tree:
        out.update(add_tree(tree=tree, path=path))

    return out

# -----------------------------------------------------------------------------
#  Save files
# -----------------------------------------------------------------------------
def save(
    obj: Dict[str, pd.DataFrame],
    path: str = '',
    savename: str = 'output',
    sep: str = ','
) -> list:
    """
    Save frequency table, taxonomy, metadata, sequences, and tree from an object.

    Parameters
    ----------
    obj : dict of str -> pandas.DataFrame
        Dictionary containing data tables. Expected keys include 'tab', 'tax',
        'meta', 'seq', and optionally 'tree'.
    path : str, optional
        Directory path where files will be saved. Defaults to the current directory.
    savename : str, optional
        Base name for output files. Defaults to "output".
    sep : str, optional
        Field separator for CSV files. Defaults to ",".

    Returns
    -------
    list of str
        List of file paths that were saved.

    Examples
    --------
    >>> files = printout(obj, path="results", savename="mydata")
    """
    if path and not os.path.exists(path):
        os.makedirs(path)
    saved_files = []
    # Tab file
    if 'tab' in obj and obj['tab'] is not None:
        filename = os.path.join(path, f"{savename}_tab.csv")
        obj['tab'].to_csv(filename, sep=sep)
        saved_files.append(filename)
    # Tax file
    if 'tax' in obj and obj['tax'] is not None:
        filename = os.path.join(path, f"{savename}_tax.csv")
        obj['tax'].to_csv(filename, sep=sep)
        saved_files.append(filename)
    # Meta file
    if 'meta' in obj and obj['meta'] is not None:
        filename = os.path.join(path, f"{savename}_meta.csv")
        obj['meta'].to_csv(filename, sep=sep)
        saved_files.append(filename)
    # Seq file (FASTA)
    if 'seq' in obj and obj['seq'] is not None:
        filename = os.path.join(path, f"{savename}_seq.fa")
        with open(filename, 'w') as f:
            for s in obj['seq'].index:
                f.write(f">{s}\n{obj['seq'].loc[s, 'seq']}\n")
        saved_files.append(filename)
    # Tree file
    if 'tree' in obj and obj['tree'] is not None:
        filename = os.path.join(path, f"{savename}_tree.nwk")
        nwk = to_newick(obj['tree'])
        with open(filename, 'w') as f:
            f.write(nwk)
        saved_files.append(filename)
    return saved_files

# -----------------------------------------------------------------------------
#  Include sintax output as taxonomy
# -----------------------------------------------------------------------------
def add_tax_from_sintax(
    filename: str,
    *,
    path: str = "",
) -> pd.DataFrame:
    """
    Read a SINTAX output file and returns a taxonomy DataFrame.

    Parameters
    ----------
    filename : str
        Path to the SINTAX output file.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If the file cannot be read or parsed.

    Notes
    -----
    - Taxonomy levels supported: Kingdom, Domain, Phylum, Class, Order, Family, Genus, Species.

    Examples
    --------
    >>> df = read_sintax("sintax_output.txt")
    """
    file_path = Path(path) / filename
    if not file_path.exists():
        raise ValueError(f"Taxonomy file not found: {file_path}")


    headings = ['Kingdom', 'Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    heading_dict = {'k': 'Kingdom', 'd': 'Domain', 'p': 'Phylum', 'c': 'Class', 'o': 'Order',
                    'f': 'Family', 'g': 'Genus', 's': 'Species'}
    read_in_lines = {}
    try:
        with open(file_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        i = 0
        while i < len(lines):
            line = lines[i]
            # Case 1: ASV and taxonomy on same line
            if '\t' in line:
                parts = line.split('\t')
                asv = parts[0].strip()
                if '+' in line:
                    tax_string = line.split('+', 1)[1].strip()
                    taxlist = tax_string.replace(':', '__').split(',')
                    taxlist = [t.strip() for t in taxlist if t]
                    read_in_lines[asv] = taxlist
                i += 1
            # Case 2: ASV on one line, taxonomy on next line
            else:
                asv = line.strip()
                if i + 1 < len(lines):
                    next_line = lines[i+1]
                    if '+' in next_line:
                        tax_string = next_line.split('+', 1)[1].strip()
                        taxlist = tax_string.replace(':', '__').split(',')
                        taxlist = [t.strip() for t in taxlist if t]
                        read_in_lines[asv] = taxlist
                    i += 2
                else:
                    i += 1
        # Build DataFrame
        df = pd.DataFrame(pd.NA, index=read_in_lines.keys(), columns=headings)
        for ix, taxlist in read_in_lines.items():
            for tax in taxlist:
                if not tax:
                    continue
                firstletter = tax[0].lower()
                if firstletter in heading_dict:
                    df.loc[ix, heading_dict[firstletter]] = tax
        df.dropna(axis=1, how='all', inplace=True)
        return df

    except Exception as e:
        raise ValueError(f"Error in read_sintax. Cannot read input file: {e}")

# -----------------------------------------------------------------------------
#  Include qiime output as taxonomy
# -----------------------------------------------------------------------------
def add_tax_from_qiime(
        filename: str,
        *,
        path: str = "",
) -> pd.DataFrame:
    """
    Read a QIIME2-style taxonomy file and return a formatted taxonomy DataFrame.

    Parameters
    ----------
    taxonomy_file : str
        Path to the QIIME2 taxonomy file (TSV format, columns: Feature ID, Taxon, Confidence).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by feature ID, with columns for each taxonomic rank (as detected) and 'Confidence'.
        If no prefix letters are found, defaults to ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'].

    Example
    -------
    >>> df = add_tax_from_qiime("taxonomy.tsv")
    >>> df.head()
    """
    file_path = Path(path) / filename
    if not file_path.exists():
        raise ValueError(f"Taxonomy file not found: {file_path}")

    heading_dict = {'k': 'Kingdom', 'd': 'Domain', 'p': 'Phylum', 'c': 'Class', 'o': 'Order',
                    'f': 'Family', 'g': 'Genus', 's': 'Species'}

    default_order = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

    df = pd.read_csv(file_path, sep='\t', comment='#', skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]
    if 'Feature ID' in df.columns:
        df = df.rename(columns={'Feature ID': 'Feature'})
    if 'Taxon' not in df.columns or 'Feature' not in df.columns:
        raise ValueError("Input file must have columns 'Feature ID' and 'Taxon'.")

    # Detect all unique prefixes in the Taxon column
    all_prefixes = []
    for taxon in df['Taxon']:
        for p in taxon.split(';'):
            m = re.match(r'^([a-z])__', p.strip())
            if m:
                all_prefixes.append(m.group(1))
    seen = set()
    col_order = []
    for prefix in all_prefixes:
        if prefix not in seen and prefix in heading_dict:
            seen.add(prefix)
            col_order.append(heading_dict[prefix])
    # If no prefixes found, fallback to default order
    if not col_order:
        col_order = default_order

    # Parse the taxonomy string into columns
    def split_tax(taxon):
        parts = {}
        found_any_prefix = False
        for p in taxon.split(';'):
            p = p.strip()
            m = re.match(r'^([a-z])__([\s\S]*)', p)
            if m and m.group(1) in heading_dict:
                found_any_prefix = True
                label = heading_dict[m.group(1)]
                value = m.group(2).strip() if m.group(2).strip() not in ['', '_', 'Unassigned'] else ''
                parts[label] = value
        # If no prefixes found in this taxon, split by ';' and assign to default_order
        if not found_any_prefix:
            raw_parts = [p.strip() if p.strip() not in ['', '_', 'Unassigned'] else '' for p in taxon.split(';')]
            for i, label in enumerate(default_order):
                if i < len(raw_parts):
                    parts[label] = raw_parts[i]
        # Fill missing columns with empty string
        return pd.Series([parts.get(col, '') for col in col_order], index=col_order)

    result = df['Taxon'].apply(split_tax)
    result.index = df['Feature']
    result.index.name = 'feature'
    return result

# -----------------------------------------------------------------------------
#  Include GTDB-Tk output as taxonomy
# -----------------------------------------------------------------------------
def add_tax_from_gtdbtk(
    filenames: Union[str, List[str]],
    *,
    path: str = "",
) -> pd.DataFrame:
    """
    Parse one or more GTDB-Tk summary files into a DataFrame with taxonomic levels.

    Parameters
    ----------
    filenames : str or list of str
        Path(s) to GTDB-Tk summary .tsv file(s). Each file must contain at least
        'user_genome' and 'classification' columns.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame indexed by bin ID ('user_genome'), with columns for each
        taxonomic rank: 'Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'.
        Missing levels are filled with None.

    Examples
    --------
    >>> df = parse_gtdbtk_summary(['gtdbtk.bac120.summary.tsv', 'gtdbtk.ar53.summary.tsv'])
    >>> df.head()
    """
    if isinstance(filenames, str):
        filenames = [filenames]

    file_paths = []
    for f in filenames:
        fp = Path(path) / f
        if not fp.exists():
            raise ValueError(f"Taxonomy file not found: {fp}")
        file_paths.append(fp)

    tax_ranks = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    pattern = re.compile(r'([dpcfogs])__([^;]*)')

    dfs = []
    for path in file_paths:
        df = pd.read_csv(path, sep='\t', dtype=str)
        # Parse classification into taxonomic levels
        tax_split = df['classification'].apply(
            lambda x: dict(pattern.findall(x)) if pd.notnull(x) else {}
        )
        tax_df = pd.DataFrame(list(tax_split))
        # Rename columns to full rank names
        tax_df = tax_df.rename(columns={
            'd': 'Domain', 'p': 'Phylum', 'c': 'Class', 'o': 'Order',
            'f': 'Family', 'g': 'Genus', 's': 'Species'
        })
        # Ensure all ranks are present
        for rank in tax_ranks:
            if rank not in tax_df.columns:
                tax_df[rank] = None
        # Set index to bin ID
        tax_df.index = df['user_genome']
        dfs.append(tax_df[tax_ranks])

    result = pd.concat(dfs)
    return result

# -----------------------------------------------------------------------------
#  Include relative abundance table from CoverM
# -----------------------------------------------------------------------------
def add_tab_from_coverm(
    filename: str,
    *,
    path: str = "",
    first_sep: Optional[str] = None,
    second_sep: str = " ",
    detection_threshold: Optional[float] = None
) -> Dict[str, pd.DataFrame]:
    """
    Extract relative abundance data from a CoverM output file.

    Parameters
    ----------
    filename : str
        Path to the CoverM output file (TSV or CSV).
    path : str, default ""
        Directory path (absolute or relative) containing file(s). Can be "" for CWD.
    first_sep : str or None
        Separator used to find sample name in CoverM-file column headings.
        Defaults to None.
    second_sep : str
        Separator used to find sample name in CoverM-file column headings.
        Defaults to " ".
    detection_threshold : float, optional
        If set, all relative abundances below this threshold are set to zero,
        but only if both 'Covered Bases' and 'Length' columns are available for the sample.

    Returns
    -------
    out : dict
        Dictionary with keys:
        - 'tab': DataFrame with genomes as row indices and samples as columns.
        - 'unmapped': Series with unmapped abundances (if present).
    """
    file_path = Path(path) / filename
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")

    _, ext = os.path.splitext(filename)
    delimiter = '\t' if ext.lower() in ['.tsv', '.txt'] else ','

    df = pd.read_csv(file_path, sep=delimiter, dtype=str, index_col=0)
    df = df.replace('NA', np.nan)

    relabund_cols = [col for col in df.columns if re.search(r'Relative Abundance', col, re.IGNORECASE)]
    sample_names = []
    for col in relabund_cols:
        # Case 1: Both separators present
        if first_sep and first_sep in col and second_sep in col:
            start = col.rfind(first_sep) + len(first_sep)
            end = col.find(second_sep, start)
            sample = col[start:end] if end != -1 else col[start:]
            sample_names.append(sample)
        # Case 2: Only second_sep present (or first_sep is None)
        elif (not first_sep or first_sep not in col) and second_sep in col:
            end = col.find(second_sep)
            sample = col[:end] if end != -1 else col
            sample_names.append(sample)
        # Fallback: Use the whole column name
        else:
            sample_names.append(col)

    abundance_df = df[relabund_cols].copy()
    abundance_df.columns = sample_names
    abundance_df = abundance_df.apply(pd.to_numeric, errors='coerce')

    unmapped = None
    if "unmapped" in abundance_df.index:
        unmapped = abundance_df.loc["unmapped"]
        abundance_df = abundance_df.drop("unmapped", errors="ignore")
        df = df.drop("unmapped", errors="ignore")
        
    if detection_threshold is not None:
        for col in sample_names:
            # Use regex for stricter matching
            covered_col = [c for c in df.columns if re.search(rf'{re.escape(col)}.*Covered Bases', c)]
            length_col = [c for c in df.columns if re.search(rf'{re.escape(col)}.*Length', c)]
            if covered_col and length_col:
                covered = pd.to_numeric(df[covered_col[0]], errors='coerce')
                length = pd.to_numeric(df[length_col[0]], errors='coerce')
                detected = (covered / length) >= detection_threshold
                abundance_df.loc[~detected, col] = 0.0

    abundance_df.index.name = 'Feature'
    out = {'tab': abundance_df}
    if unmapped is not None:
        out['unmapped'] = unmapped
    return out

# -----------------------------------------------------------------------------
#  Include EBD style count table generated with SingleM
# -----------------------------------------------------------------------------
def add_ebd_tab_from_singlem(
    filename: str,
    *,
    path: str = "",
    first_sep: Optional[str] = None,
    second_sep: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Add count data (tab) and taxonomic information (tax) from an EBD file generated
    with SingleM.

    Parameters
    ----------
    filename : str
        Path to the file (TSV).
    path : str, default ""
        Directory path (absolute or relative) containing file(s). Can be "" for CWD.
    first_sep : str or None
        Separator used to find sample name in CoverM-file column headings.
        Defaults to None.
    second_sep : str
        Separator used to find sample name in CoverM-file column headings.
        Defaults to None.

    Returns
    -------
    out : dict
        Dictionary with keys:
            
        - 'tab': DataFrame with features as row indices and samples as columns.
        - 'tax': DataFrame with features as row indices and taxonomic levels as columns.
    """
    file_path = Path(path) / filename
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")

    _, ext = os.path.splitext(filename)
    delimiter = "," if ext.lower() in [".csv"] else '\t'

    df = pd.read_csv(file_path, sep=delimiter, dtype=str, index_col=0)
    df = df.replace('NA', np.nan)
    df = df.T
    
    if df.empty:
        raise ValueError("No data in loaded dataframe.")

    colnames = df.columns.tolist()

    if first_sep is not None or second_sep is not None:
        new_colnames = []
        for col in colnames:
            if first_sep is None:
                start = 0
            else:
                start = col.rfind(first_sep) + len(first_sep)
            if second_sep is None:
                end = -1
            else:
                end = col.find(second_sep, start)
            new_colnames.append(col[start:end] if end != -1 else col[start:])
        df.columns = new_colnames

    index_dict = {}
    for i, name in enumerate(df.index):
        index_dict[name] = "OTU"+str(i+1)

    df = df.rename(index=index_dict)

    inverted = {v: k for k, v in index_dict.items()}
    tax = pd.DataFrame.from_dict(inverted, orient="index")

    tax = tax[0].str.split(';', expand=True).apply(lambda col: col.str.strip())
    # If the first token is "Root" or similar, drop it
    if (tax.iloc[:, 0] == 'Root').all():
        tax = tax.iloc[:, 1:]

    taxlevels = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    if len(tax.columns) <= len(taxlevels):
        tax.columns = taxlevels[:len(tax.columns)]
    else:
        raise ValueError("Could not set taxonomic levels. Too many levels found in input data.")

    df.index.name = "Feature"
    tax.index.name = "Feature"
    for c in tax.columns:
        tax.loc[(tax[c]=="None")|(tax[c]=="NA"), c] = np.nan
    out = {"tab": df.astype(float), "tax": tax}
    
    return out
