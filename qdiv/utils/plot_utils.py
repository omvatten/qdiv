import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from typing import Dict, List, Literal, Union, Any
from .data_utils import get_df

# -----------------------------------------------------------------------------
# Get list of colors or markers
# -----------------------------------------------------------------------------
def get_colors_markers(
    get_type: Literal["colors", "markers"] = "colors",
    plot: bool = False
) -> Union[List[str], None]:
    """
    Return predefined color or marker lists, or optionally plot them.

    Parameters
    ----------
    get_type : {'colors', 'markers'}, default='colors'
        Whether to return color names or marker symbols.
    plot : bool, default=False
        If True, display a figure showing all available options.
        If False, return a list of colors or markers.

    Returns
    -------
    list of str or None
        - If ``plot=False``: a list of color names or marker symbols.
        - If ``plot=True``: displays a figure and returns None.

    Notes
    -----
    Colors are sorted by HSV (hue, saturation, value) for visual coherence.
    """

    # --- Validate input ------------------------------------------------------
    if get_type not in {"colors", "markers"}:
        raise ValueError("get_type must be either 'colors' or 'markers'.")

    # --- Color handling ------------------------------------------------------
    if get_type == "colors":
        # Combine base and CSS colors
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

        # Sort by HSV
        by_hsv = sorted(
            (tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(c)[:3])), name)
            for name, c in colors.items()
        )
        color_names = [name for _, name in by_hsv]

        if not plot:
            # Preselected indices for consistent color palettes
            idx = [128, 24, 38, 79, 146, 49, 152, 117, 58, 80,
                   119, 20, 97, 57, 138, 120, 153, 60, 16]
            return [color_names[i] for i in idx]

        # --- Plot colors -----------------------------------------------------
        n = len(color_names)
        ncols = 4
        nrows = n // ncols

        fig, ax = plt.subplots(figsize=(12, 10))

        # Pixel dimensions
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

            ax.text(
                xi_text, y, f"{i}:{name}",
                fontsize=h * 0.6,
                ha="left", va="center"
            )
            ax.hlines(
                y + h * 0.1, xi_line, xf_line,
                color=colors[name], linewidth=h * 0.8
            )

        ax.set_xlim(0, X)
        ax.set_ylim(0, Y)
        ax.set_axis_off()

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()
        return None

    # --- Marker handling -----------------------------------------------------
    if get_type == "markers" and not plot:
        return [
            'o', 's', 'v', 'X', '.', '*', 'P', 'D', '<', ',', '^', '>',
            '1', '2', '3', '4', '8', 'h', 'H', '+'
        ]

    if get_type == "markers" and plot:
        markers = ['o', 's', 'v', 'X', '*', 'P', 'D', '<',
                   '1', '^', '2', '>', '3', '4', '.']

        for i, m in enumerate(markers):
            plt.scatter(i + 1, i + 1, marker=m, s=60)
            plt.text(i + 1, i + 1.4, m, ha="center")

        plt.axis("equal")
        plt.show()
        return None

# -----------------------------------------------------------------------------
# Group by taxonomic classification
# -----------------------------------------------------------------------------
def groupbytaxa(
    obj: Union[Dict[str, Any], Any],
    levels: Union[List[str], str, None] = None,
    include_index: bool = False,
    italics: bool = False
) -> dict:
    """
    Group features (e.g., ASVs/OTUs) by taxonomic levels in a microbiome dataset.

    This function aggregates features in an abundance table and associated metadata
    according to specified taxonomic levels. It accepts either a MicrobiomeData object
    or a dictionary with keys such as 'tab', 'tax', 'seq', and 'meta'. The output is a
    dictionary with grouped abundance, taxonomy, and optionally sequence and metadata tables.

    Parameters
    ----------
    obj : dict or MicrobiomeData
        Input data containing at least:
            - 'tab': pandas.DataFrame
                Abundance table (features x samples).
            - 'tax': pandas.DataFrame
                Taxonomy table (features x taxonomic levels).
        Optionally:
            - 'seq': pandas.DataFrame
                Sequence table (features x sequence).
            - 'meta': pandas.DataFrame
                Sample metadata table (samples x variables).
        The input can be a dictionary or a MicrobiomeData object.

    levels : list of str or str, default=['Phylum', 'Genus']
        Taxonomic levels to use for grouping. The last level in the list determines
        the grouping resolution. If a single string is provided, it is converted to a list.

    include_index : bool, default=False
        If True, append the original feature index to the lowest-level taxonomic name
        in the output labels.

    italics : bool, default=False
        If True, italicize taxonomic names where appropriate.

    Returns
    -------
    out : dict
        Dictionary with grouped tables:
            - 'tab': pandas.DataFrame
                Grouped abundance table (grouped features x samples).
            - 'tax': pandas.DataFrame
                Grouped taxonomy table (grouped features x taxonomic levels).
            - 'seq': pandas.DataFrame, optional
                Grouped sequence table (if present in input).
            - 'meta': pandas.DataFrame, optional
                Sample metadata table (if present in input).

    Raises
    ------
    ValueError
        If required keys ('tab', 'tax') are missing, or if specified taxonomic levels
        are not present in the taxonomy table.

    Notes
    -----
    - Taxonomy columns are matched case-insensitively.
    - Missing taxonomy at any level is filled by propagating the parent level.
    - Taxonomic names are formatted for LaTeX-style italics for plotting.
    - If `include_index` is True, the original feature index is appended to the group label.
    - The function is compatible with both dictionary and MicrobiomeData inputs.

    Examples
    --------
    >>> out = groupbytaxa(obj, levels=['Phylum', 'Genus'])
    >>> out['tab'].head()
    >>> out['tax'].head()

    >>> # With a MicrobiomeData object and including feature indices
    >>> out = groupbytaxa(mb_obj, levels='Genus', include_index=True)
    >>> out['tab'].head()
    """
    # --- Normalize levels to list ---
    if isinstance(levels, str):
        levels = [levels]

    # --- Use get_df to extract components ---
    tab = get_df(obj, "tab")
    tax = get_df(obj, "tax")
    seq = get_df(obj, "seq")
    meta = get_df(obj, "meta")

    # --- Validate input ---
    if tab is None:
        raise ValueError("obj must contain a 'tab' DataFrame.")
    if levels is None:
        out: Dict[str, pd.DataFrame] = {}
        out["tab"] = tab
        if isinstance(tax, pd.DataFrame):
            out["tax"] = tax
        if isinstance(seq, pd.DataFrame):
            out["seq"] = seq
        if isinstance(meta, pd.DataFrame):
            out["meta"] = meta
        return out

    if tax is None:
        raise ValueError("obj must contain a 'tax' DataFrame.")

    tab = tab.copy()
    tax = tax.copy()
    if seq is not None:
        seq = seq.copy()
    if meta is not None:
        meta = meta.copy()

    # Mapping of taxonomic levels to prefixes
    levdict = {
        "superkingdom": "sk__", "clade": "cl__", "kingdom": "k__", "domain": "d__",
        "realm": "r__", "phylum": "p__", "class": "c__", "order": "o__",
        "family": "f__", "subfamily": "sf__", "genus": "g__", "species": "s__"
    }

    # --- Normalize taxonomy column names ---
    tax.columns = [c.lower() for c in tax.columns]
    taxlevels = tax.columns.tolist()

    group_level = levels[-1].lower()
    if group_level not in taxlevels:
        raise ValueError(f"Grouping level '{group_level}' not found in taxonomy.")

    # Restrict taxonomy to relevant levels
    pos = taxlevels.index(group_level)
    taxlevels = taxlevels[:pos + 1]
    tax = tax[taxlevels]

    highest = taxlevels[0]
    if highest not in levdict:
        raise ValueError(f"Highest level '{highest}' not recognized.")

    # --- Fill missing taxonomy with '_unclassified' suffix ---
    # Highest level
    mask_missing = tax[highest].isna() | (tax[highest].str.strip() == "")
    tax.loc[mask_missing, highest] = levdict[taxlevels[0]]+"Unclassified"

    # Other levels
    for i in range(1, len(taxlevels)):
        parent, child = taxlevels[i - 1], taxlevels[i]

        missing_child = tax[tax[child].isna()].index.tolist()
        missing_child = missing_child + tax[(tax[child].notna())&(tax[child].str.len()< 4)].index.tolist()
        parent_uncl = tax[tax[parent].str.contains('unclassified', case=False)].index
        pumc = list(set(missing_child).intersection(parent_uncl))
        if len(pumc) > 0:
            tax.loc[pumc, child] = tax.loc[pumc, parent]

        parent_ok = list(set(tax.index)-set(parent_uncl))
        pomc = list(set(missing_child).intersection(parent_ok))
        if len(pomc) > 0:
            tax.loc[pomc, child] = tax.loc[pomc, parent] + ' unclassified'

    # --- Build accumulated taxonomy names ---
    taxAcc = tax.copy()
    if len(taxlevels) > 1:
        for i in range(1, len(taxlevels)):
            parent, child = taxlevels[i - 1].lower(), taxlevels[i].lower()
            taxAcc[child] = taxAcc[parent] + taxAcc[child]
    if include_index:
        taxAcc[group_level] = taxAcc[group_level] + ": " + taxAcc.index
    taxAcc["index"] = taxAcc.index

    # Group name
    tax["gName"] = taxAcc[group_level]
    tab["gName"] = taxAcc[group_level]

    tax = tax.groupby("gName").first()
    tab = tab.groupby("gName").sum()

    # --- Group sequences if present ---
    if seq is not None:
        seq["gName"] = taxAcc[group_level]
        seq = seq.groupby("gName").first()
    
    taxAcc = taxAcc.groupby(group_level).first()
    tax['index'] = taxAcc['index']
    tax = tax.set_index('index')
    tab['index'] = taxAcc['index']
    tab = tab.set_index('index')

    if seq is not None:
        seq['index'] = taxAcc['index']
        seq = seq.set_index('index')

    # Group taxonomy and abundance

    # --- Fix italics in taxonomy for Matplotlib mathtext (build fragments without $) ---
    if italics:
        n_tuple = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
        for lev in levels:
            c = lev.lower()
    
            #Extract prefix to add back later
            prefix = tax[[c]].copy()
            prefix['prefix'] = ""
            prefix.loc[prefix[c].str.contains("__"), "prefix"] = prefix.loc[prefix[c].str.contains("__"), c].str.split("__").str[0] + "__"
    
            #Keep only the main part of the name, without the prefix
            tax.loc[tax[c].str.contains("__"), c] = tax.loc[tax[c].str.contains("__"), c].str.split("__").str[1]
    
            # Fix candidatus
            candidatus = tax[(tax[c].str.contains("Candidatus", regex=False))|(tax[c].str.contains("Ca.", regex=False))].index
            if len(candidatus) > 0:
                tax.loc[candidatus, c] = tax.loc[candidatus, c].str.replace("Ca.", "\\mathit{Ca.}\\mathrm{").str.replace("Candidatus", "\\mathit{Ca.}\\mathrm{") + "}"
    
            # Two part names with space that should not be italics 
            ws = tax[tax[c].str.contains(" ", case=False)].index.tolist()
            if len(ws) > 0:
                temp = tax.loc[ws].copy()
                ws1 = temp[temp[c].str.split(" ").str[1].str.startswith(n_tuple)].index.tolist()
                ws2 = temp[temp[c].str.split(" ").str[1].str.endswith(n_tuple)].index.tolist()
                ws3 = temp[temp[c].str.split(" ").str[1].str.contains("unclassified", case=False)].index.tolist()
                ws_p2 = list(set(ws1 + ws2 + ws3))
                ws1 = temp[temp[c].str.split(" ").str[0].str.startswith(n_tuple)].index.tolist()
                ws2 = temp[temp[c].str.split(" ").str[0].str.endswith(n_tuple)].index.tolist()
                ws_p1 = list(set(ws1 + ws2))
    
    
            # Two part names with underscore that should not be italics 
            wu = tax[tax[c].str.contains("_", case=False)].index.tolist()
            wu = list(set(wu) - set(ws))
            if len(wu) > 0:
                temp = tax.loc[wu].copy()
                wu1 = temp[temp[c].str.split("_").str[1].str.startswith(n_tuple)].index.tolist()
                wu2 = temp[temp[c].str.split("_").str[1].str.endswith(n_tuple)].index.tolist()
                wu3 = temp[temp[c].str.split("_").str[1].str.contains("unclassified", case=False)].index.tolist()
                wu_p2 = list(set(wu1 + wu2 + wu3))
                wu1 = temp[temp[c].str.split("_").str[0].str.startswith(n_tuple)].index.tolist()
                wu2 = temp[temp[c].str.split("_").str[0].str.endswith(n_tuple)].index.tolist()
                wu_p1 = list(set(wu1 + wu2))
    
            # Completely unclassified, should not be italics
            unclassified = tax.loc[tax[c].str.contains("unclassified", case=False), c].index
            unclassified = list(set(unclassified) - set(ws_p1+ws_p2+wu_p1+wu_p2))
            if len(unclassified) > 0:
                tax.loc[unclassified, c] = "\\mathrm{" + tax.loc[unclassified, c] + "}"
    
            # Some one-part placeholder name
            ph1 = tax[tax[c].str.startswith(n_tuple)].index.tolist()
            ph2 = tax[tax[c].str.endswith(n_tuple)].index.tolist()
            ph = list(set(ph1+ph2) - set(ws_p1+ws_p2+wu_p1+wu_p2))
            tax.loc[ph, c] = "\\mathrm{" + tax.loc[ph, c] + "}"
    
            #Without problematic spaces and no unclassified      
            w = list(set(tax.index) - set(ws_p1+ws_p2+wu_p1+wu_p2) - set(ph) - set(unclassified) - set(candidatus))
            if len(w) > 0:
                tax.loc[w, c] = "\\mathit{" + tax.loc[w, c] + "}"
    
            #With problematic 2nd part after space
            w = list(set(ws_p2) - set(candidatus) - set(unclassified) - set(ws_p1))
            if len(w) > 0:
                part1 = "\\mathit{" + tax.loc[w, c].str.split(" ").str[0] + "}"
                part2 = " \\mathrm{" + tax.loc[w, c].str.split(" ").str[1] + "}"
                tax.loc[w, c] = part1 + part2
    
            #With problematic 2nd part after underscore
            w = list(set(wu_p2) - set(candidatus) - set(unclassified) - set(wu_p1))
            if len(w) > 0:
                part1 = "\\mathit{" + tax.loc[w, c].str.split("_").str[0] + "}"
                part2 = "_\\mathrm{" + tax.loc[w, c].str.split("_").str[1] + "}"
                tax.loc[w, c] = part1 + part2
    
            tax.loc[prefix["prefix"].str.len()>1, c] = "\\mathrm{" + prefix.loc[prefix["prefix"].str.len() > 1, "prefix"] + "}" + tax.loc[prefix["prefix"].str.len()>1, c]

    # Combine taxonomy for the levels
    if len(levels) > 1:
        for i in range(1, len(levels)):
            parent, child = levels[i-1].lower(), levels[i].lower()
            tax.loc[tax[parent]!=tax[child], child] = tax.loc[tax[parent]!=tax[child], parent] + "; " + tax.loc[tax[parent]!=tax[child], child]
    tax["Name"] = tax[group_level]

    if italics:
        tax["Name"] = "$" + tax["Name"].str.replace("-", "_").str.replace(" ","\\ ").str.replace("_","\\_") + "$"

    # --- Final grouping ---
    out: Dict[str, pd.DataFrame] = {}
    tab["Name"] = tax["Name"]
    out["tab"] = tab.groupby("Name").sum()
    if seq is not None:
        seq["Name"] = tax["Name"]
        out["seq"] = seq.groupby("Name").first()
    out["tax"] = tax.groupby("Name").first()
    if meta is not None:
        out["meta"] = meta
    return out

