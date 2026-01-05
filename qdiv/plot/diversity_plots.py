from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ..io import subset_samples
from ..diversity import naive_alpha, phyl_alpha, func_alpha
from ..diversity import dissimilarity_by_feature, naive_multi_beta, phyl_multi_beta, func_multi_beta
from ..utils import get_df, parse_leaves, get_colors_markers

# -----------------------------------------------------------------------------
# Plot dissimilarity contribution of features
# -----------------------------------------------------------------------------
def dissimilarity_contributions(
    obj: Union[Dict[str, Any], Any],
    *,
    by: Optional[str] = None,
    q: float = 1.0,
    div_type: str = "naive",
    index: str = "local",
    n: int = 20,
    levels: Optional[List[str]] = None,
    from_file: Optional[str] = None,
    figsize: Tuple[float, float] = (18 / 2.54, 14 / 2.54),
    fontsize: int = 10,
    savename: Optional[str] = None,
) -> Tuple["plt.figure.Figure", "pd.DataFrame"]:
    """
    Plot contributions of taxa to observed dissimilarity within categories.

    This function visualizes how individual taxa contribute to dissimilarity
    (e.g., Bray-Curtis or Hill-based) across sample groups defined by metadata.

    Parameters
    ----------
    obj : dict or MicrobiomeData
        Input data containing at least:
            - 'tab': pandas.DataFrame
                Abundance table (features x samples).
            - 'tax': pandas.DataFrame
                Taxonomy table (features x taxonomic levels).
            - ``meta`` (pandas.DataFrame): metadata table.
    by : str, optional
        Metadata column used to categorize samples. Dissimilarity is calculated within each category.
    q : float, default=1.0
        Diversity order (Hill number).
    div_type : {'naive', 'phyl'}, default='naive'
        Diversity type:
        - 'naive': taxonomic dissimilarity.
        - 'phyl': phylogenetic dissimilarity.
    index : {'local', 'regional'}, default='local'
        Index type for dissimilarity calculation.
    n : int, default=20
        Number of top taxa to include in the plot.
    levels : list of str, optional
        Taxonomic levels to include in y-axis labels (e.g., ['Genus']).
    from_file : str, optional
        Path to a CSV file with precomputed dissimilarity contributions.
        If None, contributions are computed from `obj`.
    figsize : tuple of float, default=(18/2.54, 14/2.54)
        Figure size in inches.
    fontsize : int, default=10
        Font size for plot text.
    savename : str, optional
        If provided, save the figure to this path and also as PDF.

    Returns
    -------
    fig : matplotlib.figure.Figure
    df : pandas.DataFrame
        DataFrame of contributions for plotted taxa and categories.
        Returns None if computation or plotting fails.

    Notes
    -----
    - If `from_file` is provided, the function reads contributions from that file.
    - If `levels` is provided and `div_type='naive'`, taxonomy names are appended to feature IDs.
    - For phylogenetic diversity, node names or feature sets are used for labeling.

    Examples
    --------
    >>> df = dissimilarity_contributions(obj, by='Treatment', q=1, div_type='naive', levels=['Genus'])
    >>> print(df.head())
    """
    tax = get_df(obj, "tax")
    tree = get_df(obj, "tree")
    if levels is not None and isinstance(levels, str):
        levels = [levels]

    # Load or compute dissimilarity contributions
    if from_file is None:
        # Compute contributions using your diversity function
        dis_data = dissimilarity_by_feature(obj, by=by, q=q, div_type=div_type, index=index)
    else:
        dis_data = pd.read_csv(from_file, index_col=0)

    # Prepare data
    df = dis_data.drop(["N", "dis"], axis=0)
    catlist = [x for x in df.columns if 'nodes' not in x]
    df["avg"] = df[catlist].mean(axis=1)
    
    df = df.sort_values(by="avg", ascending=False).iloc[:n]

    # Categories and taxa labels
    ylist = range(len(df.index))
    taxlist = df.index.tolist()

    # Append taxonomy names if requested
    if levels is not None and tax is not None and div_type == "naive":
        tax_df = tax.loc[df.index].fillna("").astype(str)
        for i, asv in enumerate(df.index):
            taxname = "; ".join([tax_df.loc[asv, lvl] for lvl in levels if len(tax_df.loc[asv, lvl]) > 3])
            taxlist[i] = f"{taxname}; {asv}" if taxname else asv

    elif levels is not None and div_type == "phyl" and tree is not None:
        tree_df = tree.loc[df.index]
        tax = tax.fillna("").astype(str)
        for i, ix in enumerate(tree_df.index):
            asvlist = tree_df.loc[ix, "leaves"]
            if len(asvlist) == 1:
                taxname = "; ".join([tax.loc[asvlist[0], lvl] for lvl in levels if len(tax.loc[asvlist[0], lvl]) > 3])
                taxlist[i] = f"{taxname}; {asvlist[0]}" if taxname else asvlist[0]
            else:
                taxlist[i] = tree_df.loc[ix, "nodes"]

    # Plot
    plt.rcParams.update({"font.size": fontsize})
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, len(catlist))
    gs.update(wspace=0, hspace=0)

    for cat_nr, cat in enumerate(catlist):
        ax = fig.add_subplot(gs[0, cat_nr], frame_on=True)
        ax.barh(ylist, df[cat])
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(taxlist if cat_nr == 0 else [])
        ax.set_xlabel("%")

        title_text = f"{cat}\nN={int(dis_data.loc['N', cat])}\n$^{{{q}}}$d={round(dis_data.loc['dis', cat], 2)}"
        ax.set_title(title_text)

    if savename:
        plt.savefig(savename, dpi=240)
        try:
            plt.savefig(f"{savename}.pdf", format="pdf")
        except Exception:
            # Fallback silently if a PDF backend is not available in the environment
            pass

    return fig, df


# -----------------------------------------------------------------------------
# Plot phylogram
# -----------------------------------------------------------------------------
def phyl_tree(
    obj: Union[Dict[str, Any], Any],
    *,
    width: float = 12,
    name_internal_nodes: bool = False,
    abundance_info: Optional[str] = None,
    xlog: bool = False,
    savename: Optional[str] = None,
) -> Tuple["plt.figure.Figure", "pd.DataFrame"]:
    """
    Plot a phylogram from a tree DataFrame with optional abundance bars.

    Parameters
    ----------
    obj : dict or MicrobiomeData
        Input object with required key:
        - ``tree`` (pandas.DataFrame): tree structure with columns ['nodes', 'leaves', 'branchL'].
        Optional keys:
        - ``tab`` (pandas.DataFrame): abundance table (features x samples).
        - ``meta`` (pandas.DataFrame): metadata table for sample grouping.
    width : float, default=12
        Width of the plot in centimeters. Height is set automatically based on number of ASVs.
    name_internal_nodes : bool, default=False
        If True, labels are added to internal nodes.
    abundance_info : {'index'} or str, optional
        If 'index', plot relative abundance bars for each ASV.
        If a metadata column name, plot grouped abundance bars for each category.
    xlog : bool, default=False
        If True, abundance bars use a logarithmic x-axis.
    savename : str, optional
        If provided, save the figure to this path and also as PDF.

    Returns
    -------
    fig : matplotlib.figure.Figure
    df_endN : pandas.DataFrame
        DataFrame of end nodes with positions and optional abundance info.

    Notes
    -----
    - The tree DataFrame must contain columns: 'nodes', 'leaves', 'branchL'.
    - If `abundance_info` is provided, relative abundances are computed per leaf or category.
    - Bars are plotted to the right of the tree when `abundance_info` is not None.

    Examples
    --------
    >>> phyl_tree(obj, width=15, name_internal_nodes=True, abundance_info='Treatment', xlog=True, savename='phylogram')
    """
    # Validate tree
    tree = get_df(obj, "tree")
    tab = get_df(obj, "tab")
    meta = get_df(obj, "meta")
    if tree is None:
        raise ValueError("Error: 'tree' not found in obj.")
    
    df = tree.copy()

    # Separate end nodes and internal nodes
    df_endN = df[(~df["nodes"].str.startswith('in'))&(df["nodes"] != 'Root')].set_index("nodes")
    df_intN = df[df["nodes"].str.startswith('in')].set_index("nodes")

    # Assign initial positions
    df_endN["ypos"] = range(len(df_endN.index))
    df_endN["xpos"] = df_endN["dist_to_root"].astype(float)

    # Sort internal nodes by size
    df_intN['asv_count'] = 0
    df_intN['asv_count'] = df_intN['leaves'].apply(lambda x: len(parse_leaves(x)))
    df_intN = df_intN.sort_values("asv_count", ascending=True)

    # Compute abundance info if requested
    catlist = []
    if abundance_info and tab is not None and meta is not None:
        if abundance_info != "index":
            catlist = meta[abundance_info].dropna().unique().tolist()
        else:
            catlist = meta.index.tolist()

        for cat in catlist:
            df_endN[f"ra:{cat}"] = 0.0
            temp_obj = subset_samples(obj, by=abundance_info, values=[cat])
            temp_tab = get_df(temp_obj, "tab")
            ra = temp_tab / temp_tab.sum()
            ra = ra.mean(axis=1)
            df_endN.loc[ra.index, f"ra:{cat}"] = ra

    # Plot tree
    textspacing = df_endN["xpos"].max() / 50
    plt.rcParams.update({"font.size": 10})
    fig = plt.figure(figsize=(width / 2.54, 0.7 * len(df_endN.index) / 2.54), constrained_layout=True)
    gs = fig.add_gridspec(1, 10)
    gs.update(wspace=0, hspace=0)

    ax = fig.add_subplot(gs[0, :9] if abundance_info else gs[0, :10], frame_on=True)

    # Plot end nodes
    for node in df_endN.index:
        ypos = df_endN.loc[node, "ypos"]
        xpos = df_endN.loc[node, "xpos"]
        ax.text(xpos + textspacing, ypos, node, va="center", color="red")
        node_BL = df_endN.loc[node, "branchL"]
        ax.plot([xpos - node_BL, xpos], [ypos, ypos], lw=1, color="black")
        df_endN.loc[node, "xpos"] = xpos - node_BL

    # Plot internal nodes
    for intN in df_intN.index:
        asvlist = df_intN.loc[intN, "leaves"]
        xpos = df_endN.loc[asvlist, "xpos"].mean()
        ymax = df_endN.loc[asvlist, "ypos"].max()
        ymin = df_endN.loc[asvlist, "ypos"].min()
        ax.plot([xpos, xpos], [ymin, ymax], lw=1, color="black")
        ymean = (ymax + ymin) / 2
        xmin = xpos - df_intN.loc[intN, "branchL"]
        ax.plot([xmin, xpos], [ymean, ymean], lw=1, color="black")
        df_endN.loc[asvlist, ["ypos", "xpos"]] = [float(ymean), float(xmin)]
        if name_internal_nodes:
            ax.text(xpos, ymean, df_intN.loc[intN, "nodes"], va="center", color="red")

    ax.plot([0, 0], [df_endN["ypos"].min(), df_endN["ypos"].max()], lw=1, color="black")
    ax.set_ylim(-1, len(df_endN.index))
    ax.axis("off")

    # Plot abundance bars
    if abundance_info:
        ax2 = fig.add_subplot(gs[0, 9], frame_on=True)
        bars_leg1, bars_leg2 = [], []
        orig_ypos = range(len(df_endN.index))
        for cat_nr, cat in enumerate(catlist):
            bar_thickness = 0.8 / len(catlist)
            bar_yoffset = bar_thickness * (cat_nr - (len(catlist) - 1) / 2)
            ylist = np.array(orig_ypos) + bar_yoffset
            xlist = df_endN[f"ra:{cat}"]
            bl = ax2.barh(ylist, xlist, height=0.95 * bar_thickness, label=cat)
            bars_leg1.append(bl)
            bars_leg2.append(cat)

        if xlog:
            ax2.set_xscale("log")
        ax2.set_ylim(-1, len(df_endN.index))
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax.legend(bars_leg1, bars_leg2, loc="lower right", bbox_to_anchor=(1, 1), ncol=4, frameon=False)

    if savename:
        plt.savefig(savename, dpi=240)
        try:
            plt.savefig(f"{savename}.pdf", format="pdf")
        except Exception:
            # Fallback silently if a PDF backend is not available in the environment
            pass

    return fig, df_endN

# -----------------------------------------------------------------------------
# Plot harvey balls from metadata
# -----------------------------------------------------------------------------
def harvey_balls(
    meta: Union[pd.DataFrame, Dict[str, Any], Any],
    columns_by: List[str] = None,
    *,
    rows_by: str = "index",
    row_colors: Optional[str] = None,
    column_colors: Optional[List[str]] = None,
    row_label_width: int = 4,
    figsize: Tuple[float, float] = (18 / 2.54, 14 / 2.54),
    fontsize: int = 10,
    savename: Optional[str] = None,
) -> Tuple["plt.figure.Figure", "pd.DataFrame"]:
    """
    Plot Harvey balls (fraction-of-circle indicators) for percentage columns in metadata.

    Parameters
    ----------
    meta : DataFrame | MicrobiomeData-like | dict
        Object with metadata table. Must contain the `columns_by` fields and optionally
        a `rows_by` field used to derive row labels.
    columns_by : list of str
        List of metadata column names containing percentages (0–100) to visualize
        as Harvey balls across rows.
    rows_by : str, default='index'
        Name of the metadata column used as row labels. If 'index', the
        DataFrame index is used as row labels.
    row_colors : str, optional
        Name of metadata column containing per-row text colors (e.g., 'red', '#333').
        If None, all row labels are drawn in black.
    column_colors : list of str, optional
        Colors for the column headers (one per `columns_by`). If None, defaults to black
        for all headers; if provided but shorter than `columns_by`, the list is padded with black.
    row_label_width : int, default=4
        Number of GridSpec columns reserved for the row label area (left-hand text).
    figsize : tuple of float, default=(18/2.54, 14/2.54)
        Figure size in inches.
    fontsize : int, default=10
        Base font size for the figure.
    savename : str, optional
        If provided, saves the figure (PNG) to this path and also as PDF (`savename + '.pdf'`).

    Returns
    -------
    fig : matplotlib.figure.Figure
    plot_data : pandas.DataFrame
        A DataFrame containing the row labels and selected percentage values used for plotting:
        columns ['__label__', *columns_by]. Returns None if validation fails.

    Notes
    -----
    - Harvey balls are drawn using pie charts where the **black** wedge represents the percentage,
      and the **white** wedge represents the complement to 100%.
    - All values in `columns_by` must be numeric (0–100). Non-numeric rows are coerced if possible;
      rows with missing values will still be plotted (missing values treated as 0).
    - If `rows_by='index'`, row labels are taken from `meta.index`; otherwise, from `meta[rows_by]`.

    Examples
    --------
    >>> df = harvey_balls(
    ...     meta,
    ...     rows_by='Treatment',
    ...     columns_by=['PFAS_%', 'DOC_%'],
    ...     row_colors='TreatmentColor',
    ...     column_colors=['#1f77b4', '#ff7f0e'],
    ...     savename='harvey_balls'
    ... )
    >>> print(df.head())
    """
    # ---- Validation ---------------------------------------------------------
    meta = get_df(meta, "meta")
    if meta is None or meta.empty:
        raise ValueError("Error: meta is missing.")

    if columns_by is None:
        raise ValueError("Error: columns_by must be specified.")

    if isinstance(columns_by, str):
        columns_by = [columns_by]

    if not set(columns_by) & set(meta.columns):
        raise ValueError("columns_by must be a column in metadata")

    # rows_by must exist unless using index
    if rows_by != "index" and rows_by not in meta.columns:
        raise ValueError("columns_by must be a column in metadata")

    # prepare column header colors
    if column_colors is None or len(column_colors) == 0:
        column_colors = ["black"] * len(columns_by)
    elif len(column_colors) < len(columns_by):
        column_colors = column_colors + ["black"] * (len(columns_by) - len(column_colors))

    # ---- Prepare labels and colors -----------------------------------------
    if rows_by == "index":
        row_labels = meta.index.tolist()
    else:
        row_labels = meta[rows_by].astype(str).tolist()

    if row_colors is None:
        per_row_text_colors = ["black"] * len(row_labels)
    else:
        if row_colors not in meta.columns:
            print(f"Error: 'row_colors' column '{row_colors}' not found in metadata.")
            return None
        per_row_text_colors = meta[row_colors].astype(str).tolist()

    # coerce percentage columns to numeric, fill NAs with 0, clip to [0, 100]
    plot_data = meta[columns_by].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    plot_data.insert(0, "__label__", row_labels)

    # ---- Plot ---------------------------------------------------------------
    plt.rcParams.update({"font.size": fontsize})
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    # rows: +1 for header row; columns: label area + number of metrics
    gs = GridSpec(len(meta) + 1, row_label_width + len(columns_by), figure=fig)

    # Header: left label
    ax01 = plt.subplot(gs[0, :row_label_width])
    ax01.text(0, 0, rows_by, va="center")
    ax01.set_ylim(-1, 1)
    ax01.axis("off")

    # Header: metric names
    for i, g in enumerate(columns_by):
        ax = plt.subplot(gs[0, row_label_width + i])
        ax.text(0, 0, g, color=column_colors[i], ha="center", va="center")
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
        ax.axis("off")

    # Rows
    for j, label in enumerate(row_labels):
        # Row label area
        ax1 = plt.subplot(gs[j + 1, :row_label_width])
        ax1.text(0, 0, label, color=per_row_text_colors[j], va="center")
        ax1.set_ylim(-1, 1)
        ax1.axis("off")

        # Harvey balls per metric
        for i, g in enumerate(columns_by):
            ax = plt.subplot(gs[j + 1, row_label_width + i])
            black = float(plot_data.loc[meta.index[j], g])  # percentage
            white = 100.0 - black
            if white == 100.0:
                ax.pie([white], colors=["white"], startangle=90,
                       wedgeprops={"linewidth": 1, "edgecolor": "black"})
            else:
                ax.pie([white, black], colors=["white", "black"], startangle=90,
                       wedgeprops={"linewidth": 1, "edgecolor": "black"})
            ax.set_aspect("equal")

    # Save
    if savename:
        plt.savefig(savename, dpi=240)
        try:
            plt.savefig(f"{savename}.pdf", format="pdf")
        except Exception:
            # Fallback silently if a PDF backend is not available in the environment
            pass

    return fig, plot_data

# -----------------------------------------------------------------------------
# Plot alpha diversity profiles
# -----------------------------------------------------------------------------
def alpha_diversity_profile(
    obj: Union[dict, Any],
    *,
    q_range: Tuple[float, float] = (0.0, 2.0),
    q_step: float = 0.05,
    distmat: Optional[pd.DataFrame] = None,
    div_type: str = "naive",
    color_by: Optional[str] = None,
    order: Optional[str] = None,
    ylog: bool = False,
    figsize: Tuple[float, float] = (18 / 2.54, 14 / 2.54),
    fontsize: int = 10,
    colorlist: Optional[List[str]] = None,
    use_values_in_tab: bool = False,
    savename: Optional[str] = None,
) -> Tuple["plt.Figure", "plt.Axes", "pd.DataFrame"]:
    """
    Plot alpha diversity vs diversity order across samples.

    This function computes alpha diversity (Hill numbers) for a range of
    diversity orders q and plots the curves per sample. It supports taxonomic,
    phylogenetic, and functional diversity depending on `div_type`.

    Parameters
    ----------
    obj : dict or MicrobiomeData
        Input data structure containing at least:
            - ``tab`` (pandas.DataFrame): abundance table (features x samples).
            - ``meta`` (pandas.DataFrame): sample metadata.
        If `div_type='phyl'`, must also contain:
            - ``tree`` (pandas.DataFrame or compatible structure): phylogenetic tree info.
    q_range : tuple of float, default=(0.0, 2.0)
        Inclusive range (start, end) of diversity orders to evaluate.
    q_step : float, default=0.05
        Step size between q values. Must be positive.
    distmat : pandas.DataFrame, optional
        Functional distance matrix (features x features). Required when `div_type='func'`.
    div_type : {'naive', 'phyl', 'func'}, default='naive'
        Diversity type:
        - 'naive': taxonomic alpha diversity.
        - 'phyl' : phylogenetic alpha diversity (requires ``tree`` in `obj`).
        - 'func' : functional alpha diversity (requires `distmat`).
    color_by : str, optional
        Metadata column name used to group legend colors. If None, each sample
        is labeled individually.
    order : str, optional
        Metadata column name used to sort samples before plotting.
    ylog : bool, default=False
        If True, plot alpha diversity on a logarithmic y-scale.
    figsize : tuple of float, default=(18/2.54, 14/2.54)
        Figure size in inches.
    fontsize : int, default=10
        Font size for plot text.
    colorlist : list of str, optional
        List of colors to use for groups or samples. If None, uses package
        defaults via ``get_colors_markers('colors')`` or Matplotlib's cycle.
    use_values_in_tab : bool, default=False
        Pass-through flag to alpha diversity backends (e.g., whether `tab` is
        already normalized).
    savename : str, optional
        If provided, saves the figure to this path and also as PDF
        (i.e., `savename` and `savename + '.pdf'`).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object for the figure.
    df : pandas.DataFrame
        DataFrame with rows = q-values and columns = samples, containing
        computed alpha diversity values.

    Notes
    -----
    - For `div_type='phyl'`, `get_df(obj, 'tree')` must exist.
    - For `div_type='func'`, `distmat` must be provided and compatible with `tab`.
    - The legend groups are deduplicated using the values of `color_by`.
      Only the first occurrence of each group is shown in the legend.

    Examples
    --------
    >>> fig, ax, df = alpha_diversity(obj, q_range=(0, 2), q_step=0.1,
    ...                           div_type='naive', color_by='Treatment')
    >>> df.head()
    """
    # --- Validate inputs ------------------------------------------------------
    tab = get_df(obj, "tab")
    meta = get_df(obj, "meta")
    if tab is None or not isinstance(tab, pd.DataFrame):
        raise ValueError("`obj` must contain a 'tab' DataFrame.")
    if meta is None or not isinstance(meta, pd.DataFrame):
        raise ValueError("`obj` must contain a 'meta' DataFrame.")

    if not isinstance(q_range, (tuple, list)) or len(q_range) != 2:
        raise ValueError("`q_range` must be a (start, end) tuple of floats.")
    q_start, q_end = float(q_range[0]), float(q_range[1])
    if q_start > q_end:
        raise ValueError("`q_range` must satisfy start <= end.")
    if q_step <= 0:
        raise ValueError("`q_step` must be positive.")

    div_type = str(div_type).lower()
    if div_type not in {"naive", "phyl", "func"}:
        raise ValueError("`div_type` must be one of {'naive', 'phyl', 'func'}.")

    tree = None
    if div_type == "phyl":
        tree = get_df(obj, "tree")
        if tree is None:
            raise ValueError("`div_type='phyl'` requires `obj` to provide a 'tree' via get_df(obj, 'tree').")

    if div_type == "func":
        if distmat is None or not isinstance(distmat, pd.DataFrame):
            raise ValueError("`div_type='func'` requires `distmat` as a pandas.DataFrame.")
        # Optional compatibility check: feature axis alignment
        common = set(distmat.index).intersection(tab.index)
        if len(common) == 0:
            raise ValueError("No overlapping features between `tab` rows and `distmat` index. "
                             "Ensure both use the same feature IDs.")

    # Sort samples if requested
    if order is not None:
        if order not in meta.columns:
            raise KeyError(f"`order` column '{order}' not found in metadata.")
        meta = meta.sort_values(by=order)

    smplist = meta.index.tolist()
    missing_cols = [c for c in smplist if c not in tab.columns]
    if missing_cols:
        raise ValueError(f"The following samples from meta are missing in tab columns: {missing_cols}")

    # --- Prepare q values and result DataFrame --------------------------------
    # Include end point with a small epsilon to avoid floating point exclusion
    xvalues = np.arange(q_start, q_end + (q_step / 2.0), q_step)
    df = pd.DataFrame(index=xvalues, columns=smplist, dtype=float)

    # Subset abundance table to the sorted samples
    tab_use = tab[smplist]

    # --- Compute alpha diversity per q ----------------------------------------
    for q in xvalues:
        if div_type == "naive":
            alphadiv = naive_alpha(tab_use, q=q, use_values_in_tab=use_values_in_tab)
        elif div_type == "phyl":
            alphadiv = phyl_alpha(tab_use, tree=obj["tree"], q=q, use_values_in_tab=use_values_in_tab)
        elif div_type == "func":
            alphadiv = func_alpha(tab_use, distmat=distmat, q=q, use_values_in_tab=use_values_in_tab)
        else:
            # Defensive programming (unreachable due to earlier validation)
            raise RuntimeError(f"Unsupported div_type: {div_type}")

        # Expect alphadiv to be array-like or Series aligned to smplist
        try:
            df.loc[q, smplist] = alphadiv
        except Exception as e:
            raise ValueError(f"Failed to assign alpha diversity values for q={q}. "
                             f"Ensure the backend returns values aligned to samples. Error: {e}")

    # --- Plotting --------------------------------------------------------------
    plt.rcParams.update({"font.size": fontsize})
    fig, ax = plt.subplots(figsize=figsize)

    # Determine colors
    if colorlist is None:
        try:
            colorlist = get_colors_markers("colors")
        except Exception:
            # Fallback to Matplotlib default color cycle
            colorlist = plt.rcParams.get("axes.prop_cycle", None)
            colorlist = colorlist.by_key()["color"] if colorlist is not None else ["C0", "C1", "C2", "C3", "C4"]

    seen_groups: List[str] = []

    for s in df.columns:
        group = meta.loc[s, color_by] if (color_by is not None) else s
        # Legend label de-duplication
        label = group if group not in seen_groups else "_nolegend_"
        if group not in seen_groups:
            seen_groups.append(group)
        colnr = seen_groups.index(group)  # stable order by first appearance
        col = colorlist[colnr % len(colorlist)]

        if ylog:
            ax.semilogy(df.index, df[s].values, lw=1, color=col, label=label)
        else:
            ax.plot(df.index, df[s].values, lw=1, color=col, label=label)

    # Axis labels
    if div_type == "naive":
        ax.set_ylabel(r"Diversity ($^{q}$D)")
    elif div_type == "phyl":
        ax.set_ylabel(r"Diversity ($^{q}$PD)")
    elif div_type == "func":
        ax.set_ylabel(r"Diversity ($^{q}$FD)")

    ax.set_xlabel("Diversity order (q)")

    # Ticks and limits based on q_range
    tick_step = 0.5
    xticks = np.arange(q_start, q_end + 1e-12, tick_step)
    ax.set_xticks(xticks)
    ax.set_xlim(q_start, q_end)

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", frameon=False)
    plt.tight_layout()

    # Saving
    if savename:
        plt.savefig(savename, dpi=240)
        try:
            plt.savefig(f"{savename}.pdf", format="pdf")
        except Exception:
            # Fallback silently if a PDF backend is not available in the environment
            pass

    return fig, ax, df

# -----------------------------------------------------------------------------
# Plot beta diversity profiles
# -----------------------------------------------------------------------------
def beta_diversity_profile(
    obj: Union[Dict[str, Any], Any],
    *,
    q_range: Tuple[float, float] = (0.0, 2.0),
    q_step: float = 0.05,
    distmat: Optional[pd.DataFrame] = None,
    group_by: Optional[str] = None,
    order: Optional[str] = None,
    dis: bool = True,
    viewpoint: str = "regional",   # {'local', 'regional'} only used when dis=True
    div_type: str = "naive",
    ylog: bool = False,
    figsize: Tuple[float, float] = (18 / 2.54, 14 / 2.54),
    fontsize: int = 10,
    colorlist: Optional[List[str]] = None,
    savename: Optional[str] = None,
    drop_na_groups: bool = True,
) -> Tuple["plt.Figure", "plt.Axes", pd.DataFrame]:
    """
    Plot multi-sample β-diversity (or its dissimilarity transform) vs diversity order q.

    This function evaluates β_q across a grid of q-values using `naive_multi_beta`
    and plots a curve per group (or a single curve labeled 'all' if `group_by=None`).
    It can optionally convert β to dissimilarity using the "local" or "regional"
    viewpoints (as returned by `naive_multi_beta`).

    Parameters
    ----------
    obj : dict or MicrobiomeData-like
        Must support `get_df(obj, 'tab')` -> DataFrame (features × samples) and
        `get_df(obj, 'meta')` -> DataFrame (sample metadata).
    q_range : (float, float), default=(0.0, 2.0)
        Inclusive (start, end) range of q-values.
    q_step : float, default=0.05
        Step size between consecutive q values. Must be positive.
    distmat : pandas.DataFrame, optional
        Functional distance matrix (features x features). Required when `div_type='func'`.
    group_by : str or None, default=None
        Metadata column defining groups of samples. If None, treats all samples as one group.
    order : str or None, default=None
        Metadata column used to sort samples before computing group order. The
        order of first appearance of groups in the (optionally) sorted metadata
        determines the plotting order.
    dis : bool, default=True
        If True, plot dissimilarity instead of raw β. Uses the `viewpoint` column
        ('local_dis' or 'regional_dis') returned by `naive_multi_beta`.
    viewpoint : {'local', 'regional'}, default='regional'
        Which dissimilarity column to use when `dis=True`.
    div_type : {'naive', 'phyl', 'func'}, default='naive'
        Diversity type:
        - 'naive': taxonomic alpha diversity.
        - 'phyl' : phylogenetic alpha diversity (requires ``tree`` in `obj`).
        - 'func' : functional alpha diversity (requires `distmat`).
    ylog : bool, default=False
        If True, use a logarithmic y-scale. Note that dissimilarities may include
        zeros, which cannot be shown on a log scale; such points will be omitted.
    figsize : (float, float), default=(18/2.54, 14/2.54)
        Figure size in inches.
    fontsize : int, default=10
        Base font size.
    colorlist : list of str or None, default=None
        Colors for groups. If None, uses Matplotlib's default color cycle.
    savename : str or None, default=None
        If provided, saves the figure as `savename` (raster) and `savename + '.pdf'`.
    drop_na_groups : bool, default=True
        If True, drops groups that are entirely NaN across all q (e.g., groups with <2 samples).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object for the figure.
    df : pandas.DataFrame
        DataFrame with rows = q-values and columns = groups. Contains β (if `dis=False`)
        or dissimilarity (if `dis=True`) for each group at each q.
    """
    # --- Validate input -------------------------------------------------------
    tab = get_df(obj, "tab")
    meta = get_df(obj, "meta")
    if not isinstance(tab, pd.DataFrame):
        raise ValueError("`obj` must provide a 'tab' DataFrame via get_df(obj, 'tab').")
    if not isinstance(meta, pd.DataFrame):
        raise ValueError("`obj` must provide a 'meta' DataFrame via get_df(obj, 'meta').")

    if tab.shape[1] < 2:
        raise ValueError("At least two samples are required to compute multi-sample β-diversity.")

    # q-range
    if not isinstance(q_range, (tuple, list)) or len(q_range) != 2:
        raise ValueError("`q_range` must be a (start, end) tuple of floats.")
    q_start, q_end = float(q_range[0]), float(q_range[1])
    if q_start > q_end:
        raise ValueError("`q_range` must satisfy start <= end.")
    if q_step <= 0:
        raise ValueError("`q_step` must be positive.")

    # viewpoint selection for dissimilarity
    viewpoint = str(viewpoint).lower()
    if dis:
        if viewpoint not in {"local", "regional"}:
            raise ValueError("`viewpoint` must be 'local' or 'regional' when `dis=True`.")
        ycol = "local_dis" if viewpoint == "local" else "regional_dis"
        y_label = f"Dissimilarity ({viewpoint})"
    else:
        ycol = "beta"
        y_label = r"Multi-sample $\beta_q$"

    # Determine plotting order via metadata
    if order is not None:
        if order not in meta.columns:
            raise KeyError(f"`order` column '{order}' not found in metadata.")
        meta_sorted = meta.sort_values(by=order)
    else:
        meta_sorted = meta
    
    # Build the group order based on first appearance in the (optionally) sorted meta
    if group_by is None:
        groups_order = ["all"]
    else:
        if group_by not in meta.columns:
            raise KeyError(f"`group_by` column '{group_by}' not found in metadata.")
        # Preserve order of first appearance
        groups_order = pd.unique(meta_sorted[group_by].astype(str))

    # Prepare q grid and results container
    q_values = np.arange(q_start, q_end + (q_step / 2.0), q_step)  # inclusive of end
    df = pd.DataFrame(index=np.round(q_values, 10), columns=groups_order, dtype=float)

    # --- Compute β (or dissimilarity) curves ----------------------------------
    for q in df.index:
        if div_type == "naive":
            betadf = naive_multi_beta(obj, by=group_by, q=float(q))
        elif div_type == "phyl":
            betadf = phyl_multi_beta(obj, by=group_by, q=float(q))
        elif div_type == "func":
            betadf = func_multi_beta(obj, distmat, by=group_by, q=float(q))
        else:
            # Defensive programming (unreachable due to earlier validation)
            raise RuntimeError(f"Unsupported div_type: {div_type}")

        # Reindex to our desired plotting order
        y = betadf[ycol].reindex(groups_order)

        # Optionally drop groups that have <2 samples => NaN, but do that once at the end
        df.loc[q, groups_order] = y.values

    # Drop all-NaN groups if requested (e.g., groups with <2 samples throughout)
    if drop_na_groups:
        all_nan_cols = df.columns[df.isna().all()]
        if len(all_nan_cols) > 0:
            df = df.drop(columns=all_nan_cols)
            # Update groups order to match what remains
            groups_order = [g for g in groups_order if g in df.columns]

    # --- Plotting --------------------------------------------------------------
    plt.rcParams.update({"font.size": fontsize})
    fig, ax = plt.subplots(figsize=figsize)

    # Colors
    if colorlist is None:
        prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
        colorlist = prop_cycle.by_key()["color"] if prop_cycle is not None else ["C0", "C1", "C2", "C3", "C4"]

    # Draw one line per group
    for i, grp in enumerate(groups_order):
        if grp not in df.columns:
            continue
        yvals = df[grp].values

        # For log scale, omit non-positive values silently (Matplotlib will skip them)
        if ylog:
            ax.semilogy(df.index.values, yvals, lw=1.5, color=colorlist[i % len(colorlist)], label=str(grp))
        else:
            ax.plot(df.index.values, yvals, lw=1.5, color=colorlist[i % len(colorlist)], label=str(grp))

    # Axes and labels
    ax.set_xlabel("Diversity order (q)")
    ax.set_ylabel(y_label)

    # Ticks and limits
    xticks = np.arange(q_start, q_end + 1e-12, 0.5)
    ax.set_xticks(xticks)
    ax.set_xlim(q_start, q_end)

    # Legend
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", frameon=False, title=(group_by if group_by else "Group"))

    plt.tight_layout()

    # Save
    if savename:
        plt.savefig(savename, dpi=240)
        try:
            plt.savefig(f"{savename}.pdf", format="pdf")
        except Exception:
            # Fallback silently if a PDF backend is not available in the environment
            pass

    return fig, ax, df
