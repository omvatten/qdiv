from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
import math
import copy
from ..io import merge_samples, subset_features, subset_taxa
from ..utils import groupbytaxa, get_colors_markers, get_df
from ..diversity import naive_alpha, phyl_alpha, func_alpha


def _get_ra_table(
    obj: Union[Dict[str, Any], Any],
    *,
    group_by: Optional[str] = None,
    value_aggregation: Literal["sum", "mean"] = "sum",
    order: Optional[str] = None,
    levels: Optional[List[str]] = None,
    include_index: bool = False,
    levels_shown: Optional[str] = None,
    subset_levels: Optional[Union[str, List[str]]] = None,
    subset_patterns: Optional[Union[str, List[str]]] = None,
    n: int = 20,
    featurelist: Optional[List[str]] = None,
    method: Literal["max", "mean"] = "max",
    sorting: Literal["abundance", "alphabetical"] = "abundance",
    use_values_in_tab: bool = False,
    italics: bool = False,
) -> pd.DataFrame:
    """
    Plot a heatmap of taxa abundances.

    Parameters
    ----------
    obj : dict or MicrobiomeData
        Input data containing at least:
            - 'tab': pandas.DataFrame
                Abundance table (features x samples).
            - 'tax': pandas.DataFrame
                Taxonomy table (features x taxonomic levels).
    group_by : str or list, optional
        Metadata column(s) used to merge samples.
    value_aggregation : {'sum', 'mean'}, default = 'sum'
    order : str, optional
        Metadata column used to order samples along the x-axis.
    levels : list of str, optional
        Taxonomic levels used for y-axis grouping.
    include_index : bool, default=False
        Whether to include the feature index in labels.
    levels_shown : {'number', None}, optional
        If 'number', show numeric labels instead of taxonomic names.
    subset_levels : str or list of str, optional
        Taxonomic levels to filter by.
    subset_patterns : str or list of str, optional
        Text patterns to filter taxa.
    n : int, default=20
        Number of top taxa to plot (ignored if `featurelist` is provided).
    featurelist : list of str, optional
        Specific features to plot.
    method : {'max', 'min'}, default = 'max'
    sorting : {'abundance', 'alphabetical'}, default = 'abundance'
    use_values_in_tab : bool, default = False
    italics : bool, default=False
        If True, italicize taxonomic names where appropriate.

    Returns
    -------
    table : pandas.DataFrame
    """
    tab = get_df(obj, "tab")
    if tab is None:
        raise ValueError("Input must contain a 'tab' dataframe")

    if hasattr(obj, "to_dict"):
        merged_obj = obj.to_dict()
    elif isinstance(obj, dict):
        merged_obj = obj
    else:
        raise ValueError("Input must be a dictionary or a MicrobiomeData object")

    # --- Merge samples ---
    if group_by is not None:
        merged_obj = merge_samples(merged_obj, by=group_by, method=value_aggregation)

    # --- Normalize to relative abundance ---
    if not use_values_in_tab:
        merged_obj["tab"] = 100 * merged_obj["tab"] / merged_obj["tab"].sum()

    # --- Order samples ---
    logiclist = None
    if order and "meta" in merged_obj:
        md = merged_obj["meta"].copy()
        md[order] = md[order].astype(float)
        md = md.sort_values(by=order)
        logiclist = list(dict.fromkeys(md[group_by] if group_by else md.index))
        merged_obj["meta"] = md

    # --- Subset features ---
    if featurelist:
        merged_obj = subset_features(merged_obj, featurelist=featurelist)
    elif subset_patterns:
        merged_obj = subset_taxa(
            merged_obj,
            subset_levels=subset_levels,
            subset_patterns=subset_patterns,
        )

    # --- Group by taxa ---
    if isinstance(levels, str) and levels is not None:
        levels = [levels]
    taxa_obj = groupbytaxa(merged_obj, levels=levels, include_index=include_index, italics=italics)
    ra = taxa_obj["tab"]
    table = ra.copy()
    if table.empty:
        raise ValueError("Data is missing in table after groupbytaxa.")

    # --- Select top taxa ---
    if not featurelist:
        if method == "max":
            ra["rank"] = ra.max(axis=1)
        elif method == "mean":
            ra["rank"] = ra.mean(axis=1)
        ra = ra.sort_values(by="rank", ascending=False)
        retain = ra.index[:n]
        table = table.loc[retain]

    # --- Sort taxa (y-axis) ---
    if sorting == "abundance":
        table["avg"] = table.mean(axis=1)
        table = table.sort_values(by="avg", ascending=True).drop(columns="avg")
    elif sorting == "tax":
        tax = taxa_obj["tax"].loc[table.index].fillna("zzz")
        tax = tax.sort_values(tax.columns.tolist())
        table = table.loc[tax.index]

    # --- Sort samples (x-axis) ---
    if logiclist:
        table = table.loc[:, logiclist]

    # --- Replace labels with numbers ---
    if levels_shown == "number":
        table.index = list(range(len(table.index), 0, -1))

    return table

# -----------------------------------------------------------------------------
# Plot heatmap of taxa relative abundances
# -----------------------------------------------------------------------------
def heatmap(
    obj: Union[Dict[str, Any], Any],
    *,
    group_by: Optional[Union[str, List[str]]] = None,
    value_aggregation: Literal["sum", "mean"] = "sum",
    order: Optional[str] = None,
    levels: Optional[List[str]] = None,
    include_index: bool = False,
    levels_shown: Optional[str] = None,
    subset_levels: Optional[Union[str, List[str]]] = None,
    subset_patterns: Optional[Union[str, List[str]]] = None,
    n: int = 20,
    featurelist: Optional[List[str]] = None,
    method: Literal["max", "mean"] = "max",
    sorting: Literal["abundance", "alphabetical"] = "abundance",
    use_values_in_tab: bool = False,
    italics: bool = False,

    figsize: Tuple[float, float] = (14, 10),
    fontsize: int = 15,
    sep_col: Union[List[int], int, None] = None,
    sep_line: Union[List[int], int, None] = None,
    labels: bool = True,
    labelsize: int = 10,
    color_threshold: float = 8.0,
    cmap: str = "Reds",
    gamma: float = 0.5,
    colorbar_ticks: Optional[List[float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 240,
    savename: Optional[str] = None,
) -> Tuple["plt.Figure", "plt.Axes", "pd.DataFrame"]:
    """
    Plot a heatmap of taxa abundances.

    Parameters
    ----------
    obj : dict or MicrobiomeData
        Input data containing at least:
            - 'tab': pandas.DataFrame
                Abundance table (features x samples).
            - 'tax': pandas.DataFrame
                Taxonomy table (features x taxonomic levels).
    group_by : str or list, optional
        Metadata column(s) used to merge samples.
    value_aggregation : {'sum', 'mean'}, default = 'sum'
    order : str, optional
        Metadata column used to order samples along the x-axis.
    levels : list of str, optional
        Taxonomic levels used for y-axis grouping.
    include_index : bool, default=False
        Whether to include the feature index in labels.
    levels_shown : {'number', None}, optional
        If 'number', show numeric labels instead of taxonomic names.
    subset_levels : str or list of str, optional
        Taxonomic levels to filter by.
    subset_patterns : str or list of str, optional
        Text patterns to filter taxa.
    n : int, default=20
        Number of top taxa to plot (ignored if `featurelist` is provided).
    featurelist : list of str, optional
        Specific features to plot.
    method : {'max', 'min'}, default = 'max'
    sorting : {'abundance', 'alphabetical'}, default = 'abundance'
    italics : bool, default=False
        If True, italicize taxonomic names where appropriate.

    figsize : tuple of float, default=(14, 10)
        Figure size in inches.
    fontsize : int, default=15
        Font size for axis labels.
    sep_col : list of int, optional
        Column indices where separators are inserted.
    sep_line : list of int, optional
        Column indices where vertical lines are drawn.
    labels : bool, default=True
        Whether to show abundance values in cells.
    labelsize : int, default=10
        Font size of cell labels.
    color_threshold : float, default=8.0
        Threshold for switching label color (black/white).
    cmap : str, default='Reds'
        Colormap for heatmap.
    gamma : float, default=0.5
        Gamma for PowerNorm scaling.
    colorbar_ticks : list of float, optional
        Tick marks for colorbar.
    vmin : float, optional
        Minimum value for cplor normalization (passed to PowerNorm).
    vmax : float, optional
        Maximum value for cplor normalization (passed to PowerNorm).
    dpi : int, default 240
        Resolution of saved figure.
    savename : str, optional
        Filename to save figure (PNG and PDF). If None, figure is not saved.
    use_values_in_tab : bool, default = False


    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object for the figure.
    table : pandas.DataFrame
        The final abundance table (after grouping, filtering, and sorting) that was plotted.

    Examples
    --------
    >>> heatmap(obj, group_by='Treatment', levels=['Genus'], n=30, savename='heatmap.png')
    """

    obj = copy.deepcopy(obj)
    meta = get_df(obj, "meta")
    if meta is None and group_by is not None:
        raise ValueError('meta is missing in obj')

    if group_by is None:
        combined = None
    elif isinstance(group_by, str):
        combined = group_by
    elif isinstance(group_by, list):
        combined = "_".join(group_by)
        meta[combined] = meta[group_by[0]].astype(str)
        if len(group_by) > 1:
            for i in range(1, len(group_by)):
                meta[combined] = meta[combined] + "_" + meta[group_by[i]].astype(str)
        if hasattr(obj, "meta"):
            obj.meta = meta
        elif isinstance(obj, dict):
            obj["meta"] = meta
    else:
        raise ValueError('group_by is unknown format.')

    table = _get_ra_table(
        obj=obj,
        group_by=combined,
        value_aggregation=value_aggregation,
        order=order,
        levels=levels,
        include_index=include_index,
        levels_shown=levels_shown,
        subset_levels=subset_levels,
        subset_patterns=subset_patterns,
        n=n,
        featurelist=featurelist,
        method=method,
        sorting=sorting,
        use_values_in_tab=use_values_in_tab,
        italics=italics
    )

    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("Error in constructing relative abundance table.")

    # --- Format cell labels ---
    labelvalues = None
    if labels:
        labelvalues = table.copy()
        labelvalues = labelvalues.astype(str)
        for r in table.index:
            for c in table.columns:
                value = float(table.loc[r, c])
                if 0 < value < 0.1:
                    labelvalues.loc[r, c] = "<0.1"
                elif 0.1 <= value < 9.95:
                    labelvalues.loc[r, c] = str(round(value, 1))
                elif value >= 9.95 and value <= 99:
                    labelvalues.loc[r, c] = str(int(round(value, 0)))
                elif value > 99:
                    labelvalues.loc[r, c] = "99"
                else:
                    labelvalues.loc[r, c] = "0"

    # --- Insert separators ---
    if isinstance(sep_col, int):
        sep_col = [sep_col]
    if isinstance(sep_col, list) and sep_col is not None and max(sep_col) < len(table.columns):
        for i, col in enumerate(sep_col):
            table.insert(loc=col + i, column=" " * (i + 1), value=0)
            if labels and labelvalues is not None:
                labelvalues.insert(loc=col + i, column=" " * (i + 1), value="")

    # --- Plot heatmap ---
    plt.rcParams.update({"font.size": fontsize})
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(table, cmap=cmap, norm=mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax), aspect="auto")
    if colorbar_ticks:
        fig.colorbar(im, ticks=colorbar_ticks)

    # Axes
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_xticklabels(table.columns, rotation=90)
    ax.set_yticklabels(table.index)

    # Grid
    ax.set_xticks(np.arange(-0.5, len(table.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(table.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)

    if isinstance(sep_col, list) and sep_col is not None and max(sep_col) < len(table.columns):
        for i, col in enumerate(sep_col):
            for j in range(6):
                ax.axvline(col + i - 0.5 + j / 5, 0, len(table.index), linestyle="-", lw=1, color="white")
    if isinstance(sep_line, int):
        sep_line = [sep_line]
    if isinstance(sep_line, list) and sep_line is not None and max(sep_line) < len(table.columns):
        for col in sep_line:
            ax.axvline(col - 0.5, 0, len(table.index), linestyle="-", color='black')

    # Fix labels inside the heatmap cells
    if labels and labelvalues is not None:
        for r in range(len(table.index)):
            for c in range(len(table.columns)):
                textcolor = "white" if table.iloc[r, c] > color_threshold else "black"
                ax.text(
                    c, r,
                    labelvalues.iloc[r, c],
                    fontsize=labelsize,
                    ha="center", va="center",
                    color=textcolor
                )

    # Adjust layout
    fig.tight_layout()

    # Save figure if requested
    if savename:
        plt.savefig(savename, dpi=dpi)
        try:
            plt.savefig(f"{savename}.pdf", format="pdf")
        except Exception:
            # Fallback silently if a PDF backend is not available in the environment
            pass
    return fig, ax, table
                       

# -----------------------------------------------------------------------------
# Plot rarefaction curves
# -----------------------------------------------------------------------------
def rarefactioncurve(
    obj: Union[Dict[str, Any], Any],
    distmat: Optional[Union[str, pd.DataFrame]] = None,
    *,
    step: Union[str, int] = "flexible",
    div_type: str = "naive",
    q: float = 0.0,
    figsize: Tuple[float, float] = (14, 10),
    fontsize: int = 18,
    color_by: Optional[str] = None,
    order: Optional[str] = None,
    tag: Optional[str] = None,
    colorlist: Optional[List[str]] = None,
    only_return_data: bool = False,
    only_plot_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    savename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate and plot rarefaction curves for alpha diversity (Hill numbers).

    The function subsamples (without replacement) individual reads within each
    sample to compute the rarefaction curve for a chosen diversity type, then
    plots per-sample curves. If `only_return_data=True`, it returns the computed
    curves instead of plotting them. You can also supply precomputed curves via
    `only_plot_data` to plot without recomputation.

    Parameters
    ----------
    obj : dict or MicrobiomeData
        Input data containing at least:
            - 'tab': pandas.DataFrame
                Abundance table (features x samples).
            - `meta` (pd.DataFrame): metadata with sample IDs as index matching ``tab`` columns.
        Optional keys depending on `div_type`:
        - ``tree``: phylogenetic tree object (required if ``div_type='phyl'``).
    distmat : str or pandas.DataFrame or None, optional
        Distance matrix required when ``div_type='func'``. Can be a preloaded DataFrame
        or a path-like string handled by your ``func_alpha`` implementation.
    step : {'flexible'} or int, default='flexible'
        Subsampling step size (depth increments).
        If 'flexible', the total reads of each sample are divided by 20 (min 1).
        If an integer, it must be a positive step size in reads.
    div_type : {'naive', 'phyl', 'func'}, default='naive'
        Diversity measure to compute:
        - 'naive' : taxonomic (plain) diversity via ``naive_alpha``.
        - 'phyl'  : phylogenetic diversity via ``phyl_alpha`` (requires `tree`).
        - 'func'  : functional diversity via ``func_alpha`` (requires ``distmat``).
    q : float, default=0.0
        Order of diversity (Hill number).
    figsize : tuple of float, default=(14, 10)
        Figure size (width, height) in inches.
    fontsize : int, default=18
        Base font size for the plot.
    color_by : str, optional
        Metadata column in used to color-code lines (group legend).
    order : str, optional
        Metadata column in used to order samples along the legend
        or visual grouping in the plot.
    tag : {'index'} or str, optional
        If 'index', annotate curve endpoints with sample IDs.
        If a metadata column name, annotate with that column's values.
    colorlist : list of str, optional
        Colors used for plotting. If not provided, colors are drawn from
        ``get_colors_markers('colors')``. Ensure the list is long enough for all
        groups/samples.
    only_return_data : bool, default=False
        If True, return the computed data dictionary and do not plot.
    only_plot_data : dict, optional
        Precomputed data dictionary to plot (skips computation). The format is:
        ``{sample_id: (xvals: np.ndarray, yvals: np.ndarray)}``.
    savename : str, optional
        If provided, save the plot to ``savename`` and also to a PDF file
        ``savename + '.pdf'`` (unless ``savename`` already ends with ``.pdf``).

    Returns
    -------
    dict
        Returns a dictionary with the keys 'meta', which holds the metadata dataframe
        and 'samples', which is another dictionary mapping sample IDs to
        (x, y) arrays for the rarefaction curves.

    Notes
    -----
    - The function shuffles individual reads per sample using ``numpy.random.shuffle``.
      For reproducibility, set the global NumPy random seed before calling.
    - Helper functions ``naive_alpha``, ``phyl_alpha``, and ``func_alpha`` are assumed
      to be available in the current namespace.
    - The count table ``obj['tab']`` must contain non-negative integers; zero-count
      features are ignored per sample during accumulation.

    Examples
    --------
    Compute and plot, coloring by a metadata column:

    >>> data = rarefactioncurve(
    ...     obj,
    ...     step='flexible',
    ...     div_type='naive',
    ...     q=0,
    ...     color_by='Treatment',
    ...     savename='rarefaction.png'
    ... )

    >>> rd = rarefactioncurve(obj, step=500, only_return_data=True)

    Plot from precomputed data:

    >>> _ = rarefactioncurve(obj, only_plot_data=rd)  # uses obj['meta'] for annotations
    """
    # --- Validation ---
    if only_plot_data is None:
        tab = get_df(obj, "tab")
        meta = get_df(obj, "meta")
        if tab is None:
            raise ValueError("obj must contain a 'tab' pandas.DataFrame (features x samples).")
        if meta is None:
            raise ValueError("obj must contain a 'meta' pandas.DataFrame (samples as index).")
        if div_type not in {"naive", "phyl", "func"}:
            raise ValueError("div_type must be one of {'naive', 'phyl', 'func'}.")
    
        if div_type == "phyl":
            tree = get_df(obj, "tree")
            if tree is None:
                raise ValueError("div_type='phyl' requires obj['tree'].")
    
        if div_type == "func":
            if distmat is None:
                raise ValueError("div_type='func' requires distmat.")
    
        # Ensure meta index covers all samples present in tab
        missing_meta = [c for c in tab.columns if c not in meta.index]
        if missing_meta:
            raise ValueError(f"Samples missing in meta index: {missing_meta}")

    # --- Plotting helper ---
    def _plot_rarefactioncurve(mxyd) -> None:
        # Sorting by metadata column if provided
        _meta_plot = mxyd["meta"]
        rd = mxyd["samples"]
        if order is not None:
            if order not in _meta_plot.columns:
                raise ValueError(f"'order' column '{order}' not found in meta.")
            _meta_plot = _meta_plot.sort_values(by=[order])

        nonlocal colorlist
        if colorlist is None:
            # User-provided utility assumed available
            colorlist = get_colors_markers("colors")

        plt.rcParams.update({"font.size": fontsize})
        fig, ax = plt.subplots(figsize=figsize)

        # Color coding by group (color_by) or unique colors per sample
        if color_by is not None:
            if color_by not in _meta_plot.columns:
                raise ValueError(f"'color_by' column '{color_by}' not found in meta.")
            smpcats = _meta_plot[color_by].dropna().astype(str).unique().tolist()
            for cat_nr, cat in enumerate(smpcats):
                ax.plot([], [], label=str(cat), color=colorlist[cat_nr % len(colorlist)])
                smplist = _meta_plot[_meta_plot[color_by].astype(str) == str(cat)].index.tolist()
                for smp in smplist:
                    x, y = rd[smp]
                    ax.plot(x, y, label="_nolegend_", color=colorlist[cat_nr % len(colorlist)])
        else:
            # One color per sample (cycled as needed)
            for smp_nr, smp in enumerate(_meta_plot.index.tolist()):
                x, y = rd[smp]
                ax.plot(x, y, label="_nolegend_", color=colorlist[smp_nr % len(colorlist)])

        # Tagging endpoints
        if tag == "index":
            for smp, (x, y) in rd.items():
                ax.annotate(smp, (x[-1], y[-1]), color="black")
        elif tag is not None:
            if tag not in _meta_plot.columns:
                raise ValueError(f"'tag' column '{tag}' not found in meta.")
            for smp, (x, y) in rd.items():
                antext = _meta_plot.loc[smp, tag]
                ax.annotate(str(antext), (x[-1], y[-1]), color="black")

        if color_by is not None:
            ax.legend(bbox_to_anchor=(1, 1), loc="upper left", frameon=False)

        ax.set_xlabel("Reads")
        ax.set_ylabel(rf"$^{{{q}}}D$")  # Hill number notation
        plt.tight_layout()

        if savename:
            plt.savefig(savename)
            if not str(savename).lower().endswith(".pdf"):
                plt.savefig(f"{savename}.pdf", format="pdf")
        plt.show()

    # --- Compute or plot-only ---
    if only_plot_data is not None:
        _plot_rarefactioncurve(only_plot_data)
        return only_plot_data

    res_di = {}
    print("Working on rarefaction curve for sample: ", end="")

    for smp in tab.columns:
        print(f"{smp}.. ", end="")
        smp_series = tab[smp]
        smp_series = smp_series[smp_series > 0]  # positive counts only
        totalreads = int(smp_series.sum())

        # Skip empty samples gracefully
        if totalreads <= 0:
            res_di[smp] = (np.array([0, 1], dtype=int), np.array([0.0, 1.0], dtype=float))
            continue

        # Create per-read labels by expanding counts, then shuffle
        name_arr = smp_series.index.to_list()
        counts_arr = smp_series.to_numpy(dtype=int)
        cumreads2 = np.cumsum(counts_arr)
        cumreads1 = cumreads2 - counts_arr
        ind_reads_arr = np.empty(totalreads, dtype=object)
        for i, (v1, v2) in enumerate(zip(cumreads1, cumreads2)):
            ind_reads_arr[int(v1):int(v2)] = name_arr[i]
        np.random.shuffle(ind_reads_arr)

        # Determine step size
        if isinstance(step, str):
            if step != "flexible":
                raise ValueError("When 'step' is a string, only 'flexible' is supported.")
            step_size = max(1, totalreads // 20)
        else:
            if not isinstance(step, int) or step <= 0:
                raise ValueError("'step' must be a positive integer or 'flexible'.")
            step_size = min(step, totalreads)  # cap at totalreads

        # Build x and y values for the rarefaction curve
        xvals = np.arange(step_size, totalreads, step_size, dtype=int)
        yvals = np.zeros(len(xvals), dtype=float)

        for i, depth in enumerate(xvals):
            uniq, counts = np.unique(ind_reads_arr[:depth], return_counts=True)
            temp_tab = pd.DataFrame(counts, index=uniq, columns=[smp])

            if div_type == "naive":
                div_val = naive_alpha(temp_tab, q=q)
            elif div_type == "phyl":
                div_val = phyl_alpha({'tab': temp_tab, 'tree': tree}, q=q)
            else:  # 'func'
                div_val = func_alpha(temp_tab, distmat, q=q)
            yvals[i] = float(div_val[smp])

        # Add true value at totalreads and seed initial points at 0 and 1
        if div_type == "naive":
            div_val_full = naive_alpha(tab[[smp]], q=q)
        elif div_type == "phyl":
            div_val_full = phyl_alpha({'tab': tab[[smp]], 'tree': tree}, q=q)
        else:
            div_val_full = func_alpha(tab[[smp]], distmat, q=q)

        xvals = np.append(xvals, totalreads)
        yvals = np.append(yvals, float(div_val_full[smp]))
        xvals = np.insert(xvals, 0, [0, 1])
        yvals = np.insert(yvals, 0, [0.0, 1.0])

        res_di[smp] = (xvals, yvals)
    print("Done")
    out = {}
    out["samples"] = res_di
    out["meta"] = meta
    
    if not only_return_data:
        _plot_rarefactioncurve(out)

    return out

# -----------------------------------------------------------------------------
# Octave plot
# -----------------------------------------------------------------------------
def octave(
    obj: Union[Dict[str, Any], Any],
    *,
    group_by: Optional[str] = None,
    values: Optional[List[str]] = None,
    nrows: int = 2,
    ncols: int = 2,
    fontsize: int = 11,
    figsize: Tuple[float, float] = (10, 6),
    xlabels: bool = True,
    ylabels: bool = True,
    title: bool = True,
    color: str = "blue",
    savename: Optional[str] = None,
) -> Tuple["plt.figure.Figure", "pd.DataFrame"]:
    """
    Plot octave distributions of ASV abundances according to Edgar & Flyvbjerg (DOI: 10.1101/38983).

    This function bins feature counts into logarithmic intervals (powers of 2) and plots
    histograms for each sample or merged group of samples. Useful for visualizing
    abundance distributions across samples.

    Parameters
    ----------
    obj : dict or MicrobiomeData
        Input data containing at least:
            - 'tab': pandas.DataFrame. Abundance table (features x samples).
        Optional key:
        - ``meta`` (pandas.DataFrame): metadata table for sample grouping.
    group_by : str, optional
        Metadata column name used to merge samples by category. If None, each sample
        is plotted individually.
    values : list of str, optional
        Subset of sample names or metadata values to include. If None, all samples
        or all categories in ``group_by`` are used.
    nrows : int, default=2
        Number of rows in the subplot grid.
    ncols : int, default=2
        Number of columns in the subplot grid. ``nrows * ncols`` must be >= number of panels.
    fontsize : int, default=11
        Font size for plot text.
    figsize : tuple of float, default=(10, 6)
        Figure size in inches.
    xlabels : bool, default=True
        Whether to show x-axis labels (k bins).
    ylabels : bool, default=True
        Whether to show y-axis labels (ASV counts).
    title : bool, default=True
        Whether to display sample name or group name as subplot title.
    color : str, default='blue'
        Color of the bars in the histograms.
    savename : str, optional
        If provided, save the figure to this path and also as PDF. Additionally,
        export the bin counts as a CSV file (``savename + '.csv'``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    df : pandas.DataFrame
        DataFrame containing bin definitions and counts per sample/group.
        Columns: ['k', 'min_count', 'max_count', sample1, sample2, ...].
        Returns None if plotting fails due to insufficient panels.

    Notes
    -----
    - Bins are defined as intervals [2^k, 2^(k+1)).
    - If the number of samples exceeds ``nrows * ncols``, the function prints a warning
      and returns None without plotting.

    Examples
    --------
    >>> df = octave(obj, group_by='Treatment', nrows=2, ncols=3, color='green', savename='octave_plot')
    >>> print(df.head())
    """
    # --- Prepare data ---
    tab = get_df(obj, "tab")
    if tab is None or tab.empty:
        raise ValueError("tab is missing.")

    meta = get_df(obj, "meta")
    if group_by is not None and meta is None:
        raise ValueError("meta is missing.")
 
    if group_by is None:
        smplist = tab.columns.tolist()
    else:
        merged_obj = merge_samples({'tab':tab, 'meta':meta}, group_by=group_by, values=values)
        tab = merged_obj["tab"].copy()
        smplist = tab.columns.tolist()

    if len(smplist) > nrows * ncols:
        print(f"Too few panels: {len(smplist)} needed, but only {nrows * ncols} available.")
        return None

    # Compute bin range
    max_read = tab.max().max()
    max_k = math.floor(math.log(max_read, 2)) if max_read >= 1 else math.ceil(math.log(max_read, 2))

    min_read = tab[tab > 0].min().min()
    min_k = math.floor(math.log(min_read, 2)) if min_read >= 1 else math.ceil(math.log(min_read, 2))
    min_k = min(min_k, 0)

    k_index = np.arange(min_k, max_k + 1)
    df = pd.DataFrame(0, index=k_index, columns=["k", "min_count", "max_count"] + smplist)
    df["k"] = k_index
    df["min_count"] = 2.0 ** k_index
    df["max_count"] = 2.0 ** (k_index + 1)

    # --- Plotting ---
    plt.rcParams.update({"font.size": fontsize})
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(nrows, ncols)
    gs.update(wspace=0, hspace=0)

    for smp_nr, smp in enumerate(smplist):
        row = smp_nr // ncols
        col = smp_nr % ncols
        ax = fig.add_subplot(gs[row, col], frame_on=True)

        # Count ASVs per bin
        for k in df.index:
            bin_min = df.loc[k, "min_count"]
            bin_max = df.loc[k, "max_count"]
            temp = tab.loc[(tab[smp] >= bin_min) & (tab[smp] < bin_max), smp]
            df.loc[k, smp] = len(temp)

        ax.bar(df["k"], df[smp], color=color)
        ax.set_xticks(k_index[::2])

        if xlabels and row == nrows - 1:
            ax.set_xticklabels(k_index[::2])
            ax.set_xlabel(r"k (bin [$\geq 2^k$ and < $2^{k+1}$])")
        elif xlabels:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xticklabels([])

        if ylabels and col == 0:
            ax.set_ylabel("Count")
        elif ylabels:
            ax.set_ylabel("")
        else:
            ax.set_yticklabels([])

        if title:
            ax.text(
                0.97 * ax.get_xlim()[1],
                0.97 * ax.get_ylim()[1],
                str(smp),
                verticalalignment="top",
                horizontalalignment="right",
            )

    # --- Save outputs ---
    if savename:
        plt.savefig(savename)
        try:
            plt.savefig(f"{savename}.pdf", format="pdf")
        except Exception:
            # Fallback silently if a PDF backend is not available in the environment
            pass

        df.to_csv(f"{savename}.csv", index=False)

    return fig, df

# -----------------------------------------------------------------------------
# Pie charts of relative abundances
# -----------------------------------------------------------------------------
def pie(
    obj: Union[Dict[str, Any], Any],
    *,
    group_by: Optional[str] = None,
    value_aggregation: Literal["sum", "mean"] = "sum",
    order: Optional[str] = None,
    levels: Optional[List[str]] = None,
    include_index: bool = False,
    levels_shown: Optional[str] = None,
    subset_levels: Optional[Union[str, List[str]]] = None,
    subset_patterns: Optional[Union[str, List[str]]] = None,
    n: int = 6,
    featurelist: Optional[List[str]] = None,
    method: Literal["max", "mean"] = "max",
    sorting: Literal["abundance", "alphabetical"] = "abundance",
    use_values_in_tab: bool = False,

    nrows: int = 1,
    ncols: int = 1,
    figsize: Tuple[float, float] = (18 / 2.54, 10 / 2.54),
    fontsize: int = 10,
    colorlist: Optional[List[str]] = None,
    other_color: str = "grey",
    legend_columns: int = 1,
    show_legend: bool = True,
    savename: Optional[str] = None,
) -> Tuple["plt.figure.Figure", "pd.DataFrame"]:
    """
    Plot pie charts of taxonomic composition for samples or merged groups.

    Parameters
    ----------
    obj : dict or MicrobiomeData
        Input data containing at least:
            - 'tab': pandas.DataFrame
                Abundance table (features x samples).
            - 'tax': pandas.DataFrame
                Taxonomy table (features x taxonomic levels).
    group_by : str, optional
        Metadata column used to merge samples.
    value_aggregation : {'sum', 'mean'}, default = 'sum'
    order : str, optional
        Metadata column used to order samples along the x-axis.
    levels : list of str, optional
        Taxonomic levels used for grouping.
    include_index : bool, default=False
        Whether to include the feature index in labels.
    levels_shown : {'number', None}, optional
        If 'number', show numeric labels instead of taxonomic names.
    subset_levels : str or list of str, optional
        Taxonomic levels to filter by.
    subset_patterns : str or list of str, optional
        Text patterns to filter taxa.
    n : int, default=20
        Number of top taxa to plot (ignored if `featurelist` is provided).
    featurelist : list of str, optional
        Specific features to plot.
    method : {'max', 'min'}, default = 'max'
    sorting : {'abundance', 'alphabetical'}, default = 'abundance'


    nrows : int, default=1
        Number of rows in the subplot grid.
    ncols : int, default=1
        Number of columns in the subplot grid.
    figsize : tuple of float, default=(18/2.54, 10/2.54)
        Figure size in inches.
    fontsize : int, default=10
        Font size for titles and legend.
    colorlist : list of str, optional
        Colors for taxa slices. If None, defaults from `get_colors_markers('colors')` are used.
    other_color : Color for 'Other' slice.
    legend_columns : Number of columns in the legend.
    show_legend : Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
    table : pandas.DataFrame
        DataFrame of relative abundances for plotted taxa and samples.
        Returns None if required keys are missing.

    Notes
    -----
    - Taxa are grouped by the specified level using `groupbytaxa`.
    - Remaining taxa beyond `n` are aggregated into 'Other'.
    - If `order` is provided, samples are sorted by that metadata column.

    Examples
    --------
    >>> df = pie(obj, group_by='Treatment', level='Genus', n=8, savename='pie_chart')
    >>> print(df.head())
    """

    table = _get_ra_table(
        obj=obj,
        group_by=group_by,
        value_aggregation=value_aggregation,
        order=order,
        levels=levels,
        include_index=include_index,
        levels_shown=levels_shown,
        subset_levels=subset_levels,
        subset_patterns=subset_patterns,
        n=n,
        featurelist=featurelist,
        method=method,
        sorting=sorting,
        use_values_in_tab=use_values_in_tab
    )

    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("Error in constructing relative abundance table.")

    # Add 'Other'
    table = table.iloc[::-1]
    table.loc["Other"] = 100 - table.sum()

    # Colors
    if colorlist is None:
        colorlist = get_colors_markers("colors")
    colorlist = colorlist[:n] + [other_color]

    # --- Plot ---
    plt.rcParams.update({"font.size": fontsize})
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    if ncols * nrows < len(table.columns) + 1:
        ncols = 3
        nrows = math.ceil((len(table.columns) + 1) / ncols)
    gs = GridSpec(nrows, ncols, figure=fig)

    for i, c in enumerate(table.columns):
        row, col = divmod(i, ncols)
        ax = plt.subplot(gs[row, col])
        ax.pie(
            table[c],
            colors=colorlist,
            startangle=90,
            wedgeprops={"linewidth": 0.5, "edgecolor": "black"},
            counterclock=False,
        )
        ax.set_title(c, ha="center", va="center", fontsize=fontsize)

    # Legend panel
    if show_legend:
        row, col = divmod(i + 1, ncols)
        ax = plt.subplot(gs[row, col:])
        ax.axis("off")
        legend_patches = [Patch(color=color, label=label) for color, label in zip(colorlist, table.index)]
        ax.legend(
            handles=legend_patches,
            fontsize=fontsize,
            bbox_to_anchor=(0, 1),
            loc="upper left",
            frameon=False,
            ncol=legend_columns,
        )

    # Save outputs
    if savename:
        plt.savefig(savename, dpi=240)
        try:
            plt.savefig(f"{savename}.pdf", format="pdf")
        except Exception:
            # Fallback silently if a PDF backend is not available in the environment
            pass
        table.to_csv(f"{savename}.csv")

    return fig, table
