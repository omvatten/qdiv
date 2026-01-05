import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Dict, List, Optional, Tuple, Union, Any
from ..stats import pcoa_lingoes
from ..utils import get_colors_markers, get_df

# ------------------------------
# Helpers for ordination plot
# ------------------------------
def _extract_ordination_payload(ordination_results):
    """
    Normalize ordination input into a common payload:
      - coords_df      : DataFrame of site scores (rows=samples, cols=axes)
      - axis_names     : list[str] of axis names (e.g., ['PCo1','PCo2'] or ['dbRDA1','dbRDA2'])
      - pct_explained  : pd.Series (% per axis, aligned to axis_names)
      - eigenvalues    : pd.Series (aligned to axis_names)
      - biplot_df      : optional DataFrame (rows=predictors, cols=axes)
      - kind           : 'pcoa' or 'dbrda'

    Accepts:
      - a distance DataFrame (square), or
      - dict from pcoa_lingoes, or
      - dict from dbrda
    """
    # Case A: a distance matrix → compute PCoA using provided pcoa_fn
    if isinstance(ordination_results, pd.DataFrame) and ordination_results.shape[0] == ordination_results.shape[1]:
        res = pcoa_lingoes(ordination_results)
        coords_df = res['site_scores']
        ev = res['eigenvalues']
        pct = res['pct_explained']
        axis_names = list(coords_df.columns)
        eigenvalues = pd.Series(np.array(ev), index=axis_names)
        pct_explained = pd.Series(np.array(pct), index=axis_names)
        return {
            'coords_df': coords_df,
            'axis_names': axis_names,
            'pct_explained': pct_explained,
            'eigenvalues': eigenvalues,
            'biplot_df': None,
            'kind': 'pcoa'}

    # Case B: dict from pcoa_lingoes or dbrda
    if isinstance(ordination_results, dict):
        coords_df = ordination_results.get('site_scores', None)
        if coords_df is None or not isinstance(coords_df, pd.DataFrame):
            raise ValueError("Could not find site scores in ordination dict.")

        biplot_df = ordination_results.get('biplot_scores', None)
        kind = 'dbrda' if biplot_df is not None else 'pcoa'
        axis_names = list(coords_df.columns)

        # Eigenvalues can be Series (PCoA) or ndarray (dbRDA)
        ev = ordination_results.get('eigenvalues', None)
        if ev is None:
            eigenvalues = pd.Series(index=axis_names, dtype=float)
        else:
            eigenvalues = pd.Series(np.array(ev).ravel(), index=axis_names[:len(np.array(ev).ravel())])

        # Explained % under various keys
        if 'pct_explained' in ordination_results:
            pct = ordination_results['pct_explained']  # already in %
        else:
            pct = pd.Series(ordination_results['explained_ratio'] * 100, index=axis_names).round(2)

        return {
            'coords_df': coords_df,
            'axis_names': axis_names,
            'pct_explained': pct,
            'eigenvalues': eigenvalues,
            'biplot_df': biplot_df,
            'kind': kind
        }
    raise TypeError("ordination must be a square distance DataFrame or a dict returned by pcoa_lingoes/dbrda.")


# ------------------------------
# Helpers: arrows & scaling
# ------------------------------
def _compute_pcoa_biplot(coords_2d: pd.DataFrame, meta: pd.DataFrame, variables: list, eigen_x: float, eigen_y: float):
    """
    Compute PCoA biplot arrows for two axes from numeric metadata columns.
    """
    # Standardize U (site scores)
    U_std = coords_2d.copy()
    for j in range(2):
        col = U_std.columns[j]
        std = U_std[col].std()
        U_std[col] = (U_std[col] - U_std[col].mean()) / (std if std and std > 0 else 1.0)

    # Standardize Y from meta
    Y = pd.DataFrame(index=coords_2d.index)
    for mh in variables:
        vals = pd.to_numeric(meta[mh], errors='coerce')
        std = vals.std()
        Y[mh] = (vals - vals.mean()) / (std if std and std > 0 else 1.0)
    Y_cent = Y.transpose()

    # Project
    Spc = (1 / (len(coords_2d.index) - 1)) * np.matmul(Y_cent, U_std.to_numpy())
    biglambda = np.array([[eigen_x ** -0.5 if eigen_x > 0 else 1.0, 0.0],
                          [0.0, eigen_y ** -0.5 if eigen_y > 0 else 1.0]])
    Uproj_arr = (len(coords_2d.index) - 1) ** 0.5 * np.matmul(Spc, biglambda)
    return pd.DataFrame(Uproj_arr, index=Y.columns, columns=coords_2d.columns.tolist())

def _scale_arrows_to_limits(Uproj: pd.DataFrame, xlims, ylims, margin=0.9):
    """
    Scale arrow coordinates to fit inside axis limits.
    """
    if Uproj is None or Uproj.empty:
        return Uproj
    xn, yn = Uproj.columns.tolist()
    max_abs_x = max(1e-12, np.max(np.abs(Uproj[xn])))
    max_abs_y = max(1e-12, np.max(np.abs(Uproj[yn])))
    scale_x = margin * (xlims[1] - xlims[0]) / (2 * max_abs_x)
    scale_y = margin * (ylims[1] - ylims[0]) / (2 * max_abs_y)
    return Uproj * min(scale_x, scale_y)

# ------------------------------
# Helpers: ellipses
# ------------------------------

def _covariance_ellipse_params(x: np.ndarray, y: np.ndarray, n_std: float = 2.0):
    """
    Compute covariance ellipse parameters (center, width, height, angle in degrees).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    width = 2 * n_std * np.sqrt(max(vals[0], 1e-12))
    height = 2 * n_std * np.sqrt(max(vals[1], 1e-12))
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return mean_x, mean_y, width, height, angle

def _draw_ellipses(ax, coords: pd.DataFrame, meta: pd.DataFrame, group_col: str,
                   n_std: float = 2.0, edge_color='grey', lw=1.0,
                   label_centers=False, connect_by: str = None, colors=None):
    """
    Draw confidence ellipses for each category in `group_col`.
    Optionally connect ellipse centers ordered by `connect_by`.

    Handles:
      - Single-group case (skips connecting centers)
      - Non-numeric connect_by (skips connection)
      - Insufficient points for ellipse (draw centroid only)
    """
    xn, yn = coords.columns.tolist()
    cats = pd.unique(meta[group_col])
    centers = []

    # Draw ellipses or centroids
    for i, cat in enumerate(cats):
        mask = meta[group_col] == cat
        xs = coords.loc[mask, xn].values
        ys = coords.loc[mask, yn].values
        color = colors[i % len(colors)] if colors is not None else edge_color

        if len(xs) >= 3:  # enough points for ellipse
            cx, cy, w, h, ang = _covariance_ellipse_params(xs, ys, n_std=n_std)
            e = Ellipse((cx, cy), width=w, height=h, angle=ang,
                        facecolor='none', edgecolor=color, lw=lw)
            ax.add_patch(e)
            if label_centers:
                ax.annotate(str(cat), (cx, cy))
            centers.append((cat, cx, cy))
        else:
            # fallback: just mark centroid
            cx, cy = np.mean(xs), np.mean(ys)
            if label_centers:
                ax.annotate(str(cat), (cx, cy))
            centers.append((cat, cx, cy))

    # Connect centers if requested and valid
    if connect_by is not None and len(centers) > 1:
        try:
            # Compute mean of connect_by per group
            order_vals = meta.groupby(group_col)[connect_by].mean()
            if isinstance(order_vals, pd.Series):
                order_vals = order_vals.sort_values()
            else:
                # Single group fallback
                order_vals = pd.Series([order_vals], index=[meta[group_col].unique()[0]])
            print(order_vals)
            df_centers = pd.DataFrame(centers, columns=[group_col, 'x', 'y']).set_index(group_col)
            ordered = df_centers.loc[order_vals.index]
            ax.plot(ordered['x'].values, ordered['y'].values, color='black', lw=lw)
        except Exception as e:
            # Gracefully skip if connect_by is invalid
            print(f"Skipping ellipse connection due to error: {e}")

# ------------------------------
# Helpers: points & legends
# ------------------------------
def _default_colors(n):
    return get_colors_markers(get_type="colors")

def _default_markers(n):
    return get_colors_markers(get_type="markers")

def _draw_points(ax, coords: pd.DataFrame, meta: pd.DataFrame,
                 color_by: str = None, shape_by: str = None,
                 colors=None, markers=None, markersize=50, lw=1.0,
                 connect=None, legend=True, legend_pos_colors=(1, 1), legend_pos_shapes=(1, 0.4),
                 legend_titles=(None, None), markerscale=1.1, fontsize=12):
    """
    Scatter plot of points grouped by color_by and shape_by, with optional connection lines.
    """
    xn, yn = coords.columns.tolist()

    # Prepare grouping
    if color_by is None:
        meta['_all_'] = 'all'
        color_by = '_all_'
    cats1 = list(pd.unique(meta[color_by]))
    colors = colors or _default_colors(len(cats1))
    markers = markers or _default_markers(max(len(cats1), len(pd.unique(meta[shape_by])) if shape_by else 1))

    # Legend containers
    linesColor = [[], []]
    linesShape = [[], []]
    shapeTracker = []

    for i, cat1 in enumerate(cats1):
        sel1 = meta[color_by] == cat1
        meta_i = meta.loc[sel1]
        coords_i = coords.loc[sel1]

        # Connect lines by a numeric column
        if connect is not None:
            order_vals = pd.to_numeric(meta_i[connect], errors='coerce')
            sorter = np.argsort(order_vals.values)
            ax.plot(coords_i.iloc[sorter, 0], coords_i.iloc[sorter, 1], color=colors[i], lw=lw)

        if shape_by is not None:
            cats2 = list(pd.unique(meta_i[shape_by]))
            # add color legend stub
            linesColor[0].append(ax.scatter([], [], label=str(cat1), color=colors[i]))
            linesColor[1].append(cat1)
            for j, cat2 in enumerate(cats2):
                sel2 = meta_i[shape_by] == cat2
                xi = coords_i.loc[sel2, xn]
                yi = coords_i.loc[sel2, yn]
                mk = markers[j % len(markers)]
                ax.scatter(xi, yi, label=None, color=colors[i], marker=mk, s=markersize)
                # shape legend stub (one per shape)
                if j not in shapeTracker:
                    linesShape[0].append(ax.scatter([], [], label=str(cat2), color='black', marker=mk))
                    linesShape[1].append(cat2)
                    shapeTracker.append(j)
        else:
            mk = markers[i % len(markers)]
            linesColor[0].append(ax.scatter([], [], label=str(cat1), color=colors[i], marker=mk))
            linesColor[1].append(cat1)
            ax.scatter(coords_i[xn], coords_i[yn], label=None, color=colors[i], marker=mk, s=markersize)

    # Legends
    if legend:
        # Colors legend (var1)
        title1 = legend_titles[0] if legend_titles[0] else (color_by if color_by != '_all_' else '')
        ax.legend(linesColor[0], linesColor[1], ncol=1, bbox_to_anchor=legend_pos_colors, title=title1,
                  frameon=False, markerscale=markerscale, fontsize=fontsize, loc=2)
        # Shapes legend (var2)
        if shape_by is not None and len(linesShape[0]) > 0:
            from matplotlib.legend import Legend
            title2 = legend_titles[1] if legend_titles[1] else shape_by
            leg = Legend(ax, linesShape[0], linesShape[1], ncol=1, bbox_to_anchor=legend_pos_shapes, title=title2,
                         frameon=False, markerscale=markerscale, fontsize=fontsize, loc=2)
            ax.add_artist(leg)

# ------------------------------
# Main: ordination plot function
# ------------------------------
def ordination(
    ordination_results: Union[pd.DataFrame, Dict[str, Union[pd.DataFrame, dict]]] = None,
    meta: Union[pd.DataFrame, Dict[str, Any], Any] = None,
    *,
    color_by: Optional[str] = None,
    shape_by: Optional[str] = None,
    biplot: Optional[List[str]] = None,
    ellipse: Optional[str] = None,
    title: str = "",
    savename: Optional[str] = None,
    show_legend: bool = True,
    figsize: Tuple[float, float] = (9, 6),
    fontsize: int = 12,
    markersize: float = 50,
    markerscale: float = 1.1,
    lw: float = 1.0,
    pad: float = 1.1,
    flipx: bool = False,
    flipy: bool = False,
    hide_ticks: bool = False,
    connect: Optional[str] = None,
    ellipse_connect: Optional[str] = None,
    tag: Optional[str] = None,
    return_data: bool = False,
    ax: Optional[plt.Axes] = None,
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    which_axes: Tuple[int, int] = (0, 1),
) -> Tuple["plt.Figure", "plt.axes", "pd.DataFrame", "pd.DataFrame"]:
    """
    Create an ordination plot (PCoA or db-RDA) with optional grouping, biplots, and annotations.

    Parameters
    ----------
    ordination_results : pandas.DataFrame or dict
        Ordination results from PCoA or db-RDA. For PCoA, provide a distance DataFrame;
        for db-RDA, provide a dictionary containing scores.
    meta : DataFrame | MicrobiomeData-like | dict
        Metadata table with sample annotations.
    color_by : str, optional
        Column in `meta` used to color points by group.
    shape_by : str, optional
        Column in `meta` used to vary marker shapes by group.
    biplot : list of str, optional
        For PCoA: list of numeric metadata columns to display as biplot vectors.
        For db-RDA: set to None to use 'biplot_scores' from ordination results.
    ellipse : str, optional
        Column in `meta` used to group samples for drawing confidence ellipses.
    title : str, optional
        Plot title.
    savename : str, optional
        Filename to save the figure. Extension determines format (e.g., `.png`, `.pdf`).
    show_legend : bool, default=True
        Whether to display the legend.
    figsize : tuple of float, default=(9, 6)
        Figure size in inches.
    fontsize : int, default=12
        Font size for labels and title.
    markersize : float, default=50
        Size of scatter plot markers.
    markerscale : float, default=1.1
        Scaling factor for legend markers.
    lw : float, default=1.0
        Line width for ellipses and connections.
    pad : float, default=1.1
        Padding factor for axis limits.
    flipx : bool, default=False
        Flip the X-axis.
    flipy : bool, default=False
        Flip the Y-axis.
    hide_ticks : bool, default=False
        Hide axis ticks and labels.
    connect : str, optional
        Column in `meta` to connect points in order (e.g., time series).
    ellipse_connect : str, optional
        Column in `meta` to connect ellipse centers in order.
    tag : str, optional
        Column in `meta` or 'index' to annotate points.
    return_data : bool, default=False
        If True, return processed plotting data instead of the figure.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw the plot on. If None, a new figure is created.
    colors : list of str, optional
        Custom list of colors for groups.
    markers : list of str, optional
        Custom list of marker styles for groups.
    which_axes : tuple of int, default=(0, 1)
        Indices of ordination axes to plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object for the ordination.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object for the ordination.
    meta : pandas.DataFrame
        meta data with ordination coordinates.
    Uproj: pandas.DataFrame
        biplot coordinates (if used)

    Notes
    -----
    - Supports both PCoA and db-RDA ordination results.
    - Ellipses represent group dispersion; biplots show variable contributions.
    - Axis flipping and padding allow fine control over plot appearance.

    Examples
    --------
    >>> fig = ordination(pcoa_results, meta, color_by='Treatment', ellipse='Group')
    >>> fig.savefig('ordination_plot.png')
    """
    # Validation
    if ordination_results is None:
        raise ValueError('ordination_results are missing.')
    meta = get_df(meta, "meta")
    if meta is None:
        raise ValueError('meta data is missing.')

    # Extract normalized ordination payload
    payload = _extract_ordination_payload(ordination_results)
    coords_df = payload['coords_df'].copy()
    axis_names = payload['axis_names']
    pct_explained = payload['pct_explained']
    eigenvalues = payload['eigenvalues']
    biplot_df = payload['biplot_df']
    kind = payload['kind']  # 'pcoa' or 'dbrda'

    # Align meta to ordination sample order
    if not coords_df.index.equals(meta.index):
        common = coords_df.index.intersection(meta.index)
        if len(common) != len(coords_df.index):
            raise ValueError("Samples in metadata don't match samples in ordination site scores.")
        meta = meta.loc[coords_df.index]

    # Pick axes
    if len(axis_names) < max(which_axes) + 1:
        raise ValueError(f"Requested axes {which_axes} exceed available axes ({len(axis_names)}).")
    xn_name = axis_names[which_axes[0]]
    yn_name = axis_names[which_axes[1]]

    # Subset coordinates to 2D
    coords = coords_df[[xn_name, yn_name]].copy()

    # Axis labels with explained %
    def _axis_label(name):
        pct = pct_explained.get(name, np.nan)
        suffix = "" if pd.isna(pct) else f" ({pct:.2f}%)"
        return f"{name}{suffix}"

    xlab = _axis_label(xn_name)
    ylab = _axis_label(yn_name)

    # Plot bounds
    xaxislims = [coords.iloc[:, 0].min() * pad, coords.iloc[:, 0].max() * pad]
    yaxislims = [coords.iloc[:, 1].min() * pad, coords.iloc[:, 1].max() * pad]

    # Compute biplot arrows
    Uproj = None
    if biplot_df is not None:
        # dbRDA: use given biplot scores
        arrows2d = biplot_df[[xn_name, yn_name]].copy()
        Uproj = _scale_arrows_to_limits(arrows2d, xaxislims, yaxislims)
    elif biplot and kind == 'pcoa':
        evx = float(eigenvalues.get(xn_name, 1.0))
        evy = float(eigenvalues.get(yn_name, 1.0))
        Uproj = _compute_pcoa_biplot(coords, meta, biplot, evx, evy)
        Uproj = _scale_arrows_to_limits(Uproj, xaxislims, yaxislims)

    # Plot setup
    if ax is None:
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Points & legends
    _draw_points(ax, coords, meta, color_by=color_by, shape_by=shape_by,
                 colors=colors, markers=markers, markersize=markersize, lw=lw,
                 connect=connect, legend=show_legend, markerscale=markerscale, fontsize=fontsize)

    # Ellipses
    if ellipse is not None and ellipse in meta.columns:
        _draw_ellipses(ax, coords, meta, group_col=ellipse, n_std=2.0, edge_color='grey', lw=lw,
                       label_centers=False, connect_by=ellipse_connect, colors=colors or _default_colors(len(pd.unique(meta[ellipse]))))

    # Arrow overlay
    if Uproj is not None and len(Uproj) > 0:
        xn, yn = Uproj.columns.tolist()
        for var_name in Uproj.index:
            vx = float(Uproj.loc[var_name, xn])
            vy = float(Uproj.loc[var_name, yn])
            ha = 'left' if vx > 0 else ('right' if vx < 0 else 'center')
            va = 'bottom' if vy > 0 else ('top' if vy < 0 else 'center')
            ax.arrow(0, 0, vx, vy, color='black', width=0.001)
            ax.annotate(var_name, (1.03 * vx, 1.03 * vy), horizontalalignment=ha, verticalalignment=va)
        ax.axhline(0, 0, 1, linestyle='--', color='grey', lw=0.5)
        ax.axvline(0, 0, 1, linestyle='--', color='grey', lw=0.5)

    # Point/ellipse tags
    if tag is not None:
        if tag == 'index':
            for ix in meta.index:
                ax.annotate(str(ix), (coords.loc[ix, xn_name], coords.loc[ix, yn_name]))
        elif tag in meta.columns:
            for ix in meta.index:
                tagtext = str(meta.loc[ix, tag])
                ax.annotate(tagtext, (coords.loc[ix, xn_name], coords.loc[ix, yn_name]))

    # Final formatting
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(xaxislims)
    ax.set_ylim(yaxislims)
    if flipx:
        ax.invert_xaxis()
    if flipy:
        ax.invert_yaxis()
    if hide_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.set_title(title)
    plt.tight_layout()

    # Save
    if savename is not None:
        fig.savefig(savename)
        fig.savefig(savename + ".pdf", format="pdf")

    # Return
    if Uproj is not None:
        return fig, ax, pd.concat([meta, coords], axis=1), Uproj
    return fig, ax, pd.concat([meta, coords], axis=1), None
