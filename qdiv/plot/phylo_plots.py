import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from ..utils import get_df, ladderize_tree_df
from ..utils.phylo_utils import _normalize_tree_df, _is_missing_id

def phylo_tree(
    tree,
    *, 
    ax: Optional[plt.Axes] = None,
    label_tips: bool = True, 
    label_internals: bool = False,
    tip_order: str = "as_is",
    leaf_prefix: str = "in",
    linewidth: float = 1.5, 
    color: str = 'k',
    figsize: Tuple[float, float] | None = None,
    fontsize: int = 10,
    ladderize: bool = True,
    invert: bool = False,
    scale_bar: float | None = None,
    savename: str | None = None,
) -> Tuple["plt.Figure", "plt.Axes", "pd.DataFrame"]:

    """
    Plot a rooted phylogenetic tree as a rectangular phylogram.

    This function renders a rooted phylogenetic tree using branch lengths
    as horizontal distances (rectangular layout) and returns the Matplotlib
    Axes object. Tip (leaf) positions are laid out vertically, and internal
    node positions are computed as the mean of their descendant tips.

    Parameters
    ----------
    tree : MicrobiomeData, dict, or compatible object
        Input data. Must contain a 'tree' dataframe
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw the tree on. If ``None``, a new figure and
        axes are created.
    label_tips : bool, default True
        If True, label leaf (tip) nodes with their node names.
    label_internals : bool, default False
        If True, label internal nodes with their node names.
    tip_order : {"as_is", "alpha"}, default "as_is"
        Ordering of tips along the y-axis.
        - ``"as_is"`` preserves planar depth-first order from the root
          (respecting child order in the tree).
        - ``"alpha"`` orders tips alphabetically by node name.
    leaf_prefix : str, default "in"
        Prefix intended for leaf naming. Currently not used for filtering
        or labeling logic, but reserved for future extensions.
    linewidth : float, default 1.5
        Line width used when drawing branches.
    color : str or matplotlib-compatible color, default "k"
        Color used for branches and labels.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches when creating a new figure.
        If None, the height is scaled automatically based on the number of tips.
    fontsize : int, default 10
        Font size used for tip and internal node labels.
    ladderize : bool, default True
        If True, reorder child subtrees to produce a ladderized tree layout
        before plotting.
    invert : bool, default False
        If True, invert y-axis of tree.
    scale_bar : float, optional
        Length of the scale bar to draw (in branch-length units). If None,
        a scale bar corresponding to 10% of the tree width is drawn.
    savename : str, optional
        If provided, save the figure to this file path using
        ``bbox_inches="tight"``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure with the phylogenetic tree.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object for the figure.
    T_plot : pandas.DataFrame
        The dataframe used to generate the plot.

    Raises
    ------
    ValueError
        If the tree DataFrame is missing, malformed, or does not contain
        exactly one root.

    RuntimeError
        If y-positions for internal nodes cannot be resolved, indicating
        an invalid or cyclic tree structure.

    Notes
    -----
    - Axis spines, ticks, and labels are removed for a clean tree layout.
    """
    
    # -- Normalize input to DataFrame -----------------------------------------
    T = get_df(tree, "tree")
    if T is None:
        raise ValueError("Tree DataFrame missing.")
    T_plot = _normalize_tree_df(T)  # your normalizer (ensures parent None, leaves sets, floats)
    if ladderize:
        T_plot = ladderize_tree_df(T_plot)

    # Expect columns: nodes, parent, branchL, leaves, dist_to_root
    required = {"nodes", "parent", "branchL", "leaves", "dist_to_root"}
    missing = required - set(T_plot.columns)
    if missing:
        raise ValueError(f"Tree DataFrame missing columns: {sorted(missing)}")

    # -- Build children map (preserve order as it appears in T) ----------------
    children_map: dict[str, list[str]] = {}
    for n, p in zip(T_plot["nodes"].values, T_plot["parent"].values):
        if _is_missing_id(p):
            continue
        children_map.setdefault(p, []).append(n)


    # -- Find the (single) root ------------------------------------------------
    roots = T_plot.loc[T_plot['parent'].isna(), 'nodes'].tolist()
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root, found: {roots}")
    root = roots[0]

    # -- Identify tips vs internals -------------------------------------------
    node_names = T_plot['nodes']
    is_tip_series = ~node_names.isin(children_map.keys())

    # -- Determine a good tip order (planar DFS from the root) -----------------
    def _tips_in_planar_order(r: str) -> list[str]:
        tips: list[str] = []
        # Use explicit recursion to preserve children order; here iterative DFS:
        def dfs(u: str):
            kids = children_map.get(u, [])
            if not kids:
                tips.append(u)
                return
            for c in kids:
                dfs(c)
        dfs(r)
        return tips

    if tip_order == "alpha":
        tip_nodes = sorted(node_names[is_tip_series].tolist())
    else:
        tip_nodes = _tips_in_planar_order(root)
    print(tip_nodes)
    # -- Assign y positions ----------------------------------------------------
    y_pos: dict[str, float] = {n: float(i) for i, n in enumerate(tip_nodes)}

    # Internal node y = mean(child y), fill bottom-up until all resolved
    remaining = set(node_names[~is_tip_series].tolist())
    progressed = True
    while remaining and progressed:
        progressed = False
        # try resolving any internal whose children all have y
        for n in list(remaining):
            childs = children_map.get(n, [])
            if childs and all((c in y_pos) for c in childs):
                y_pos[n] = float(np.mean([y_pos[c] for c in childs]))
                remaining.remove(n)
                progressed = True
    if remaining:
        raise RuntimeError("Could not resolve y-positions; graph may be malformed.")

    # -- x positions from dist_to_root ----------------------------------------
    x_pos = dict(zip(T_plot['nodes'], T_plot['dist_to_root'].astype(float)))

    # -- Prepare axes ----------------------------------------------------------
    if ax is None:
        if figsize is None:
            figsize = (8, max(3, 0.4 * max(1, len(tip_nodes))))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.set_axis_off()

    # -- Draw branches ---------------------------------------------------------
    # 1) Horizontal segments: parent.x -> node.x at y=node.y
    for n, p in zip(T_plot['nodes'], T_plot['parent']):
        if _is_missing_id(p):
            continue
        p = str(p)
        x0 = x_pos[p]; x1 = x_pos[n]; y = y_pos[n]
        ax.plot([x0, x1], [y, y], color=color, linewidth=linewidth)

    # 2) Vertical segments: at x=parent.x from min(child y) to max(child y)
    for p, childs in children_map.items():
        if not childs:
            continue
        y_low  = min(y_pos[c] for c in childs)
        y_high = max(y_pos[c] for c in childs)
        x      = x_pos[p]
        ax.plot([x, x], [y_low, y_high], color=color, linewidth=linewidth)

    # -- Labels ----------------------------------------------------------------
    if label_tips:
        for n in tip_nodes:
            ax.text(x_pos[n], y_pos[n], f"  {n}", va='center', ha='left',
                    color=color, fontsize=fontsize)

    if label_internals:
        for n in node_names[~is_tip_series]:
            ax.text(x_pos[n], y_pos[n], f"{n}", va='center', ha='right',
                    color=color, fontsize=fontsize)

    # -- Bounds & optional scale bar ------------------------------------------
    xmin = min(x_pos.values()) if x_pos else 0.0
    xmax = max(x_pos.values()) if x_pos else 1.0
    span = xmax - xmin
    ax.set_xlim(xmin - 0.02 * span, xmax + 0.10 * span)
    ax.set_ylim(-0.5, len(tip_nodes) - 0.5)

    if invert: #Invert y-axis
        ax.invert_yaxis()

    # Simple scale bar (10% of span)
    if span > 0:
        if scale_bar is None:
            bar = 0.1 * span
        else:
            bar = scale_bar
        # Convert branch-length units to fraction of axis width
        bar_frac = min(bar / span, 0.5)
        x0 = 0.02
        x1 = x0 + bar_frac
        ax.plot(
            [x0, x1],
            [-0.05, -0.05],
            transform=ax.transAxes,
            clip_on=False,
            color=color,
            lw=linewidth
        )
        ax.text(
            (x0 + x1) / 2,
            -0.03,
            f"{bar:.3g}",
            transform=ax.transAxes,
            clip_on=False,
            ha="center",
            va="bottom",
            color=color
        )

    # -- Save if requested -----------------------------------------------------
    if savename:
        fig.savefig(savename, dpi=300, bbox_inches="tight")
        fig.savefig(savename + '.pdf', format="pdf")

    return fig, ax, T_plot
