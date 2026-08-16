import pandas as pd
import numpy as np
import math
from typing import Union, Any, Dict
from ..utils import rao, get_df, compute_Tmean
from ..utils import ra_to_branches, rebuild_leaf_order

# -----------------------------------------------------------------------------
# Naive alpha diversity
# -----------------------------------------------------------------------------
def naive_alpha(
    tab: Union[pd.DataFrame, Dict[str, Any], Any],
    *,
    q: float = 1,
    use_values_in_tab: bool = False
) -> Union[pd.Series, float]:
    """
    Compute naive alpha diversity of order *q* for all samples.

    Parameters
    ----------
    tab : DataFrame | MicrobiomeData-like | dict
        Object containing abundance table (features x samples).
    q : float, default=1
        Diversity order. Determines emphasis on relative abundance of features.
    use_values_in_tab : bool, default=False
        If False (default), values are converted to relative abundances.
        If True, values in `tab` are assumed to already be relative abundances.

    Returns
    -------
    pandas.Series or float
        Hill numbers for each sample. If input has one sample/column, returns a float.
    """
    # --- Get DataFrame (features x samples) -----------------------------
    tab = get_df(tab, "tab")

    # Ensure numeric
    try:
        tab = tab.astype(float)
    except Exception as e:
        raise TypeError(
            "Abundance table contains non-numeric values. Ensure counts/abundances are numeric."
        ) from e

    # --- Relative abundances --------------------------------------------------
    if use_values_in_tab:
        ra = tab
    else:
        col_sums = tab.sum(axis=0)
        if (col_sums == 0).any():
            bad = col_sums.index[col_sums == 0].tolist()
            raise ValueError(f"One or more samples have zero total abundance: {bad}")
        ra = tab.div(col_sums, axis=1)

    if q == 1:
        H = -(ra.where(ra > 0) * np.log(ra.where(ra > 0))).sum()
        return np.exp(H)
    else:
        rapow = ra.copy()
        rapow[ra > 0] = ra[ra > 0].pow(q)
        rapow = rapow.sum()
        Hillvalues = rapow
        Hillvalues[Hillvalues > 0] = Hillvalues[Hillvalues > 0].pow(1.0 / (1.0 - q))
        return Hillvalues

# -----------------------------------------------------------------------------
# Phylogenetic alpha diversity
# -----------------------------------------------------------------------------
def phyl_alpha(
    obj: Union[Dict[str, Any], Any],
    *,
    q: float = 1,
    index: str = "D",
    use_values_in_tab: bool = False,
) -> Union[pd.Series, float]:
    """
    Compute phylogenetic alpha diversity based on Hill numbers.

    This function implements the abundance-weighted phylogenetic diversity
    framework of Chao et al. (2010, Phil. Trans. R. Soc. B). Diversity is
    computed at the level of tree branches, where each branch is weighted by
    its length and by the total relative abundance of all descendant features.

    The primary quantity returned is the *mean phylogenetic diversity*
    D̄_q(T), which is a true Hill number (dimensionless, continuous, and
    monotone in q). A branch-length–scaled quantity (phylogenetic diversity,
    PD_q) can optionally be returned as a derived measure.

    Parameters
    ----------
    obj : MicrobiomeData-like | dict
        Object containing an abundance table (`tab`) and a tree dataframe
        (`tree`). The tree dataframe must include:
            - 'leaves'   : list of descendant leaves for each branch
            - 'branchL'  : branch length
    q : float, default=1
        Diversity order. 
        Determines emphasis on relative abundance of features descending from each branch of the tree.
    index : {'D', 'PD', 'H'}, default='D'
        Quantity to return
            - 'D'  : mean phylogenetic diversity D̄_q(T) (dimensionless; Hill number)
            - 'PD' : branch diversity PD_q(T) = T · D̄_q(T)
            - 'H'  : entropy-like intermediate quantity:
    use_values_in_tab : bool, default=False
        If False, abundances are converted to relative abundances per sample.
        If True, the abundance table is assumed to already contain relative
        abundances.

    Returns
    -------
    pandas.Series
        A vector of diversity values, one per sample.

    Notes
    -----
    The branch diversity PD_q(T) = T_j · D̄_q(T) has units of branch length
    (or evolutionary time) and represents effective evolutionary work.
    Unlike D̄_q(T), PD_q(T) is not a Hill number for q ≠ 0, 1 and is not
    guaranteed to be monotone in q.
    """

    # Get input
    tab = get_df(obj, "tab")
    tree = get_df(obj, "tree")
    if tree is None:
        raise ValueError('tree is missing.')
    leaf_order = get_df(obj, "leaf_order")
    if leaf_order is None:
        tree, leaf_order = rebuild_leaf_order(tree)
    
    # Confirm input is ok
    required_tree_cols = {"branchL", "leaf_start", "leaf_end"}
    missing = required_tree_cols - set(tree.columns)
    if missing:
        raise ValueError(
            f"`tree` must contain columns {sorted(required_tree_cols)}. "
            f"Missing: {sorted(missing)}."
        )
    if leaf_order is None:
        raise ValueError("`leaf_order` is required for trees.")

    # Ensure numeric
    try:
        tab = tab.astype(float)
    except Exception as e:
        raise TypeError(
            "Abundance table contains non-numeric values. Ensure counts/abundances are numeric."
        ) from e

    # --- Relative abundances --------------------------------------------------
    if use_values_in_tab:
        ra = tab
    else:
        col_sums = tab.sum(axis=0)
        if (col_sums == 0).any():
            bad = col_sums.index[col_sums == 0].tolist()
            raise ValueError(f"One or more samples have zero total abundance: {bad}")
        ra = tab.div(col_sums, axis=1)

    # Build branch × sample abundance matrix
    tree2 = ra_to_branches(ra, tree, leaf_order)

    # Get Tmean
    Tmean = compute_Tmean(tree, tree2) #Function from utils

    if abs(q - 1.0) < 1e-6:
        mask = tree2 > 0
        logp = pd.DataFrame(0.0, index=tree2.index, columns=tree2.columns)
        logp[mask] = np.log(tree2[mask])
        # - sum_b (L_b * a_{b,j} * log a_{b,j}) / T_j
        term = ((tree2 * logp).mul(tree["branchL"], axis=0).div(Tmean, axis=1).sum(axis=0))
        D = np.exp(-term)

    else:
        tree_calc = tree2.copy()
        # a_{b,j}^q
        tree_calc[tree_calc > 0] = tree_calc[tree_calc > 0].pow(q)
        # L_b * a_{b,j}^q
        tree_calc = tree_calc.mul(tree["branchL"], axis=0)
        # (L_b * a_{b,j}^q) / T_j
        tree_calc = tree_calc.div(Tmean, axis=1)
        # sum_b L_b a_{b,j}^q / T_j
        term = tree_calc.sum(axis=0)
        # D̄_q(T) = [term]^(1/(1−q))
        D = term.copy()
        D[D > 0] = D[D > 0].pow(1.0 / (1.0 - q))

    # Return requested index
    if index == "PD":
        return D * Tmean
    elif index == "D":
        return D
    elif index == "H":
        return term # entropy at q=1, moment at q≠1
    else:
        raise ValueError("`index` must be one of: 'PD', 'D', 'H'.")

# -----------------------------------------------------------------------------
# Functional alpha diversity
# -----------------------------------------------------------------------------
def func_alpha(
    tab: Union[pd.DataFrame, Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1,
    index: str = "D",
    use_values_in_tab: bool = False
) -> Union[pd.Series, float]:
    """
    Compute functional alpha diversity of order *q* following
    Chiu et al. (2014). It is calculated from pairwise trait distances
    between features and their relative abundances. 
    
    Parameters
    ----------
    tab : DataFrame | MicrobiomeData-like | dict
        Abundance table (features × samples) or convertible structure.
    distmat : pandas.DataFrame
        Functional distance matrix (features × features).
    q : float, default=1
        Diversity order. Determines emphasis on relative abundances.
    index : {'D', 'MD', 'FD'}, default='D'
        Quantity to return.
            - 'D'  : functional Hill number (effective number of
                     functionally distinct features)
            - 'MD' : mean functional diversity (D × Q)
            - 'FD' : total functional diversity (D² × Q)
    use_values_in_tab : bool, default=False
        If False, abundances are converted to relative abundances.
        If True, values in `tab` are assumed to already be relative
        abundances.
    
    Returns
    -------
    pandas.Series
        Functional diversity values for each sample.
    
    Notes
    -----
    - The implementation follows Chiu et al. (2014, PLoS ONE).
    - The default output ('D') corresponds to the functional Hill number
      and is directly comparable to taxonomic and phylogenetic Hill
      numbers.
    """
    # Get input
    tab = get_df(tab, "tab")
    if not isinstance(distmat, pd.DataFrame):
        raise TypeError("`distmat` must be a pandas DataFrame.")

    # Ensure numeric
    try:
        tab = tab.astype(float)
    except Exception as e:
        raise TypeError(
            "Abundance table contains non-numeric values. Ensure counts/abundances are numeric."
        ) from e

    # --- Relative abundances --------------------------------------------------
    if use_values_in_tab:
        ra = tab
    else:
        col_sums = tab.sum(axis=0)
        if (col_sums == 0).any():
            bad = col_sums.index[col_sums == 0].tolist()
            raise ValueError(f"One or more samples have zero total abundance: {bad}")
        ra = tab.div(col_sums, axis=1)

    if isinstance(ra, pd.Series):
        raise ValueError("`tab` must be a DataFrame, not a Series.")

    # Align distance matrix to ASVs
    asvs = ra.index.tolist()
    distmat = distmat.loc[asvs, asvs]

    # Output container
    out = pd.Series(0.0, index=ra.columns)

    svlist = ra.index.tolist()
    distmat = distmat.loc[svlist, svlist]
    Qframe = rao(ra, distmat)

    if q == 1:
        for smp in ra.columns:
            ra2mat = pd.DataFrame(np.outer(ra[smp].to_numpy(), ra[smp].to_numpy()), index=ra.index, columns=ra.index)
            ra2Lnmat = ra2mat.copy()
            mask = ra2Lnmat > 0
            ra2Lnmat[mask] = ra2Lnmat[mask].map(math.log)
            ra2ochLn = ra2mat.mul(ra2Lnmat)
            dQmat = distmat.mul(1 / Qframe.loc[smp])
            dQ_ra2_Ln = dQmat.mul(ra2ochLn)
            Chiuvalue = math.exp(-0.5 * sum(dQ_ra2_Ln.sum()))
            out.loc[smp] = Chiuvalue
    else:
        for smp in ra.columns:
            ra2mat = pd.DataFrame(np.outer(ra[smp].to_numpy(), ra[smp].to_numpy()), index=ra.index, columns=ra.index)
            mask = ra2mat > 0
            ra2mat[mask] = ra2mat[mask].pow(q)
            dQmat = distmat.mul(1 / Qframe.loc[smp])
            ra2dq = (ra2mat.mul(dQmat))
            Chiuvalue = pow(sum(ra2dq.sum()), 1 / (2 * (1 - q)))
            out.loc[smp] = Chiuvalue
    if index == 'D':
        return out
    elif index == 'MD':
        MD = out.mul(Qframe)
        return MD
    elif index == 'FD':
        MD = out.mul(Qframe)
        return out.mul(MD)
    else:
        raise ValueError("`index` must be one of: 'D', 'MD', 'FD'.")

# -----------------------------------------------------------------------------
# Phylo indices
# -----------------------------------------------------------------------------
def mpdq(
    obj: Union[Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1.0,
) -> pd.DataFrame:
    """
    Mean phylogenetic distance (MPD) with q-weighting of relative abundances.

    Parameters
    ----------
    obj : MicrobiomeData, dict, or compatible object
        Input data. Must provide at least an abundance table ('tab').
    distmat : pd.DataFrame
        Square distance matrix indexed/columned by feature ids.
    q : float, default=1.0
        Order of diversity weighting applied to relative abundances.

    Returns
    -------
    pandas.DataFrame

    Notes
    ----------
    The index is a relative abundance-weighted development of the 
    mean phylogenetic distance index described by 
    Webb et al. (2002) *American Naturalist*.
    """
    from ..model import nriq
    return nriq(obj, distmat, q=q, iterations=0)

def mntdq(
    obj: Union[Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1.0,
) -> pd.DataFrame:
    """
    Mean nearest taxon distance (MNTD) with q-weighting of relative abundances.

    Parameters
    ----------
    obj : MicrobiomeData, dict, or compatible object
        Input data. Must provide at least an abundance table ('tab').
    distmat : pd.DataFrame
        Square distance matrix indexed/columned by feature ids.
    q : float, default=1.0
        Order of diversity weighting applied to relative abundances.

    Returns
    -------
    pandas.DataFrame

    Notes
    ----------
    The index is a relative abundance-weighted development of 
    mean nearest taxon index described by 
    Webb et al. (2002) *American Naturalist*.
    """
    from ..model import ntiq
    return ntiq(obj, distmat, q=q, iterations=0)
