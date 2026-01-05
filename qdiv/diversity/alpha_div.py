import pandas as pd
import numpy as np
import math
from typing import Union, Any, Dict
from ..utils import rao, get_df, subset_tree, parse_leaves

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

    Accepts:
      - DataFrame: features x samples
      - MicrobiomeData-like object: must expose a DataFrame in .tab / .table / .counts / .abundance
      - dict-of-dicts: either {feature: {sample: count}} or {sample: {feature: count}}

    Parameters
    ----------
    tab : DataFrame | MicrobiomeData-like | dict
        Abundance table (features x samples) or convertible structure.
    q : float, default=1
        Diversity order:
        - q = 0 : species richness
        - q = 1 : exponential of Shannon entropy
        - q = 2 : inverse Simpson
        - general q : Hill number of order q
    use_values_in_tab : bool, default=False
        If False (default), values are converted to relative abundances.
        If True, values in `tab` are assumed to already be relative abundances.

    Returns
    -------
    pandas.Series or float
        Hill numbers for each sample. If input has one sample/column, returns a float.

    Notes
    -----
    - For q = 1, the limit definition is used:
          H₁ = exp( - Σ pᵢ ln pᵢ )
    - For q ≠ 1:
          H_q = ( Σ pᵢ^q )^( 1 / (1 - q) )
    - Zero abundances are ignored safely.
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
        raLn = ra.copy()
        raLn[ra > 0] = ra[ra > 0] * np.log(ra[ra > 0])
        Hillvalues = raLn.sum()
        Hillvalues[Hillvalues < 0] = np.exp(-Hillvalues[Hillvalues < 0])
        return Hillvalues
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
    index: str = "PD",
    use_values_in_tab: bool = False
) -> Union[pd.Series, float]:
    """
    Compute phylogenetic alpha diversity (Hill numbers) of order *q*.

    Implements the framework of Chao et al. (2010, Phil. Trans. R. Soc. B),
    where branch lengths are weighted by the relative abundances of all features
    descending from each branch.

    Parameters
    ----------
    obj : MicrobiomeData-like | dict
        Abundance table and tree dataframe.
        The tree dataframe must contain:
            - 'leaves'     : list of tree leaves descending from each branch
            - 'branchL'  : branch length
    q : float, default=1
        Diversity order:
        - q = 0 : Faith's PD (if index='PD')
        - q = 1 : exponential of phylogenetic Shannon entropy
        - q = 2 : phylogenetic inverse Simpson
        - general q : phylogenetic Hill number
    index : {'PD', 'D', 'H'}, default='PD'
        Output type:
        - 'PD' : Hill number × Tavg  (phylogenetic diversity)
        - 'D'  : Hill number only
        - 'H'  : entropy-like quantity before exponentiation
    use_values_in_tab : bool, default=False
        If False, convert abundances to relative abundances.
        If True, assume `tab` already contains relative abundances.

    Returns
    -------
    pandas.Series or float
        Phylogenetic diversity values for each sample.

    Notes
    -----
    - For q = 1, the limit definition is used:
          H₁ = exp( - Σ_b L_b * p_b * ln(p_b) / Tavg )
    - For q ≠ 1:
          H_q = ( Σ_b L_b * p_b^q / Tavg )^( 1 / (1 - q) )
    - p_b is the total relative abundance of all features descending from branch b.
    """

    # Get input
    tab = get_df(obj, "tab")
    tree = get_df(obj, "tree")
    if "leaves" not in tree.columns or "branchL" not in tree.columns:
        raise ValueError("`tree` must contain columns 'leaves' and 'branchL'.")

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

    #Subset tree to features in tab
    tree = subset_tree(tree, ra.index.tolist())

    # Get Tmean
    Tmean = tree[~tree['nodes'].str.startswith('in')].copy()
    Tmean = Tmean.set_index('nodes')
    asv_set = set(ra.index)
    in_common = list(asv_set.intersection(Tmean.index.tolist()))
    if len(in_common) < len(asv_set):
        raise ValueError("Not all features in tab are found in the tree")
    Tmean = Tmean.loc[in_common]
    Tmean = Tmean['dist_to_root'].sum() / len(Tmean)

    # Build branch × sample abundance matrix
    tree2 = pd.DataFrame(0.0, index=tree.index, columns=ra.columns)
    asv_set = set(ra.index)

    for branch in tree.index:
        asvlist = parse_leaves(tree.loc[branch, "leaves"])

        # Keep only ASVs present in the abundance table
        asvlist = list(asv_set.intersection(asvlist))

        if asvlist:
            tree2.loc[branch] = ra.loc[asvlist].sum()
        else:
            tree2.loc[branch] = 0.0

    #Calculate diversities
    tree_calc = tree2.copy()
    
    if q == 1:
        tree_calc[tree_calc > 0] = tree_calc[tree_calc > 0].map(math.log)
        tree_calc = tree2.mul(tree_calc)
        tree_calc = tree_calc.mul(tree['branchL'], axis=0) #Multiply with branch length
        tree_calc = tree_calc.div(Tmean, axis=1) #Divide by Tmean
        tree_calc = -tree_calc.sum()
        hill_div = tree_calc
        hill_div[hill_div > 0] = hill_div[hill_div > 0].apply(math.exp)
    else:
        tree_calc[tree_calc > 0] = tree_calc[tree_calc > 0].pow(q) #Take power of q
        tree_calc = tree_calc.mul(tree['branchL'], axis=0) #Multiply with branch length
        tree_calc = tree_calc.div(Tmean, axis=1) #Divide by Tmean
        tree_calc = tree_calc.sum()
        hill_div = tree_calc
        hill_div[hill_div > 0] = hill_div[hill_div > 0].pow(1.0 / (1.0 - q))

    # Return requested index
    if index == 'PD':
        return hill_div.mul(Tmean)
    elif index == 'D':
        return hill_div
    elif index == 'H':
        return tree_calc
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
    index: str = "FD",
    use_values_in_tab: bool = False
) -> Union[pd.Series, float]:
    """
    Compute functional alpha diversity (Hill numbers) of order *q*.

    Implements the framework of Chiu et al. (2014, PLoS ONE), where functional
    diversity is derived from pairwise trait distances and species abundances.

    For each sample, functional diversity is computed from:

        Q = Σᵢ Σⱼ pᵢ pⱼ dᵢⱼ        (Rao's quadratic entropy)

    and the functional Hill number of order q:

        q = 1:
            FD₁ = exp( -½ Σᵢ Σⱼ (pᵢ pⱼ ln(pᵢ pⱼ)) dᵢⱼ / Q )

        q ≠ 1:
            FD_q = ( Σᵢ Σⱼ (pᵢ pⱼ)ᵠ dᵢⱼ / Q )^( 1 / (2(1−q)) )

    Parameters
    ----------
    tab : DataFrame | MicrobiomeData-like | dict
        Abundance table (features x samples) or convertible structure.
    distmat : pandas.DataFrame
        Functional distance matrix (features × features).
    q : float, default=1
        Diversity order.
    index : {'FD', 'D', 'MD'}, default='FD'
        Output type:
        - 'D'  : functional Hill number
        - 'MD' : mean functional diversity (D × Q)
        - 'FD' : functional diversity (D × MD)
    use_values_in_tab : bool, default=False
        If False, convert abundances to relative abundances.
        If True, assume `tab` already contains relative abundances.

    Returns
    -------
    pandas.Series
        Functional diversity values for each sample.

    Notes
    -----
    - Uses Rao's Q as implemented in your `rao()` function.
    - Zero abundances are handled safely.
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

