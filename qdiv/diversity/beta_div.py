import pandas as pd
import numpy as np
import math
from typing import Optional, Dict, Any, Union
from ..io import subset_samples
from ..utils import rao, beta2dist, get_df
from ..utils import ra_to_branches, compute_Tmean, rebuild_leaf_order
from .alpha_div import naive_alpha, phyl_alpha, func_alpha

def _get_tqdm(use_tqdm: bool):
    """
    Internal helper that returns tqdm if available and requested; otherwise provides
    a minimal stub compatible with tqdm's API.
    """
    if use_tqdm:
        try:
            from tqdm import tqdm
            return tqdm
        except Exception:
            pass

    class _DummyTqdm:  # fallback with same constructor signature
        def __init__(self, iterable=None, total=None, desc=None, unit=None, leave=False):
            self._iterable = iterable if iterable is not None else range(total or 0)

        def __iter__(self):
            return iter(self._iterable)

        def update(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

    return _DummyTqdm

# -----------------------------------------------------------------------------
# Naive beta diversity
# -----------------------------------------------------------------------------
def naive_beta(
    tab: Union[pd.DataFrame, Dict[str, Any], Any],
    *,
    q: float = 1,
    dis: bool = True,
    viewpoint: str = "regional",
    use_values_in_tab: bool = False,
    use_numba: bool = False,
) -> pd.DataFrame:
    """
    Compute naive (taxonomic) pairwise beta diversity of order *q*.
    Implements the two‑community Hill‑number beta diversity framework
    described in Chao et al. (2014), using only species abundances
    (no phylogenetic or functional information).

    Parameters
    ----------
    tab : DataFrame | MicrobiomeData-like | dict
        Must contain abundance table (features x samples).
    q : float, default=1
        Diversity order. Determines emphasis on relative abundances.
    dis : bool, default=True
        If True, converts beta diversity to a dissimilarity scaled from 0 to 1.
        If False, returns raw beta values scaled from 1 to 2.
    viewpoint : {'local', 'regional'}, default='regional'
        Viewpoint for converting beta diversity to dissimilarity. 
    use_values_in_tab : bool, default=False
        If False, convert abundances to relative abundances.
        If True, assume `tab` already contains relative abundances.
    use_numba : bool, default=False
        If True, accelerates calculation using Numba (which requires that Numba is installed).
        Very useful for large datasets.

    Returns
    -------
    pandas.DataFrame
        Pairwise beta diversity (or dissimilarity) matrix.
    """
    # Validate input
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

    if isinstance(ra, pd.Series) or ra.shape[1] < 2:
        raise ValueError("`tab` must contain ≥ 2 samples (columns).")

    # Check for duplicate sample names
    smplist = ra.columns.tolist()
    if len(smplist) != len(set(smplist)):
        raise ValueError("Duplicate sample names detected in `tab`.")

    # Output matrix
    out = pd.DataFrame(0.0, index=smplist, columns=smplist)

    # Acceleration
    if use_numba:
        try:
            from .accelerate_div import naive_beta_numba
        except Exception:
            print('Numba failed, falling back to Python.')
            naive_beta_numba = None
    else:
        naive_beta_numba = None

    # Pairwise beta diversity
    if naive_beta_numba is not None:
        out = naive_beta_numba(ra.to_numpy(), q)
        out = pd.DataFrame(out, index=smplist, columns=smplist)

    else:
        for i in range(len(smplist) - 1):
            s1 = smplist[i]
            p1 = ra[s1]
    
            for j in range(i + 1, len(smplist)):
                s2 = smplist[j]
                p2 = ra[s2]
    
                # q = 1 (Shannon case)
                if q == 1:
                    # α-diversity
                    mask1 = p1 > 0
                    H1 = (p1[mask1] * np.log(p1[mask1])).sum()
    
                    mask2 = p2 > 0
                    H2 = (p2[mask2] * np.log(p2[mask2])).sum()
    
                    alpha = math.exp(-0.5 * H1 - 0.5 * H2)
    
                    # γ-diversity
                    m = (p1 + p2) / 2
                    maskg = m > 0
                    Hg = (m[maskg] * np.log(m[maskg])).sum()
                    gamma = math.exp(-Hg)
    
                    beta = gamma / alpha
    
                # q ≠ 1 (General Hill case)
                else:
                    # α-diversity
                    p1q = p1[p1 > 0].pow(q).sum()
                    p2q = p2[p2 > 0].pow(q).sum()
                    alpha = (0.5 * p1q + 0.5 * p2q) ** (1.0 / (1.0 - q))
    
                    # γ-diversity
                    m = (p1 + p2) / 2
                    mq = m[m > 0].pow(q).sum()
                    gamma = mq ** (1.0 / (1.0 - q))
    
                    beta = gamma / alpha
    
                out.loc[s1, s2] = beta
                out.loc[s2, s1] = beta

    # Convert β to dissimilarity if requested
    # Ensure beta diagonal is 1
    for s in out.index:
        out.loc[s, s] = 1.0

    if dis:
        return beta2dist(beta=out, q=q, N=2, div_type="naive", viewpoint=viewpoint)
    return out

# -----------------------------------------------------------------------------
# Phylogenetic beta diversity
# -----------------------------------------------------------------------------
def phyl_beta(
    obj: Union[Dict[str, Any], Any],
    *,
    q: float = 1,
    dis: bool = True,
    viewpoint: str = "regional",
    use_values_in_tab: bool = False,
    use_numba: bool = False,
) -> pd.DataFrame:
    """
    Compute phylogenetic pairwise beta diversity of order *q*.
    Implements the two‑community phylogenetic Hill‑number beta framework
    described in Chao et al. (2014), where branch lengths are weighted by
    the relative abundances of all features descending from each branch.

    Parameters
    ----------
    obj : MicrobiomeData-like | dict
        Must include

          - ``'tab'``, feature × sample abundance DataFrame
          - ``'tree'``, branch information DataFrame
          - ``'leaf_order'``, list of leaf names corresponding to the tree

    q : float, default=1
        Diversity order. Determines emphasis on relative abundances.
    dis : bool, default=True
        If True, converts beta diversity to a dissimilarity scaled from 0 to 1.
        If False, returns raw beta values scaled from 1 to 2.
    viewpoint : {'local', 'regional'}, default='regional'
        Viewpoint for converting beta diversity to dissimilarity.
    use_values_in_tab : bool, default=False
        If False, convert abundances to relative abundances.
        If True, assume `tab` already contains relative abundances.
    use_numba : bool, optional
        If True, accelerates calculation using Numba (which requires that Numba is installed).
        Very useful for large datasets.

    Returns
    -------
    pandas.DataFrame
        Pairwise phylogenetic beta diversity (or dissimilarity) matrix.
    """
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
        raise TypeError("Abundance table contains non-numeric values.") from e

    # --- Relative abundances --------------------------------------------------
    if use_values_in_tab:
        ra = tab
    else:
        col_sums = tab.sum(axis=0)
        if (col_sums == 0).any():
            bad = col_sums.index[col_sums == 0].tolist()
            raise ValueError(f"One or more samples have zero total abundance: {bad}")
        ra = tab.div(col_sums, axis=1)

    if isinstance(ra, pd.Series) or ra.shape[1] < 2:
        raise ValueError("`tab` must contain ≥ 2 samples (columns).")

    #Subset tree to features in tab
    tree2 = ra_to_branches(ra, tree, leaf_order)

    # Align branch lengths to tree2 index
    branchL = tree["branchL"].reindex(tree2.index)
    if branchL.isna().any():
        missing = branchL.index[branchL.isna()].tolist()
        raise ValueError(f"'branchL' missing for branches: {missing}")

    # Acceleration
    if use_numba:
        try:
            from .accelerate_div import phyl_beta_numba
        except Exception:
            print('Numba failed, falling back to Python.')
            phyl_beta_numba = None
    else:
        phyl_beta_numba = None

    # --- Pairwise phylogenetic beta diversity --------------------------------
    smplist = ra.columns.tolist()

    if phyl_beta_numba is not None:
        A = tree2.to_numpy(dtype=np.float64)
        L = branchL.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(A)):
            raise ValueError("Branch abundance matrix contains non-finite values.")
        if not np.all(np.isfinite(L)):
            raise ValueError("Branch lengths contain non-finite values.")
        out_arr = phyl_beta_numba(A, L, float(q))
        out = pd.DataFrame(out_arr, index=smplist, columns=smplist)

    else:
        out = pd.DataFrame(0.0, index=smplist, columns=smplist)
    
        for i in range(len(smplist) - 1):
            s1 = smplist[i]
            for j in range(i + 1, len(smplist)):
                s2 = smplist[j]
    
                # Subtree abundances per branch
                sub = tree2[[s1, s2]].copy()
                pooled = str(s1)+str(s2)
                sub[pooled] = sub.mean(axis=1)
                Tmean = compute_Tmean(tree, sub)
                Tgamma = Tmean[pooled]
    
                # --- γ-diversity ---------------------------------------------------
                g = sub[pooled]
                if q == 1.0:
                    mask = g > 0
                    term = g.where(mask, 0.0) * np.log(g.where(mask, 1.0))
                    term = (term * (branchL / Tgamma)).sum()
                    gamma_div = math.exp(-term)
                elif q == 0:
                    occupied_gamma = (g > 0).astype(float)
                    gamma_div = (occupied_gamma.mul(branchL)).sum() / Tgamma
                else:
                    term = (branchL * g.clip(lower=0).pow(q)).sum() / Tgamma
                    gamma_div = term ** (1.0 / (1.0 - q))
    
                # --- α-diversity ---------------------------------------------------
                a1 = sub[s1]
                a2 = sub[s2]
                
                if q == 1.0:
                    term1 = np.zeros_like(a1)
                    mask = a1 > 0
                    term1[mask] = a1[mask] * np.log(a1[mask])

                    term2 = np.zeros_like(a2)
                    mask = a2 > 0
                    term2[mask] = a2[mask] * np.log(a2[mask])
                    
                    H = -((branchL * (term1 + term2)).sum() / (2.0 * Tgamma))
                    alpha_div = math.exp(H)
                elif q == 0:
                    pos_counts = ((a1 > 0).astype(float) + (a2 > 0).astype(float))
                    alpha_div = (branchL * pos_counts).sum() / (2.0 * Tgamma)
                else:
                    term = (
                        branchL *
                        ((a1.clip(lower=0).pow(q) + a2.clip(lower=0).pow(q)) / 2.0)
                    ).sum() / Tgamma
                
                    alpha_div = term ** (1.0 / (1.0 - q))
    
                # β-diversity
                beta_val = gamma_div / alpha_div
                out.loc[s1, s2] = beta_val
                out.loc[s2, s1] = beta_val

    # --- Convert β to dissimilarity if requested ------------------------------
    # Ensure beta diagonal is 1
    for s in out.index:
        out.loc[s, s] = 1.0

    if dis:
        return beta2dist(beta=out, q=q, N=2, div_type="phyl", viewpoint=viewpoint)

    return out

# -----------------------------------------------------------------------------
# Functional beta diversity
# -----------------------------------------------------------------------------
def func_beta(
    tab: Union[pd.DataFrame, Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1,
    dis: bool = True,
    viewpoint: str = "regional",
    use_values_in_tab: bool = False,
    use_tqdm: bool = True,
    use_numba: bool = False,
) -> pd.DataFrame:
    """
    Compute functional pairwise beta diversity of order *q*.

    Implements the two‑community functional Hill‑number beta framework
    based on local functional overlaps as in Chao et al. (2014). Functional
    diversity is derived from pairwise trait distances between ASVs and
    their abundances.

    For each pair of samples (A, B), the method computes:

        - Dg : functional Hill number for the pooled community (gamma)
        - Da : functional Hill number for the "average" community (alpha)
        - beta = Dg / Da

    For q = 1, the Shannon-type limit is used; for q ≠ 1, the general
    Hill-number form is used.

    Parameters
    ----------
    tab : DataFrame | MicrobiomeData-like | dict
        Abundance table (features x samples) or convertible structure.
    distmat : pandas.DataFrame
        Functional distance matrix (ASVs × ASVs), symmetric and
        indexed by the same ASVs as `tab`.
    q : float, default=1
        Diversity order. Determines emphasis on relative abundances.
    dis : bool, default=True
        If True, converts beta diversity to a dissimilarity scaled from 0 to 1.
        If False, returns raw beta values scaled from 1 to 2.
    viewpoint : {'local', 'regional'}, default='regional'
        Viewpoint for converting beta diversity to dissimilarity.
    use_values_in_tab : bool, default=False
        If False, convert abundances to relative abundances.
        If True, assume `tab` already contains relative abundances.
    use_tqdm : bool, default=True
        Use `tqdm` for progress bars.
    use_numba : bool, optional
        If True, accelerates calculation using Numba (which requires that Numba is installed).
        Very useful for large datasets.

    Returns
    -------
    DataFrame with pairwise functional dissimilarities ``dis=True``.
        If ``dis=False``, returns the pairwise functional beta diversity
        matrix, with diagonal values equal to 1.
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

    if isinstance(ra, pd.Series) or ra.shape[1] < 2:
        raise ValueError("`tab` must contain ≥ 2 samples (columns).")

    # Align distance matrix to features
    missing = set(ra.index) - set(distmat.index)
    if missing:
        raise ValueError(
            f"Features in tab are missing from distmat. Examples: {list(missing)[:5]}"
        )
    distmat = distmat.loc[ra.index, ra.index]
    smplist = list(ra.columns)

    # Check accelerator
    if use_numba:
        try:
            from .accelerate_div import func_beta_numba
        except ImportError:
            print("Numba not available, falling back to Python.")
            func_beta_numba = None
    else:
        func_beta_numba = None

    if func_beta_numba is not None:
        D = np.ascontiguousarray(distmat.to_numpy(dtype=np.float64, copy=True))
        R = np.ascontiguousarray(ra.to_numpy(dtype=np.float64, copy=True))
        out_arr = func_beta_numba(D, R, float(q))
        outD = pd.DataFrame(out_arr, index=smplist, columns=smplist)

    else:
        outD = pd.DataFrame(0.0, index=smplist, columns=smplist)
    
        # Pairwise functional beta diversity
        tqdm = _get_tqdm(use_tqdm)
    
        for i in tqdm(range(len(smplist) - 1), desc="func_beta", unit="sample"):
            for j in range(i + 1, len(smplist)):
                s1 = smplist[i]
                s2 = smplist[j]
    
                # Subset abundances for the two samples
                ra12 = ra[[s1, s2]].copy()
                ra12["mean"] = ra12.mean(axis=1)
    
                # Rao's Q for each column and for the mean
                Qvals = rao(ra12, distmat)
                Q_pooled = Qvals["mean"]
                if Q_pooled <= 0 or not np.isfinite(Q_pooled):
                    raise ValueError(
                        f"Functional Rao's Q is zero or invalid for pair '{s1}' and '{s2}'."
                    )
                dqmat = distmat * (1.0 / Q_pooled)
    
                # -------------------------
                # Gamma component (Dg)
                # -------------------------
                mask_g = ra12["mean"] > 0
                p_mean = ra12.loc[mask_g, "mean"].to_numpy()
                outer_mean = np.outer(p_mean, p_mean)
    
                if q == 1:
                    # Shannon-type functional gamma
                    log_outer = np.log(outer_mean)
                    term = outer_mean * log_outer
                    # dqmat restricted to nonzero rows/cols
                    d_sub = dqmat.loc[mask_g, mask_g].to_numpy()
                    Dg = math.exp(-0.5 * np.sum(term * d_sub))
                else:
                    outer_q = outer_mean ** q
                    d_sub = dqmat.loc[mask_g, mask_g].to_numpy()
                    val = np.sum(outer_q * d_sub)
                    Dg = val ** (1.0 / (2.0 * (1.0 - q)))
    
                # -------------------------
                # Alpha component (Da)
                # -------------------------
                # A: p1 × p1
                mask1 = ra12[s1] > 0
                p1 = ra12.loc[mask1, s1].to_numpy()
                outer11 = np.outer(p1, p1) / 4.0
                d11 = dqmat.loc[mask1, mask1].to_numpy()
    
                # B: p2 × p2
                mask2 = ra12[s2] > 0
                p2 = ra12.loc[mask2, s2].to_numpy()
                outer22 = np.outer(p2, p2) / 4.0
                d22 = dqmat.loc[mask2, mask2].to_numpy()
    
                # C: p1 × p2
                # note: indices differ; use full submatrix
                outer12 = np.outer(p1, p2) / 4.0
                d12 = dqmat.loc[mask1, mask2].to_numpy()
    
                if q == 1:
                    # Shannon-type functional alpha
                    # A
                    log11 = np.log(outer11)
                    term11 = outer11 * log11 * d11
                    asum1 = term11.sum()
    
                    # B
                    log22 = np.log(outer22)
                    term22 = outer22 * log22 * d22
                    asum2 = term22.sum()
    
                    # C
                    log12 = np.log(outer12)
                    term12 = outer12 * log12 * d12
                    asum12 = term12.sum()
    
                    Da = 0.5 * math.exp(-0.5 * (asum1 + asum2 + 2.0 * asum12))
                else:
                    # General q alpha
                    a11_q = outer11 ** q
                    asum1 = np.sum(a11_q * d11)
    
                    a22_q = outer22 ** q
                    asum2 = np.sum(a22_q * d22)
    
                    a12_q = outer12 ** q
                    asum12 = np.sum(a12_q * d12)
    
                    Da = 0.5 * (asum1 + asum2 + 2.0 * asum12) ** (1.0 / (2.0 * (1.0 - q)))
    
                # -------------------------
                # Beta component
                # -------------------------
                beta_val = Dg / Da
                outD.loc[s1, s2] = beta_val
                outD.loc[s2, s1] = beta_val

    # Ensure beta diagonal is 1
    for s in outD.index:
        outD.loc[s, s] = 1.0
    
    # Convert beta to dissimilarity if requested
    if dis:
        return beta2dist(
            beta=outD,
            q=q,
            N=2,
            div_type="func",
            viewpoint=viewpoint,
        )
    
    return outD

# -----------------------------------------------------------------------------
# Bray-Curtis
# -----------------------------------------------------------------------------
def bray(
    tab: Union[pd.DataFrame, Dict[str, Any], Any],
    *,
    use_values_in_tab: bool = False
) -> pd.DataFrame:
    """
    Compute the Bray–Curtis dissimilarity matrix between all samples.

    Parameters
    ----------
    tab : DataFrame | MicrobiomeData-like | dict
        Abundance table (features x samples) or convertible structure.
    use_values_in_tab : bool, default=False
        If False, convert abundances to relative abundances.
        If True, assume `tab` already contains relative abundances.

    Returns
    -------
    pandas.DataFrame
        Symmetric Bray–Curtis dissimilarity matrix.
    """

    # --- Validate input ------------------------------------------------------
    tab = get_df(tab, "tab")
    if tab.shape[1] < 2:
        raise ValueError("`tab` must contain at least two samples (columns).")

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

    # --- Compute Bray–Curtis -------------------------------------------------
    samples = ra.columns.tolist()
    out = pd.DataFrame(0.0, index=samples, columns=samples)

    # Efficient pairwise computation
    for i in range(len(samples) - 1):
        s1 = samples[i]
        p1 = ra[s1]

        for j in range(i + 1, len(samples)):
            s2 = samples[j]
            p2 = ra[s2]

            # BC = 1 − Σ min(p_iA, p_iB)
            bc = 1.0 - np.minimum(p1, p2).sum()

            out.loc[s1, s2] = bc
            out.loc[s2, s1] = bc

    return out

# -----------------------------------------------------------------------------
# Jaccard
# -----------------------------------------------------------------------------
def jaccard(
    tab: Union[pd.DataFrame, Dict[str, Any], Any],
    *,
    use_values_in_tab: bool = False
) -> pd.DataFrame:
    """
    Compute the Jaccard dissimilarity matrix between all samples.

    Parameters
    ----------
    tab : DataFrame | MicrobiomeData-like | dict
        Abundance table (features x samples) or convertible structure.
    use_values_in_tab : bool, default=False
        Ignored for Jaccard (presence/absence only), included for API symmetry.

    Returns
    -------
    pandas.DataFrame
        Symmetric Jaccard dissimilarity matrix.
    """

    # --- Validate input ------------------------------------------------------
    tab = get_df(tab, "tab")
    if tab.shape[1] < 2:
        raise ValueError("`tab` must contain at least two samples (columns).")

    # Ensure numeric
    try:
        tab = tab.astype(float)
    except Exception as e:
        raise TypeError(
            "Abundance table contains non-numeric values. Ensure counts/abundances are numeric."
        ) from e

    # --- Convert to presence/absence ----------------------------------------
    bintab = (tab > 0).astype(int)

    samples = bintab.columns.tolist()
    out = pd.DataFrame(0.0, index=samples, columns=samples)

    # --- Compute Jaccard dissimilarity --------------------------------------
    for i in range(len(samples) - 1):
        s1 = samples[i]
        v1 = bintab[s1]

        for j in range(i + 1, len(samples)):
            s2 = samples[j]
            v2 = bintab[s2]

            shared = np.sum((v1 == 1) & (v2 == 1))
            total = np.sum((v1 == 1) | (v2 == 1))

            # Avoid division by zero (no species in either sample)
            jac = 1.0 if total == 0 else 1.0 - shared / total

            out.loc[s1, s2] = jac
            out.loc[s2, s1] = jac

    return out

# -----------------------------------------------------------------------------
# Naive beta diversity for multiple samples
# -----------------------------------------------------------------------------
def naive_multi_beta(
    obj: Union[Dict[str, Any], Any],
    *,
    by: Optional[str] = None,
    q: float = 1,
    use_values_in_tab: bool = False,
) -> pd.DataFrame:
    """
    Compute naive (taxonomic) multi‑sample beta diversity for groups of samples.

    Parameters
    ----------
    obj : MicrobiomeData-like | dict
        Must contain
        
            - ``'meta'`` : pandas.DataFrame with sample metadata
            - ``'tab'``  : pandas.DataFrame with feature counts (features × samples)
    
    by : str or None, default=None
        Column in metadata defining sample groups.
        If None, all samples are treated as one group.
    q : float, default=1
        Diversity order. Determines emphasis on relative abundances.
    use_values_in_tab : bool, default=False
        If False, abundances are converted to relative abundances per sample.
        If True, the abundance table is assumed to already contain relative
        abundances.

    Returns
    -------
    pandas.DataFrame
        Index = categories in `by` (or 'all' if by=None).
        Columns:
            
            - N, number of samples in group
            - beta, multi‑sample beta diversity
            - local_dis, local‑viewpoint dissimilarity
            - regional_dis, regional‑viewpoint dissimilarity

    Notes
    -----
    - Groups with <2 samples return NaN.
    """

    # Validate input
    meta = get_df(obj, "meta")
    tab = get_df(obj, "tab")

    # Confirm tab input is ok
    if tab.shape[1] < 2:
        raise ValueError("At least two samples are required.")
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

    # Build dictionary of subtables by category
    if by is None:
        categories = ["all"]
        tabdict = {"all": ra.copy()}
    else:
        if by not in meta.columns:
            raise ValueError(f"Column '{by}' not found in metadata.")
        tabdict = {}
        categories = meta[by].unique().tolist()
        for cat in categories:
            smplist = meta[meta[by]==cat].index
            tabdict[cat] = ra[smplist]

    # Output container
    out = pd.DataFrame(
        np.nan,
        index=categories,
        columns=["N", "beta", "local_dis", "regional_dis"]
    )

    # Compute multi‑sample beta for each category
    for cat in categories:
        subtab = tabdict[cat]

        # Need at least 2 samples
        if subtab.shape[1] < 2:
            continue

        N = subtab.shape[1]
        out.loc[cat, "N"] = N

        # Build alpha/gamma table for naive_alpha()
        # gamma row = pooled abundances
        gamma_row = subtab.sum(axis=1).to_numpy()

        # alpha rows = each sample's abundances
        alpha_rows = subtab.to_numpy().T.reshape(-1)

        df_temp = pd.DataFrame({
            "gamma": np.concatenate([gamma_row, np.zeros_like(alpha_rows)]),
            "alpha": np.concatenate([np.zeros_like(gamma_row), alpha_rows])
        })

        # Compute alpha and gamma Hill numbers
        divs = naive_alpha(df_temp, q=q, use_values_in_tab=False)

        gamma_div = divs["gamma"]
        alpha_div = divs["alpha"] / N

        beta = gamma_div / alpha_div
        out.loc[cat, "beta"] = beta

        # Convert to dissimilarities
        out.loc[cat, "local_dis"] = beta2dist(
            beta, q=q, N=N, div_type="naive", viewpoint="local"
        )
        out.loc[cat, "regional_dis"] = beta2dist(
            beta, q=q, N=N, div_type="naive", viewpoint="regional"
        )

    return out

# -----------------------------------------------------------------------------
# Phylogenetic beta diversity for multiple samples
# -----------------------------------------------------------------------------
def phyl_multi_beta(
    obj: Union[Dict[str, Any], Any],
    *,
    by: Optional[str] = None,
    q: float = 1,
    use_values_in_tab: bool = False,
) -> pd.DataFrame:
    """
    Compute phylogenetic multi‑sample beta diversity for groups of samples.
    Implements the multi‑sample phylogenetic Hill‑number beta framework
    described in Chao et al. (2014), where branch lengths are weighted by
    the relative abundances of all ASVs descending from each branch.

    Parameters
    ----------
    obj : MicrobiomeData-like | dict
        Must include

          - ``'tab'``, feature × sample abundance DataFrame
          - ``'tree'``, branch information DataFrame
          - ``'leaf_order'``, list of leaf names corresponding to the tree

    by : str or None, default=None
        Column in metadata defining sample groups.
        If None, all samples are treated as one group.
    q : float, default=1
        Diversity order. Determines emphasis on relative abundances.
    use_values_in_tab : bool, default=False
        If False, abundances are converted to relative abundances per sample.
        If True, the abundance table is assumed to already contain relative
        abundances.

    Returns
    -------
    pandas.DataFrame
        Index = categories in `by` (or 'all' if by=None).
        Columns:
            
            - N, number of samples in group
            - beta, multi‑sample beta diversity
            - local_dis, local‑viewpoint dissimilarity
            - regional_dis, regional‑viewpoint dissimilarity

    Notes
    -----
    - Groups with <2 samples return NaN.
    """

    # Validate input
    tab = get_df(obj, "tab")
    meta = get_df(obj, "meta")
    tree = get_df(obj, "tree")
    if tree is None:
        raise ValueError('tree is missing.')
    leaf_order = get_df(obj, "leaf_order")
    if leaf_order is None:
        tree, leaf_order = rebuild_leaf_order(tree)

    # Confirm tree input is ok
    required_tree_cols = {"branchL", "leaf_start", "leaf_end"}
    missing = required_tree_cols - set(tree.columns)
    if missing:
        raise ValueError(
            f"`tree` must contain columns {sorted(required_tree_cols)}. "
            f"Missing: {sorted(missing)}."
        )
    if leaf_order is None:
        raise ValueError("`leaf_order` is required for trees.")

    # Confirm tab input is ok
    if tab.shape[1] < 2:
        raise ValueError("At least two samples are required.")
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

    # Build dictionary of tree2 tables by category
    if by is None:
        categories = ["all"]
        tabdict = {"all": tree2.copy()}
    else:
        if by not in meta.columns:
            raise ValueError(f"Column '{by}' not found in metadata.")
        tabdict = {}
        categories = meta[by].unique().tolist()
        for cat in categories:
            smplist = meta[meta[by]==cat].index
            tabdict[cat] = tree2[smplist]

    # Output container
    out = pd.DataFrame(
        np.nan,
        index=categories,
        columns=["N", "beta", "local_dis", "regional_dis"]
    )

    # Compute multi‑sample phylogenetic beta for each category
    for cat in categories:
        subtree2 = tabdict[cat]

        # Need at least 2 samples
        if subtree2.shape[1] < 2:
            continue

        N = subtree2.shape[1]
        out.loc[cat, "N"] = N

        # --- Compute Tavg = Σ_b L_b * mean(p_b) -------------------------------------
        # Align branch lengths to tree2 (branch × sample) and ensure numeric
        branchL = pd.to_numeric(tree["branchL"], errors="raise")
        if branchL.isna().any():
            missing = branchL.index[branchL.isna()].tolist()
            raise ValueError(f"'branchL' missing for branches: {missing}")
        
        mean_ra = subtree2.mean(axis=1)                  # γ_b: mean of per-branch RA across N samples
        Tavg = float(mean_ra.mul(branchL).sum())      # Σ L_b * γ_b
        
        # --- γ-diversity -------------------------------------------------------------
        g = mean_ra
        if abs(q - 1.0) < 1e-6:
            mask = g > 0
            H = -((g[mask] * np.log(g[mask]) * branchL[mask]).sum() / Tavg)
            gamma_div = math.exp(H)
        
        elif q == 0:
            gamma_div = branchL[g > 0].sum() / Tavg
        
        else:
            term = (branchL * g.clip(lower=0).pow(q)).sum() / Tavg
            gamma_div = term ** (1.0 / (1.0 - q))
        
        # --- α-diversity -------------------------------------------------------------
        K = subtree2.shape[1]
        
        if q == 1.0:
            mask = subtree2 > 0
            term = (
                (subtree2[mask] * np.log(subtree2[mask]))
                .mul(branchL, axis=0)
                .sum().sum()
                / (K * Tavg)
            )
            alpha_div = math.exp(-term)
        
        elif q == 0:
            pos_counts = (subtree2 > 0).sum(axis=1).astype(float)
            alpha_div = (branchL * pos_counts).sum() / (K * Tavg)
        
        else:
            term = (
                branchL *
                (subtree2.clip(lower=0).pow(q).sum(axis=1) / K)
            ).sum() / Tavg
            alpha_div = term ** (1.0 / (1.0 - q))

        # β-diversity
        beta = gamma_div / alpha_div
        out.loc[cat, "beta"] = beta

        out.loc[cat, "local_dis"] = beta2dist(
            beta=beta, q=q, N=N, div_type="phyl", viewpoint="local"
        )
        out.loc[cat, "regional_dis"] = beta2dist(
            beta=beta, q=q, N=N, div_type="phyl", viewpoint="regional"
        )

    return out

# -----------------------------------------------------------------------------
# Functional beta diversity for multiple samples
# -----------------------------------------------------------------------------
def func_multi_beta(
    obj: Union[Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    by: Optional[str] = None,
    q: float = 1,
    use_values_in_tab: bool = False,
) -> pd.DataFrame:
    """
    Compute functional multi‑sample beta diversity for groups of samples.

    Implements the multi‑sample functional Hill‑number beta framework
    described in Chiu et al. (2014), where functional diversity is derived
    from pairwise trait distances and species abundances.

    Parameters
    ----------
    obj : MicrobiomeData-like | dict
        Must contain
        
            - ``'meta'``, pandas.DataFrame with sample metadata
            - ``'tab'``, pandas.DataFrame (features × samples)

    distmat : pandas.DataFrame
        Functional distance matrix (features × features).
    by : str or None, default=None
        Column in metadata defining sample groups.
        If None, all samples are treated as one group.
    q : float, default=1
        Diversity order. Determines emphasis on relative abundances.
    use_values_in_tab : bool, default=False
        If False, abundances are converted to relative abundances per sample.
        If True, the abundance table is assumed to already contain relative
        abundances.

    Returns
    -------
    pandas.DataFrame
        Index = categories in `by` (or 'all' if by=None)
        Columns:
            
            - N, number of samples in group
            - beta, multi‑sample beta diversity
            - local_dis, local‑viewpoint dissimilarity
            - regional_dis, regional‑viewpoint dissimilarity

    Notes
    -----
    - Groups with <2 samples return NaN.
    """

    # Validate input
    tab = get_df(obj, "tab")
    meta = get_df(obj, "meta")

    # Confirm tab input is ok
    if tab.shape[1] < 2:
        raise ValueError("At least two samples are required.")
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

    # Make sure tab and distmat have the same index
    in_common = ra.index.intersection(distmat.index)
    if len(in_common) < len(ra.index):
        raise ValueError("Features in tab are missing in distmat.")
    ra = ra.loc[in_common]
    distmat = distmat.loc[in_common, in_common].copy()

    # Build dictionary of subtables by category
    if by is None:
        categories = ["all"]
        tabdict = {"all": ra.copy()}
    else:
        if by not in meta.columns:
            raise ValueError(f"Column '{by}' not found in metadata.")
        tabdict = {}
        categories = meta[by].unique().tolist()
        for cat in categories:
            smplist = meta[meta[by]==cat].index
            tabdict[cat] = ra[smplist]

    # Output container
    out = pd.DataFrame(
        np.nan,
        index=categories,
        columns=["N", "beta", "local_dis", "regional_dis"]
    )

    # Compute multi‑sample functional beta for each category
    for cat in categories:
        subtab = tabdict[cat]

        # Need at least 2 samples
        if subtab.shape[1] < 2:
            continue

        N = subtab.shape[1]
        out.loc[cat, "N"] = N

        smplist = subtab.columns.tolist()

        # Compute pooled mean abundances
        ra_mean = subtab.mean(axis=1)

        # Rao's Q for pooled community
        Q_pooled = rao(ra_mean, distmat)
        if Q_pooled <= 0 or not np.isfinite(Q_pooled):
            raise ValueError(
                f"Functional Rao's Q is zero or invalid for group '{cat}'. "
                "Functional beta diversity cannot be computed."
            )
        dqmat = distmat * (1.0 / Q_pooled)

        # γ-diversity (pooled)
        p = ra_mean.to_numpy()
        outer = np.outer(p, p)

        if q == 1:
            mask = outer > 0
            log_outer = np.zeros_like(outer)
            log_outer[mask] = np.log(outer[mask])
            term = outer * log_outer
            Dg = math.exp(-0.5 * np.sum(term * dqmat.values))
        else:
            mask = outer > 0
            outer_q = np.zeros_like(outer)
            outer_q[mask] = outer[mask] ** q
            Dg = (np.sum(outer_q * dqmat.values)) ** (1.0 / (2.0 * (1.0 - q)))

        # α-diversity (mean over all ordered sample pairs)
        asum = 0.0

        for s1 in smplist:
            p1 = subtab[s1].to_numpy()
            for s2 in smplist:
                p2 = subtab[s2].to_numpy()

                outer12 = np.outer(p1, p2) / (N * N)

                if q == 1.0:
                    mask = outer12 > 0
                    log_outer = np.zeros_like(outer12)
                    log_outer[mask] = np.log(outer12[mask])
                    term = outer12 * log_outer
                    asum += np.sum(term * dqmat.values)
                else:
                    mask = outer12 > 0
                    outer_q = np.zeros_like(outer12)
                    outer_q[mask] = outer12[mask] ** q
                    asum += np.sum(outer_q * dqmat.values)

        if q == 1:
            Da = (1.0 / N) * math.exp(-0.5 * asum)
        else:
            Da = (1.0 / N) * (asum ** (1.0 / (2.0 * (1.0 - q))))

        # β-diversity
        beta = Dg / Da
        out.loc[cat, "beta"] = beta

        out.loc[cat, "local_dis"] = beta2dist(
            beta=beta, q=q, N=N, div_type="func", viewpoint="local"
        )
        out.loc[cat, "regional_dis"] = beta2dist(
            beta=beta, q=q, N=N, div_type="func", viewpoint="regional"
        )

    return out

# -----------------------------------------------------------------------------
# Evenness
# -----------------------------------------------------------------------------
def evenness(
    obj: Union[pd.DataFrame, Dict[str, Any], Any],
    distmat: Optional[pd.DataFrame] = None,
    *,
    q: float = 1,
    div_type: str = "naive",
    index: str = "pielou",
    perspective: str = "samples",
    use_values_in_tab: bool = False
) -> pd.Series:
    """
    Compute evenness measures from Chao & Ricotta (2019, Ecology 100:e02852),
    with optional support for Pielou’s classical evenness index.
    
    Supports
    
        - naive (taxonomic) evenness
        - phylogenetic evenness
        - functional evenness
    
    Supported evenness indices
    
        - CR1  (regional evenness)
        - CR2  (local evenness)
        - CR3
        - CR4
        - CR5
        - pielou  (Pielou’s J; defined only for q = 1)
    
    Parameters
    ----------
    obj : DataFrame | MicrobiomeData-like | dict
        Including abundance table (features × samples) and optionally
        tree (pandas.DataFrame, required if div_type='phyl')
    distmat : pandas.DataFrame, optional
        Required if div_type='func'. Functional distance matrix.
    q : float, default=1
        Diversity order. Determines emphasis on relative abundances.
    div_type : 'naive', 'phyl', or 'func', default='naive'
        Type of diversity measure used to compute D.
    index : 'CR1','CR2','CR3','CR4','CR5','local','regional', or 'pielou', default='pielou'
        Evenness index to compute.
    perspective : 'samples'or 'taxa', default='samples'
        Whether to compute evenness across samples (columns)
        or across taxa/branches (rows).
    use_values_in_tab : bool, default=False
        If False, convert abundances to relative abundances.
    
    Returns
    -------
    pandas.Series
        Evenness values indexed by sample or taxon.
    
    Notes
    -----
    - CR1 = regional evenness
    - CR2 = local evenness
    - CR3–CR5 are alternative evenness formulations from Chao & Ricotta (2019)
    - Pielou’s index is included for convenience and corresponds to
    
          J = H' / ln(S) = ln(D₁) / ln(S)
          
      where D₁ is the Hill number of order q = 1.
    """

    # Validate inputs
    tab = get_df(obj, "tab")

    if perspective not in ("samples", "taxa"):
        raise ValueError("perspective must be 'samples' or 'taxa'.")

    if index in ("CR1", "regional"):
        power = 1 - q
    elif index in ("CR2", "local"):
        power = q - 1
    else:
        power = None  # used only for CR3–CR5

    # Compute S (richness) and D (Hill number)
    if perspective == "samples":
        # Richness per sample
        S_series = (tab > 0).sum(axis=0)

        # Diversity per sample
        if div_type == "naive":
            D_series = naive_alpha(tab, q=q, use_values_in_tab=use_values_in_tab)

        elif div_type == "phyl":
            tree = get_df(obj, "tree")
            if not isinstance(tree, pd.DataFrame):
                raise ValueError("tree must be provided for div_type='phyl'.")
            D_series = phyl_alpha(obj, q=q, index="D",
                                  use_values_in_tab=use_values_in_tab)

        elif div_type == "func":
            if not isinstance(distmat, pd.DataFrame):
                raise ValueError("distmat must be provided for div_type='func'.")
            D_series = func_alpha(tab, distmat, q=q, index="D",
                                  use_values_in_tab=use_values_in_tab)

        else:
            raise ValueError("divType must be 'naive', 'phyl', or 'func'.")

    # Perspective = taxa
    else:
        if div_type == "naive":
            tabT = tab.T
            S_series = tabT.count().astype(float)
            D_series = naive_alpha(tabT, q=q, use_values_in_tab=use_values_in_tab)

        elif div_type == "phyl":
            tree = get_df(obj, "tree")
            if not isinstance(tree, pd.DataFrame):
                raise ValueError("tree must be provided for divType='phyl'.")
            leaf_order = get_df(obj, "leaf_order")
            if not isinstance(leaf_order, list):
                raise ValueError("leaf_order must be provided for divType='phyl'.")

            # Relative abundances
            ra = tab.div(tab.sum()) if not use_values_in_tab else tab.astype(float)

            # Build branch × sample matrix
            tree2 = ra_to_branches(ra, tree, leaf_order)

            # Normalize across samples
            tree2 = tree2.T
            tree2 = tree2.div(tree2.sum()).fillna(0.0)

            #S_series = tree2.count().astype(float)
            S_series = (tree2 > 0).sum().astype(float)
            D_series = naive_alpha(tree2, q=q, use_values_in_tab=True)

        else:
            raise ValueError("divType must be 'naive' or 'phyl' when perspective='taxa'.")

    # Compute evenness index
    if index in ("CR1", "CR2", "regional", "local"):
        if q == 1:
            df = pd.DataFrame({"D": D_series, "S": S_series}).astype(float)
            measure = pd.Series(np.nan, index=df.index, dtype=float)
            mask = (df["S"] > 1) & (df["D"] > 0)
            measure.loc[mask] = (
                np.log(df.loc[mask, "D"])
                / np.log(df.loc[mask, "S"])
            )
            measure.loc[df["S"] <= 1] = np.nan
        else:
            Dp = D_series.astype(float).pow(power)
            Sp = S_series.astype(float).pow(power)
            measure = (1 - Dp) / (1 - Sp)
            measure = measure.replace([np.inf, -np.inf], np.nan)
            mask = S_series <= 1
            measure.loc[mask] = np.nan

    elif index == "CR3":
        measure = (D_series - 1) / (S_series - 1)
    
    elif index == "CR4":
        measure = (1 - 1 / D_series) / (1 - 1 / S_series)
    
    elif index == "CR5":
        measure = np.log(D_series) / np.log(S_series)
    
    elif index == "pielou":
        if q != 1:
            raise ValueError("Pielou's index is only defined for q = 1.")
        measure = np.log(D_series) / np.log(S_series)
    
    else:
        raise ValueError("index must be one of: CR1, CR2, CR3, CR4, CR5, local, regional, pielou.")

    measure = measure.fillna(1.0)
    return measure

# -----------------------------------------------------------------------------
# Dissimilarity by feature
# -----------------------------------------------------------------------------
def dissimilarity_by_feature(
    obj: Union[Dict[str, Any], Any],
    *,
    by: Optional[str] = None,
    q: float = 1,
    div_type: str = "naive",
    index: str = "regional",
    use_values_in_tab: bool = False,
) -> pd.DataFrame:
    """
    Compute the contribution of individual taxa or phylogenetic branches
    to the overall dissimilarity between multiple samples, following
    Chao and Ricotta (2019).

    Supports:
        
        - naive taxonomic dissimilarity
        - phylogenetic dissimilarity

    Parameters
    ----------
    obj : DataFrame | MicrobiomeData-like | dict
        Must contain
        
            - ``'tab'``, abundance table, features x samples
            - ``'meta'``, metadata table, required if by is not None
            - ``'tree'``, phylogenetic tree, required if div_type='phyl'
            - ``'leaf_order'``, required if div_type='phyl'

    by : str or None, default None
        Metadata column defining sample groups.
        If None, all samples are treated as one group.
    q : float, default=1
        Diversity order. Determines emphasis on relative abundances.
    div_type : {'naive', 'phyl'}, default 'naive'
        Type of dissimilarity measure.
    index : {'local', 'regional', 'CR1', 'CR2'}, default 'regional'
        Dissimilarity index.
        ``'regional'`` and ``'CR1'`` are equivalent.
        ``'local'`` and ``'CR2'`` are equivalent.
    use_values_in_tab : bool, default False
        If False, convert abundances to relative abundances.
        If True, values in tab are assumed to already be relative abundances.

    Returns
    -------
    pandas.DataFrame.
        Rows

            - 'dis' : total dissimilarity
            - 'N'   : number of samples in group
            - one row per taxon for naive dissimilarity
            - one row per branch or node for phylogenetic dissimilarity

        Columns
 
            - one column per category in ``'by'``
    """

    # ---------------------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------------------
    tab = get_df(obj, "tab")

    if tab is None:
        raise ValueError("'tab' is missing.")

    if tab.shape[1] < 2:
        raise ValueError("At least two samples are required.")

    if div_type not in ("naive", "phyl"):
        raise ValueError("div_type must be 'naive' or 'phyl'.")

    if index in ("CR1", "regional"):
        idx = "regional"
    elif index in ("CR2", "local"):
        idx = "local"
    else:
        raise ValueError("index must be 'local', 'regional', 'CR1', or 'CR2'.")

    tab = tab.astype(float)

    # ---------------------------------------------------------------------
    # Helper: q-power that handles q = 0 correctly
    # ---------------------------------------------------------------------
    def positive_power(x, q):
        """
        Compute x^q for positive x, while keeping zeros as zero.

        This avoids the pandas/numpy behavior where 0**0 becomes 1.
        """
        if isinstance(x, pd.DataFrame):
            out = pd.DataFrame(0.0, index=x.index, columns=x.columns)
            mask = x > 0
            if q == 0:
                out[mask] = 1.0
            else:
                out[mask] = x[mask].pow(q)
            return out

        if isinstance(x, pd.Series):
            out = pd.Series(0.0, index=x.index)
            mask = x > 0
            if q == 0:
                out.loc[mask] = 1.0
            else:
                out.loc[mask] = x.loc[mask].pow(q)
            return out

        raise TypeError("positive_power expects a pandas Series or DataFrame.")

    # ---------------------------------------------------------------------
    # Build dictionary of subtables by category
    # ---------------------------------------------------------------------
    if by is None:
        categories = ["all"]
        tabdict = {"all": tab.copy()}
    else:
        meta = get_df(obj, "meta")

        if meta is None:
            raise ValueError("'meta' is required when 'by' is provided.")

        if by not in meta.columns:
            raise ValueError(f"Column '{by}' not found in metadata.")

        categories = meta[by].unique().tolist()

        tabdict = {
            cat: get_df(
                subset_samples(obj, by=by, values=[cat]),
                "tab",
            )
            for cat in categories
        }

    # ---------------------------------------------------------------------
    # Prepare tree if needed
    # ---------------------------------------------------------------------
    if div_type == "phyl":
        tree = get_df(obj, "tree")
        leaf_order = get_df(obj, "leaf_order")

        if tree is None:
            raise ValueError("'tree' is required when div_type='phyl'.")

        if leaf_order is None:
            tree, leaf_order = rebuild_leaf_order(tree)

        feature_index = ["dis", "N"] + tree["nodes"].tolist()

    else:
        feature_index = ["dis", "N"] + tab.index.tolist()

    out = pd.DataFrame(np.nan, index=feature_index, columns=categories)

    # ---------------------------------------------------------------------
    # Main loop over categories
    # ---------------------------------------------------------------------
    for cat in categories:
        subtab = tabdict[cat].astype(float)
        N = subtab.shape[1]
        out.loc["N", cat] = N

        if N < 2:
            continue

        # Relative abundances
        if use_values_in_tab:
            ra = subtab
        else:
            col_sums = subtab.sum(axis=0)
            if (col_sums == 0).any():
                raise ValueError(f"Group '{cat}' contains a zero-sum sample.")
            ra = subtab.div(col_sums, axis=1)

        # -----------------------------------------------------------------
        # Naive version
        # -----------------------------------------------------------------
        if div_type == "naive":
            # Use relative abundances for weights, not raw counts.
            # This avoids group-level sequencing-depth effects.
            if idx == "regional":
                z = ra.sum(axis=1)
                zq = positive_power(z, q)
                w = zq / zq.sum()

            else:
                ra_q = positive_power(ra, q)
                w = ra_q.sum(axis=1) / ra_q.sum().sum()

            ev = evenness(
                ra if use_values_in_tab else subtab,
                q=q,
                div_type="naive",
                index=idx,
                perspective="taxa",
                use_values_in_tab=use_values_in_tab,
            )

            contrib = w * (1 - ev)
            contrib = contrib.clip(lower=0)

            total = contrib.sum()
            out.loc["dis", cat] = total

            if total > 0:
                out.loc[contrib.index, cat] = 100 * contrib / total
            else:
                out.loc[contrib.index, cat] = 0.0

            out[cat] = out[cat].fillna(0)

        # -----------------------------------------------------------------
        # Phylogenetic version
        # -----------------------------------------------------------------
        elif div_type == "phyl":

            tree2 = ra_to_branches(
                ra=ra,
                tree_df=tree,
                leaf_order=leaf_order,
            )

            ev = evenness(
                {
                    "tab": subtab,
                    "tree": tree,
                    "leaf_order": leaf_order,
                },
                q=q,
                div_type="phyl",
                index=idx,
                perspective="taxa",
                use_values_in_tab=use_values_in_tab
            )

            branchL = tree["branchL"]

            if idx == "regional":
                zi = tree2.sum(axis=1)
                zi_q = positive_power(zi, q)
                w = branchL * zi_q
                w = w / w.sum()

            else:
                tree2_q = positive_power(tree2, q)
                zv = tree2_q.sum(axis=1)
                w = branchL * zv
                w = w / w.sum()

            contrib = w * (1 - ev)
            contrib = contrib.fillna(0)
            contrib = contrib.clip(lower=0)

            total = contrib.sum()
            out.loc["dis", cat] = total

            contrib.index = tree.loc[contrib.index, "nodes"]

            if total > 0:
                out.loc[contrib.index, cat] = 100 * contrib / total
            else:
                out.loc[contrib.index, cat] = 0.0

    return out


# -----------------------------------------------------------------------------
# beta MPDq and MNTDq
# -----------------------------------------------------------------------------
def beta_mpdq(
    obj: Union[Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1.0,
) -> pd.DataFrame:
    """
    Beta mean phylogenetic distance (MPD) with q-weighting of relative abundances.
    Calculated for sample pairs.

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
    pandas.DataFrame (S x S):
    """
    from ..model import beta_nriq
    return beta_nriq(obj, distmat, q=q, iterations=0)

def beta_mntdq(
    obj: Union[Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1.0,
    include_conspecifics: bool = False,
) -> pd.DataFrame:
    """
    Beta mean nearest taxon distance (MPD) with q-weighting of relative abundances.
    Calculated for sample pairs.

    Parameters
    ----------
    obj : MicrobiomeData, dict, or compatible object
        Input data. Must provide at least an abundance table ('tab').
    distmat : pd.DataFrame
        Square distance matrix indexed/columned by feature ids.
    q : float, default=1.0
        Order of diversity weighting applied to relative abundances.
    include_conspecifics : bool, default=False
        Determines whether conspecifics (identical features shared between samples) are allowed 
        to contribute zero-distance matches in the nearest-taxon calculation.

    Returns
    -------
    pandas.DataFrame (S x S):
    """
    from ..model import beta_ntiq
    return beta_ntiq(obj, distmat, q=q, iterations=0, include_conspecifics=include_conspecifics)

def beta_impdq(
    obj: Union[Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1.0,
    locality: float = 1.0,
    dist_scale: str | float = "auto",
    include_conspecifics: bool = True,
) -> Dict[str, Any]:
    """
    Beta interpolated mean phylogenetic distance (iMPD) with q-weighting of relative abundances.
    Calculated for sample pairs.

    Parameters
    ----------
    obj : MicrobiomeData, dict, or compatible object
        Input data. Must provide at least an abundance table ('tab').
    distmat : pd.DataFrame
        Square distance matrix indexed/columned by feature ids.
    q : float, default=1.0
        Order of diversity weighting applied to relative abundances.
    locality : float, default=1
        Non-negative kernel locality parameter. Larger values give stronger
        nearest-neighbour focus.
    dist_scale : "auto" or float, default="auto"
        Distance scale used to convert locality into kernel sharpness. If
        ``"auto"``, the median positive distance in ``distmat`` is used.
        Supplying a numeric value gives reproducible kernel sharpness across
        runs or datasets.
    include_conspecifics : bool, default=True
        Determines whether conspecifics (identical features shared between samples) are allowed 
        to contribute zero-distance matches in the nearest-taxon calculation.

    Returns
    -------
    Dictionary with pandas dataframes (S x S).
    
        - 'beta_iMPDq': the q-weighted beta-iMPD metric
        - 'beta_ENN': the effective number of neighbours contributing to the metric
        - 'beta_NTF': the nearest taxon focus
        - 'beta_ENN_min': the minimum ENN value possible for the give q-weighting
        - 'beta_ENN_max': the maximum ENN value possible for the give q-weighting
        - 'dist_scale': the dist_scale used in the calculations
    """
    from ..model import beta_inriq
    out = beta_inriq(obj, distmat, q=q, locality=locality, dist_scale=dist_scale, iterations=0, include_conspecifics=include_conspecifics)
    return out
