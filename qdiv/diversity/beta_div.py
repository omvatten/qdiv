import pandas as pd
import numpy as np
import math
from typing import Optional, Dict, Any, Union, Tuple
from ..io import subset_samples
from ..utils import rao, beta2dist, get_df
from ..utils import subset_tree_df, ra_to_branches, compute_Tmean
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
    use_values_in_tab: bool = False
) -> pd.DataFrame:
    """
    Compute naive (taxonomic) pairwise beta diversity of order *q*.

    Implements the two‑community Hill‑number beta diversity framework
    described in Chao et al. (2014), using only species abundances
    (no phylogenetic or functional information).

    For two samples A and B:

        α_q = Hill number of the average of A and B
        γ_q = Hill number of the pooled community
        β_q = γ_q / α_q

    Special case q = 1 uses the Shannon limit:

        α₁ = exp( -½ Σ pᵢ ln pᵢ  - ½ Σ qᵢ ln qᵢ )
        γ₁ = exp( -Σ mᵢ ln mᵢ )

    Parameters
    ----------
    tab : DataFrame | MicrobiomeData-like | dict
        Abundance table (features x samples) or convertible structure.
    q : float, default=1
        Diversity order.
    dis : bool, default=True
        If True, convert β to a dissimilarity using `beta2dist`.
        If False, return raw β values.
    viewpoint : {'local', 'regional'}, default='regional'
        Viewpoint for converting β to dissimilarity.
    use_values_in_tab : bool, default=False
        If False, convert abundances to relative abundances.
        If True, assume `tab` already contains relative abundances.

    Returns
    -------
    pandas.DataFrame
        Pairwise β-diversity (or dissimilarity) matrix.

    Notes
    -----
    - Requires `beta2dist()` to be defined elsewhere.
    - Only works for ≥ 2 samples.
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

    # Pairwise beta diversity
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
    use_values_in_tab: bool = False
) -> pd.DataFrame:
    """
    Compute phylogenetic pairwise beta diversity of order *q*.

    Implements the two‑community phylogenetic Hill‑number beta framework
    described in Chao et al. (2014), where branch lengths are weighted by
    the relative abundances of all features descending from each branch.

    For two samples A and B:

        α_q = phylogenetic Hill number of the average of A and B
        γ_q = phylogenetic Hill number of the pooled community
        β_q = γ_q / α_q

    Special case q = 1 uses the Shannon limit.

    Parameters
    ----------
    obj : MicrobiomeData-like | dict
        Must provide:
          - 'tab': feature × sample abundance DataFrame
          - 'tree': branch × columns DataFrame with:
                * 'leaves' : iterable/list of leaf IDs under each branch
                * 'branchL': branch length (float)
    q : float, default=1
        Diversity order.
    dis : bool, default=True
        If True, convert β to a dissimilarity using `beta2dist`.
    viewpoint : {'local', 'regional'}, default='regional'
        Viewpoint for converting β to dissimilarity.
    use_values_in_tab : bool, default=False
        If False, convert abundances to relative abundances.
        If True, assume `tab` already contains relative abundances.

    Returns
    -------
    pandas.DataFrame
        Pairwise phylogenetic β-diversity (or dissimilarity) matrix.

    Notes
    -----
    - Requires `beta2dist()` to be defined elsewhere.
    - Only works for ≥ 2 samples.
    """

    tab = get_df(obj, "tab")
    tree = get_df(obj, "tree")

    if "leaves" not in tree.columns or "branchL" not in tree.columns:
        raise ValueError("`tree` must contain columns 'leaves' and 'branchL'.")

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
    tree = subset_tree_df(tree, ra.index.tolist())
    Tmean = compute_Tmean(tree, ra.index.tolist())
    tree2 = ra_to_branches(ra, tree)

    # Align branch lengths to tree2 index
    branchL = tree["branchL"].reindex(tree2.index)
    if branchL.isna().any():
        missing = branchL.index[branchL.isna()].tolist()
        raise ValueError(f"'branchL' missing for branches: {missing}")

    # --- Pairwise phylogenetic beta diversity --------------------------------
    smplist = ra.columns.tolist()
    out = pd.DataFrame(0.0, index=smplist, columns=smplist)

    for i in range(len(smplist) - 1):
        s1 = smplist[i]
        for j in range(i + 1, len(smplist)):
            s2 = smplist[j]

            # Subtree abundances per branch
            sub = tree2[[s1, s2]].copy()

            # --- γ-diversity ---------------------------------------------------
            g = sub.mean(axis=1)
            if q == 1:
                mask = g > 0
                term = g.where(mask, 0.0) * np.log(g.where(mask, 1.0))
                term = (term * (branchL / Tmean)).sum()
                gamma_div = math.exp(-term)
            elif q == 0:
                occupied_gamma = (g > 0).astype(float)
                gamma_div = (occupied_gamma.mul(branchL)).sum() / Tmean
            else:
                g = g / Tmean
                term = (g.clip(lower=0) ** q).mul(branchL)
                gamma_div = (term.sum()) ** (1.0 / (1.0 - q)) / Tmean

            # --- α-diversity ---------------------------------------------------
            if q == 1:
                mask = sub > 0
                term = sub.where(mask, 0.0) * np.log(sub.where(mask, 1.0))
                term = term.mul(branchL / Tmean, axis=0)
                term = term.sum().sum()
                alpha_div = math.exp(-term / 2.0)
            elif q == 0:
                pos_counts = (sub > 0).sum(axis=1).astype(float)
                alpha_div = (pos_counts.mul(branchL)).sum() / (2.0 * Tmean)
            else:
                aq = sub.clip(lower=0) / (2.0 * Tmean)
                aq = aq.pow(q)
                term = aq.sum(axis=1).mul(branchL)
                alpha_div = (term.sum() ** (1.0 / (1.0 - q))) / (2.0 * Tmean)

            # β-diversity
            beta_val = gamma_div / alpha_div
            out.loc[s1, s2] = beta_val
            out.loc[s2, s1] = beta_val

    # --- Convert β to dissimilarity if requested ------------------------------
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
        Diversity order.
    dis : bool, default=True
        If True, convert β to a dissimilarity using `beta2dist`.
    viewpoint : {'local', 'regional'}, default='regional'
        Viewpoint for converting β to dissimilarity.
    use_values_in_tab : bool, default=False
        If False, convert abundances to relative abundances.
        If True, assume `tab` already contains relative abundances.
    use_tqdm : bool, default=True
        Use `tqdm` for progress bars.

    Returns
    -------
    pandas.DataFrame
        Pairwise functional dissimilarity matrix (if `dis=True`) or
        squared functional beta (β²) matrix (if `dis=False`).

    Notes
    -----
    - Only works for ≥ 2 samples.
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
    asvs = ra.index.tolist()
    distmat = distmat.loc[asvs, asvs]

    smplist = list(ra.columns)
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

    # Square β to get FD-like measure (your original behavior)
    outFD = outD.pow(2)

    # Convert β to dissimilarity if requested
    if dis:
        return beta2dist(beta=outFD, q=q, N=2, div_type="func", viewpoint=viewpoint)

    return outFD

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

    Bray–Curtis dissimilarity between two samples A and B is:

        BC(A, B) = 1 − Σ_i min(p_iA, p_iB)

    where p_iA and p_iB are relative abundances of feature i in samples A and B.

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

    Notes
    -----
    - Requires at least two samples.
    - Zero-sum samples are not allowed unless `use_values_in_tab=True`.
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

    Jaccard dissimilarity between two samples A and B is:

        J(A, B) = 1 − ( |A ∩ B| / |A ∪ B| )

    where presence/absence is determined by whether abundance > 0.

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

    Notes
    -----
    - Requires at least two samples.
    - Abundances are converted to binary presence/absence.
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
) -> pd.DataFrame:
    """
    Compute naive (taxonomic) multi‑sample beta diversity for groups of samples.

    This implements the multi‑sample Hill‑number beta framework:

        β_q = γ_q / ( α_q / N )

    where:
        - γ_q is the Hill number of the pooled community
        - α_q is the mean within‑sample Hill number
        - N is the number of samples in the group

    Parameters
    ----------
    obj : MicrobiomeData-like | dict
        Must contain:
            - 'meta' : pandas.DataFrame with sample metadata
            - 'tab'  : pandas.DataFrame with feature counts (features × samples)
    by : str or None, default=None
        Column in metadata defining sample groups.
        If None, all samples are treated as one group.
    q : float, default=1
        Diversity order.

    Returns
    -------
    pandas.DataFrame
        Index = categories in `var` (or 'all' if var=None)
        Columns:
            - N             : number of samples in group
            - beta          : multi‑sample beta diversity
            - local_dis     : local‑viewpoint dissimilarity
            - regional_dis  : regional‑viewpoint dissimilarity

    Notes
    -----
    - Groups with <2 samples return NaN.
    """

    # Validate input
    meta = get_df(obj, "meta")
    tab = get_df(obj, "tab")

    if tab.shape[1] < 2:
        raise ValueError("At least two samples are required.")

    # Build dictionary of subtables by category
    if by is None:
        categories = ["all"]
        tabdict = {"all": tab.copy()}
    else:
        if by not in meta.columns:
            raise ValueError(f"Column '{by}' not found in metadata.")

        categories = meta[by].unique().tolist()
        tabdict = {
            cat: get_df(subset_samples(obj, by=by, values=[cat]), "tab")
            for cat in categories
        }

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

        # Convert to relative abundances if needed
        col_sums = subtab.sum()
        if (col_sums == 0).any():
            raise ValueError(f"Group '{cat}' contains a zero‑sum sample.")
        ra = subtab.div(col_sums)

        # Build alpha/gamma table for naive_alpha()
        # gamma row = pooled abundances
        gamma_row = ra.sum(axis=1).to_numpy()

        # alpha rows = each sample's abundances
        alpha_rows = ra.to_numpy().T.reshape(-1)

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
) -> pd.DataFrame:
    """
    Compute phylogenetic multi‑sample beta diversity for groups of samples.

    Implements the multi‑sample phylogenetic Hill‑number beta framework
    described in Chao et al. (2014), where branch lengths are weighted by
    the relative abundances of all ASVs descending from each branch.

    For each group of samples:

        β_q = γ_q / ( α_q / N )

    where:
        - γ_q is the phylogenetic Hill number of the pooled community
        - α_q is the mean within‑sample phylogenetic Hill number
        - N is the number of samples in the group

    Parameters
    ----------
    obj : MicrobiomeData-like | dict
        Must contain:
            - 'meta' : pandas.DataFrame with sample metadata
            - 'tab'  : pandas.DataFrame with ASV counts (ASVs × samples)
            - 'tree' : pandas.DataFrame with:
                * 'leaves'   : list of features under each branch
                * 'branchL'  : branch length
    by : str or None, default=None
        Metadata column defining sample groups.
        If None, all samples are treated as one group.
    q : float, default=1
        Diversity order.

    Returns
    -------
    pandas.DataFrame
        Index = categories in `by` (or 'all' if by=None)
        Columns:
            - N             : number of samples in group
            - beta          : multi‑sample phylogenetic beta diversity
            - local_dis     : local‑viewpoint dissimilarity
            - regional_dis  : regional‑viewpoint dissimilarity

    Notes
    -----
    - Only works for ≥ 2 samples per group.
    """

    # Validate input

    tab = get_df(obj, "tab")
    meta = get_df(obj, "meta")
    tree = get_df(obj, "tree")

    if tab.shape[1] < 2:
        raise ValueError("At least two samples are required.")

    if "leaves" not in tree.columns or "branchL" not in tree.columns:
        raise ValueError("`tree` must contain columns 'leaves' and 'branchL'.")

    #Subset tree to features in tab
    tree = subset_tree_df(tree, tab.index.tolist())

    # Build dictionary of subtables by category
    if by is None:
        categories = ["all"]
        tabdict = {"all": tab.copy()}
    else:
        if by not in meta.columns:
            raise ValueError(f"Column '{by}' not found in metadata.")

        categories = meta[by].unique().tolist()
        tabdict = {
            cat: get_df(subset_samples(obj, by=by, values=[cat]), "tab")
            for cat in categories
        }

    # Output container
    out = pd.DataFrame(
        np.nan,
        index=categories,
        columns=["N", "beta", "local_dis", "regional_dis"]
    )

    # Compute multi‑sample phylogenetic beta for each category
    for cat in categories:
        subtab = tabdict[cat]

        # Need at least 2 samples
        if subtab.shape[1] < 2:
            continue

        N = subtab.shape[1]
        out.loc[cat, "N"] = N

        # Relative abundances
        col_sums = subtab.sum()
        if (col_sums == 0).any():
            raise ValueError(f"Group '{cat}' contains a zero‑sum sample.")
        ra = subtab.div(col_sums)

        # Build branch × sample abundance matrix
        tree2 = ra_to_branches(ra, tree)

        # --- Compute Tavg = Σ_b L_b * mean(p_b) -------------------------------------
        # Align branch lengths to tree2 (branch × sample) and ensure numeric
        branchL = pd.to_numeric(tree["branchL"], errors="raise").reindex(tree2.index)
        if branchL.isna().any():
            missing = branchL.index[branchL.isna()].tolist()
            raise ValueError(f"'branchL' missing for branches: {missing}")
        
        mean_ra = tree2.mean(axis=1)                  # γ_b: mean of per-branch RA across N samples
        Tavg = float(mean_ra.mul(branchL).sum())      # Σ L_b * γ_b
        
        # --- γ-diversity -------------------------------------------------------------
        g = mean_ra
        
        if q == 1:
            # Shannon limit: use masked logs (zeros contribute 0)
            g_safe = g.where(g > 0)                                  # NaN where zero
            term = (g_safe * np.log(g_safe)).mul(branchL)            # L_b * γ_b ln γ_b
            gamma_div = math.exp(-(term.sum() / Tavg))
        
        elif q == 0:
            # Presence/absence: indicator of γ_b > 0 (avoid NaN**0 → 1)
            I = (g > 0).astype(float)
            term = I.mul(branchL)                                    # L_b * 1{γ_b>0}
            gamma_div = term.sum() / Tavg
        
        else:
            # General Hill number
            g_norm = g / Tavg
            term = (g_norm.clip(lower=0) ** q).mul(branchL)          # L_b * γ_b^q / Tavg^q
            gamma_div = (term.sum()) ** (1.0 / (1.0 - q)) / Tavg
        
        # --- α-diversity -------------------------------------------------------------
        # N must be the number of samples in the group (K communities)
        # Ensure N is defined earlier as: N = tree2.shape[1] or len(group_samples)
        tree2_scaled = tree2 / (N * Tavg)
        a = tree2_scaled  # branch × N matrix
        
        if q == 1:
            # Shannon limit: masked logs per branch × community, weighted by branchL
            a_safe = a.where(a > 0)
            term = (a_safe * np.log(a_safe)).mul(branchL, axis=0)    # L_b * Σ_k a_bk ln a_bk
            H = -term.sum().sum() - math.log(N * Tavg)
            alpha_div = math.exp(H)
        
        elif q == 0:
            # Presence/absence across communities
            I = (a > 0).astype(float)
            term = I.sum(axis=1).mul(branchL)                        # L_b * Σ_k 1{a_bk>0}
            alpha_div = term.sum() / (N * Tavg)
        
        else:
            # General Hill number
            aq = (a.clip(lower=0) ** q)
            term = aq.sum(axis=1).mul(branchL)                       # Σ_k a_bk^q * L_b
            alpha_div = (term.sum()) ** (1.0 / (1.0 - q)) / (N * Tavg)

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
) -> pd.DataFrame:
    """
    Compute functional multi‑sample beta diversity for groups of samples.

    Implements the multi‑sample functional Hill‑number beta framework
    described in Chiu et al. (2014), where functional diversity is derived
    from pairwise trait distances and species abundances.

    For each group of samples:

        β_q = D_gamma / D_alpha

    where:
        - D_gamma is the functional Hill number of the pooled community
        - D_alpha is the mean functional Hill number across all sample pairs
        - N is the number of samples in the group
        - NxN = N² (number of ordered sample pairs)

    Parameters
    ----------
    obj : MicrobiomeData-like | dict
        Must contain:
            - 'meta' : pandas.DataFrame with sample metadata
            - 'tab'  : pandas.DataFrame (features × samples)
    distmat : pandas.DataFrame
        Functional distance matrix (features × features).
    by : str or None, default=None
        Metadata column defining sample groups.
        If None, all samples are treated as one group.
    q : float, default=1
        Diversity order.

    Returns
    -------
    pandas.DataFrame
        Index = categories in `by` (or 'all' if by=None)
        Columns:
            - NxN          : N² (number of ordered sample pairs)
            - beta         : functional multi‑sample beta diversity
            - local_dis    : local‑viewpoint dissimilarity
            - regional_dis : regional‑viewpoint dissimilarity

    Notes
    -----
    - Only works for ≥ 2 samples per group.
    """

    # Validate input
    tab = get_df(obj, "tab")
    meta = get_df(obj, "meta")
    if tab.shape[1] < 2:
        raise ValueError("At least two samples are required.")

    # Make sure tab and distmat have the same index
    in_common = list(set(distmat.index).intersection(tab.index))
    if len(in_common) < len(tab):
        raise ValueError("Features in tab are missing in distmat.")
    tab = tab.loc[in_common]
    distmat = distmat.loc[in_common, in_common].copy()

    # Build dictionary of subtables by category
    if by is None:
        categories = ["all"]
        tabdict = {"all": tab.copy()}
    else:
        if by not in meta.columns:
            raise ValueError(f"Column '{by}' not found in metadata.")

        categories = meta[by].unique().tolist()
        tabdict = {
            cat: get_df(subset_samples(obj, by=by, values=[cat], keep_absent=True), "tab")
            for cat in categories
        }

    # Output container
    out = pd.DataFrame(
        np.nan,
        index=categories,
        columns=["NxN", "beta", "local_dis", "regional_dis"]
    )

    # Compute multi‑sample functional beta for each category
    for cat in categories:
        subtab = tabdict[cat]

        # Need at least 2 samples
        if subtab.shape[1] < 2:
            continue

        N = subtab.shape[1]
        out.loc[cat, "NxN"] = N * N

        # Relative abundances
        col_sums = subtab.sum()
        if (col_sums == 0).any():
            raise ValueError(f"Group '{cat}' contains a zero‑sum sample.")
        ra = subtab.div(col_sums)

        smplist = ra.columns.tolist()

        # Compute pooled mean abundances
        ra_mean = ra.mean(axis=1)

        # Rao's Q for pooled community
        Q_pooled = rao(ra_mean, distmat)
        dqmat = distmat * (1.0 / Q_pooled)

        # γ-diversity (pooled)
        p = ra_mean.to_numpy()
        outer = np.outer(p, p)
        print(outer)

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
        print('Dq', Dg)

        # α-diversity (mean over all ordered sample pairs)
        asum = 0.0

        for s1 in smplist:
            p1 = ra[s1].to_numpy()
            for s2 in smplist:
                p2 = ra[s2].to_numpy()

                outer12 = np.outer(p1, p2) / (N * N)

                if q == 1:
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
    
    Supports:
        - naive (taxonomic) evenness
        - phylogenetic evenness
        - functional evenness
    
    Supported evenness indices:
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
        tree (pandas.DataFrame, required if divType='phyl')
    distmat : pandas.DataFrame, optional
        Required if divType='func'. Functional distance matrix.
    q : float, default=1
        Diversity order.
    div_type : {'naive', 'phyl', 'func'}
        Type of diversity measure used to compute D.
    index : {'CR1','CR2','CR3','CR4','CR5','local','regional','pielou'}
        Evenness index to compute.
        - 'pielou' computes Pielou’s J = ln(D₁) / ln(S), valid only for q = 1.
    perspective : {'samples','taxa'}
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
    - Pielou’s index is included for convenience and corresponds to:
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
            D_series = phyl_alpha(tab, tree, q=q, index="D",
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
            S_series = tabT.count()
            D_series = naive_alpha(tabT, q=q, use_values_in_tab=use_values_in_tab)

        elif div_type == "phyl":
            tree = get_df(obj, "tree")
            if not isinstance(tree, pd.DataFrame):
                raise ValueError("tree must be provided for divType='phyl'.")

            # Relative abundances
            ra = tab.div(tab.sum()) if not use_values_in_tab else tab.astype(float)

            #Subset tree to features in tab
            tree = subset_tree_df(tree, ra.index.tolist())

            # Build branch × sample matrix
            tree2 = ra_to_branches(ra, tree)

            # Normalize across samples
            tree2 = tree2.T
            tree2 = tree2.div(tree2.sum())

            S_series = tree2.count()
            D_series = naive_alpha(tree2, q=q, use_values_in_tab=True)

        else:
            raise ValueError("divType must be 'naive' or 'phyl' when perspective='taxa'.")

    # Compute evenness index
    if index in ("CR1", "CR2", "regional", "local"):
        if q == 1:
            df = pd.DataFrame({"D": D_series, "S": S_series}).astype(float)
            mask = (df["S"] > 0) & (df["D"] > 0)
            logD = np.log(df.loc[mask, "D"])
            logS = np.log(df.loc[mask, "S"])
            measure = logD / logS
        else:
            Dp = D_series.pow(power)
            Sp = S_series.pow(power)
            measure = (1 - Dp) / (1 - Sp)
    
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
    use_values_in_tab: bool = False
) -> pd.DataFrame:
    """
    Compute the contribution of individual taxa (or phylogenetic nodes)
    to the overall dissimilarity between multiple samples, following
    Chao & Ricotta (2019, Ecology 100:e02852).

    Supports:
        - naive (taxonomic) dissimilarity
        - phylogenetic dissimilarity

    Parameters
    ----------
    obj : DataFrame | MicrobiomeData-like | dict
        Must contain:
            - 'tab' : abundance table (features × samples)
            - 'meta' : metadata table (optional if by=None)
            - 'tree' : phylogenetic tree (required if divType='phyl')
    by : str or None, default=None
        Metadata column defining sample groups.
        If None, all samples are treated as one group.
    q : float, default=1
        Diversity order.
    div_type : {'naive','phyl'}, default='naive'
        Type of dissimilarity measure.
    index : {'local','regional','CR1','CR2'}, default='regional'
        Evenness/dissimilarity index.
    use_values_in_tab : bool, default=False
        If False, convert abundances to relative abundances.

    Returns
    -------
    pandas.DataFrame
        Rows:
            - 'dis' : total dissimilarity
            - 'N'   : number of samples in group
            - one row per taxon (naive) or per node (phylogenetic)
        Columns:
            - one column per category in `by`
    """

    # Validate input
    tab = get_df(obj, "tab")

    if tab.shape[1] < 2:
        raise ValueError("At least two samples are required.")

    if div_type not in ("naive", "phyl"):
        raise ValueError("divType must be 'naive' or 'phyl'.")

    if index in ("CR1", "regional"):
        idx = "regional"
    elif index in ("CR2", "local"):
        idx = "local"
    else:
        raise ValueError("index must be 'local', 'regional', 'CR1', or 'CR2'.")

    # Build dictionary of subtables by category
    if by is None:
        categories = ["all"]
        tabdict = {"all": tab.copy()}
    else:
        meta = get_df(obj, "meta")
        if by not in meta.columns:
            raise ValueError(f"Column '{by}' not found in metadata.")

        categories = meta[by].unique().tolist()
        tabdict = {
            cat: get_df(subset_samples(obj, by=by, values=[cat]), "tab")
            for cat in categories
        }

    # Prepare output table
    if div_type == "naive":
        feature_index = ["dis", "N"] + tab.index.tolist()
    else:
        tree = get_df(obj, "tree")
        tree = subset_tree_df(tree, tab.index.tolist())
        feature_index = ["dis", "N"] + tree.index.tolist()

    out = pd.DataFrame(np.nan, index=feature_index, columns=categories)

    # Main loop over categories
    for cat in categories:
        subtab = tabdict[cat]
        N = subtab.shape[1]
        out.loc["N", cat] = N

        if N < 2:
            continue

        # Relative abundances
        if use_values_in_tab:
            ra = subtab.astype(float)
        else:
            col_sums = subtab.sum()
            if (col_sums == 0).any():
                raise ValueError(f"Group '{cat}' contains a zero-sum sample.")
            ra = subtab.div(col_sums)

        # NAIVE VERSION
        if div_type == "naive":
            # Compute weights w_i
            if idx == "regional":  # CR1
                w = subtab.sum(axis=1).pow(q)
                w = w / w.sum()
            else:  # local = CR2
                tab_q = subtab.copy()
                mask = tab_q > 0
                tab_q[mask] = tab_q[mask].pow(q)
                w = tab_q.sum(axis=1) / tab_q.sum().sum()

            # Evenness per taxon
            ev = evenness(
                subtab,
                q=q,
                div_type="naive",
                index=idx,
                perspective="taxa",
                use_values_in_tab=use_values_in_tab,
            )

            # Contribution
            contrib = w * (1 - ev)

            out.loc["dis", cat] = contrib.sum()
            out.loc[contrib.index, cat] = 100 * contrib / contrib.sum()
            out[cat] = out[cat].fillna(0)

        # PHYLOGENETIC VERSION
        elif div_type == "phyl":

            # Build branch × sample matrix
            tree2 = ra_to_branches(ra, tree)
    
            # Evenness per node
            ev = evenness(
                {"tab": subtab, "tree": tree},
                q=q,
                div_type="phyl",
                index=idx,
                perspective="taxa",
                use_values_in_tab=use_values_in_tab,
            )

            # Compute weights
            if idx == "regional":  # CR1
                zi = tree2.sum(axis=1)
                mask = zi > 0
                zi_q = np.zeros_like(zi)
                zi_q[mask] = zi[mask] ** q
                Lz = tree["branchL"] * zi_q
                w = zi_q / Lz.sum()
            else:  # local = CR2
                tree2_q = tree2.copy()
                mask = tree2_q > 0
                tree2_q[mask] = tree2_q[mask].pow(q)
                zv = tree2_q.sum(axis=1)
                Lz = tree["branchL"] * zv
                w = zv / Lz.sum()
    
            # Contribution
            contrib = tree["branchL"] * w * (1 - ev)
    
            out.loc["dis", cat] = contrib[contrib.notna()].sum()
            out.loc[contrib.index, cat] = 100 * contrib / contrib.sum()
            out.loc[tree.index, 'nodes'] = tree['nodes']
    
    return out


# -----------------------------------------------------------------------------
# Partition beta diversity
# -----------------------------------------------------------------------------

# Structure-only dissimilarity on rank-sorted abundances (identity removed)
def _get_structural_dis(
    ra: pd.DataFrame,
    *,
    q: float = 1.0,
) -> pd.DataFrame:
    """
    Compute structure-only Hill-number dissimilarity by:
      1) building relative abundances from obj['tab'] (unless already relative)
      2) sorting each sample's abundance vector descending (erases identity)
      3) computing the same Hill-number dissimilarity on the sorted table

    Parameters
    ----------
    ra : DataFrame
        Relative abundance table.
    q : float
        Diversity order.

    Returns
    -------
    dis_sorted : DataFrame (samples × samples)
        Structure-only dissimilarity (identity erased).
    """
    # Vectorized column-wise sort (ascending), then flip to descending
    a = np.sort(ra.to_numpy(dtype=float), axis=0)[::-1, :]
    ra_sorted = pd.DataFrame(a, index=range(ra.shape[0]), columns=ra.columns)

    # Compute structure-only dissimilarity using the same taxonomic Hill metric
    dis_sorted = naive_beta(ra_sorted, q=q, dis=True, viewpoint="local", use_values_in_tab=True)

    # Sanity: ensure sample order matches original
    if not (dis_sorted.index.equals(ra.columns) and dis_sorted.columns.equals(ra.columns)):
        dis_sorted = dis_sorted.reindex(index=ra.columns, columns=ra.columns)

    return dis_sorted

# A q-weighted mass-based approach get weights
def _get_unique_fractions(
    ra: pd.DataFrame,
    *,
    q: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute attribution weights u_A, u_B for two samples based on
    community-specific 'unique' fractions at Hill order q, using a
    min-based, abundance-compatible directional overlap.

    For two columns A,B in `ra` (features × 2), define
      C^{A|B}_{q,min} = sum_i min(p_iA, p_iB)^q / sum_i p_iA^q
      C^{B|A}_{q,min} = sum_i min(p_iA, p_iB)^q / sum_i p_iB^q

    Then the local unique fractions are
      U_A^loc = 1 - C^{A|B}_{q,min}
      U_B^loc = 1 - C^{B|A}_{q,min}

    and weights are
      λ_A = U_A^loc / (U_A^loc + U_B^loc),  λ_B = 1 - λ_A
    with a 0.5/0.5 tie-break when both uniques are 0.

    Parameters
    ----------
    ra : DataFrame
        Relative abundances, shape (features × 2). Columns are the two samples.
        Columns should sum to 1 (or will be normalized if eps>0 and sums>0).
    q : float, default=1.0
        Hill order.

    Returns
    -------
    (w1, w2) : Tuple[float, float]
        Weights for the first and second sample (λ_A, λ_B), in [0,1] and summing to 1.
    """

    def _get_qvector(vec, q):
        vec = np.asarray(vec, dtype=float)
        if q == 0:
            return (vec > 0).astype(float)
        if q == 1:
            return vec
        vec = np.clip(vec, 0.0, None)
        return vec ** q

    if not isinstance(ra, pd.DataFrame) or ra.shape[1] != 2:
        raise ValueError("`ra` must be a DataFrame with exactly two columns (features × 2).")

    # Extract as float arrays
    A = ra.iloc[:, 0].to_numpy(dtype=float)
    B = ra.iloc[:, 1].to_numpy(dtype=float)

    if q == 1.0:
        A_q = A
        B_q = B
    else:
        # Use power; if q<0 or weird, user should guard upstream
        A_q = _get_qvector(A, q)
        B_q = _get_qvector(B, q)

    # Shared term
    M_q = np.minimum(A, B)
    M_q = _get_qvector(M_q, q)

    sum_Aq = A_q.sum()
    sum_Bq = B_q.sum()
    sum_Mq = M_q.sum()

    # Degenerate cases: empty or zero vectors
    if sum_Aq <= 0 and sum_Bq <= 0:
        return 0.5, 0.5
    if sum_Aq <= 0:
        return 0.0, 1.0
    if sum_Bq <= 0:
        return 1.0, 0.0

    # Directional overlaps in [0,1]
    C_A_given_B = min(1.0, max(0.0, float(sum_Mq / sum_Aq)))
    C_B_given_A = min(1.0, max(0.0, float(sum_Mq / sum_Bq)))

    # Local unique fractions (bounded)
    U_A_loc = max(0.0, 1.0 - C_A_given_B)
    U_B_loc = max(0.0, 1.0 - C_B_given_A)

    tot = U_A_loc + U_B_loc
    if tot <= 0.0:
        # Identical communities under the chosen q (or both unique=0)
        return 0.5, 0.5

    return U_A_loc, U_B_loc

# Naive decomposition
def partition_naive_beta(
    tab: Union[pd.DataFrame, Dict[str, Any], Any],
    *,
    q: float = 1.0,
    use_values_in_tab: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    POD-style split of Hill-number dissimilarity (replacement/diversity-difference)
    compared to regional dissimilarity,
    identity–structure divergence (ISD_q) compared to local dissimilarity.

    Returns
    -------
    dict of DataFrames:
      - 'd_regional'  : observed regional dissimilarity
      - 'd_replacement' : Podani-Schmera replacement component
      - 'd_diversity' : diversity-difference component
      - 'd_local' : observed local dissimilarity
      - 'd_isd' : identity-structure divergence
      - 'd_sorted' : dissimilarity caused by structure, calculated on sorted vectors
    """

    tab = get_df(tab, "tab").astype(float)

    # Relative abundances
    if not use_values_in_tab:
        col_sums = tab.sum(axis=0)
        if (col_sums == 0).any():
            bad = col_sums.index[col_sums == 0].tolist()
            raise ValueError(f"Zero-total samples: {bad}")
        ra = tab.div(col_sums, axis=1)
    else:
        ra = tab

    if ra.shape[1] < 2:
        raise ValueError("`tab` must contain ≥ 2 samples (columns).")

    # Observed alpha (for weights) and observed dissimilarity
    alpha = naive_alpha(ra, q=q, use_values_in_tab=True)
    disL = naive_beta(ra, q=q, dis=True, viewpoint="local", use_values_in_tab=True)
    disR = naive_beta(ra, q=q, dis=True, viewpoint="regional", use_values_in_tab=True)

    # Structure-only dissimilarity on sorted abundances
    ds = _get_structural_dis(ra, q=q)
    d_isd = disL - ds

    # Replacement / Diversity-difference split on the observed dissimilarity
    smplist = alpha.index.tolist()
    dr = pd.DataFrame(0.0, index=smplist, columns=smplist)
    dd = pd.DataFrame(0.0, index=smplist, columns=smplist)

    for i in range(len(smplist) - 1):
        s1 = smplist[i]
        for j in range(i + 1, len(smplist)):
            s2 = smplist[j]

            a1, a2 = alpha[s1], alpha[s2]
            if a1 + a2 == 0.0:
                raise ValueError("Alpha diversity is zero for both samples.")
            
            u1, u2 = _get_unique_fractions(ra[[s1, s2]], q=q)
            ud_a1 = u1 * a1
            ud_a2 = u2 * a2

            w1 = ud_a1 / (ud_a1 + ud_a2)
            w2 = 1.0 - w1

            D = float(disR.at[s1, s2])  # observed dissimilarity
            repl = 2.0 * min(w1, w2) * D
            rich = abs(w1 - w2) * D

            dr.loc[s1, s2] = dr.loc[s2, s1] = repl
            dd.loc[s1, s2] = dd.loc[s2, s1] = rich

    return {"regional_d": disR, 
            "regional_replacement": dr, 
            "regional_diversity_difference": dd, 
            "local_d": disL, 
            "local_isd": d_isd, 
            "local_structural_d": ds}

# Phylogenetic decomposition
def partition_phyl_beta(
    obj: Union[Dict[str, Any], Any],
    *,
    q: float = 1.0,
    viewpoint: str = "regional",
    use_values_in_tab: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Phylogenetic analogue:
      - 'd'  : observed phylogenetic dissimilarity  (on the real tree)
      - 'dr' : replacement component
      - 'dd' : diversity-difference component
      - 'ds' : structure-only dissimilarity (sorted abundances; collapsed tree)
      - 'di' : PD-ISD_q = d - ds
    """
    # Observed PD alpha (for weights) and PD dissimilarity
    alpha = phyl_alpha(obj, q=q, use_values_in_tab=use_values_in_tab)                                # Series indexed by sample
    dis   = phyl_beta(obj, q=q, dis=True, viewpoint=viewpoint, use_values_in_tab=use_values_in_tab)  # DataFrame

    # Structure-only (collapsed phylogeny): ordinary Hill dissimilarity on sorted abundances
    ds = _get_structural_dis(obj, q=q, viewpoint=viewpoint, use_values_in_tab=use_values_in_tab)

    # PD-ISD_q (can be negative)
    di = dis - ds

    # Replacement / Diversity-difference split on the observed PD dissimilarity
    smplist = alpha.index.tolist()     # <- FIX: use index, not columns
    dr = pd.DataFrame(0.0, index=smplist, columns=smplist)
    dd = pd.DataFrame(0.0, index=smplist, columns=smplist)

    for i in range(len(smplist) - 1):
        s1 = smplist[i]
        for j in range(i + 1, len(smplist)):
            s2 = smplist[j]

            a1, a2 = alpha[s1], alpha[s2]
            if a1 + a2 == 0.0:
                raise ValueError("Alpha diversity is zero for both samples.")

            w1 = a1 / (a1 + a2)
            w2 = 1.0 - w1

            D = float(dis.at[s1, s2])  # observed PD dissimilarity
            repl = 2.0 * min(w1, w2) * D
            rich = abs(w1 - w2) * D

            dr.loc[s1, s2] = dr.loc[s2, s1] = repl
            dd.loc[s1, s2] = dd.loc[s2, s1] = rich

    return {"d": dis, "dr": dr, "dd": dd, "ds": ds, "di": di}

# Functional decomposition
def partition_func_beta(
    tab: Union[pd.DataFrame, Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1.0,
    viewpoint: str = "regional",
    use_values_in_tab: bool = False,
) -> Dict[str, pd.DataFrame]:

    """
    Functional analogue:
      - 'd'  : observed functional dissimilarity  (based on distmat)
      - 'dr' : replacement component
      - 'dd' : diversity-difference component
      - 'ds' : structure-only dissimilarity (sorted abundances)
      - 'di' : functional ISD_q = d - ds
    """

    # Observed PD alpha (for weights) and PD dissimilarity
    alpha = func_alpha(tab, distmat=distmat, q=q, use_values_in_tab=use_values_in_tab)                                # Series indexed by sample
    dis   = func_beta(tab, distmat=distmat, q=q, dis=True, viewpoint=viewpoint, use_values_in_tab=use_values_in_tab)  # DataFrame

    # Structure-only (collapsed phylogeny): ordinary Hill dissimilarity on sorted abundances
    ds = _get_structural_dis(tab, q=q, viewpoint=viewpoint, use_values_in_tab=use_values_in_tab)

    # PD-ISD_q (can be negative)
    di = dis - ds

    # Replacement / Diversity-difference split on the observed PD dissimilarity
    smplist = alpha.index.tolist()     # <- FIX: use index, not columns
    dr = pd.DataFrame(0.0, index=smplist, columns=smplist)
    dd = pd.DataFrame(0.0, index=smplist, columns=smplist)

    for i in range(len(smplist) - 1):
        s1 = smplist[i]
        for j in range(i + 1, len(smplist)):
            s2 = smplist[j]

            a1, a2 = alpha[s1], alpha[s2]
            if a1 + a2 == 0.0:
                raise ValueError("Alpha diversity is zero for both samples.")

            w1 = a1 / (a1 + a2)
            w2 = 1.0 - w1

            D = float(dis.at[s1, s2])  # observed PD dissimilarity
            repl = 2.0 * min(w1, w2) * D
            rich = abs(w1 - w2) * D

            dr.loc[s1, s2] = dr.loc[s2, s1] = repl
            dd.loc[s1, s2] = dd.loc[s2, s1] = rich

    return {"d": dis, "dr": dr, "dd": dd, "ds": ds, "di": di}

def beta_mpdq(
    obj: Union[Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1.0,
) -> pd.DataFrame:
    """
    Computes beta-MPD_q for all sample pairs.

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
    Computes beta-MNTD_q for all sample pairs.

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
    pandas.DataFrame (S x S)
    """
    from ..model import beta_ntiq
    return beta_ntiq(obj, distmat, q=q, iterations=0, include_conspecifics=include_conspecifics)
