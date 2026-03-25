"""
Statistical calculations on the meta data.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
from itertools import combinations, product
import warnings
from typing import List, Optional, Dict, Union, Any, Tuple
from ..utils import get_df

__all__ = ["corr", "bootstrap_sample_matrix"]

# ------Helpers for corr ---------------------------
def _is_categorical(series: pd.Series, max_unique: int = 20) -> bool:
    s = series.dropna()
    if pd.api.types.is_categorical_dtype(series) or series.dtype == "object" or series.dtype == "bool":
        return True
    if pd.api.types.is_integer_dtype(series) and s.nunique() <= max_unique:
        return True
    return False


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    tab = pd.crosstab(x, y)
    n = tab.values.sum()
    if n == 0 or tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan

    expected = np.outer(tab.sum(1), tab.sum(0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_mat = ((tab - expected) ** 2) / expected
        chi2 = np.nan_to_num(chi2_mat.to_numpy(copy=False)).sum()

    phi2 = chi2 / n
    r, k = tab.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / max(1, (n - 1)))
    rcorr = r - (r - 1) ** 2 / max(1, (n - 1))
    kcorr = k - (k - 1) ** 2 / max(1, (n - 1))
    denom = max(1e-12, min(kcorr - 1, rcorr - 1))
    return float(np.sqrt(phi2corr / denom))


def _point_biserial(x: pd.Series, y: pd.Series) -> float:
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    levels = s["x"].unique()
    if len(levels) != 2:
        return np.nan
    y1 = s.loc[s["x"] == levels[0], "y"]
    y2 = s.loc[s["x"] == levels[1], "y"]
    n1, n2 = len(y1), len(y2)
    if n1 < 2 or n2 < 2:
        return np.nan
    sd = s["y"].std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return float((y1.mean() - y2.mean()) / sd * np.sqrt((n1 * n2) / (n1 + n2) ** 2))

def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    s = pd.DataFrame({"g": categories, "y": values}).dropna()
    if len(s) < 2:
        return np.nan
    groups = s.groupby("g")["y"]
    if groups.ngroups < 2:
        return np.nan
    y_mean = s["y"].mean()
    ss_between = np.sum(groups.size() * (groups.mean() - y_mean) ** 2)
    ss_total = np.sum((s["y"] - y_mean) ** 2)
    if ss_total <= 0:
        return np.nan
    eta_sq = ss_between / ss_total
    return float(np.sqrt(eta_sq))

def _stat_numeric_numeric(x: pd.Series, y: pd.Series, *, method: str = "spearman") -> float:
    xv = x.to_numpy(copy=False)
    yv = y.to_numpy(copy=False)
    if len(xv) < 3 or len(yv) < 3:
        return np.nan
    if np.nanstd(xv, ddof=1) == 0 or np.nanstd(yv, ddof=1) == 0:
        return np.nan
    return x.corr(y, method=method)

def _stat_cat_cat(x: pd.Series, y: pd.Series) -> float:
    if x.dropna().nunique() < 2 or y.dropna().nunique() < 2:
        return np.nan
    return _cramers_v(x, y)

def _stat_cat_num(x: pd.Series, y: pd.Series) -> float:
    yv = y.to_numpy(copy=False)
    if len(yv) < 3 or np.nanstd(yv, ddof=1) == 0:
        return np.nan
    k = x.dropna().nunique()
    if k < 2:
        return np.nan
    if k == 2:
        return _point_biserial(x, y)
    return _correlation_ratio(x, y)

def _perm_test_stat(
    x: pd.Series,
    y: pd.Series,
    stat_func,
    *,
    n_perm: int = 999,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Generic permutation test: returns (stat, two-sided p)."""
    stat = stat_func(x, y)
    if not np.isfinite(stat):
        return np.nan, np.nan

    y_arr = y.to_numpy(copy=False)
    x_arr = x.to_numpy(copy=False)
    n = len(y_arr)
    perm_stats = np.empty(n_perm, dtype=float)

    for i in range(n_perm):
        idx = rng.permutation(n)
        perm_stats[i] = stat_func(pd.Series(x_arr, copy=False), pd.Series(y_arr[idx], copy=False))

    # +1 correction to avoid 0/1 extremes on finite permutations
    b = np.count_nonzero(np.abs(perm_stats) >= np.abs(stat))
    p = (b + 1) / (n_perm + 1)
    return float(stat), float(p)


def _fdr_bh_offdiag(P: pd.DataFrame) -> pd.DataFrame:
    """
    Benjamini–Hochberg FDR on off-diagonal entries (i<j). Returns a symmetric
    matrix of adjusted p-values with NaN on the diagonal.
    """
    m = P.shape[0]
    padj = pd.DataFrame(np.full_like(P, np.nan, dtype=float), index=P.index, columns=P.columns)

    # Collect off-diagonal upper-triangular positions and their p-values
    idx_pairs = [(i, j) for i in range(m) for j in range(i+1, m)]
    pvals = np.array([
        float(P.iat[i, j]) if pd.notna(P.iat[i, j]) else np.nan
        for (i, j) in idx_pairs
    ], dtype=float)

    # Finite mask and number of tests
    finite_mask = np.isfinite(pvals)
    k = int(finite_mask.sum())
    if k == 0:
        # nothing to adjust
        return padj

    # Order finite p-values
    finite_idx = np.nonzero(finite_mask)[0]
    order = np.argsort(pvals[finite_mask])
    ranked = pvals[finite_mask][order]

    # BH adjustment (step-up) + monotone step-down
    q = ranked * k / np.arange(1, k + 1, dtype=float)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    # Write adjusted values back to the linear array aligned with idx_pairs
    padj_lin = np.full_like(pvals, np.nan, dtype=float)
    padj_lin[finite_idx[order]] = q

    # Assign both upper and lower triangles explicitly
    for val, (i, j) in zip(padj_lin, idx_pairs):
        if np.isfinite(val):
            padj.iat[i, j] = val
            padj.iat[j, i] = val

    # Diagonal stays NaN
    np.fill_diagonal(padj.values, np.nan)
    return padj

# -----------------------------------------------------------------------------
# Correlations between meta data variables
# -----------------------------------------------------------------------------
def corr(
    meta: Union[Dict[str, Any], Any],
    columns: Optional[List[str]] = None,
    column_types: Optional[List[str]] = None,
    method: str = "spearman",
    permutations: int = 999,
    random_state: int | None = None,
    *,
    return_coding: bool = True,
) -> Dict[str, pd.DataFrame | Dict[tuple, Dict[str, Any]]]:
    """
    Compute mixed-type correlation/association matrix + permutation p-values + BH-FDR padj + counts.

    Pairwise rules:
        - num <-> num        => Pearson/Spearman (pandas)
        - cat (binary) <-> num => point-biserial
        - cat (multi)  <-> num => correlation ratio (eta)
        - cat <-> cat        => bias-corrected Cramér's V

    Parameters
    ----------
    meta : dict-like or object recognized by get_df(meta, "meta")
        Must contain a pandas DataFrame with metadata under key/name "meta".
    columns : list[str], optional
        Columns to include. Defaults to all columns.
    column_types : list[str], optional
        List of types corresponding to the data colums. Can be 'cat' for categorical data or 'num' for numerical.
        If not provided, column types will be inferred automatically.
    numeric_method : {"pearson", "spearman"}, default "spearman"
        Method for numeric–numeric pairs.
    permutations : int, default 999
        Number of permutations per pair for p-values.
    random_state : int, default None
        Random seed for reproducibility.

    Returns
    -------
    dict : {'R': DataFrame, 'p': DataFrame, 'padj': DataFrame, 'N': DataFrame, 
            'coding': dict (optional if return_coding=True)}
        R    : effect sizes (symmetric; diag=1.0)
        P    : raw permutation p-values (symmetric; diag=NaN)
        padj : BH-FDR adjusted p-values (symmetric; diag=NaN)
        N    : pairwise complete case counts (symmetric; diag=#non-missing in column)
        coding : [('var1', 'var2')] shows information about how to interpret the correlations
    """
    df = get_df(meta, "meta")
    if df is None:
        raise ValueError("'meta' is needed in input.")
    df = df.copy()

    if columns is None:
        columns = df.columns.tolist()

    cols: List[str] = []
    types: Dict[str, str] = {}

    if column_types is None:
        for c in columns:
            types[c] = "cat" if _is_categorical(df[c]) else "num"
            cols.append(c)
    elif isinstance(column_types, list) and len(column_types) == len(columns):
        for i, c in enumerate(columns):
            types[c] = column_types[i]
            cols.append(c)
    else:
        raise ValueError("columns don't match column_types")

    R = pd.DataFrame(index=cols, columns=cols, dtype=float)
    P = pd.DataFrame(index=cols, columns=cols, dtype=float)
    N = pd.DataFrame(index=cols, columns=cols, dtype=float)

    # --- Storage for coding info
    coding: Dict[Tuple[str, str], Dict[str, Any]] = {}

    rng = np.random.default_rng(random_state) #Random state

    # Go through all pairs
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j < i:
                continue
            if j == i:
                R.loc[a, b] = 1.0
                P.loc[a, b] = np.nan
                N.loc[a, b] = df[a].notna().sum()
                continue

            s = pd.DataFrame({"a": df[a], "b": df[b]}).dropna()
            n_pair = len(s)
            N.loc[a, b] = N.loc[b, a] = n_pair
            if n_pair < 3:
                R.loc[a, b] = R.loc[b, a] = np.nan
                P.loc[a, b] = P.loc[b, a] = np.nan
                continue

            x = s["a"].reset_index(drop=True)
            y = s["b"].reset_index(drop=True)
            ta, tb = types[a], types[b]

            if ta == "num" and tb == "num":
                stat_fun = lambda xv, yv: _stat_numeric_numeric(xv, yv, method=method)

            elif ta == "cat" and tb == "cat":
                stat_fun = _stat_cat_cat

            else:
                # cat <-> num case
                stat_fun = _stat_cat_num

                # --- Record coding information if requested
                if return_coding:
                    # Identify which side is the categorical variable and its levels
                    if ta == "cat" and tb == "num":
                        cat_var = a
                        cat_series = x
                    else:
                        cat_var = b
                        cat_series = y

                    # levels in the exact order used by the computation
                    levels = pd.Series(cat_series).unique()
                    k = len(levels)

                    if k == 2:
                        # point-biserial sign meaning
                        interp = (
                            f"R>0 ⇒ mean(Y|{cat_var}=={levels[0]!r}) > mean(Y|{cat_var}=={levels[1]!r}); "
                            f"R<0 ⇒ mean(Y|{cat_var}=={levels[0]!r}) < mean(Y|{cat_var}=={levels[1]!r})"
                        )
                        info = {
                            "pair_type": "binary_cat_num",
                            "cat_var": cat_var,
                            "levels_order": [levels[0], levels[1]],
                            "sign_interpretation": interp,
                        }
                    elif k >= 3:
                        info = {
                            "pair_type": "multi_cat_num",
                            "cat_var": cat_var,
                            "levels_order": list(levels),
                            "sign_interpretation": None,  # eta has no sign
                        }
                    else:
                        info = {
                            "pair_type": "degenerate",
                            "cat_var": cat_var,
                            "levels_order": list(levels),
                            "sign_interpretation": None,
                        }

                    # store under both (a,b) and (b,a) for symmetry
                    coding[(a, b)] = info
                    coding[(b, a)] = info

            stat, p = _perm_test_stat(x, y, stat_fun, n_perm=permutations, rng=rng)
            R.loc[a, b] = R.loc[b, a] = stat
            P.loc[a, b] = P.loc[b, a] = p

    # Mirror diagonal counts (optional: keep as per-column non-missing)
    for k in cols:
        N.loc[k, k] = df[k].notna().sum()

    # FDR adjust off-diagonal, mirror to symmetric matrix
    padj = _fdr_bh_offdiag(P)

    # --- assemble output
    out = {'R': R, 'p': P, 'padj': padj, 'N': N}
    if return_coding:
        out['coding'] = coding
    return out


# -----------------------------------------------------------------------------
# Bootstrap function for confidence intervals
# -----------------------------------------------------------------------------
def _percentile_ci(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    lo = np.nanpercentile(x, 100 * (alpha / 2.0))
    hi = np.nanpercentile(x, 100 * (1 - alpha / 2.0))
    return float(lo), float(hi)

def _upper_tri_mean(sub: np.ndarray) -> float:
    n = sub.shape[0]
    if n < 2:
        return np.nan
    tri = sub[np.triu_indices(n, k=1)]
    return float(tri.mean()) if tri.size else np.nan

def _weighted_mean_distance(
    D: np.ndarray,
    blocks: List[Tuple[np.ndarray, np.ndarray]],
    *,
    within: bool,
) -> Tuple[float, int]:
    """
    Compute a pair-count-weighted mean distance over a collection of index blocks.

    Parameters
    ----------
    D : numpy.ndarray
        Square distance matrix of shape (n, n).
    blocks : list of (idx1, idx2)
        Each element defines a block of distances via index arrays.
        Indices are deduplicated within this function to avoid duplicate->zero
        inflation when bootstrap sampling is done with replacement.

        - If within=True, each block is treated as within-group distances:
          idx1 is used (idx2 ignored) and the mean of the upper triangle of
          D[idx1, idx1] is computed.
        - If within=False, each block is treated as between-group distances:
          the mean of the full rectangular block D[idx1, idx2] is computed.
    within : bool
        Whether to compute within-group (upper triangle) or between-group
        (rectangular block) distances.

    Returns
    -------
    mean : float
        Weighted mean distance across blocks, or NaN if no valid pairs exist.
    total_pairs : int
        Total number of contributing pairs (sum of weights).
    """
    weighted_sum = 0.0
    total_pairs = 0

    for idx1, idx2 in blocks:
        idx1 = np.unique(idx1)
        idx2 = np.unique(idx2)

        if within:
            n = idx1.size
            w = n * (n - 1) // 2
            if w <= 0:
                continue
            sub = D[np.ix_(idx1, idx1)]
            m = _upper_tri_mean(sub)
        else:
            n1, n2 = idx1.size, idx2.size
            w = n1 * n2
            if w <= 0:
                continue
            sub = D[np.ix_(idx1, idx2)]
            m = float(sub.mean()) if sub.size else np.nan

        if not np.isfinite(m):
            continue

        weighted_sum += m * w
        total_pairs += w

    if total_pairs == 0:
        return np.nan, 0

    return weighted_sum / total_pairs, total_pairs


def bootstrap_sample_matrix(
    df: pd.DataFrame,
    meta: Union[pd.DataFrame, Dict[str, Any], Any],
    by: Union[str, List[str]],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: Union[int, np.random.Generator, None] = None,
    return_boot: bool = False,
    warn_small: bool = True,
    **kwargs,
) -> Dict[str, Dict[str, Dict[str, Union[float, Tuple[float, float], np.ndarray, int]]]]:
    """
    Compute bootstrap confidence intervals for within‑ and between‑group
    summaries of a square sample×sample matrix.
    
    For each variable listed in `by`, the function computes two aggregated
    summary statistics:
    
    * **within**: the average pairwise value among samples sharing the same
      category of that variable, and
    * **between**: the average pairwise value among samples belonging to
      different categories of that variable.
    
    These summaries are estimated using a **nested bootstrap**. Samples are
    resampled *with replacement* within each fully crossed cell defined by
    `by`. For each bootstrap replicate, category‑level values are pooled
    across the remaining factor(s), and summary statistics are computed using
    **pair‑count weighting**.
    
    To avoid artificial inflation of zero or repeated distances caused by
    bootstrap duplicates, resampled indices are **deduplicated prior to
    computing pairwise distances**. As a consequence, the effective number
    of contributing sample pairs may vary across bootstrap replicates.
    
    In addition to per‑variable summaries, the function returns a single
    **“Crossed”** aggregation that pools across all fully crossed cells,
    providing one overall within‑cell and one overall between‑cell summary
    computed using the same bootstrap resamples.
    
    Parameters
    ----------
    df : (n x n) pandas.DataFrame
        Symmetric sample×sample matrix with identical row/column labels and
        order.
    meta : pandas.DataFrame | dict | MicrobiomeData-like
        Metadata indexed by sample IDs matching `df.index`. Metadata will be
        aligned to `df.index` without reordering `df`.
    by : str | list[str]
        One or more categorical metadata columns defining the fully crossed
        cells used for nested resampling.
    n_boot : int, default 1000
        Number of bootstrap replicates.
    alpha : float, default 0.05
        Percentile confidence level (95% CI if alpha = 0.05).
    random_state : int | numpy.random.Generator | None
        Random seed or NumPy random generator for reproducible permutations.
    return_boot : bool, default False
        If False, the returned dictionary omits the bootstrap replicate arrays
        to reduce memory usage.
    warn_small : bool, default True
        If True, issue warnings when cells or categories contain very few
        samples, which may lead to unstable bootstrap estimates.
    
    Returns
    -------
    out : dict
        Dictionary with the following structure:
    
        {
          "<var>": {
            "within": {
              "mean": float,
              "ci": (lo, hi),
              "total_pairs": int,
              "boot": np.ndarray | None
            },
            "between": {
              "mean": float,
              "ci": (lo, hi),
              "total_pairs": int,
              "boot": np.ndarray | None
            }
          },
          ...,
          "Crossed": {
            "within": {
              "mean": float,
              "ci": (lo, hi),
              "total_pairs": int,
              "boot": np.ndarray | None
            },
            "between": {
              "mean": float,
              "ci": (lo, hi),
              "total_pairs": int,
              "boot": np.ndarray | None
            }
          }
        }
    
        One entry is returned for each variable in `by`, each containing a
        single within‑ and between‑category summary. If more than one variable
        is supplied, an additional “Crossed” entry provides summaries pooled
        across all fully crossed cells.
    
    Notes
    -----
    * Bootstrap resampling is performed at the **sample level** within fully
      crossed cells, but duplicate resampled indices are **deduplicated before
      computing pairwise distances**. This yields a bootstrap of pairwise
      distance summaries rather than a bootstrap of raw samples.
    * Because of deduplication, the effective number of contributing sample
      pairs can differ between bootstrap replicates. The reported
      `total_pairs` value corresponds to the **average number of distinct
      pairs contributing per bootstrap replicate**, not a fixed property of
      the original dataset.
    * Pairwise summaries are computed using the upper triangle for within‑group
      blocks and the full rectangular block for between‑group blocks.
    """

    if "seed" in kwargs:
        if random_state is not None:
            raise TypeError("Specify only one of 'random_state' or 'seed'.")
        random_state = kwargs.pop("seed")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

    # ---- Retrieve & validate metadata ----
    M = get_df(meta, "meta")
    if not isinstance(M, pd.DataFrame) or M.empty:
        raise ValueError("`meta` must be a non-empty pandas DataFrame (via get_df(meta, 'meta')).")

    if not isinstance(df, pd.DataFrame) or df.shape[0] != df.shape[1]:
        raise ValueError("`df` must be a square pandas DataFrame.")
    if not df.index.equals(df.columns):
        raise ValueError("`df` must have identical row and column labels (same order).")

    # Align M to df (do not reorder df)
    if set(M.index) != set(df.index):
        missing_in_meta = set(df.index) - set(M.index)
        missing_in_df   = set(M.index) - set(df.index)
        if missing_in_meta or missing_in_df:
            raise ValueError("Index mismatch between meta and df.")
    if not M.index.equals(df.index):
        M = M.loc[df.index]

    # Normalize by → list[str]
    if isinstance(by, str):
        by = [by]
    if len(by) < 1:
        raise ValueError("`by` must contain at least one metadata column.")
    for var in by:
        if var not in M.columns:
            raise ValueError(f"Column '{var}' not found in metadata.")
        if not is_categorical_dtype(M[var]):
            M[var] = M[var].astype("category")

    # ---- Build fully crossed cells across all variables in `by` ----
    level_lists = [list(M[v].cat.categories) for v in by]  # deterministic order
    cell_levels = list(product(*level_lists))
    # Keep only present cells
    present_cells: List[Tuple[Any, ...]] = []
    for lev in cell_levels:
        mask = np.ones(len(M), dtype=bool)
        for v, l in zip(by, lev):
            mask &= (M[v].to_numpy() == l)
        if mask.any():
            present_cells.append(lev)
    if len(present_cells) == 0:
        raise ValueError("No non-empty cells found for the provided `by` variables.")

    # Map: cell_key -> positions
    label_to_pos = {lab: i for i, lab in enumerate(df.index)}
    pos_by_cell: Dict[Tuple[Any, ...], np.ndarray] = {}
    n_by_cell: Dict[Tuple[Any, ...], int] = {}
    for lev in present_cells:
        mask = np.ones(len(M), dtype=bool)
        for v, l in zip(by, lev):
            mask &= (M[v].to_numpy() == l)
        labels = M.index[mask]
        pos = np.fromiter((label_to_pos[s] for s in labels), dtype=int, count=len(labels))
        pos_by_cell[lev] = pos
        n_by_cell[lev] = len(pos)

    # Small-sample warnings (cells)
    if warn_small:
        for lev, n in n_by_cell.items():
            if n < 2:
                warnings.warn(
                    f"Cell {dict(zip(by, lev))} has only {n} sample(s); "
                    "within-cell pairs do not exist and nested resampling may be unstable.",
                    UserWarning
                )
            elif n == 2:
                warnings.warn(
                    f"Cell {dict(zip(by, lev))} has 2 samples; bootstrap within this cell is limited.",
                    UserWarning
                )

    # ---- Per-variable category→cells mapping ----
    cats_present_by_var: Dict[str, List[Any]] = {}
    cat_cells: Dict[str, Dict[Any, List[Tuple[Any, ...]]]] = {}
    for var_i, var in enumerate(by):
        cats = [c for c in M[var].cat.categories if (M[var] == c).any()]
        cats_present_by_var[var] = cats
        cat_cells[var] = {c: [] for c in cats}
        for lev in present_cells:
            c = lev[var_i]
            if c in cat_cells[var]:
                cat_cells[var][c].append(lev)

    # ---- RNG and storage ----
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    D = df.to_numpy()

    # Storage per variable
    boot_within: Dict[str, np.ndarray]  = {var: np.empty(n_boot, dtype=float) for var in by}
    boot_between: Dict[str, np.ndarray] = {var: np.empty(n_boot, dtype=float) for var in by}

    # Storage for Crossed aggregation
    boot_cross_within  = np.empty(n_boot, dtype=float)
    boot_cross_between = np.empty(n_boot, dtype=float)

    # ---- Precompute denominators (pair counts) per variable ----
    n_c_by_var: Dict[str, Dict[Any, int]] = {}
    for var_i, var in enumerate(by):
        n_c: Dict[Any, int] = {}
        for c in cats_present_by_var[var]:
            pooled_pos = np.concatenate([pos_by_cell[lev] for lev in cat_cells[var][c]]) if cat_cells[var][c] else np.array([], dtype=int)
            n_c[c] = int(pooled_pos.size)
        n_c_by_var[var] = n_c

    within_den_by_var: Dict[str, int] = {}
    between_den_by_var: Dict[str, int] = {}
    for var in by:
        n_c = n_c_by_var[var]
        within_den = int(sum(max(n * (n - 1) // 2, 0) for n in n_c.values()))
        cats = cats_present_by_var[var]
        cat_pairs = list(combinations(cats, 2))
        between_den = int(sum(n_c[a] * n_c[b] for (a, b) in cat_pairs))
        within_den_by_var[var] = within_den
        between_den_by_var[var] = between_den

        if warn_small:
            for c, n in n_c.items():
                if n < 2:
                    warnings.warn(
                        f"Category {var}={c!r} has only {n} sample(s) pooled across other factors; "
                        "within-category dissimilarity is undefined.",
                        UserWarning
                    )
                elif n == 2:
                    warnings.warn(
                        f"Category {var}={c!r} has 2 samples pooled across other factors; "
                        "bootstrap within-category will be unstable.",
                        UserWarning
                    )

    # ---- Bootstrap loop ----
    cells_list = list(present_cells)
    boot_within_den   = {var: np.empty(n_boot, dtype=int) for var in by}
    boot_between_den  = {var: np.empty(n_boot, dtype=int) for var in by}
    boot_cross_within_den  = np.empty(n_boot, dtype=int)
    boot_cross_between_den = np.empty(n_boot, dtype=int)

    for b in range(n_boot):
        # 1) Nested resampling within each fully crossed cell
        boot_pos_by_cell = {
            lev: rng.choice(pos_by_cell[lev], size=n_by_cell[lev], replace=True)
            for lev in cells_list
        }

        # 2) Per-variable pooled sets and weighted means
        for var_i, var in enumerate(by):
            cats = cats_present_by_var[var]
            pooled_by_cat: Dict[Any, np.ndarray] = {}
            n_by_cat: Dict[Any, int] = {}
            for c in cats:
                cells = cat_cells[var][c]
                if cells:
                    pooled = np.concatenate([boot_pos_by_cell[lev] for lev in cells])
                else:
                    pooled = np.array([], dtype=int)
                pooled_by_cat[c] = pooled
                n_by_cat[c] = int(pooled.size)

            # WITHIN(var): weighted mean of within-category upper-triangle dissimilarities
            # (deduplicating resampled indices to avoid duplicate->zero inflation)
            blocks = [(pooled_by_cat[c], pooled_by_cat[c]) for c in cats]
            boot_within[var][b], within_den = _weighted_mean_distance(D, blocks, within=True)
            boot_within_den[var][b]  = within_den

            # BETWEEN(var): weighted mean of between-category distances (deduplicated)
            blocks = [(pooled_by_cat[c1], pooled_by_cat[c2]) for (c1, c2) in combinations(cats, 2)]
            boot_between[var][b], between_den = _weighted_mean_distance(D, blocks, within=False)
            boot_between_den[var][b] = between_den

        # 3) Crossed aggregation (pooled across all cells): WITHIN
        blocks = [(boot_pos_by_cell[lev], boot_pos_by_cell[lev]) for lev in cells_list]
        boot_cross_within[b], cross_within_den = _weighted_mean_distance(D, blocks, within=True)
        boot_cross_within_den[b]  = cross_within_den
        
        blocks = [(boot_pos_by_cell[c1], boot_pos_by_cell[c2]) for (c1, c2) in combinations(cells_list, 2)]
        boot_cross_between[b], cross_between_den = _weighted_mean_distance(D, blocks, within=False)
        boot_cross_between_den[b] = cross_between_den

    # ---- Summaries & output ----
    out: Dict[str, Dict[str, Dict[str, Union[float, Tuple[float, float], np.ndarray, int]]]] = {}
    # Per-variable blocks
    for var in by:
        wvec = boot_within[var]
        bvec = boot_between[var]
        w_ci = _percentile_ci(wvec, alpha=alpha) if np.isfinite(wvec).any() else (np.nan, np.nan)
        b_ci = _percentile_ci(bvec, alpha=alpha) if np.isfinite(bvec).any() else (np.nan, np.nan)
        out[var] = {
            "within": {
                "mean": float(np.nanmean(wvec)),
                "ci": w_ci,
                "total_pairs": int(np.nanmean(boot_within_den[var])),
                "boot": wvec if return_boot else None,
            },
            "between": {
                "mean": float(np.nanmean(bvec)),
                "ci": b_ci,
                "total_pairs": int(np.nanmean(boot_between_den[var])),
                "boot": bvec if return_boot else None,
            },
        }

    # Crossed block
    if len(by) > 1:
        cw_ci = _percentile_ci(boot_cross_within,  alpha=alpha) if np.isfinite(boot_cross_within).any()  else (np.nan, np.nan)
        cb_ci = _percentile_ci(boot_cross_between, alpha=alpha) if np.isfinite(boot_cross_between).any() else (np.nan, np.nan)
        out["Crossed"] = {
            "within": {
                "mean": float(np.nanmean(boot_cross_within)),
                "ci": cw_ci,
                "total_pairs": int(np.nanmean(boot_cross_within_den)),
                "boot": boot_cross_within if return_boot else None,
            },
            "between": {
                "mean": float(np.nanmean(boot_cross_between)),
                "ci": cb_ci,
                "total_pairs": int(np.nanmean(boot_cross_between_den)),
                "boot": boot_cross_between if return_boot else None,
            },
        }

    return out
