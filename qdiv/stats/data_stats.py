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
from .distance_tests import mantel

__all__ = ["corr", "bootstrap_sample_matrix", "phylo_signal_mantel"]

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

def bootstrap_sample_matrix(
    df: pd.DataFrame,
    meta: Union[pd.DataFrame, Dict[str, Any], Any],
    by: Union[str, List[str]],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
    return_boot: bool = False,
    warn_small: bool = True,
) -> Dict[str, Dict[str, Dict[str, Union[float, Tuple[float, float], np.ndarray, int]]]]:
    """
    For each variable listed in `by`, the function computes two aggregated
    summary statistics—one describing variation within the categories of that
    variable, and one describing variation between those categories.
    These summaries are derived by applying a nested bootstrap (stratified
    resampling within all fully crossed groups defined by `by`) to the input
    square matrix. For each bootstrap draw, category‑level values are pooled and
    combined using pair‑count weighting, and percentile confidence intervals are
    then computed across all bootstrap replicates.

    Additionally, it returns a single “Crossed” aggregation that pools across all
    fully crossed cells (one `within`, one `between`) using the same nested resamples.

    Parameters
    ----------
    df : (n x n) pandas.DataFrame
        Symmetric sample×sample matrix with identical row/column labels and order.
    meta : DataFrame | dict | MicrobiomeData-like
        Metadata indexed by sample IDs matching `df.index`. Will be aligned to `df.index`.
    by : str | list[str]
        One or more metadata columns to define the fully crossed cells.
    n_boot : int, default 1000
        Number of bootstrap replicates.
    alpha : float, default 0.05
        Percentile CI level (95% CI if alpha=0.05).
    random_state : int | None
        Random seed for reproducibility.
    return_boot : bool, default False
        If False, the returned dict omits the large "boot" arrays to save memory.
    warn_small : bool, default True
        If True, warn when a cell or category has very few samples.

    Returns
    -------
    out : dict
        {
          "<var>": {
            "within":  { "mean": float, "ci": (lo, hi), "total_pairs": int, "boot": np.ndarray | None },
            "between": { "mean": float, "ci": (lo, hi), "total_pairs": int, "boot": np.ndarray | None },
          },
          ...,
          "Crossed": {
            "within":  { "mean": float, "ci": (lo, hi), "total_pairs": int, "boot": np.ndarray | None },
            "between": { "mean": float, "ci": (lo, hi), "total_pairs": int, "boot": np.ndarray | None }
          }
        }
        One entry per variable in `by` (each with a single within and between summary),
        plus a pooled “Crossed” summary across all fully crossed cells.

    Notes
    -----
    * This function performs nested resampling across the fully crossed cells of all
      variables in `by`, then collapses results to per-variable summaries by pooling
      across the other factor(s), and also provides a single pooled "Crossed" summary.
    """
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
    rng = np.random.default_rng(random_state)
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

    # ---- Denominators for Crossed aggregation (fixed) ----
    # Within: sum over cells C(n_cell,2); Between: sum over unordered cell pairs n_i*n_j
    cells_list = list(present_cells)
    cross_within_den = int(sum(max(n_by_cell[c] * (n_by_cell[c] - 1) // 2, 0) for c in cells_list))
    cross_between_den = int(sum(n_by_cell[c1] * n_by_cell[c2] for (c1, c2) in combinations(cells_list, 2)))

    # ---- Bootstrap loop ----
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

            # WITHIN(var)
            within_den = within_den_by_var[var]
            if within_den > 0:
                num = 0.0
                for c in cats:
                    n = n_by_cat[c]
                    w = n * (n - 1) // 2
                    if w <= 0:
                        continue
                    sub = D[np.ix_(pooled_by_cat[c], pooled_by_cat[c])]
                    num += _upper_tri_mean(sub) * w
                boot_within[var][b] = num / within_den
            else:
                boot_within[var][b] = np.nan

            # BETWEEN(var)
            between_den = between_den_by_var[var]
            if between_den > 0:
                num = 0.0
                for (c1, c2) in combinations(cats, 2):
                    n1, n2 = n_by_cat[c1], n_by_cat[c2]
                    w = n1 * n2
                    if w <= 0:
                        continue
                    sub = D[np.ix_(pooled_by_cat[c1], pooled_by_cat[c2])]
                    num += (float(sub.mean()) if sub.size else 0.0) * w
                boot_between[var][b] = num / between_den
            else:
                boot_between[var][b] = np.nan

        # 3) Crossed aggregation (pooled across all cells)
        if cross_within_den > 0:
            num = 0.0
            for lev in cells_list:
                n = n_by_cell[lev]
                w = n * (n - 1) // 2
                if w <= 0:
                    continue
                sub = D[np.ix_(boot_pos_by_cell[lev], boot_pos_by_cell[lev])]
                num += _upper_tri_mean(sub) * w
            boot_cross_within[b] = num / cross_within_den
        else:
            boot_cross_within[b] = np.nan

        if cross_between_den > 0:
            num = 0.0
            for (c1, c2) in combinations(cells_list, 2):
                n1, n2 = n_by_cell[c1], n_by_cell[c2]
                w = n1 * n2
                if w <= 0:
                    continue
                sub = D[np.ix_(boot_pos_by_cell[c1], boot_pos_by_cell[c2])]
                num += (float(sub.mean()) if sub.size else 0.0) * w
            boot_cross_between[b] = num / cross_between_den
        else:
            boot_cross_between[b] = np.nan

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
                "total_pairs": within_den_by_var[var],
                "boot": wvec if return_boot else None,
            },
            "between": {
                "mean": float(np.nanmean(bvec)),
                "ci": b_ci,
                "total_pairs": between_den_by_var[var],
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
                "total_pairs": cross_within_den,
                "boot": boot_cross_within if return_boot else None,
            },
            "between": {
                "mean": float(np.nanmean(boot_cross_between)),
                "ci": cb_ci,
                "total_pairs": cross_between_den,
                "boot": boot_cross_between if return_boot else None,
            },
        }
    return out


# -----------------------------------------------------------------------------
# Bootstrap function for confidence intervals
# -----------------------------------------------------------------------------
def phylo_signal_mantel(
    obj: Union[Dict[str, Any], Any],
    distmat: pd.DataFrame,
    by: str,
    *,
    method: str = "pearson",
    permutations: int = 999,
    relative_abundance: bool = True,
    min_total: float = 0.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Estimate phylogenetic signal of species environmental niches using Mantel test.

    Steps:
        1. Extract abundance table (tab), metadata (meta), and phylo distmat.
        2. Extract environmental variable from meta (by column).
        3. Compute species-level niche value as abundance-weighted mean environment:
               niche_i = sum_s(abundance_{i,s} * env_s) / sum_s(abundance_{i,s})
        4. Compute pairwise absolute-niche-distance matrix between species.
        5. Mantel test between species-niche-distance and phylogenetic distance matrix.
    
    Parameters
    ----------
    obj : MicrobiomeData or dict
        Must contain 'tab' and 'meta'.
    distmat : DataFrame
        Square phylogenetic distance matrix (features × features).
    by : str
        Column name in metadata to extract environmental values (e.g., "pH").
    method : {"pearson","spearman"}, optional
        Correlation method for Mantel.
    permutations : int, optional
        Number of null permutations.

    Returns
    -------
    dict
        {
            "by"        : environmental variable name,
            "niche"      : species niche vector (aligned with distmat),
            "niche_dist_vec": distance vector that could be used for plotting
            "phylo_dist_vec": distance vector that could be used for plotting
            "mantel_r"   : Mantel correlation,
            "mantel_p"   : p-value,
            "mantel_out" : full mantel output
        }
    """
    # --- extract data ---
    tab = get_df(obj, "tab")
    meta = get_df(obj, "meta")

    if tab is None or meta is None:
        raise ValueError("Need 'tab' and 'meta' in object.")

    if by not in meta.columns:
        raise ValueError(f"Environmental variable '{by}' not found in metadata.")

    # Align metadata values to tab sample order
    env_raw = meta.loc[tab.columns, by]
    
    # ----- abundances -----
    A = tab.copy().astype(float)
    A = A.div(A.sum(axis=0), axis=1).fillna(0.0)

    # total abundance per species
    distmat = distmat.loc[A.index, A.index]

    # abundance matrix (numpy)
    A_np = A.to_numpy()

    # ================================
    # CASE 1 — continuous environmental variable
    # ================================
    if np.issubdtype(env_raw.dtype, np.number):

        env_vec = env_raw.to_numpy()
        numer = A_np @ env_vec
        denom = A_np.sum(axis=1)
        niche = numer / denom
        niche = pd.Series(niche, index=A.index)

        niche_distmat = pd.DataFrame(
            np.abs(niche[:, None] - niche[None, :]),
            index=A.index,
            columns=A.index,
        )

    else:
        # ================================
        # CASE 2 — categorical environmental variable
        # ================================
        env_factor = env_raw.astype("category")
        levels = env_factor.cat.categories

        # build probability vector per species
        P = []
        for level in levels:
            indicator = (env_factor == level).astype(int).to_numpy()
            p_level = (A_np @ indicator) / A_np.sum(axis=1)
            P.append(p_level)

        P = np.vstack(P).T  # shape: (species x n_levels)

        niche = pd.DataFrame(P, index=A.index, columns=levels)

        # L1 (Gower-like) distance
        diff = np.abs(P[:, None, :] - P[None, :, :]).sum(axis=2)
        diff /= len(levels)  # scale to [0,1]

        niche_distmat = pd.DataFrame(diff, index=A.index, columns=A.index)
    print('Here')
    # --- Mantel test ---
    mantel_out = mantel(
        niche_distmat,
        distmat,
        method=method,
        permutations=permutations,
    )

    # --- extract upper triangular for plotting ---
    iu = np.triu_indices_from(distmat, k=1)
    phylo_dist_vec = distmat.values[iu]
    niche_dist_vec = niche_distmat.values[iu]

    return {
        "env": by,
        "niche": niche,
        "niche_dist_vec": niche_dist_vec,
        "phylo_dist_vec": phylo_dist_vec,
        "mantel_r": mantel_out["r"],
        "mantel_p": mantel_out["p"],
        "mantel_out": mantel_out,
    }


# ---------------------------
# Utilities phylo signal functions
# ---------------------------

def _align_trait_and_dist(trait: pd.Series, distmat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align a numeric trait vector and a symmetric, labeled distance matrix.
    Drops any tips missing in either input or with NaN trait.
    Returns:
        y (n,), D (n x n), ids (n,)
    """
    if not isinstance(trait, pd.Series):
        raise TypeError("trait must be a pandas Series (index: tip IDs).")
    if not isinstance(distmat, pd.DataFrame):
        raise TypeError("distmat must be a pandas DataFrame (square, tip IDs).")
    if distmat.shape[0] != distmat.shape[1]:
        raise ValueError("distmat must be square.")

    # intersect and order
    common = trait.index.intersection(distmat.index)
    if common.empty:
        raise ValueError("No overlapping tip IDs between trait and distmat.")

    y = trait.loc[common].astype(float)
    D = distmat.loc[common, common].astype(float)

    # drop missing trait values
    mask = y.notna().to_numpy()
    ids = y.index.to_numpy()[mask]
    y = y.to_numpy()[mask]
    D = D.to_numpy()[np.ix_(mask, mask)]

    # basic symmetry fix
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return y, D, ids


def _cov_bm_from_dist(D: np.ndarray, ultrametric_tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    """
    Build the Brownian-motion covariance matrix V from a patristic distance matrix D.

    For an ultrametric tree of height T: V = T*1*1^T - 0.5*D, with diag(V)=T.
    We estimate T as max(D)/2 and check (softly) ultrametricity.
    Returns:
        V (n x n), T (float)
    """
    n = D.shape[0]
    # estimate tree height
    T = 0.5 * np.nanmax(D[np.triu_indices(n, k=1)])
    # build covariance
    V = T - 0.5 * D
    # enforce symmetry and diagonals
    V = 0.5 * (V + V.T)
    np.fill_diagonal(V, T)

    # ultrametric sanity: all tips the same distance from root implies diag(V)=T and
    # row maxima of D ≈ 2T. We do a soft check; don’t fail hard if slightly off.
    row_max = np.max(D + np.eye(n)*-np.inf, axis=1)
    if not np.allclose(row_max, 2.0*T, rtol=1e-4, atol=ultrametric_tol):
        # Small deviations are common with rounding; warn via return flag if you prefer.
        pass

    return V, T


def _chol_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve V x = b given lower-triangular Cholesky factor L (V = L L^T)."""
    # forward
    y = np.linalg.solve(L, b)
    # backward
    x = np.linalg.solve(L.T, y)
    return x


def _ensure_spd_with_jitter(M: np.ndarray, jitter_start: float = 1e-10, max_tries: int = 8) -> Tuple[np.ndarray, float]:
    """
    Ensure M is symmetric positive-definite by adding jitter*I as needed.
    Returns the Cholesky factor and the final jitter used.
    """
    M = 0.5 * (M + M.T)
    jitter = 0.0
    for k in range(max_tries):
        try:
            L = np.linalg.cholesky(M + jitter * np.eye(M.shape[0]))
            return L, jitter
        except np.linalg.LinAlgError:
            jitter = jitter_start if jitter == 0.0 else jitter * 10.0
    # try a last time with a larger bump
    L = np.linalg.cholesky(M + (jitter * 10.0) * np.eye(M.shape[0]))
    return L, jitter * 10.0


# ---------------------------
# Blomberg's K
# ---------------------------

def blomberg_k(
    trait: pd.Series,
    distmat: pd.DataFrame,
    *,
    permutations: int = 0,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute Blomberg's K from a patristic distance matrix using the covariance formulation.

    Definitions:
      - y: trait vector (length n)
      - V: BM covariance among tips (unit rate) reconstructed from distances
      - 1: vector of ones
      - μ̂ = (1^T V^{-1} y) / (1^T V^{-1} 1)
      - r = y - μ̂ 1
      - H = I - 11^T/n  (centering projector)
      - c = trace(H V) / (n - 1)

    Then:
        K = (r^T V^{-1} r) / (y^T H y) * c
      which has E[K] = 1 under BM on an ultrametric tree.

    Parameters
    ----------
    trait : pd.Series
        Numeric trait, index = tip IDs.
    distmat : pd.DataFrame
        Patristic distance matrix (tips × tips), symmetric, zeros on diag.
    permutations : int, optional
        If >0, label-randomization p-value is computed by permuting y's labels.
    random_state : int, optional
        RNG seed for permutations.

    Returns
    -------
    dict with keys:
        K : float
        p  : float or np.nan  (NaN if permutations == 0)
        n  : int  (number of tips used)
        jitter : float  (jitter added to V for SPD, if any)
        ids : np.ndarray  (tip IDs used, in order)
    """
    y, D, ids = _align_trait_and_dist(trait, distmat)
    n = y.size
    if n < 3:
        raise ValueError("Need at least 3 tips to compute Blomberg's K.")

    V, _ = _cov_bm_from_dist(D)
    one = np.ones(n, dtype=float)
    H = np.eye(n) - np.outer(one, one) / n

    # Cholesky of V (with jitter if needed)
    L, jitter = _ensure_spd_with_jitter(V)

    # Compute V^{-1} * 1 and V^{-1} * y via solves
    Vinv1 = _chol_solve(L, one)
    Vinvy = _chol_solve(L, y)

    mu_hat = (one @ Vinvy) / (one @ Vinv1)
    r = y - mu_hat * one

    # r^T V^{-1} r without explicitly forming V^{-1}
    Vinvr = _chol_solve(L, r)
    num = float(r @ Vinvr)                 # (n-1) * σ^2_hat (under BM)

    den = float(y @ (H @ y))               # (n-1) * sample variance (uncentered SS)
    # c = trace(H V) / (n - 1); compute efficiently: trace(H V) = trace(V) - (1/n) * 1^T V 1
    trace_V = float(np.trace(V))
    s = float(one @ (V @ one))             # 1^T V 1  (sum of all elements of V)
    c = (trace_V - s / n) / (n - 1)

    K = (num / den) * c

    # Optional permutations (label-randomization)
    if permutations and permutations > 0:
        rng = np.random.default_rng(random_state)
        ks = 0
        for _ in range(permutations):
            yp = y[rng.permutation(n)]
            Vinvyp = _chol_solve(L, yp)
            mu_p = (one @ Vinvyp) / (one @ Vinv1)
            rp = yp - mu_p * one
            Vinvrp = _chol_solve(L, rp)
            num_p = float(rp @ Vinvrp)
            den_p = float(yp @ (H @ yp))
            Kp = (num_p / den_p) * c
            if Kp >= K:
                ks += 1
        p = (ks + 1) / (permutations + 1)
    else:
        p = np.nan

    return {"K": K, "p": p, "n": n, "jitter": jitter, "ids": ids}


# ---------------------------
# Pagel's lambda (ML)
# ---------------------------

def _profile_loglik_lambda(y: np.ndarray, V: np.ndarray, lam: float, L_cache: Dict[float, Tuple[np.ndarray, float]]) -> Tuple[float, float, float]:
    """
    Profile log-likelihood at Pagel's lambda = lam.
    Vλ = diag(V) + lam * (V - diag(V))
       = lam * V + (1 - lam) * diag(V)
    We compute μ̂, σ̂^2 and the log-likelihood at lam.
    Returns: (logLik, mu_hat, sigma2_hat)
    """
    n = y.size
    diagV = np.diag(V)
    Vlam = (lam * V) + ((1.0 - lam) * np.diag(diagV))

    # Cholesky (cache if re-evaluating same lam)
    if lam in L_cache:
        L, jitter = L_cache[lam]
    else:
        L, jitter = _ensure_spd_with_jitter(Vlam)
        L_cache[lam] = (L, jitter)

    one = np.ones(n, dtype=float)
    Vinv1 = _chol_solve(L, one)
    Vinvy = _chol_solve(L, y)

    mu_hat = (one @ Vinvy) / (one @ Vinv1)
    r = y - mu_hat * one

    # Quadratic form r^T Vλ^{-1} r via solve
    Vinvr = _chol_solve(L, r)
    q = float(r @ Vinvr)

    sigma2_hat = q / n

    # log |Vλ| from Cholesky; |Vλ| = Π L_ii^2 ⇒ log|Vλ| = 2 Σ log L_ii
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    loglik = -0.5 * (n * np.log(2.0 * np.pi * sigma2_hat) + logdet + n)

    return loglik, mu_hat, sigma2_hat


def _golden_section_max(func, a=0.0, b=1.0, tol=1e-5, max_iter=200):
    """
    Golden-section search to maximize a unimodal function on [a,b].
    Returns (x*, f(x*), iterations).
    """
    invphi = (np.sqrt(5) - 1) / 2  # 1/phi
    invphi2 = (3 - np.sqrt(5)) / 2 # 1/phi^2
    (a, b) = (float(a), float(b))
    h = b - a
    if h <= tol:
        x = (a + b) / 2
        return x, func(x), 0
    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))
    c = a + invphi2 * h
    d = a + invphi * h
    fc = func(c)
    fd = func(d)
    for k in range(n):
        if fc < fd:
            a = c
            c = d
            fc = fd
            h = invphi * h
            d = a + invphi * h
            fd = func(d)
        else:
            b = d
            d = c
            fd = fc
            h = invphi * h
            c = a + invphi2 * h
            fc = func(c)
    x = (a + b) / 2
    return x, func(x), n


def pagel_lambda(
    trait: pd.Series,
    distmat: pd.DataFrame,
    *,
    bounds: Tuple[float, float] = (0.0, 1.0),
    tol: float = 1e-5,
    max_iter: int = 200,
    test_bounds: bool = True,
) -> Dict[str, Any]:
    """
    Maximum-likelihood estimate of Pagel's lambda given a patristic distance matrix.

    Model: y ~ MVN(μ 1, σ^2 Vλ),  Vλ = diag(V) + λ (V - diag(V)).
    We optimize λ ∈ [0,1] by maximizing the profile log-likelihood.

    Parameters
    ----------
    trait : pd.Series
        Numeric trait, index = tip IDs.
    distmat : pd.DataFrame
        Patristic distance matrix (tips × tips), symmetric, zeros on diag.
    bounds : (float, float)
        Search interval for λ (default [0,1]).
    tol : float
        Tolerance for golden-section search.
    max_iter : int
        Not used by golden-section (kept for API symmetry).
    test_bounds : bool
        If True, also compute logLik at λ=0 and λ=1 and report LRTs.

    Returns
    -------
    dict with keys:
        lambda  : float (MLE)
        logLik  : float (profile log-likelihood at MLE)
        mu      : float (μ̂ at MLE)
        sigma2  : float (σ̂² at MLE)
        n       : int
        jitter  : float (jitter used at MLE)
        lrt_lambda_0 : (stat, p) or None
        lrt_lambda_1 : (stat, p) or None
    """
    from math import erfc, sqrt

    y, D, ids = _align_trait_and_dist(trait, distmat)
    n = y.size
    if n < 3:
        raise ValueError("Need at least 3 tips to estimate Pagel's lambda.")

    V, _ = _cov_bm_from_dist(D)

    # cache Cholesky factors at different λ to avoid recomputation when possible
    cache: Dict[float, Tuple[np.ndarray, float]] = {}

    # objective: profile log-likelihood at λ
    def obj(lam: float) -> float:
        ll, _, _ = _profile_loglik_lambda(y, V, lam, cache)
        return ll

    # search in [a,b]
    a, b = bounds
    lam_hat, ll_hat, _ = _golden_section_max(obj, a=a, b=b, tol=tol, max_iter=max_iter)

    # get μ̂ and σ̂² at λ̂
    ll_hat, mu_hat, sigma2_hat = _profile_loglik_lambda(y, V, lam_hat, cache)
    jitter = cache[lam_hat][1] if lam_hat in cache else 0.0

    # optional LRTs against the bounds λ=0 and λ=1
    lrt0 = lrt1 = None
    if test_bounds:
        ll0, _, _ = _profile_loglik_lambda(y, V, 0.0, cache)
        ll1, _, _ = _profile_loglik_lambda(y, V, 1.0, cache)
        # LRT statistics (approx chi^2_1)
        stat0 = 2.0 * (ll_hat - ll0)
        stat1 = 2.0 * (ll_hat - ll1)
        # chi^2_1 p-value via complementary error function: p = erfc(sqrt(stat/2))
        p0 = float(erfc(np.sqrt(max(stat0, 0.0) / 2.0)))
        p1 = float(erfc(np.sqrt(max(stat1, 0.0) / 2.0)))
        lrt0 = (stat0, p0)
        lrt1 = (stat1, p1)

    return {
        "lambda": lam_hat,
        "logLik": ll_hat,
        "mu": mu_hat,
        "sigma2": sigma2_hat,
        "n": n,
        "jitter": jitter,
        "lrt_lambda_0": lrt0,
        "lrt_lambda_1": lrt1,
        "ids": ids,
    }