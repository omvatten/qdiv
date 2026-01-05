"""
Distance tests: Mantel, Permanova, Gower

Public API:
    - mantel
    - permanova
    - gower
"""

import pandas as pd
import numpy as np
import random
from typing import Literal, List, Union, Dict, Any, Sequence
from ..utils import get_df
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype,
    is_categorical_dtype,
)

__all__ = [
    "mantel",
    "permanova",
    "gower"
]

# -----------------------------------------------------------------------------
# Mantel
# -----------------------------------------------------------------------------
def mantel(
    dis1: pd.DataFrame,
    dis2: pd.DataFrame,
    method: Literal["spearman", "pearson", "absDist"] = "spearman",
    getOnlyStat: bool = False,
    permutations: int = 999
) -> Union[float, List[float]]:
    """
    Perform a Mantel test between two dissimilarity matrices.

    The Mantel test evaluates the association between two distance/dissimilarity
    matrices by comparing their lower‑triangular entries. A permutation test
    is used to assess significance by randomly permuting sample labels in one
    matrix.

    Parameters
    ----------
    dis1 : pandas.DataFrame
        First square dissimilarity matrix (samples × samples).
    dis2 : pandas.DataFrame
        Second square dissimilarity matrix (samples × samples).
    method : {'spearman', 'pearson', 'absDist'}, default='spearman'
        Correlation/dissimilarity measure:
        - 'spearman' : Spearman rank correlation (returns 1 − ρ)
        - 'pearson'  : Pearson correlation (returns 1 − r)
        - 'absDist'  : Mean absolute difference between distances
    getOnlyStat : bool, default=False
        If True, return only the observed statistic (no permutations).
    permutations : int, default=999
        Number of permutations for the null distribution.

    Returns
    -------
    float or list [statistic, p_value]
        - If getOnlyStat=True: returns the observed statistic.
        - Otherwise: returns [observed_statistic, p_value].

    Notes
    -----
    - Matrices are automatically reordered to have identical sample order.
    - Only the lower triangular part (excluding diagonal) is used.
    - For correlation methods, the statistic is expressed as a *dissimilarity*
      (1 − r or 1 − ρ), so **smaller values indicate stronger similarity**.
    """

    # --- Input validation ----------------------------------------------------
    if not isinstance(dis1, pd.DataFrame) or not isinstance(dis2, pd.DataFrame):
        raise TypeError("dis1 and dis2 must be pandas DataFrames.")

    if dis1.shape != dis2.shape:
        raise ValueError("dis1 and dis2 must have the same shape.")

    if method not in {"spearman", "pearson", "absDist"}:
        raise ValueError("method must be 'spearman', 'pearson', or 'absDist'.")

    # --- Align matrices by sorted sample names -------------------------------
    samples = sorted(dis1.columns.tolist())
    dis1 = dis1.loc[samples, samples]
    dis2 = dis2.loc[samples, samples]

    # --- Helper: compute Mantel statistic ------------------------------------
    def get_stat(mat1: pd.DataFrame, mat2: pd.DataFrame) -> float:
        mask = np.tril(np.ones(mat1.shape, dtype=bool), k=-1)

        v1 = mat1.values[mask]
        v2 = mat2.values[mask]

        if method in ("spearman", "pearson"):
            dfcorr = pd.DataFrame({"x": v1, "y": v2})
            corr = dfcorr.corr(method=method).loc["x", "y"]
            return 1 - corr  # convert similarity → dissimilarity

        elif method == "absDist":
            return np.mean(np.abs(v1 - v2))

    # --- Observed statistic ---------------------------------------------------
    real_stat = get_stat(dis1, dis2)
    if getOnlyStat:
        return real_stat

    # --- Permutation test -----------------------------------------------------
    null_stats = np.empty(permutations, dtype=float)

    for i in range(permutations):
        perm = samples.copy()
        random.shuffle(perm)

        # permute rows and columns of dis1
        perm_dis1 = dis1.loc[perm, perm]

        null_stats[i] = get_stat(perm_dis1, dis2)

    # For correlation-based stats: smaller = stronger association
    # p-value = proportion of permuted stats <= observed
    # (two-sided is not used here; Mantel is typically one-sided)
    p_val = (np.sum(null_stats <= real_stat) + 1) / (permutations + 1)

    return [real_stat, p_val]

# -----------------------------------------------------------------------------
# Permanova
# -----------------------------------------------------------------------------
def permanova(
    dis: pd.DataFrame,
    meta: pd.DataFrame,
    by: Union[str, List[str]],
    permutations: int = 999
) -> Dict[str, Any]:
    """
    Perform PERMANOVA (Permutational Multivariate Analysis of Variance).

    This implementation supports:
    - One‑way PERMANOVA (single categorical variable)
    - Two‑way PERMANOVA with interaction (two categorical variables)

    Distances are partitioned into within‑group and between‑group components
    using sums of squares computed from the lower‑triangular entries of the
    dissimilarity matrix. Significance is assessed by permuting sample labels.

    Parameters
    ----------
    dis : pandas.DataFrame
        Square dissimilarity matrix (samples × samples).
    meta : pandas.DataFrame
        Metadata table containing grouping variables (rows = samples).
    by : str or list of str
        One or two metadata column names defining the grouping.
    permutations : int, default=999
        Number of permutations for the null distribution.

    Returns
    -------
    dict
        Dictionary containing:
        - ``by`` : variable(s) tested
        - ``F`` : array of F‑statistics (length 1 for one‑way, length 3 for two‑way)
        - ``p`` : array of p‑values corresponding to each F‑statistic

    Notes
    -----
    - Only the lower triangular part of the distance matrix is used.
    - For two‑way PERMANOVA, the returned F‑statistics correspond to:
        [main effect 1, main effect 2, interaction]
    - p‑values are one‑sided (F_perm ≥ F_obs).
    """

    # --- Input validation ----------------------------------------------------
    if not isinstance(dis, pd.DataFrame):
        raise TypeError("dis must be a pandas DataFrame.")
    if not isinstance(meta, pd.DataFrame):
        raise TypeError("meta must be a pandas DataFrame.")
    if dis.shape[0] != dis.shape[1]:
        raise ValueError("dis must be a square matrix.")
    if not all(dis.index == dis.columns):
        raise ValueError("dis must have identical row/column labels.")
    if isinstance(by, list) and len(by) not in (1, 2):
        raise ValueError("by must be a string or a list of length 1 or 2.")

    # Ensure metadata is aligned
    meta = meta.loc[dis.index]

    # --- Helper: compute sum of squares --------------------------------------
    def get_SS(dist: pd.DataFrame, variable: str, metaSS: pd.DataFrame) -> float:
        mask = np.tril(np.ones(dist.shape, dtype=bool), k=-1)

        # Total SS (no grouping)
        if variable not in metaSS.columns:
            vect = dist.values[mask]
            return np.sum(vect ** 2) / len(dist)

        # Grouped SS
        SS = 0.0
        for cat in metaSS[variable].unique():
            idx = metaSS.index[metaSS[variable] == cat]
            if len(idx) > 1:
                sub = dist.loc[idx, idx]
                vect = sub.values[np.tril(np.ones(sub.shape, dtype=bool), k=-1)]
                SS += np.sum(vect ** 2) / len(idx)
        return SS

    # --- Helper: compute F‑statistics ----------------------------------------
    def get_F(dist: pd.DataFrame, metaF: pd.DataFrame) -> np.ndarray:
        SStot = get_SS(dist, "None", metaF)

        # One‑way PERMANOVA
        if isinstance(by, str) or (isinstance(by, list) and len(by) == 1):
            v = by[0] if isinstance(by, list) else by
            SSw = get_SS(dist, v, metaF)
            SSa = SStot - SSw

            dfa = metaF[v].nunique() - 1
            dfw = len(dist) - metaF[v].nunique()

            F = (SSa / dfa) / (SSw / dfw)
            return np.array([F, np.nan, np.nan])

        # Two‑way PERMANOVA
        v1, v2 = by
        mc = metaF.copy()
        mc["interaction"] = mc[v1].astype(str) + mc[v2].astype(str)

        SS1 = SStot - get_SS(dist, v1, mc)
        SS2 = SStot - get_SS(dist, v2, mc)
        SSr = get_SS(dist, "interaction", mc)

        df1 = mc[v1].nunique() - 1
        df2 = mc[v2].nunique() - 1
        dfr = len(dist) - mc[v1].nunique() * mc[v2].nunique()

        if SSr > 0:
            SS12 = SStot - SS1 - SS2 - SSr
            df12 = df1 * df2

            F1 = (SS1 / df1) / (SSr / dfr)
            F2 = (SS2 / df2) / (SSr / dfr)
            F12 = (SS12 / df12) / (SSr / dfr)
            return np.array([F1, F2, F12])

        # No interaction variance
        SSe = SStot - SS1 - SS2
        dfe = df1 * df2

        F1 = (SS1 / df1) / (SSe / dfe)
        F2 = (SS2 / df2) / (SSe / dfe)
        return np.array([F1, F2, np.nan])

    # --- Observed F ----------------------------------------------------------
    real_F = get_F(dis, meta)

    # --- Permutation test ----------------------------------------------------
    null_F = np.zeros((permutations, 3))
    samples = dis.index.tolist()

    for i in range(permutations):
        perm = samples.copy()
        random.shuffle(perm)
        perm_dis = dis.loc[perm, perm]
        null_F[i] = get_F(perm_dis, meta)

    # p‑values: proportion of permuted F ≥ observed F
    p_val = (np.sum(null_F >= real_F, axis=0) + 1) / (permutations + 1)
    p_val[np.isnan(real_F)] = np.nan

    # --- Output --------------------------------------------------------------
    if isinstance(by, list):
        return {"by": by, "F": real_F, "p": p_val}
    else:
        return {"by": by, "F": real_F[0], "p": p_val[0]}

# -----------------------------------------------------------------------------
# Gower
# -----------------------------------------------------------------------------
def gower(
    meta: Union[pd.DataFrame, Dict[str, Any], Any] = None,
    *,
    by: Union[Sequence[str], str, None] = None,
    return_similarity: bool = False,
) -> pd.DataFrame:
    """
    Compute the Gower distance matrix for a pandas DataFrame
    containing mixed variable types (numeric, categorical/boolean, datetime).

    Parameters
    ----------
    meta : pd.DataFrame, dict, or MicrobiomeData object
        Input data. Rows are samples; columns are variables.
    by : Sequence[str] or str, optional
        Variable names (columns) to include. If None, all columns are included.
    return_similarity : bool, optional
        If True, return Gower similarity (1 - distance). Default False (distance).

    Returns
    -------
    pandas.DataFrame
        Pairwise Gower distances (or similarities) between samples (rows).

    Notes
    -----
    - Numerical variables are scaled by their range (max - min). If the range is 0
      (constant column), that variable contributes 0 for all pairs.
    - Datetime variables are converted to days (float) and treated as numeric.
    - Categorical/boolean variables contribute 0 when equal, 1 when different.
    - Missing values: a variable only contributes for row pairs where it is
      present in both rows; the per-pair denominator is the count of contributing
      variables for that pair.
    """
    # Obtain DataFrame
    df = get_df(meta, "meta")
    if df is None or df.empty:
        raise ValueError("Input 'meta' is missing or empty.")

    # Column selection
    if by is None:
        X = df.copy()
    elif isinstance(by, str):
        if by not in df.columns:
            raise ValueError(f"Variable '{by}' not found in columns.")
        X = df[[by]].copy()
    else:
        by = list(by)
        missing = [c for c in by if c not in df.columns]
        if missing:
            raise ValueError(f"Variables not found in columns: {missing}")
        X = df[by].copy()

    if X.shape[1] == 0:
        raise ValueError("No variables selected (empty 'by').")

    # Determine variable types
    cols_num: List[str] = []
    cols_cat: List[str] = []

    NS_PER_DAY = 86_400_000_000_000  # nanoseconds per day

    for col in X.columns:
        s = X[col]
        if is_datetime64_any_dtype(s):
            # Convert datetime to numeric (days, float); preserve NaT as NaN
            # .view('int64') works for tz-naive; for tz-aware, .astype('int64') is fine in recent pandas
            vals_int = s.view("int64")
            vals_days = vals_int.astype("float") / NS_PER_DAY
            X[col] = vals_days
            cols_num.append(col)
        elif is_numeric_dtype(s):
            cols_num.append(col)
        elif is_bool_dtype(s) or is_categorical_dtype(s) or s.dtype == object:
            cols_cat.append(col)
        else:
            # Fallback: treat as categorical
            cols_cat.append(col)

    n = len(X)
    # Early return for single row
    if n == 1:
        return pd.DataFrame(
            np.array([[0.0 if not return_similarity else 1.0]]),
            index=X.index,
            columns=X.index,
        )

    num = np.zeros((n, n), dtype=np.float64)  # numerator: sum of per-variable distances
    den = np.zeros((n, n), dtype=np.float64)  # denominator: count of valid variables per pair

    # --- Numerical variables ---
    for col in cols_num:
        x = X[col].to_numpy(dtype=float)
        mask = ~np.isnan(x)
        both = np.outer(mask, mask)

        # Range scaling (exclude NaNs)
        valid_vals = x[mask]
        if valid_vals.size == 0:
            # No contribution; but den remains unchanged (no both)
            continue

        r = valid_vals.max() - valid_vals.min()
        if not np.isfinite(r) or r == 0.0:
            # Constant or non-finite range -> contributes 0 where both present
            contrib = np.zeros((n, n), dtype=float)
        else:
            diffs = np.abs(x[:, None] - x[None, :]) / r
            diffs[~both] = 0.0
            contrib = diffs

        num += contrib
        den += both.astype(float)

    # --- Categorical / Boolean / Object variables ---
    # Compare factorized integer codes rather than Python objects (faster, leaner)
    for col in cols_cat:
        s = X[col]
        # Factorize with NaN as -1
        codes, uniques = pd.factorize(s, sort=False, use_na_sentinel=True)
        codes = codes.astype(np.int64)
        mask = codes != -1
        both = np.outer(mask, mask)

        # Diff = 1 where codes differ, 0 where equal
        # Using broadcasting on integer codes
        eq = (codes[:, None] == codes[None, :])
        diff = (~eq).astype(float)
        diff[~both] = 0.0

        num += diff
        den += both.astype(float)

    # Finalize
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = num / den
    dist[den == 0] = np.nan  # no comparable variables for that pair

    # Convert to similarity if requested
    result = 1.0 - dist if return_similarity else dist

    # Diagonal
    np.fill_diagonal(result, 0.0 if not return_similarity else 1.0)

    return pd.DataFrame(result, index=X.index, columns=X.index)
