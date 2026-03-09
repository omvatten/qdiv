"""
Distance tests: Mantel, Permanova, Gower

Public API:
    - mantel
    - permanova
    - gower
"""

import pandas as pd
import numpy as np
from numpy.linalg import matrix_rank, pinv
import warnings
from typing import Literal, List, Union, Dict, Any, Sequence
from ..utils import get_df
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype,
    is_categorical_dtype,
    CategoricalDtype,
)

__all__ = [
    "mantel",
    "permanova",
    "gower"
]

# -----------------------------------------------------------------------------
# Mantel
# -----------------------------------------------------------------------------
def _rank_vector(a: np.ndarray) -> np.ndarray:
    """
    Fast rank (average ranks for ties) using NumPy only.
    """
    # argsort twice trick
    order = np.argsort(a, kind="mergesort")                  # stable
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)     # 1..m

    # handle ties -> average ranks
    # find groups of equal values in sorted order
    s = a[order]
    # run-length encoding
    diff = np.ones_like(s, dtype=bool)
    diff[1:] = s[1:] != s[:-1]
    idx_start = np.flatnonzero(diff)
    idx_end = np.r_[idx_start[1:], s.size]                   # exclusive end

    for b, e in zip(idx_start, idx_end):
        if e - b > 1:
            # average rank for the block b..e-1
            avg = (b + 1 + e) / 2.0
            ranks[order[b:e]] = avg
    return ranks

def mantel(
    dis1: pd.DataFrame,
    dis2: pd.DataFrame,
    method: Literal["spearman", "pearson", "absDist"] = "spearman",
    getOnlyStat: bool = False,
    permutations: int = 999,
    *,
    random_state: Union[int, np.random.Generator, None] = None,
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
    if not isinstance(dis1, pd.DataFrame) or not isinstance(dis2, pd.DataFrame):
        raise TypeError("dis1 and dis2 must be pandas DataFrames.")
    if dis1.shape != dis2.shape:
        raise ValueError("dis1 and dis2 must have the same shape.")
    if method not in {"spearman", "pearson", "absDist"}:
        raise ValueError("method must be 'spearman', 'pearson', or 'absDist'.")

    # --- align to identical label order (sorted) like your current function ---
    samples = sorted(dis1.columns.tolist())
    dis1 = dis1.loc[samples, samples]
    dis2 = dis2.loc[samples, samples]

    # --- convert to ndarray and extract lower triangle (k=-1) once ---
    A = dis1.to_numpy(dtype=float, copy=False)
    B = dis2.to_numpy(dtype=float, copy=False)
    n = A.shape[0]
    if n < 2:
        # no pairs
        return 0.0 if getOnlyStat else [0.0, 1.0]

    tril_i, tril_j = np.tril_indices(n, k=-1)
    v1 = A[tril_i, tril_j].astype(float, copy=True)
    v2 = B[tril_i, tril_j].astype(float, copy=True)

    # --- observed statistic ---
    if method == "absDist":
        obs = float(np.mean(np.abs(v1 - v2)))
        if getOnlyStat:
            return obs
    else:
        if method == "pearson":
            # z-score once
            v1_mean, v1_std = v1.mean(), v1.std(ddof=1)
            v2_mean, v2_std = v2.mean(), v2.std(ddof=1)
            z1 = (v1 - v1_mean) / v1_std
            z2 = (v2 - v2_mean) / v2_std
            obs_r = float((z1 @ z2) / (z1.size - 1))
        else:  # spearman
            r1 = _rank_vector(v1)
            r2 = _rank_vector(v2)
            r1_mean, r1_std = r1.mean(), r1.std(ddof=1)
            r2_mean, r2_std = r2.mean(), r2.std(ddof=1)
            z1 = (r1 - r1_mean) / r1_std
            z2 = (r2 - r2_mean) / r2_std
            obs_r = float((z1 @ z2) / (z1.size - 1))
        # convert similarity -> dissimilarity like your code
        obs = 1.0 - obs_r
        if getOnlyStat:
            return obs

    # --- permutations (NumPy RNG, no pandas in loop) ---
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    null_stats = np.empty(permutations, dtype=float)

    if method == "absDist":
        # absDist doesn't need z-scores
        for b in range(permutations):
            perm = rng.permutation(n)
            v1p = A[perm][:, perm][tril_i, tril_j]
            null_stats[b] = np.mean(np.abs(v1p - v2))
    elif method == "pearson":
        # z2 fixed; z1 permutes as values permute (mean/std invariant)
        for b in range(permutations):
            perm = rng.permutation(n)
            v1p = A[perm][:, perm][tril_i, tril_j]
            z1p = (v1p - v1_mean) / v1_std
            r = (z1p @ z2) / (z1p.size - 1)
            null_stats[b] = 1.0 - r
    else:  # spearman
        # r2 fixed; need ranks of v1_perm each time (ties handled)
        for b in range(permutations):
            perm = rng.permutation(n)
            v1p = A[perm][:, perm][tril_i, tril_j]
            r1p = _rank_vector(v1p)
            # Pearson on ranks
            r1p_m, r1p_s = r1p.mean(), r1p.std(ddof=1)
            z1p = (r1p - r1p_m) / r1p_s
            r = (z1p @ z2) / (z1p.size - 1)
            null_stats[b] = 1.0 - r

    # one-sided p-value (proportion of permuted stats <= observed), +1 correction
    p = (np.sum(null_stats <= obs) + 1) / (permutations + 1)
    return [obs, p]

# -----------------------------------------------------------------------------
# Permanova
# -----------------------------------------------------------------------------
def permanova(
    dis: pd.DataFrame,
    meta: Union[pd.DataFrame, Dict[str, Any], Any],
    by: Union[str, List[str]],
    *,
    permutations: int = 999,
    include_interaction: bool = False,
    strata: Union[str, Sequence[str], None] = None,
    seed: int | None = None,
    perm_scheme: Literal["labels", "freedman-lane"] = "freedman-lane",
) -> Dict[str, Any]:
    """
    PERMANOVA (Anderson, 2001) via projection matrices on the Gower‑centered
    distance matrix. Supports one or two categorical factors (with optional
    interaction) and stratified permutations (blocks). Tests are partial
    (marginal), i.e., each term conditional on all other included terms.

    Parameters
    ----------
    dis : (n x n) pandas.DataFrame
        Symmetric distance/dissimilarity matrix with identical row/column labels.
    meta : DataFrame | dict | MicrobiomeData-like
        Metadata with rows indexed by sample IDs matching dis.index.
    by : str or list[str]
        One or two column names in `meta` defining the factor(s).
    permutations : int, default 999
        Number of permutations for the null distribution.
    include_interaction : bool, default False
        If len(by)==2 and both factors have >1 levels, include and test the interaction.
    strata : str | list[str] | None
        Column name(s) in `meta` defining permutation blocks (exchangeability strata).
        When given, permutations are performed within each stratum only.
    seed : int | None
        Random seed for reproducible permutations.
    perm_scheme : {'labels', 'freedman-lane'}, default 'labels'
        - 'labels': classical label permutations (your current implementation).
        - 'freedman-lane': residual-based permutation (permute residuals from the
          reduced model for each tested term and refit to pseudo-response). This
          makes main effects testable even when a factor is constant within strata.

    Returns
    -------
    dict
        {
          'by': [tested term names in order],
          'table': pandas.DataFrame with index=['Term(s)', 'Residual'] and columns:
                   ['df','SS','MS','F','p','R2'],
          'permutations': int,
          'strata': None | list[str],
          'perm_scheme': str
        }
    """
    # ---- validation / alignment ----
    M = get_df(meta, "meta")
    if not isinstance(M, pd.DataFrame) or M.empty:
        raise ValueError("`meta` must be a non-empty pandas DataFrame.")
    if not isinstance(dis, pd.DataFrame) or dis.shape[0] != dis.shape[1]:
        raise ValueError("`dis` must be a square pandas DataFrame.")
    if not all(dis.index == dis.columns):
        raise ValueError("`dis` must have identical row and column labels.")
    if isinstance(by, str):
        by = [by]
    if not (1 <= len(by) <= 2):
        raise ValueError("`by` must be a str or a list of length 1 or 2.")

    # Align metadata to the distance matrix order (do not reorder dis)
    in_common = list(set(M.index).intersection(dis.index))
    if len(in_common) < len(M.index) and len(in_common) == len(dis.index):
        M = M.loc[dis.index]
    elif len(in_common) < len(dis.index):
        raise ValueError("Index in meta and dis are not identical.")

    # Ensure categorical dtype for stable dummy coding
    for b in by:
        if b not in M.columns:
            raise ValueError(f"Column '{b}' not found in metadata.")
        if not isinstance(M[b].dtype, CategoricalDtype):
            M[b] = M[b].astype("category")

    # Normalize strata argument
    if strata is not None:
        if isinstance(strata, str):
            strata_cols = [strata]
        else:
            strata_cols = list(strata)
        if any(c not in M.columns for c in strata_cols):
            missing = [c for c in strata_cols if c not in M.columns]
            raise ValueError(f"strata columns not found in metadata: {missing}")
    else:
        strata_cols = None

    if perm_scheme == "labels" and strata_cols is not None:
        tested_terms = set(by) | (set([f"{by[0]}:{by[1]}"]) if len(by)==2 and include_interaction else set())
        if any(s in tested_terms for s in strata_cols):
            warnings.warn(
                "The 'strata' include a tested term; its permutations are fixed. "
                "P-values for that term will be uninformative (1.0/NaN).",
                UserWarning
            )

    # ---- helpers ----
    n = dis.shape[0]
    I = np.eye(n, dtype=float)

    def _rank(X: np.ndarray) -> int:
        return 0 if X.size == 0 else int(matrix_rank(X))

    def _hat(X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros((n, n), dtype=float)
        P = X @ pinv(X)  # numerically stable pseudo-hat
        return (P + P.T) / 2.0

    def _ss(H: np.ndarray, A: np.ndarray) -> float:
        return float(np.trace(H @ A))

    def _dummies(col: str, meta_df: pd.DataFrame) -> np.ndarray:
        Z = pd.get_dummies(meta_df[col], drop_first=True)
        return Z.to_numpy(dtype=float)

    def _interaction_products(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if X1.size == 0 or X2.size == 0:
            return np.empty((n, 0), dtype=float)
        return np.hstack(
            [X1[:, [i]] * X2[:, [j]] for i in range(X1.shape[1]) for j in range(X2.shape[1])]
        ).astype(float)

    def _build_terms(meta_df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
        terms: list[tuple[str, np.ndarray]] = []
        Z0 = np.ones((n, 1), dtype=float)  # Intercept
        terms.append(("Intercept", Z0))
        if len(by) == 1:
            terms.append((by[0], _dummies(by[0], meta_df)))
        else:
            v1, v2 = by
            X1 = _dummies(v1, meta_df)
            X2 = _dummies(v2, meta_df)
            terms.append((v1, X1))
            terms.append((v2, X2))
            if include_interaction and meta_df[v1].nunique() > 1 and meta_df[v2].nunique() > 1:
                X12 = _interaction_products(X1, X2)
                terms.append((f"{v1}:{v2}", X12))
        return terms

    # ---- Gower centering ----
    D2 = dis.to_numpy(dtype=float) ** 2
    np.fill_diagonal(D2, 0.0)
    D2 = (D2 + D2.T) / 2.0
    G = I - np.ones((n, n), dtype=float) / n
    A = -0.5 * (G @ D2 @ G)
    total_SS = float(np.trace(A))

    # ---- design & observed stats ----
    terms = _build_terms(M)
    X_all = np.concatenate([X for _, X in terms], axis=1) if terms else np.empty((n, 0))
    r_all = _rank(X_all)
    H_all = _hat(X_all)
    H_res = I - H_all
    SS_res = _ss(H_res, A)
    df_res = n - r_all

    if df_res <= 0:
        rows = []
        for name, X in terms:
            if name == "Intercept":
                continue
            SS_t = np.nan
            df_t = 0
            rows.append([name, df_t, SS_t, np.nan, np.nan, np.nan, np.nan])
        rows.append(["Residual", df_res, SS_res, np.nan, np.nan, np.nan,
                     (SS_res / total_SS) if total_SS > 0 else np.nan])
        out_df = pd.DataFrame(rows, columns=["term","df","SS","MS","F","p","R2"]).set_index("term")
        return {"by": [t for t in out_df.index if t != "Residual"],
                "table": out_df, "permutations": permutations,
                "strata": (strata_cols if strata_cols else None),
                "perm_scheme": perm_scheme}

    rows = []
    F_obs = []
    df_list = []
    # Cache per-term projections for re-use (also used by FL)
    per_term_info: dict[str, dict[str, np.ndarray | int]] = {}

    for i, (name_i, _) in enumerate(terms):
        if name_i == "Intercept":
            continue
        X_others = np.concatenate([X for j, (nm, X) in enumerate(terms) if j != i], axis=1) \
                   if len(terms) > 1 else np.empty((n, 0))
        r_others = _rank(X_others)
        H_others = _hat(X_others)
        H_term = (H_all - H_others + (H_all - H_others).T) / 2.0
        SS_t = _ss(H_term, A)
        df_t = r_all - r_others
        if df_t <= 0:
            MS_t = F_t = np.nan
        else:
            MS_t = SS_t / df_t
            MS_res = SS_res / df_res
            F_t = MS_t / MS_res
        R2_t = (SS_t / total_SS) if total_SS > 0 else np.nan
        rows.append([name_i, df_t, SS_t, MS_t, F_t, np.nan, R2_t])
        F_obs.append(F_t)
        df_list.append(df_t)

        # Store for later use (Freedman–Lane)
        per_term_info[name_i] = {
            "H_others": H_others,
            "H_term": H_term,
            "df_t": df_t,
        }

    rows.append(["Residual", df_res, SS_res, SS_res/df_res, np.nan, np.nan,
                 (SS_res / total_SS) if total_SS > 0 else np.nan])
    out_df = pd.DataFrame(rows, columns=["term","df","SS","MS","F","p","R2"]).set_index("term")

    # ---- permutations ----
    n_terms = len(F_obs)
    if permutations and n_terms > 0:
        rng = np.random.default_rng(seed)
        F_null = np.zeros((permutations, n_terms), dtype=float)

        # Helper: produce a permutation order (indices) respecting strata
        # (for label permutations we permute meta, for FL we permute residuals A_resid)
        index_to_pos = {lab: i for i, lab in enumerate(M.index)}

        def _permute_order_within_strata() -> np.ndarray:
            if strata_cols is None:
                return rng.permutation(n)
            order_list: list[int] = []
            for _, grp in M.groupby(strata_cols, sort=False, observed=True):
                pos = np.array([index_to_pos[idx] for idx in grp.index], dtype=int)
                if pos.size <= 1:
                    order_list.extend(pos.tolist())
                else:
                    perm_pos = pos[rng.permutation(pos.size)]
                    order_list.extend(perm_pos.tolist())
            return np.asarray(order_list, dtype=int)

        # ------- Branch 1: classical label permutations -------
        if perm_scheme == "labels":
            # (This is your existing block, kept intact except minor refactoring)
            def permute_meta_within_strata(M_in: pd.DataFrame) -> pd.DataFrame:
                if strata_cols is None:
                    order = rng.permutation(len(M_in))
                    return M_in.iloc[order]
                parts = []
                for _, grp in M_in.groupby(strata_cols, sort=False, observed=True):
                    idx = grp.index.to_numpy()
                    if idx.size <= 1:
                        parts.append(grp)
                    else:
                        perm = rng.permutation(idx.size)
                        parts.append(grp.iloc[perm])
                M_perm = pd.concat(parts, axis=0)
                return M_perm.loc[M_in.index]

            for b in range(permutations):
                M_perm = permute_meta_within_strata(M)
                terms_p = _build_terms(M_perm)
                X_all_p = np.concatenate([X for _, X in terms_p], axis=1) if terms_p else np.empty((n, 0))
                r_all_p = _rank(X_all_p)
                H_all_p = _hat(X_all_p)
                H_res_p = I - H_all_p
                SS_res_p = _ss(H_res_p, A)
                df_res_p = n - r_all_p
                if df_res_p <= 0:
                    F_null[b, :] = np.nan
                    continue
                term_names_p = [nm for nm, _ in terms_p if nm != "Intercept"]
                for i, name_i in enumerate(out_df.index[:-1]):  # tested terms
                    if name_i not in term_names_p:
                        F_null[b, i] = np.nan
                        continue
                    X_others_p = np.concatenate(
                        [X for (nm, X) in terms_p if nm not in ("Intercept", name_i)], axis=1
                    ) if len(terms_p) > 2 else np.empty((n, 0))
                    H_others_p = _hat(X_others_p)
                    H_term_p = (H_all_p - H_others_p + (H_all_p - H_others_p).T) / 2.0
                    SS_t_p = _ss(H_term_p, A)
                    df_t_p = _rank(X_all_p) - _rank(X_others_p)
                    if df_t_p <= 0:
                        F_null[b, i] = np.nan
                    else:
                        MS_t_p = SS_t_p / df_t_p
                        MS_res_p = SS_res_p / df_res_p
                        F_null[b, i] = MS_t_p / MS_res_p

        # ------- Branch 2: Freedman–Lane residual permutations -------
        else:  # perm_scheme == "freedman-lane" (default)
            # Precompute per-term components for FL: fitted (others) and residual (others)
            per_term_FL: dict[str, dict[str, np.ndarray | int]] = {}
            for name_i, info in per_term_info.items():
                H_others_i = info["H_others"]
                A_fit_others_i = H_others_i @ A @ H_others_i
                R = (I - H_others_i) @ A @ (I - H_others_i)  # residuals of reduced model
                # Symmetrize for numerical stability
                R = (R + R.T) / 2.0
                per_term_FL[name_i] = {
                    "A_fit_others": A_fit_others_i,
                    "R": R,
                    "H_term": info["H_term"],
                    "df_t": info["df_t"],
                }

            for b in range(permutations):
                order = _permute_order_within_strata()
                # For each term, build A* = A_fit(others) + P R P^T and compute F
                for i, name_i in enumerate(out_df.index[:-1]):
                    info = per_term_FL[name_i]
                    df_t_i = info["df_t"]
                    if df_t_i is None or df_t_i <= 0:
                        F_null[b, i] = np.nan
                        continue

                    # Permute residuals of the reduced model
                    R = info["R"]
                    R_perm = R[order, :][:, order]  # P R P^T
                    # Pseudo-response under H0 for the term
                    A_star = info["A_fit_others"] + R_perm

                    # Compute F using same H_term and H_res from observed design
                    SS_t_star = _ss(info["H_term"], A_star)
                    MS_t_star = SS_t_star / df_t_i
                    SS_res_star = _ss(H_res, A_star)
                    MS_res_star = SS_res_star / df_res
                    F_null[b, i] = MS_t_star / MS_res_star

        # ---- p-values with robustness checks ----
        for i, name in enumerate(out_df.index[:-1]):
            Fi = out_df.loc[name, "F"]
            df_t = out_df.loc[name, "df"]
            if not np.isfinite(Fi) or df_t <= 0:
                out_df.loc[name, "p"] = np.nan
                continue
            perm_vals = F_null[:, i]
            perm_vals = perm_vals[np.isfinite(perm_vals)]
            if perm_vals.size == 0:
                out_df.loc[name, "p"] = np.nan
                continue
            p = (np.sum(perm_vals >= Fi) + 1) / (perm_vals.size + 1)

            if df_t > 0:
                if perm_vals.size == 0:
                    out_df.loc[name, "p"] = np.nan
                    continue
                if np.unique(np.round(perm_vals, 12)).size == 1:
                    out_df.loc[name, "p"] = np.nan
                    continue
                if strata_cols is not None and any(
                    (col == name or name.startswith(col + ":")) and
                    any(len(g) <= 1 for _, g in M.groupby(strata_cols, observed=True))
                    for col in strata_cols
                ):
                    out_df.loc[name, "p"] = np.nan
                    continue
            out_df.loc[name, "p"] = p

    return {
        "by": [t for t in out_df.index if t != "Residual"],
        "table": out_df,
        "permutations": permutations,
        "strata": (strata_cols if strata_cols else None),
        "perm_scheme": perm_scheme,
    }

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
