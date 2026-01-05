"""
Ordination utilities: PCoA, db-RDA, and marginal (partial) permutation tests.

Public API:
    - pcoa_lingoes
    - dbrda
    - summarize_dbrda
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from numpy.linalg import svd, lstsq
from typing import Union, List, Optional, Any, Dict
from ..utils import get_df

__all__ = [
    "pcoa_lingoes",
    "dbrda",
    "marginal_factor_tests_dbrda",
    "summarize_dbrda"]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _encode_metadata_df(meta: pd.DataFrame, drop_first: bool = True):
    """
    Encode metadata into a design matrix X:
      - numeric columns: centered
      - categorical columns: dummy-coded (treatment coding if drop_first=True)

    Returns
    -------
    X_df : pandas.DataFrame
        Encoded design matrix aligned to meta.index.
    names : list[str]
        Column names of X_df.
    cont_mask : numpy.ndarray[bool]
        Mask for columns that came from numeric covariates (continuous).
    """
    num_cols = meta.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in meta.columns if c not in num_cols]

    X_num = pd.DataFrame(index=meta.index)
    if num_cols:
        X_num = meta[num_cols].astype(float)
        X_num = X_num - X_num.mean(axis=0)

    X_cat = pd.DataFrame(index=meta.index)
    if cat_cols:
        X_cat = pd.get_dummies(meta[cat_cols], drop_first=drop_first)

    X = pd.concat([X_num, X_cat], axis=1)
    if X.shape[1] == 0:
        raise ValueError("Metadata produced an empty design matrix.")
    names = list(X.columns)
    cont_mask = np.array([c in num_cols for c in names], dtype=bool)
    return X, names, cont_mask


def _hat_matrix(X: np.ndarray) -> np.ndarray:
    """Projection (hat) matrix for possibly rank-deficient X via SVD."""
    if X.size == 0 or X.shape[1] == 0:
        # No columns: projection is a zero matrix
        return np.zeros((X.shape[0], X.shape[0]), dtype=float)
    U, s, VT = svd(X, full_matrices=False)
    tol = np.finfo(float).eps * max(X.shape) * (s[0] if s.size > 0 else 0.0)
    s_inv = np.array([1.0/si if si > tol else 0.0 for si in s])
    X_pinv = (VT.T * s_inv) @ U.T
    return X @ X_pinv


def _residualize(M: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Residualize matrix M with respect to Z."""
    HZ = _hat_matrix(Z)
    return (np.eye(Z.shape[0]) - HZ) @ M


def _inertia_from_projection(H: np.ndarray, Y: np.ndarray) -> float:
    """
    Constrained inertia for multivariate response Y under projection H:
    sum of column variances of H @ Y (ddof=0).
    """
    fitted = H @ Y
    return float(np.sum(np.var(fitted, axis=0)))


def _build_design_with_interactions(meta: pd.DataFrame,
                                    interactions=None,
                                    drop_first: bool = True):
    """
    Build design matrix with main effects and optional categorical interactions.

    Parameters
    ----------
    meta : pd.DataFrame
        Explanatory variables (numeric + categorical).
    interactions : list[tuple[str, str]] | None
        Pairs of CATEGORICAL factor names for which to create interaction dummies.
        Example: [('Biochar', 'Plant')]
    drop_first : bool
        Treatment coding for dummies. Keep True unless you handle collinearity manually.

    Returns
    -------
    X_df : pd.DataFrame
        Design matrix with interaction columns appended (if any).
    groups : dict[str, list[str]]
        Mapping from factor name (and interaction name "A×B") to column names in X_df.
    """
    interactions = interactions or []
    X_df, names, cont_mask = _encode_metadata_df(meta, drop_first=drop_first)

    # Identify original categorical columns
    num_cols = meta.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in meta.columns if c not in num_cols]

    # Grouping: numeric 1:1, categorical by dummy prefix
    groups = {}
    for col in num_cols:
        if col in X_df.columns:
            groups[col] = [col]
    for col in cat_cols:
        cols = [c for c in X_df.columns if c.startswith(col + "_")]
        if cols:
            groups[col] = cols

    # Interactions only for categorical factors
    for (a, b) in interactions:
        if a not in cat_cols or b not in cat_cols:
            raise ValueError(f"Interactions are only supported between categorical factors. Got: {a}, {b}.")
        a_cols = [c for c in X_df.columns if c.startswith(a + "_")]
        b_cols = [c for c in X_df.columns if c.startswith(b + "_")]
        inter_cols = []
        for ac in a_cols:
            for bc in b_cols:
                new_name = f"{a}:{ac.split(a+'_',1)[1]}×{b}:{bc.split(b+'_',1)[1]}"
                X_df[new_name] = X_df[ac].values * X_df[bc].values
                inter_cols.append(new_name)
        if inter_cols:
            groups[f"{a}×{b}"] = inter_cols

    return X_df, groups

# -----------------------------------------------------------------------------
# PCoA
# -----------------------------------------------------------------------------
def pcoa_lingoes(
    dis: pd.DataFrame
) -> pd.DataFrame:
    """
    Perform Principal Coordinates Analysis (PCoA) using the Lingoes correction.

    The Lingoes correction transforms a non‑Euclidean distance matrix into a
    Euclidean one by adding a constant to all squared distances, ensuring that
    all eigenvalues are non‑negative. PCoA is then performed on the corrected
    matrix to obtain principal coordinate axes.

    Parameters
    ----------
    dis : pandas.DataFrame
        Square distance matrix (rows and columns represent samples). Values
        must be non‑negative and the matrix must be symmetric.

    Returns
    -------
    coords_df : pandas.DataFrame
        Principal coordinate scores (samples × axes), ordered by decreasing
        eigenvalue magnitude.
    eigvals : pandas.Series
        Eigenvalues associated with each axis (only the positive eigenvalues
        after Lingoes correction).
    pct_explained : pandas.Series
        Percentage of total variance explained by each axis (positive
        eigenvalues only).
    total_variance : float
        Sum of all positive eigenvalues after correction.

    Notes
    -----
    - The Lingoes correction is applied only if negative eigenvalues are
      detected.
    - The output coordinates are centered and scaled according to standard
      PCoA conventions.
    """

    if not isinstance(dis, pd.DataFrame):
        raise TypeError("dist_df must be a pandas DataFrame.")

    if dis.shape[0] != dis.shape[1]:
        raise ValueError("Distance matrix must be square.")

    # Copy & coerce to float
    D_df = dis.copy().astype(float)

    # Enforce symmetry and zero diagonal
    D_df = (D_df + D_df.T) / 2.0
    np.fill_diagonal(D_df.values, 0.0)
    if (D_df.values < 0).any():
        raise ValueError("Distances must be non‑negative.")

    labels = D_df.index.to_list()
    n = D_df.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n

    def double_center(d):
        # Gower centering on squared distances
        return -0.5 * J @ (d ** 2) @ J

    # Initial Gower matrix
    B = double_center(D_df.values)

    # Numerical symmetry safeguard before eigh
    B = (B + B.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(B)

    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Lingoes correction if negative eigenvalues exist
    if eigvals.min() < 0:
        c = 2.0 * abs(eigvals.min())                 # Lingoes: add 2|λ_min| to squared distances
        D_corr = np.sqrt(np.maximum(D_df.values**2 + c, 0.0))
        B = double_center(D_corr)
        B = (B + B.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(B)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

    # Keep positive eigenvalues (allow tiny negative due to round‑off)
    tol = max(1e-12, 1e-12 * np.max(np.abs(eigvals)))
    pos = eigvals > tol
    if not np.any(pos):
        raise ValueError("No positive eigenvalues after Lingoes correction; check the distance matrix.")

    eigvals_pos = eigvals[pos]
    eigvecs_pos = eigvecs[:, pos]

    # Coordinates scaled by sqrt(λ)
    coords = eigvecs_pos * np.sqrt(eigvals_pos)

    axis_names = [f"PCo{i+1}" for i in range(len(eigvals_pos))]
    coords_df = pd.DataFrame(coords, index=labels, columns=axis_names)
    eigvals_series = pd.Series(eigvals_pos, index=axis_names)
    explained_var_series = pd.Series((eigvals_pos / eigvals_pos.sum()) * 100.0, index=axis_names).round(2)
    out = {
        'site_scores': coords_df,
        'eigenvalues': eigvals_series,
        'pct_explained': explained_var_series,
        'total_variance': eigvals_series.sum()}
    return out

# -----------------------------------------------------------------------------
# db-RDA (global model + scores + per-variable contributions)
# -----------------------------------------------------------------------------
def dbrda(
    dis: pd.DataFrame = None,
    meta: Union[pd.DataFrame, Dict[str, Any], Any] = None,
    *,
    by: Optional[Union[str, List[str]]] = None,
    condition: Optional[pd.DataFrame] = None,
    n_axes: int = 2,
    scale: str = "site",
    perm_n: int = 999,
    perm_seed: int = 42,
    pcoa_fn=pcoa_lingoes,
    per_var_perm: bool = False,
    interactions: Optional[List[str]] = None,
    drop_first: bool = True
) -> Dict[str, Any]:
    """
    Distance‑based Redundancy Analysis (db‑RDA).

    This function performs constrained ordination on a distance matrix by:
    1. Converting the distance matrix into principal coordinates (PCoA)
       using the specified PCoA function (default: Lingoes correction).
    2. Regressing the PCoA coordinates onto explanatory variables.
    3. Extracting constrained axes, biplot scores, and variance components.
    4. Performing a global permutation test (Freedman–Lane).
    5. Optionally computing per‑variable permutation p‑values.
    6. Optionally including categorical interaction terms.

    Parameters
    ----------
    dis : pandas.DataFrame
        Square distance matrix (samples × samples). Must have matching row/column labels.
    meta : pandas.DataFrame
        Metadata table containing explanatory variables (rows = samples).
    by : str or list of str, optional
        Subset of metadata columns to use as explanatory variables.
        If None, all columns in `meta` are used.
    condition : pandas.DataFrame, optional
        Conditioning variables for partial db‑RDA. Must align with `meta`.
    n_axes : int, default=2
        Number of constrained axes to return.
    scale : {'site', 'species'}, default='site'
        Scaling for biplot scores.
    perm_n : int, default=999
        Number of permutations for the global test.
    perm_seed : int, default=42
        Random seed for reproducibility.
    pcoa_fn : callable, default=pcoa_lingoes
        Function used to compute PCoA. Must return a dict with
        'site_scores' and 'eigenvalues'.
    per_var_perm : bool, default=False
        If True, compute permutation p‑values for each predictor.
    interactions : list of str, optional
        Variables for which interaction terms should be generated.
    drop_first : bool, default=True
        Whether to drop the first dummy level when encoding categorical variables.

    Returns
    -------
    dict
        {
            'site_scores' : pandas.DataFrame,
            'biplot_scores' : pandas.DataFrame,
            'variable_contributions' : pandas.DataFrame,
            'eigenvalues' : numpy.ndarray,
            'explained_ratio' : numpy.ndarray,
            'total_inertia' : float,
            'constrained_inertia' : float,
            'unconstrained_inertia' : float,
            'F_global' : float,
            'p_global' : float
        }

    Notes
    -----
    - The global permutation test uses the Freedman–Lane procedure.
    - Partial db‑RDA is performed by residualizing both the response
      coordinates and the design matrix against the conditioning variables.
    - Interaction terms are constructed before dummy encoding.
    """

    meta = get_df(meta, "meta")
    if meta is None:
        raise ValueError('meta is missing.')

    # Select metadata columns
    if by is None:
        meta_use = meta.copy()
    elif isinstance(by, str):
        meta_use = meta[[by]].copy()
    elif isinstance(by, list):
        meta_use = meta[by].copy()
    else:
        raise TypeError("`by` must be None, a string, or a list of strings.")

    # Align metadata to distance matrix
    if not isinstance(dis, pd.DataFrame):
        raise TypeError("`dis` must be a pandas DataFrame.")

    if not meta_use.index.equals(dis.index):
        meta_use = meta_use.loc[dis.index]

    # PCoA
    pcoa_res = pcoa_fn(dis)
    coords = pcoa_res["site_scores"]
    eig_vals = pcoa_res["eigenvalues"]

    U_all = coords.values.astype(float)
    sample_index = coords.index
    total_inertia = float(np.sum(eig_vals))

    # Build design matrix (with optional interactions)
    if interactions:
        X_df, _ = _build_design_with_interactions(
            meta_use, interactions=interactions, drop_first=drop_first
        )
        xnames = list(X_df.columns)
    else:
        X_df, xnames, _ = _encode_metadata_df(meta_use, drop_first=drop_first)

    X = X_df.values.astype(float)

    # Partial db‑RDA (conditioning)
    if condition is not None:
        if condition.shape[0] != meta_use.shape[0]:
            raise ValueError("`condition` must have same number of rows as `meta`.")

        Z_df, _, _ = _encode_metadata_df(condition.loc[meta_use.index], drop_first=drop_first)
        Z = Z_df.values.astype(float)

        U_basis = _residualize(U_all, Z)
        X_basis = _residualize(X, Z)
    else:
        U_basis = U_all
        X_basis = X

    # Projection onto constrained space
    H = _hat_matrix(X_basis)
    fitted = (H @ U_basis).astype(float)

    # Constrained covariance matrix
    if fitted.shape[1] == 1:
        warnings.warn(
            "Only one positive eigenvalue detected. dbRDA will return a single axis.",
            UserWarning
        )
        Cc = np.array([[np.var(fitted[:, 0], ddof=0)]])
    else:
        Cc = np.cov(fitted.T, bias=True)

    # Eigen decomposition
    lam, V = np.linalg.eigh(Cc)
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    V = V[:, order]

    sites_all = fitted @ V
    a = min(n_axes, sites_all.shape[1])
    eigvals = lam[:a]
    sites = sites_all[:, :a]
    constrained_inertia = float(np.sum(lam))
    explained_ratio = eigvals / total_inertia

    # Biplot scores
    biplot = []
    for ax_i in range(a):
        coef, _, _, _ = lstsq(X_basis, sites[:, ax_i], rcond=None)
        biplot.append(coef)
    biplot = np.array(biplot).T

    if scale == "species":
        maxlen = np.max(np.linalg.norm(biplot, axis=0))
        if maxlen > 0:
            biplot = biplot / maxlen

    # Global permutation test (Freedman–Lane)
    Hr = _hat_matrix(np.zeros((X_basis.shape[0], 0)))
    Hf = _hat_matrix(X_basis)

    Fitted0 = Hr @ U_basis
    E = U_basis - Fitted0

    df_model = int(np.linalg.matrix_rank(X_basis))
    df_resid = int(U_basis.shape[0] - df_model - 1)

    I_f = _inertia_from_projection(Hf, U_basis)
    I_unconstr = total_inertia - I_f

    F_obs = (I_f / max(df_model, 1)) / (I_unconstr / max(df_resid, 1))

    rng = np.random.RandomState(perm_seed)
    hits = 0

    for _ in range(perm_n):
        perm = rng.permutation(U_basis.shape[0])
        Y_star = Fitted0 + E[perm, :]
        I_f_star = _inertia_from_projection(Hf, Y_star)
        F_star = (I_f_star / max(df_model, 1)) / (
            (total_inertia - I_f_star) / max(df_resid, 1)
        )
        hits += (F_star >= F_obs)

    pval_global = (hits + 1) / (perm_n + 1)

    # Per‑variable contributions
    contributions = []
    pvals = []

    for i in range(X_basis.shape[1]):
        Xi = X_basis[:, [i]]
        Hi = _hat_matrix(Xi)
        inertia_i = float(np.sum(np.var(Hi @ U_basis, axis=0)))
        contributions.append(100.0 * inertia_i / total_inertia)

        if per_var_perm:
            h = 0
            for _ in range(perm_n):
                Xp = Xi[rng.permutation(Xi.shape[0]), :]
                Hp = _hat_matrix(Xp)
                inertia_p = float(np.sum(np.var(Hp @ U_basis, axis=0)))
                if inertia_p >= inertia_i:
                    h += 1
            pvals.append((h + 1) / (perm_n + 1))
        else:
            pvals.append(None)

    # Output
    axis_names = [f"dbRDA{i+1}" for i in range(a)]

    return {
        "site_scores": pd.DataFrame(sites, index=sample_index, columns=axis_names),
        "biplot_scores": pd.DataFrame(biplot, index=xnames, columns=axis_names),
        "variable_contributions": pd.DataFrame({
            "Predictor": xnames,
            "pct_explained": contributions,
            "p-value": pvals
        }),
        "eigenvalues": eigvals,
        "explained_ratio": explained_ratio,
        "total_inertia": total_inertia,
        "constrained_inertia": constrained_inertia,
        "unconstrained_inertia": total_inertia - constrained_inertia,
        "F_global": F_obs,
        "p_global": pval_global
    }

# -----------------------------------------------------------------------------
# Marginal (partial) permutation tests (Freedman–Lane)
# -----------------------------------------------------------------------------
def marginal_factor_tests_dbrda(
    dis: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    by: Optional[Union[str, List[str]]] = None,
    condition: Optional[pd.DataFrame] = None,
    interactions: Optional[List[str]] = None,
    pcoa_fn=pcoa_lingoes,
    perm_n: int = 999,
    perm_seed: int = 42,
    drop_first: bool = True,
    return_F: bool = True
) -> pd.DataFrame:
    """
    Marginal (partial) permutation tests for db‑RDA factors.

    Performs Freedman–Lane marginal tests for each factor block in the design
    matrix, controlling for all other terms. Also computes diagnostics based on
    the factor block alone (XG):

        - inertia_alone
        - pct_explained (alone)
        - p-alone (simple permutation test ignoring other terms)

    Parameters
    ----------
    dis : pandas.DataFrame
        Square distance matrix (samples × samples).
    meta : pandas.DataFrame
        Metadata table (rows = samples).
    by : str or list of str, optional
        Subset of metadata columns to include as explanatory variables.
        If None, all columns in `meta` are used.
    condition : pandas.DataFrame, optional
        Conditioning variables for partial db‑RDA.
    interactions : list of str, optional
        Variables for which interaction terms should be generated.
    pcoa_fn : callable, default=pcoa_lingoes
        Function returning {'site_scores', 'eigenvalues'}.
    perm_n : int, default=999
        Number of permutations.
    perm_seed : int, default=42
        Random seed.
    drop_first : bool, default=True
        Whether to drop the first dummy level when encoding categorical variables.
    return_F : bool, default=True
        If True, compute F‑statistics; otherwise use raw delta inertia.

    Returns
    -------
    pandas.DataFrame
        Columns:
            Factor
            df_added
            delta_inertia
            pct_explained (marginal)
            inertia_alone
            pct_explained (alone)
            F
            p-marginal
            p-alone
    """

    # Select metadata columns
    if by is None:
        meta_use = meta.copy()
    elif isinstance(by, str):
        meta_use = meta[[by]].copy()
    elif isinstance(by, list):
        meta_use = meta[by].copy()
    else:
        raise TypeError("`by` must be None, a string, or a list of strings.")

    # Align metadata to distance matrix
    if not meta_use.index.equals(dis.index):
        meta_use = meta_use.loc[dis.index]

    # PCoA
    pcoa_res = pcoa_fn(dis)
    coords = pcoa_res["site_scores"]
    eig_vals = pcoa_res["eigenvalues"]

    Y_all = coords.values.astype(float)
    total_inertia = float(np.sum(eig_vals))
    n = Y_all.shape[0]

    # Build design matrix
    X_df, groups = _build_design_with_interactions(
        meta_use, interactions=interactions, drop_first=drop_first
    )
    X_all = X_df.values.astype(float)

    # Conditioning (partial db‑RDA)
    if condition is not None:
        if condition.shape[0] != meta_use.shape[0]:
            raise ValueError("`condition` must have same number of rows as `meta`.")

        Z_df, _, _ = _encode_metadata_df(condition.loc[meta_use.index], drop_first=drop_first)
        Z = Z_df.values.astype(float)

        Y = _residualize(Y_all, Z)
        X = _residualize(X_all, Z)
    else:
        Y = Y_all
        X = X_all

    rng = np.random.default_rng(perm_seed)
    rows = []

    def rank(A):
        return int(np.linalg.matrix_rank(A)) if A.size > 0 else 0

    # Loop over each factor block
    for factor, cols in groups.items():

        # Identify columns belonging to this factor
        idx = [X_df.columns.get_loc(c) for c in cols]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[idx] = True

        XG = X[:, mask]       # factor block
        Xr = X[:, ~mask]      # reduced model without factor

        rk_r = rank(Xr)
        rk_full = rank(X)
        df_added = rk_full - rk_r

        # Aliased factor (no degrees of freedom)
        if df_added <= 0:
            rows.append({
                "Factor": factor,
                "df_added": 0,
                "delta_inertia": 0.0,
                "pct_explained (marginal)": 0.0,
                "inertia_alone": 0.0,
                "pct_explained (alone)": 0.0,
                "F": np.nan,
                "p-marginal": np.nan,
                "p-alone": np.nan,
                "note": "aliased (df_added=0)"
            })
            continue

        # Marginal contribution (Freedman–Lane)
        Hr = _hat_matrix(Xr)
        Hf = _hat_matrix(X)

        I_r = _inertia_from_projection(Hr, Y)
        I_f = _inertia_from_projection(Hf, Y)
        delta_obs = I_f - I_r

        df_resid = n - rk_full - 1

        if return_F:
            denom = (total_inertia - I_f) / max(df_resid, 1)
            F_obs = (delta_obs / df_added) / denom if denom > 0 else np.inf
        else:
            F_obs = np.nan

        # Permutation test (marginal)
        Fitted_r = Hr @ Y
        E = Y - Fitted_r
        hits = 0

        for _ in range(perm_n):
            perm = rng.permutation(n)
            Y_star = Fitted_r + E[perm, :]

            I_r_star = _inertia_from_projection(Hr, Y_star)
            I_f_star = _inertia_from_projection(Hf, Y_star)
            delta_star = I_f_star - I_r_star

            if return_F:
                denom_star = (total_inertia - I_f_star) / max(df_resid, 1)
                F_star = (delta_star / df_added) / denom_star if denom_star > 0 else np.inf
                if F_star >= F_obs:
                    hits += 1
            else:
                if delta_star >= delta_obs:
                    hits += 1

        pval_marginal = (hits + 1) / (perm_n + 1)

        # Diagnostics: factor alone (XG)
        Hi = _hat_matrix(XG)
        inertia_alone = _inertia_from_projection(Hi, Y)
        pct_alone = 100.0 * inertia_alone / total_inertia

        # Simple permutation test for factor alone
        hits_alone = 0
        for _ in range(perm_n):
            perm = rng.permutation(n)
            inertia_star = _inertia_from_projection(Hi, Y[perm, :])
            if inertia_star >= inertia_alone:
                hits_alone += 1

        pval_alone = (hits_alone + 1) / (perm_n + 1)

        rows.append({
            "Factor": factor,
            "df_added": int(df_added),
            "delta_inertia": float(delta_obs),
            "pct_explained (marginal)": 100.0 * float(delta_obs) / total_inertia,
            "inertia_alone": float(inertia_alone),
            "pct_explained (alone)": pct_alone,
            "F": float(F_obs),
            "p-marginal": float(pval_marginal),
            "p-alone": float(pval_alone)
        })

    # Output
    out = (
        pd.DataFrame(rows)
        .sort_values(["p-marginal", "pct_explained (marginal)"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return out

# -----------------------------------------------------------------------------
# Summarize db-RDA
# -----------------------------------------------------------------------------
def summarize_dbrda(
    dis: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    by: Optional[Union[str, List[str]]] = None,
    condition: Optional[pd.DataFrame] = None,
    interactions: Optional[List[str]] = None,
    pcoa_fn=pcoa_lingoes,
    perm_n: int = 999,
    perm_seed: int = 42,
    drop_first: bool = True,
    include_interpretation: bool = True,
    include_alone: bool = True
) -> pd.DataFrame:
    """
    Summarize db‑RDA (global model + marginal factor tests).

    This function:
      1. Runs dbRDA once (global model).
      2. Runs marginal (partial) permutation tests per factor (Freedman–Lane).
      3. Aggregates % explained by original factors (from the full model).
      4. Computes R² and adjusted R².
      5. Returns a tidy DataFrame, optionally with textual interpretation.

    Parameters
    ----------
    dist : pandas.DataFrame
        Square distance matrix (rows/cols = samples). Index must match columns.
    meta : pandas.DataFrame
        Metadata indexed by sample IDs.
    by : str or list of str, optional
        Subset of metadata columns to use as explanatory variables.
        If None, all columns in `meta` are used.
    condition : pandas.DataFrame, optional
        Covariates to partial out (same index as `meta`).
    interactions : list of str, optional
        Variables for which interaction terms should be generated.
    pcoa_fn : callable, default=pcoa_lingoes
        Function for the PCoA step; must return 'site_scores' and 'eigenvalues'.
    perm_n : int, default=999
        Number of permutations for marginal tests.
    perm_seed : int, default=42
        Random seed for permutations.
    drop_first : bool, default=True
        Drop first level in categorical encoding (reference coding).
    include_interpretation : bool, default=True
        If True, adds a textual interpretation column.
    include_alone : bool, default=True
        If True, keeps “alone” diagnostics (factor-alone %-explained, p-alone).

    Returns
    -------
    pandas.DataFrame
        Columns (by default):
          - Factor
          - pct_explained (full model)
          - df_added
          - delta_inertia
          - pct_explained (marginal)
          - F
          - p-marginal
          - inertia_alone
          - pct_explained (alone)
          - p-alone
          - Interpretation (optional)

        Attributes (df.attrs):
          - 'R²'                    : float
          - 'Adjusted R²'           : float
          - 'F_global'              : float
          - 'p_global'              : float
          - 'Total inertia'         : float
          - 'Constrained inertia'   : float
          - 'Unconstrained inertia' : float
          - 'n'                     : int (samples)
          - 'df_model'              : int (approx. number of fitted parameters)
    """

    # ---------------------------
    # Run dbRDA (global model)
    # ---------------------------
    res = dbrda(
        dis=dis,
        meta=meta,
        by=by,
        condition=condition,
        interactions=interactions,
        pcoa_fn=pcoa_fn,
        per_var_perm=False,   # keep dbRDA fast; marginal perms are run below
        drop_first=drop_first,
        perm_n=perm_n,
        perm_seed=perm_seed
    )

    # Sample size and df_model (approx.)
    n = dis.shape[0] if isinstance(dis, pd.DataFrame) else len(dis)

    try:
        df_model = int(np.linalg.matrix_rank(res["biplot_scores"].values))
    except Exception:
        df_model = int(res.get("biplot_scores", pd.DataFrame()).shape[1])

    # Global fit metrics
    total_inertia = float(res["total_inertia"])
    constr_inertia = float(res["constrained_inertia"])
    unconstr_inertia = float(res["unconstrained_inertia"])
    R2 = constr_inertia / total_inertia if total_inertia > 0 else 0.0

    # Adjusted R²
    denom = (n - df_model - 1)
    if denom > 0:
        adjR2 = 1.0 - (1.0 - R2) * ((n - 1.0) / denom)
    else:
        adjR2 = np.nan
    if isinstance(adjR2, float) and adjR2 < 0 and adjR2 > -1e-12:
        adjR2 = 0.0

    # Aggregate %-Explained by original factors (full model)
    vc = res["variable_contributions"].copy()
    if not {"Predictor", "pct_explained"}.issubset(vc.columns):
        raise ValueError(
            "`dbrda` result missing expected 'variable_contributions' columns."
        )

    # Use only selected columns for factor grouping
    if by is None:
        meta_use = meta
    elif isinstance(by, str):
        meta_use = meta[[by]]
    else:
        meta_use = meta[by]

    vc["Predictor"] = vc["Predictor"].astype(str)
    num_cols = meta_use.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in meta_use.columns if c not in num_cols]

    rows: List[Dict[str, Any]] = []

    # Numeric predictors: aggregate by exact name
    for col in num_cols:
        mask = (vc["Predictor"] == col)
        if mask.any():
            rows.append({
                "Factor": col,
                "pct_explained (full model)": float(vc.loc[mask, "pct_explained"].sum())
            })

    # Categorical predictors: aggregate across one-hot columns (prefix match)
    for col in cat_cols:
        mask = vc["Predictor"].str.startswith(col + "_")
        if mask.any():
            rows.append({
                "Factor": col,
                "pct_explained (full model)": float(vc.loc[mask, "pct_explained"].sum())
            })

    # Optional: interactions aggregation (if naming convention matches)
    if interactions:
        for a in interactions:
            # if interactions is list[str], treat each as name used in design
            mask = vc["Predictor"].str.contains(a)
            if mask.any():
                rows.append({
                    "Factor": a,
                    "pct_explained (full model)": float(vc.loc[mask, "pct_explained"].sum())
                })

    agg_full = pd.DataFrame(rows)

    if agg_full.empty:
        # Fallback: group by a simple "root" of predictor name
        def _root(p: str) -> str:
            for sep in ["×", ":", "_"]:
                if sep in p:
                    return p.split(sep)[0]
            return p

        tmp = vc.groupby(vc["Predictor"].map(_root))["pct_explained"].sum().reset_index()
        tmp.columns = ["Factor", "pct_explained (full model)"]
        agg_full = tmp

    # Marginal tests (per factor)
    mt = marginal_factor_tests_dbrda(
        dis=dis,
        meta=meta,
        by=by,
        condition=condition,
        interactions=interactions,
        pcoa_fn=pcoa_fn,
        perm_n=perm_n,
        perm_seed=perm_seed,
        drop_first=drop_first,
        return_F=True
    )

    # Merge and finalize
    out = mt.merge(agg_full, on="Factor", how="left")

    if "pct_explained (full model)" not in out.columns:
        out["pct_explained (full model)"] = np.nan

    # Optional interpretation column
    if include_interpretation:
        def _interpret(row: pd.Series) -> str:
            marginal = row.get("pct_explained (marginal)", np.nan)
            alone = row.get("pct_explained (alone)", np.nan)
            p_marg = row.get("p-marginal", np.nan)
            p_alone = row.get("p-alone", np.nan)

            if pd.notna(p_marg) and p_marg < 0.05:
                return f"Significant unique effect (marginal {marginal:.2f}%)"
            if pd.notna(p_alone) and p_alone < 0.05:
                return f"Significant alone ({alone:.2f}%), overlaps with others"
            return "Not significant; effect likely weak or redundant"

        out["Interpretation"] = out.apply(_interpret, axis=1)

    # Column order
    keep_cols = [
        "Factor",
        "pct_explained (full model)",
        "df_added",
        "delta_inertia",
        "pct_explained (marginal)",
        "F",
        "p-marginal",
    ]
    if include_alone:
        keep_cols += ["inertia_alone", "pct_explained (alone)", "p-alone"]
    if include_interpretation:
        keep_cols += ["Interpretation"]

    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].copy()

    # Sort by significance then effect size
    sort_keys = [k for k in ["p-marginal", "pct_explained (marginal)"] if k in out.columns]
    sort_asc = [True, False][:len(sort_keys)]
    if sort_keys:
        out = out.sort_values(sort_keys, ascending=sort_asc).reset_index(drop=True)

    # Attach attributes
    out.attrs["R²"] = round(float(R2), 4)
    out.attrs["Adjusted R²"] = None if pd.isna(adjR2) else round(float(adjR2), 4)
    out.attrs["F_global"] = float(res.get("F_global", np.nan))
    out.attrs["p_global"] = float(res.get("p_global", np.nan))
    out.attrs["Total inertia"] = round(total_inertia, 6)
    out.attrs["Constrained inertia"] = round(constr_inertia, 6)
    out.attrs["Unconstrained inertia"] = round(unconstr_inertia, 6)
    out.attrs["n"] = int(n)
    out.attrs["df_model"] = int(df_model)

    return out
