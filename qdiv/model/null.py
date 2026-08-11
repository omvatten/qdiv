import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Any, Literal, Tuple
from ..diversity import bray, jaccard, naive_beta, phyl_beta, func_beta
from ..utils import get_df

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
        def __init__(self, iterable=None, total=None, desc=None, unit=None, leave=False, 
                     ncols=None, ascii=True, mininterval=None, position=None, miniters=None):
            self._iterable = iterable if iterable is not None else range(total or 0)

        def __iter__(self):
            return iter(self._iterable)

        def update(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

    return _DummyTqdm

# ---------------------------------------------------------------------------
# RCQ: Null comparisons for beta-diversity
# ---------------------------------------------------------------------------
def rcq(
    obj: Union[Dict[str, Any], Any],
    *,
    constrain_by: Optional[str] = None,
    randomization: Literal["frequency", "abundance"] = "frequency",
    iterations: int = 999,
    div_type: Literal["Jaccard", "Bray", "naive", "phyl", "func"] = "naive",
    distmat: Optional[pd.DataFrame] = None,
    q: float = 1.0,
    use_tqdm: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    use_numba: bool = True,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Raup–Crick-style null comparisons for beta-diversity.

    Randomizes the abundance table while preserving each sample's richness and
    total reads, then contrasts the observed beta-diversity matrix against a
    null distribution built via randomization.

    Parameters
    ----------
    obj : MicrobiomeData | dict | Any
        Input with at least an abundance table under key 'tab'. Optionally may include
        'meta' (sample metadata) and 'tree' (for phylogenetic measures).
    constrain_by : str, optional
        Column in metadata to constrain randomization within categories; if None, randomize across all samples.
    randomization : {"frequency", "abundance"}, default="frequency"
        Randomization strategy for selecting the set of taxa per randomized sample:
          - "abundance": probabilities proportional to group-level summed abundances
          - "frequency": probabilities proportional to group-level presence frequency
        Within the selected set, additional reads are allocated proportional to
        the selected taxa's group-level abundances to match each sample's total reads.
    iterations : int, default=999
        Number of randomization iterations used to build the null distribution.
    div_type : {"Jaccard", "Bray", "naive", "phyl", "func"}, default="naive"
        Dissimilarity index to compute for observed and null tables.
        - "Jaccard", "Bray": classic indices on the (randomized) count table
        - "naive": Hill-number-based (requires q)
        - "phyl": phylogenetic beta diversity (requires 'tree' in obj)
        - "func": functional beta diversity (requires distmat)
    distmat : pandas.DataFrame, optional
        Square functional distance matrix (features × features); required if div_type="func".
    q : float, default=1.0
        Diversity order for Hill-number-based indices (used by "naive", "phyl", "func").
    use_tqdm : bool, default=True
        Use `tqdm` for progress bars.
    random_state : int | numpy.random.Generator, optional
        Random seed or Generator for reproducibility.
    use_numba : bool, optional
        If True, uses Numba path; otherwise uses pure Python implementation.

    Returns
    -------
    dict
        {
          "div_type": str,
          "obs_d":    DataFrame (S × S), observed beta-diversity,
          "p":        DataFrame (S × S), Raup–Crick probability  P(null < obs) + 0.5·P(null == obs),
          "null_mean":DataFrame (S × S), mean of null,
          "null_std": DataFrame (S × S), std of null,
          "ses":      DataFrame (S × S), (null_mean - obs) / null_std
        }

    Notes
    -----
    - Per-sample constraints: if `constrain_by` is given, randomization is performed within each
      metadata category independently to preserve structure. Otherwise, all samples are randomized together.
    - Richness & read preservation: for each sample, we draw a set of taxa matching the original
      richness, then allocate extra reads to match the original total reads.
    - Raup–Crick p-index: counts how often the null dissimilarity is strictly lower than observed,
      ties contribute 0.5, normalized by `iterations`.
    - A p value close to zero means observed dissimilarity is lower than the null expectation.
    - A p value close to one means observed dissimilarity is higher than the null expectation.
    - A positive ses means observed dissimilarity is lower than the null expectation.
    - A negative ses means observed dissimilarity is higher than the null expectation.
    """
    if "seed" in kwargs:
        if random_state is not None:
            raise TypeError("Specify only one of 'random_state' or 'seed'.")
        random_state = kwargs.pop("seed")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

    # --- Extract tables & context
    tab = get_df(obj, "tab")
    if tab is None:
        raise ValueError("'tab' is needed in input.")
    tab = tab.copy()
    if tab.empty:
        raise ValueError("'tab' must be a non-empty DataFrame.")

    meta = None
    tree = None
    if hasattr(obj, "meta") or (isinstance(obj, dict) and "meta" in obj):
        try:
            meta = get_df(obj, "meta")
        except Exception:
            meta = None
    if div_type == "phyl":
        tree = get_df(obj, "tree")

    if div_type == "func":
        if not isinstance(distmat, pd.DataFrame):
            raise ValueError("div_type='func' requires a pandas DataFrame 'distmat'.")
        # Align to feature order
        distmat = distmat.loc[tab.index, tab.index].copy()

    if constrain_by is not None:
        if meta is None or constrain_by not in meta.columns:
            raise ValueError("constrain_by requires a metadata DataFrame containing the specified column.")

    if iterations < 1:
        raise ValueError("iterations must be >= 1.")
    if randomization not in {"abundance", "frequency"}:
        raise ValueError("randomization must be 'abundance' or 'frequency'.")

    # --- RNG + tqdm
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    tqdm = _get_tqdm(use_tqdm)

    # --- Partition samples into constraint groups
    if constrain_by is None:
        groups = [tab.columns.tolist()]
    else:
        cats = pd.unique(meta[constrain_by])
        groups = [meta.index[meta[constrain_by] == cat].tolist() for cat in cats]

    # --- Helper: compute beta-diversity for a table (dispatch)
    def _beta_for_table(t: pd.DataFrame) -> pd.DataFrame:
        if div_type.lower() == "bray":
            return bray(t)
        if div_type.lower() == "jaccard":
            return jaccard(t)
        if div_type == "naive":
            return naive_beta(t, q=q, use_numba=use_numba)
        if div_type == "phyl":
            return phyl_beta({"tab": t, "tree": tree}, q=q, use_numba=use_numba)
        if div_type == "func":
            return func_beta(t, distmat, q=q, use_numba=use_numba, use_tqdm=False)
        raise ValueError("Unsupported div_type. Choose among {'Jaccard','Bray','naive','phyl','func'}.")

    # --- Observed beta-diversity
    obs_beta = _beta_for_table(tab)
    obs_arr = obs_beta.to_numpy()

    # --- Streaming accumulators (S × S)
    mu = np.zeros_like(obs_arr, dtype=np.float64)   # null mean
    M2 = np.zeros_like(obs_arr, dtype=np.float64)   # for variance
    count_lt = np.zeros_like(obs_arr, dtype=np.int64)  # p-index counts (null < obs)
    count_eq = np.zeros_like(obs_arr, dtype=np.int64)  # p-index ties

    # --- Convert table to NumPy for faster iteration-loop randomization
    if not np.all(np.isfinite(tab.to_numpy(dtype=float))):
        raise ValueError("'tab' contains non-finite values.")
    
    tab_float = tab.to_numpy(dtype=float)
    
    if not np.allclose(tab_float, np.round(tab_float)):
        raise ValueError(
            "rcq requires an integer count table because richness and total reads "
            "are preserved during randomization."
        )

    tab_arr = tab.to_numpy(dtype=np.int64)
    n_features, n_samples = tab_arr.shape
    
    feature_index = np.arange(n_features)
    sample_names_all = tab.columns.tolist()
    feature_names_all = tab.index
    
    smp_index = {smp: i for i, smp in enumerate(sample_names_all)}
    
    # per-sample richness and read totals, preserved by randomization
    richness_vec = (tab_arr > 0).sum(axis=0).astype(np.int64)
    reads_vec = tab_arr.sum(axis=0).astype(np.int64)
    
    # --- Precompute group-level probabilities as NumPy arrays
    group_specs = []
    
    for smp_list in groups:
        smp_idx = np.array([smp_index[smp] for smp in smp_list], dtype=np.int64)
        subarr = tab_arr[:, smp_idx]
    
        # Group-level abundances, used for allocating reads within selected taxa
        abund_arr = subarr.sum(axis=1).astype(float)
    
        if randomization == "abundance":
            total = abund_arr.sum()
    
            if total > 0:
                sel_p = abund_arr / total
            else:
                sel_p = np.full(n_features, 1.0 / n_features, dtype=float)
    
        else:  # randomization == "frequency"
            freq_counts = (subarr > 0).sum(axis=1).astype(float)
            total = freq_counts.sum()
    
            if total > 0:
                sel_p = freq_counts / total
            else:
                sel_p = np.full(n_features, 1.0 / n_features, dtype=float)
    
        group_specs.append((smp_idx, sel_p, abund_arr))
    

    for t in tqdm(
            range(1, iterations + 1),
            desc="iterations",
            unit="iter",
            leave=False,
            ncols=80,
            ascii=True,
            mininterval=0.5,
            position=0,
            miniters=1,
    ):
        # Fast randomized table as NumPy array
        rtab_arr = np.zeros((n_features, n_samples), dtype=np.int64)
    
        # Randomize within each constraint group
        for smp_idx, sel_p, abund_arr in group_specs:
            for sidx in smp_idx:
                richness = int(richness_vec[sidx])
                reads = int(reads_vec[sidx])
    
                if richness <= 0 or reads <= 0:
                    continue
    
                if richness > n_features:
                    raise ValueError(
                        f"Sample '{sample_names_all[sidx]}' has richness larger "
                        f"than the number of available features."
                    )
    
                # 1) Draw taxa without replacement
                rows = rng.choice(
                    feature_index,
                    size=richness,
                    replace=False,
                    p=sel_p,
                )
    
                # Each selected taxon gets one read first
                rtab_arr[rows, sidx] = 1
    
                # 2) Allocate remaining reads among selected taxa
                extra = reads - richness
    
                if extra > 0:
                    sub_abund = abund_arr[rows]
                    sub_total = sub_abund.sum()
    
                    if sub_total > 0:
                        sub_p = sub_abund / sub_total
                    else:
                        sub_p = np.full(richness, 1.0 / richness, dtype=float)
    
                    # Much faster than rng.choice(..., size=extra) + np.unique(...)
                    extra_counts = rng.multinomial(extra, sub_p)
    
                    rtab_arr[rows, sidx] += extra_counts
    
        # Convert back to DataFrame only once per iteration
        rtab = pd.DataFrame(
            rtab_arr,
            index=feature_names_all,
            columns=sample_names_all,
        )
    
        # Compute beta-diversity for this randomized table
        null_beta = _beta_for_table(rtab)
        x = null_beta.to_numpy()
    
        # --- Welford updates
        delta = x - mu
        mu += delta / t
        M2 += delta * (x - mu)
    
        # --- Raup-Crick counts
        count_lt += x < obs_arr
        count_eq += x == obs_arr

    # --- Finalize statistics
    denom_var = max(1, iterations - 1)
    null_mean = mu
    null_std = np.sqrt(np.maximum(M2 / denom_var, 0.0))
    p = (count_lt + 0.5 * count_eq) / iterations

    with np.errstate(invalid="ignore", divide="ignore"):
        ses_arr = np.where(null_std > 0, (null_mean - obs_arr) / null_std, np.nan)

    # --- Pack DataFrames
    index_cols = tab.columns.tolist()
    out = {
        "div_type": f"{div_type}_q={q}" if div_type in {"naive", "phyl", "func"} else div_type,
        "obs_d":    pd.DataFrame(obs_arr, index=index_cols, columns=index_cols),
        "p":        pd.DataFrame(p,       index=index_cols, columns=index_cols),
        "null_mean":pd.DataFrame(null_mean, index=index_cols, columns=index_cols),
        "null_std": pd.DataFrame(null_std,  index=index_cols, columns=index_cols),
        "ses":      pd.DataFrame(ses_arr,   index=index_cols, columns=index_cols),
    }
    return out

# -- Helper function for q-weighting
def _q_weight(R: np.ndarray, q: float) -> np.ndarray:
    """
    Convert relative abundances to normalized q-weighted abundances.
    """
    if q == 1.0:
        return R.copy()
    if q == 0.0:
        present = R > 0
        richness = present.sum(axis=0, keepdims=True)
        return np.divide(
            present.astype(float),
            richness,
            out=np.zeros_like(R, dtype=float),
            where=richness > 0,
        )
    Rq = np.zeros_like(R, dtype=float)
    pos = R > 0
    Rq[pos] = np.power(R[pos], q)
    denom = Rq.sum(axis=0, keepdims=True)
    return np.divide(
        Rq,
        denom,
        out=np.zeros_like(Rq),
        where=denom > 0,
    )

# ---------------------------------------------------------------------------
# MPDq and NRIq
# ---------------------------------------------------------------------------
def nriq(
    obj: Union[Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1.0,
    iterations: int = 999,
    randomization: Literal["features", "abundances"] = "features",
    use_tqdm: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    use_numba: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Net Relatedness Index (NRI) with q-weighting of relative abundances.
    Accepts either a MicrobiomeData object or a dict with at least a 'tab' DataFrame.

    Parameters
    ----------
    obj : MicrobiomeData, dict, or compatible object
        Input data. Must provide at least an abundance table ('tab').
    distmat : pd.DataFrame
        Square distance matrix indexed/columned by feature ids.
    q : float, default=1.0
        Order of diversity weighting applied to relative abundances.
    iterations : int, default=999
        Number of random permutations of distmat.
    randomization : {'features', 'abundances'}, default='features'
        Randomization strategy. Shuffle features in the phylogenetic tree
        or relative abundance values in each sample.
    use_tqdm : bool, default=True
        Use tqdm for progress bars.
    random_state : int or np.random.Generator, optional
        Random seed or generator for reproducibility.
    use_numba : bool, optional
        If True, uses Numba path; otherwise uses pure Python implementation.

    Returns
    -------
    pandas.DataFrame
        Indexed by sample names with columns:
        - 'MPDq'
        - 'null_mean'
        - 'null_std'
        - 'p'        (Pr[ null < observed ] + 0.5*ties) / iterations
        - 'ses'      (null_mean - observed) / null_std

    Notes
    -----
    - A p value close to zero means that the observed MPD is lower than the null expectation
    - A p value close to one means that the observed MPD is higher than the null expectation
    - A positive ses means that the observed MPD is lower than the null expectation
    - A negative ses means that the observed MPD is higher than the null expectation

    References
    ----------
    Webb et al. (2002) *American Naturalist*.
    """
    if "seed" in kwargs:
        if random_state is not None:
            raise TypeError("Specify only one of 'random_state' or 'seed'.")
        random_state = kwargs.pop("seed")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

    # Robustly extract the abundance table using get_df
    tab = get_df(obj, "tab")

    if tab is None or tab.empty:
        raise ValueError("obj must contain a pandas DataFrame under key 'tab'.")

    if not set(tab.index).issubset(set(distmat.index)) or not set(tab.index).issubset(set(distmat.columns)):
        missing = sorted(list(set(tab.index) - set(distmat.index)))
        raise ValueError(
            f"distmat must include all feature ids from tab.index. Missing count: {len(missing)} (e.g., {missing[:5]})"
        )

    smplist = tab.columns
    D = distmat.loc[tab.index, tab.index].to_numpy()
    R = (tab / tab.sum(axis=0)).fillna(0).to_numpy(float)
    Rq = _q_weight(R, q)
    N, S = Rq.shape

    if use_numba:
        try:
            from .accelerate_model import mpdq_null_numba
            backend = "numba"
        except Exception:
            print('Numba failed, falling back to Python.')
            mpdq_null_numba = None
            backend = "python"
    else:
        mpdq_null_numba = None
        backend = "python"

    def _alpha_mpdq(_D, _Rq):
        # sum_i w_i
        z = _Rq.sum(axis=0)
    
        # full numerator w^T D w  (includes diagonal)
        num_full = (_Rq * (_D @ _Rq)).sum(axis=0)
    
        w2 = (_Rq * _Rq).sum(axis=0)
        den = z*z - w2      # denominator excluding diagonal
    
        present_counts = (_Rq > 0).sum(axis=0)  # (S,)
        too_small = present_counts < 2          # boolean mask

        out = np.full_like(den, np.nan, dtype=float)
        valid = (~too_small) & (den > 0)
    
        out[valid] = num_full[valid] / den[valid]
        return out
    
    present_counts = (R > 0).sum(axis=0)  # shape (S,)
    obs = _alpha_mpdq(D, Rq)
    obs[present_counts < 2] = np.nan

    # streaming stats init
    mu = np.zeros(S, dtype=float)
    M2 = np.zeros(S, dtype=float)
    n_valid = np.zeros(S, dtype=int)
    count_lt = np.zeros(S, dtype=int)
    count_eq = np.zeros(S, dtype=int)

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    tqdm = _get_tqdm(use_tqdm)

    if iterations < 1:
        return(pd.DataFrame({"MPDq": obs}, index=smplist))
    if randomization not in {"features", "abundances"}:
        raise ValueError("randomization must be 'features' or 'abundances'.")

    # null loop
    for t in tqdm(
            range(1, iterations + 1),
            desc="iterations",
            unit="iter",
            leave=False,
            ncols=80,
            ascii=True,
            mininterval=0.5,
            position=0,
            miniters=1,
    ):
        if randomization == "features":
            perm = rng.permutation(N)
            R_perm = R[perm, :]
            Rq_perm = Rq[perm, :]
        elif randomization == "abundances":
            R_perm = np.empty_like(R)
            Rq_perm = np.empty_like(Rq)
            for j in range(S):
                perm = rng.permutation(N)
                R_perm[:, j] = R[perm, j]
                Rq_perm[:, j] = Rq[perm, j]

        if mpdq_null_numba is not None:
            x = mpdq_null_numba(D, Rq_perm)
        else:
            x = _alpha_mpdq(D, Rq_perm)
    
        # --- Welford updates (vectorized) ---
        valid_x = np.isfinite(x) & np.isfinite(obs)
        n_valid[valid_x] += 1
        delta = x[valid_x] - mu[valid_x]
        mu[valid_x] += delta / n_valid[valid_x]
        M2[valid_x] += delta * (x[valid_x] - mu[valid_x])
        count_lt[valid_x] += x[valid_x] < obs[valid_x]
        count_eq[valid_x] += x[valid_x] == obs[valid_x]

    # Finalize stats
    null_mean = np.full(S, np.nan)
    null_std = np.full(S, np.nan)
    p = np.full(S, np.nan)
    
    has_mean = n_valid > 0
    null_mean[has_mean] = mu[has_mean]
    
    has_var = n_valid > 1
    null_std[has_var] = np.sqrt(
        M2[has_var] / (n_valid[has_var] - 1)
    )
    
    p[has_mean] = (
        count_lt[has_mean]
        + 0.5 * count_eq[has_mean]
    ) / n_valid[has_mean]

    with np.errstate(invalid="ignore", divide="ignore"):
        ses = np.where(
            null_std > 0,
            (null_mean - obs) / null_std,
            np.nan
        )
    print('Iterations done with backend '+backend)

    output = pd.DataFrame(
        {
            "MPDq":      obs,
            "null_mean": null_mean,
            "null_std":  null_std,
            "p":         p,
            "ses":       ses,
        },
        index=smplist
    )
    return output

# ---------------------------------------------------------------------------
# NTIq
# ---------------------------------------------------------------------------
def ntiq(
    obj: Union[Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1.0,
    iterations: int = 999,
    randomization: Literal["features", "abundances"] = "features",
    use_tqdm: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    use_numba: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Nearest Taxon Index (NTI) with q-weighting of relative abundances.
    Computes MNTD_q (mean nearest-taxon distance with q-weighted abundances),
    then compares to a null obtained by either permuting feature labels
    ("features") or shuffling abundances within each sample ("abundances").

    Parameters
    ----------
    obj : MicrobiomeData, dict, or compatible object
        Input data. Must provide at least an abundance table ('tab').
    distmat : pd.DataFrame
        Square distance matrix indexed/columned by feature ids.
    q : float, default=1.0
        Order of diversity weighting applied to relative abundances.
    iterations : int, default=999
        Number of random permutations of distmat.
    randomization : {'features', 'abundances'}, default='features'
        Randomization strategy. Shuffle features in the phylogenetic tree
        or relative abundance values in each sample.
    use_tqdm : bool, default=True
        Use tqdm for progress bars.
    random_state : int or np.random.Generator, optional
        Random seed or generator for reproducibility.
    use_numba : bool, optional
        If True, uses Numba path; otherwise uses pure Python implementation.

    Returns
    -------
    pandas.DataFrame
        Indexed by sample names with columns:
        - 'MNTDq'
        - 'null_mean'
        - 'null_std'
        - 'p'        (Pr[ null < observed ] + 0.5*ties) / iterations
        - 'ses'      (null_mean - observed) / null_std

    Notes
    -----
    - A p value close to zero means that the observed MNTD is lower than the null expectation
    - A p value close to one means that the observed MNTD is higher than the null expectation
    - A positive ses means that the observed MNTD is lower than the null expectation
    - A negative ses means that the observed MNTD is higher than the null expectation
    """
    if "seed" in kwargs:
        if random_state is not None:
            raise TypeError("Specify only one of 'random_state' or 'seed'.")
        random_state = kwargs.pop("seed")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

    # --- Input & alignment ---
    tab = get_df(obj, "tab")
    if tab is None or tab.empty:
        raise ValueError("obj must contain a pandas DataFrame under key 'tab'.")
    if not set(tab.index).issubset(set(distmat.index)) or not set(tab.index).issubset(set(distmat.columns)):
        missing = sorted(list(set(tab.index) - set(distmat.index)))
        raise ValueError(
            f"distmat must include all feature ids from tab.index. Missing count: {len(missing)} (e.g., {missing[:5]})"
        )

    smplist = tab.columns
    D = distmat.loc[tab.index, tab.index].to_numpy()  # (N x N)
    R = (tab / tab.sum(axis=0)).fillna(0).to_numpy(float)
    Rq = _q_weight(R, q)
    N, S = Rq.shape

    if use_numba:
        try:
            from .accelerate_model import mntdq_null_numba
            backend = "numba"
        except Exception:
            print('Numba failed, falling back to Python.')
            mntdq_null_numba = None
            backend = "python"
    else:
        mntdq_null_numba = None
        backend = "python"

    # --- Helper: compute vector of MNTD_q for all samples in one go ---
    # For each sample s:
    #  1) take presence mask m = (R[:, s] > 0)
    #  2) within D[m, m], set diagonal to +inf and take rowwise min -> dmin (per-present feature)
    #  3) aggregate: sum( w_i^q * dmin_i ) / sum( w_i^q ) where w_i = R[:, s]
    def _mntdq_all(D: np.ndarray, R_used: np.ndarray, Rq_used: np.ndarray) -> np.ndarray:
        S = R_used.shape[1]
        out = np.full(S, np.nan, dtype=float)
        for s in range(S):
            m = R_used[:, s] > 0.0
            k = int(m.sum())
            if k < 2:
                out[s] = np.nan
                continue
            # then:
            D_sub = D[np.ix_(m, m)].copy()
            np.fill_diagonal(D_sub, np.inf)
            dmin = D_sub.min(axis=1)

            # q-weighted aggregation
            wq = Rq_used[m, s]
            denom = float(wq.sum())
            if denom > 0.0:
                out[s] = float((wq * dmin).sum() / denom)
            else:
                out[s] = np.nan
        return out

    # --- Observed MNTD_q ---
    obs = _mntdq_all(D, R, Rq)

    # --- Streaming (Welford) statistics over null iterations ---
    mu = np.zeros(S, dtype=float)
    M2 = np.zeros(S, dtype=float)
    n_valid = np.zeros(S, dtype=int)
    count_lt = np.zeros(S, dtype=int)
    count_eq = np.zeros(S, dtype=int)

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    tqdm = _get_tqdm(use_tqdm)

    if iterations < 1:
        return(pd.DataFrame({"MNTDq": obs}, index=smplist))
    if randomization not in {"features", "abundances"}:
        raise ValueError("randomization must be 'features' or 'abundances'.")

    # Null loop
    for t in tqdm(
            range(1, iterations + 1),
            desc="iterations",
            unit="iter",
            leave=False,
            ncols=80,
            ascii=True,
            mininterval=0.5,
            position=0,
            miniters=1,
    ):

        if randomization == "features":
            perm = rng.permutation(N)
            R_perm = R[perm, :]
            Rq_perm = Rq[perm, :]
        elif randomization == "abundances":
            R_perm = np.empty_like(R)
            Rq_perm = np.empty_like(Rq)
            for j in range(S):
                perm = rng.permutation(N)
                R_perm[:, j] = R[perm, j]
                Rq_perm[:, j] = Rq[perm, j]

        if mntdq_null_numba is not None:
            x = mntdq_null_numba(D, R_perm, Rq_perm)
        else:
            x = _mntdq_all(D, R_perm, Rq_perm)

        # --- Welford updates (vectorized) ---
        valid_x = np.isfinite(x) & np.isfinite(obs)
        n_valid[valid_x] += 1
        delta = x[valid_x] - mu[valid_x]
        mu[valid_x] += delta / n_valid[valid_x]
        M2[valid_x] += delta * (x[valid_x] - mu[valid_x])
        count_lt[valid_x] += x[valid_x] < obs[valid_x]
        count_eq[valid_x] += x[valid_x] == obs[valid_x]

    # Finalize stats
    null_mean = np.full(S, np.nan)
    null_std = np.full(S, np.nan)
    p = np.full(S, np.nan)
    
    has_mean = n_valid > 0
    null_mean[has_mean] = mu[has_mean]
    
    has_var = n_valid > 1
    null_std[has_var] = np.sqrt(
        M2[has_var] / (n_valid[has_var] - 1)
    )
    
    p[has_mean] = (
        count_lt[has_mean]
        + 0.5 * count_eq[has_mean]
    ) / n_valid[has_mean]

    with np.errstate(invalid="ignore", divide="ignore"):
        ses = np.where(
            null_std > 0,
            (null_mean - obs) / null_std,
            np.nan
        )
    print('Iterations done with backend '+backend)

    # Pack results
    output = pd.DataFrame(
        {
            "MNTDq": obs,
            "null_mean": null_mean,
            "null_std": null_std,
            "p": p,
            "ses": ses,
        },
        index=smplist,
    )
    return output

# ---------------------------------------------------------------------------
# Interpolated net relatedness index
# ---------------------------------------------------------------------------
def inriq(
    obj: Union[Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1.0,
    locality: float = 1,
    dist_scale: float | str = "auto",
    iterations: int = 999,
    randomization: Literal["features", "abundances"] = "features",
    use_tqdm: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    use_numba: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate interpolated MPDq and an NRI-style null standardized effect size.
    The metric uses an exponential soft-min kernel to interpolate continuously
    between a broad MPD-like endpoint and a local MNTD-like endpoint. For each
    focal taxon i, neighbour probabilities are

        p_ij ∝ exp(-r*d_ij)*w_j

    where d_ij is the pairwise distance, w_j is the q-weighted relative
    abundance of neighbour j, and

        r = (exp(locality)-1) / dist_scale.

    With locality = 0, all non-self neighbours contribute according to their
    q-weighted abundances. As locality increases, weight is progressively
    concentrated on nearby neighbours.

    Parameters
    ----------
    obj : dict or object
        Object containing a feature abundance table accessible as ``"tab"``.
        Rows are taxa/features and columns are samples.
    distmat : pandas.DataFrame
        Square pairwise distance matrix with taxa matching ``tab.index``.
    q : float, default=1.0
        Diversity order used for abundance weighting.
    locality : float, default=1
        Non-negative kernel locality parameter. Larger values give stronger
        nearest-neighbour focus.
    dist_scale : {"auto"} or float, default="auto"
        Distance scale used to convert locality into kernel sharpness. If
        ``"auto"``, the median positive distance in ``distmat`` is used.
        Supplying a numeric value gives reproducible kernel sharpness across
        runs or datasets.
    iterations : int, default=999
        Number of null randomizations. If less than 1, only observed values are
        returned.
    randomization : {"features", "abundances"}, default="features"
        Null model. ``"features"`` permutes feature labels across the distance
        matrix. ``"abundances"`` permutes abundances independently within each
        sample.
    use_tqdm : bool, default=True
        Show a progress bar for null iterations.
    random_state : int or numpy.random.Generator, optional
        Random seed or generator.
    use_numba : bool, optional
        If True, uses Numba path; otherwise uses pure Python implementation.

    Returns
    -------
    pandas.DataFrame
        Results indexed by sample. Columns include:

        iMPDq
            Observed interpolated mean pairwise distance.
        ENN
            Effective Number of Neighbours, calculated as the Shannon effective
            number of the neighbour probability distribution.
        NTF
            Nearest Taxon Focus. Normalized position of ENN between its broad
            and nearest-neighbour attainable endpoints.
        null_mean
            Mean null expectation of iMPDq.
        null_std
            Standard deviation of null iMPDq values.
        p
            Mid-p left-tail probability, calculated as
            (Pr[null < observed] + 0.5 Pr[null = observed]).        
        ses
            NRI-style standardized effect size, calculated as
            ``(null_mean - observed) / null_std``. Positive values indicate
            lower iMPDq than expected under the null model.

    Notes
    -----
    - ENN and NTF are diagnostics of the realized kernel. They are not fixed by
      ``locality`` and may differ among samples because each sample has a distinct
      distance and abundance structure. ENN may remain greater than one at high
      locality when several neighbours are tied or nearly tied as nearest taxa.
    - A p value close to zero means that the observed iMPDq is lower than
      expected under the null model.
    - A p value close to one means that the observed iMPDq is higher than
      expected under the null model.
    - A positive ses means that the observed iMPDq is lower than the null
      expectation.
    - A negative ses means that the observed iMPDq is higher than the null
      expectation.
    """
    if "seed" in kwargs:
        if random_state is not None:
            raise TypeError("Specify only one of 'random_state' or 'seed'.")
        random_state = kwargs.pop("seed")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

    # ---- Extract abundance table ----
    tab = get_df(obj, "tab")
    if tab is None or tab.empty:
        raise ValueError("'tab' must be provided in the input.")
    smplist = tab.columns

    # Align distances
    missing_index = sorted(set(tab.index) - set(distmat.index))
    missing_columns = sorted(set(tab.index) - set(distmat.columns))
    if missing_index or missing_columns:
        raise ValueError(
            "distmat must contain all taxa from tab.index in both rows and columns. "
            f"Missing from index: {len(missing_index)} "
            f"(e.g., {missing_index[:5]}). "
            f"Missing from columns: {len(missing_columns)} "
            f"(e.g., {missing_columns[:5]})."
        )

    # Fix D and R
    D = distmat.loc[tab.index, tab.index].to_numpy(dtype=float, copy=True) # (N x N)
    R = (tab / tab.sum(axis=0)).fillna(0).to_numpy(float)       # (N x S)
    N, S = R.shape

    if D.shape[0] != D.shape[1]:
        raise ValueError("distmat must be square.")
    if np.any(~np.isfinite(np.diag(D))):
        raise ValueError("distmat diagonal must be finite.")
    if np.any(D[np.isfinite(D)] < 0):
        raise ValueError("distmat must not contain negative distances.")

    # q-weighting
    Rq = _q_weight(R, q)

    #Calculate distance sensitivity parameter, r
    if locality < 0:
        raise ValueError("locality must be non-negative.")
    dpos = D[np.isfinite(D) & (D > 0)]
    if dpos.size == 0:
        dist_scale_used = np.nan
        r = 0.0
    else:
        if dist_scale == "auto":
            dist_scale_used = float(np.median(dpos))
        
            if not np.isfinite(dist_scale_used) or dist_scale_used <= 0:
                raise ValueError(
                    "Unable to determine a valid automatic distance scale."
                )
        else:
            try:
                dist_scale_used = float(dist_scale)
            except (TypeError, ValueError):
                raise ValueError("dist_scale must be 'auto' or a positive finite number.")
            if not np.isfinite(dist_scale_used) or dist_scale_used <= 0:
                raise ValueError("dist_scale must be 'auto' or a positive finite number.")
        r = np.expm1(locality) / dist_scale_used
        if not np.isfinite(r):
            raise ValueError(
                "Kernel sharpness is not finite. "
                "Try a smaller locality or a larger dist_scale."
            )

    if use_numba:
        try:
            from .accelerate_model import impdq_null_numba
            backend = "numba"
        except Exception:
            print('Numba failed, falling back to Python.')
            impdq_null_numba = None
            backend = "python"
    else:
        impdq_null_numba = None
        backend = "python"

    def _impdq_sample(
        Dloc: np.ndarray,
        w_loc: np.ndarray,
        r: float,
        diagnostics: bool = False,
    ):
        """
        Calculate iMPDq for a single sample.
        """
        k_taxa = len(w_loc)
        finite = np.isfinite(Dloc)
        target = finite & (w_loc[None, :] > 0)
    
        # Kernel probabilities
        X = np.full_like(Dloc, -np.inf, dtype=float)
        X[target] = -r * Dloc[target]
        row_max = np.max(X, axis=1, keepdims=True)
        A = np.exp(X - row_max)
        A[~np.isfinite(A)] = 0.0
        Aw = A * w_loc[None, :]
        denom = Aw.sum(axis=1)

        # Soft-min distances
        numer = (A * np.where(finite, Dloc, 0.0)) @ w_loc
        dvals = np.divide(
            numer,
            denom,
            out=np.full(k_taxa, np.nan),
            where=denom > 0,
        )
        valid_d = np.isfinite(dvals) & (w_loc > 0)
        if np.any(valid_d):
            impdq = float(
                np.sum(w_loc[valid_d] * dvals[valid_d])
                / np.sum(w_loc[valid_d])
            )
        else:
            impdq = np.nan
    
        if not diagnostics:
            return impdq
    
        # ENN
        P = np.divide(
            Aw,
            denom[:, None],
            out=np.zeros_like(Aw),
            where=denom[:, None] > 0,
        )
        plogp = np.zeros_like(P)
        pos = P > 0
        plogp[pos] = P[pos] * np.log(P[pos])
        enn_rows = np.full(k_taxa, np.nan, dtype=float)
        valid_kernel = denom > 0
        enn_rows[valid_kernel] = np.exp(
            -np.sum(plogp[valid_kernel], axis=1)
        )
    
        # ENNmax
        W0 = np.where(target, w_loc[None, :], 0.0)
        denom0 = W0.sum(axis=1)
        P0 = np.divide(
            W0,
            denom0[:, None],
            out=np.zeros_like(W0),
            where=denom0[:, None] > 0,
        )
        p0logp0 = np.zeros_like(P0)
        pos0 = P0 > 0
        p0logp0[pos0] = P0[pos0] * np.log(P0[pos0])
        ennmax_rows = np.full(k_taxa, np.nan, dtype=float)
        valid0 = denom0 > 0
        ennmax_rows[valid0] = np.exp(
            -np.sum(p0logp0[valid0], axis=1)
        )
    
        # ENNmin
        Dvalid = np.where(target, Dloc, np.inf)
        dmin = np.min(Dvalid, axis=1, keepdims=True)
        nearest = (
            target
            & np.isclose(
                Dvalid,
                dmin,
                rtol=1e-10,
                atol=1e-12
            )
        )
        Wmin = np.where(
            nearest,
            w_loc[None, :],
            0.0
        )
        denom_min = Wmin.sum(axis=1)
        Pmin = np.divide(
            Wmin,
            denom_min[:, None],
            out=np.zeros_like(Wmin),
            where=denom_min[:, None] > 0,
        )
        pminlogpmin = np.zeros_like(Pmin)
        posmin = Pmin > 0
        pminlogpmin[posmin] = (
            Pmin[posmin] * np.log(Pmin[posmin])
        )
        ennmin_rows = np.full(k_taxa, np.nan, dtype=float)
        valid_min = denom_min > 0
        ennmin_rows[valid_min] = np.exp(
            -np.sum(pminlogpmin[valid_min], axis=1)
        )
    
        # NTF
        ntf_rows = np.full(
            k_taxa,
            np.nan,
            dtype=float
        )
        denom_range = (
            ennmax_rows - ennmin_rows
        )
        valid_ntf = (
            np.isfinite(enn_rows)
            & np.isfinite(ennmax_rows)
            & np.isfinite(ennmin_rows)
            & (denom_range > 1e-12)
        )
        ntf_rows[valid_ntf] = np.clip(
            (
                ennmax_rows[valid_ntf]
                - enn_rows[valid_ntf]
            )
            / denom_range[valid_ntf],
            0.0,
            1.0,
        )
        valid_enn = np.isfinite(enn_rows) & (w_loc > 0)
        valid_ennmin = np.isfinite(ennmin_rows) & (w_loc > 0)
        valid_ennmax = np.isfinite(ennmax_rows) & (w_loc > 0)
        valid_ntf = np.isfinite(ntf_rows) & (w_loc > 0)
        enn = np.nan
        ennmin = np.nan
        ennmax = np.nan
        ntf = np.nan
        if np.any(valid_enn):
            enn = float(
                np.sum(
                    w_loc[valid_enn]
                    * enn_rows[valid_enn]
                )
                / np.sum(w_loc[valid_enn])
            )
        if np.any(valid_ennmin):
            ennmin = float(
                np.sum(
                    w_loc[valid_ennmin]
                    * ennmin_rows[valid_ennmin]
                )
                / np.sum(w_loc[valid_ennmin])
            )
        if np.any(valid_ennmax):
            ennmax = float(
                np.sum(
                    w_loc[valid_ennmax]
                    * ennmax_rows[valid_ennmax]
                )
                / np.sum(w_loc[valid_ennmax])
            )
        if np.any(valid_ntf):
            ntf = float(
                np.sum(
                    w_loc[valid_ntf]
                    * ntf_rows[valid_ntf]
                )
                / np.sum(w_loc[valid_ntf])
            )
        return impdq, enn, ntf, ennmin, ennmax

    #----------------------------------------------------------
    # ---- Run a loop and calculate observed values per sample
    #----------------------------------------------------------
    obs = np.zeros(S, float)
    enn = np.zeros(S, float)
    ennmin = np.zeros(S, float)
    ennmax = np.zeros(S, float)
    ntf = np.zeros(S, float)

    for s in range(S):
        w_s = Rq[:, s]
        present_s = R[:, s] > 0

        # Low richness case:
        if present_s.sum() < 2: 
            obs[s] = np.nan
            enn[s] = np.nan
            ennmin[s] = np.nan
            ennmax[s] = np.nan
            ntf[s] = np.nan
            continue
        
        # Normal case:
        rows = np.where(present_s)[0]
        Dloc = D[np.ix_(rows, rows)].copy()
        np.fill_diagonal(Dloc, np.inf)
        w_loc = w_s[rows].copy()

        obs[s], enn[s], ntf[s], ennmin[s], ennmax[s] = _impdq_sample(
            Dloc=Dloc,
            w_loc=w_loc,
            r=r,
            diagnostics=True,
        )
        
    #----------------------------------------------------------
    # ---- null distribution ----
    #----------------------------------------------------------
    if iterations < 1:
        return pd.DataFrame(
            {
                "iMPDq": obs,
                "NTF": ntf,
                "ENN": enn,
                "ENN_min": ennmin,
                "ENN_max": ennmax,
                "dist_scale": [dist_scale_used] * len(obs)
            },
            index=smplist,
        )

    if randomization not in {"features", "abundances"}:
        raise ValueError("randomization must be 'features' or 'abundances'.")

    rng = random_state if isinstance(random_state, np.random.Generator)\
          else np.random.default_rng(random_state)
    tqdm = _get_tqdm(use_tqdm)

    mu = np.zeros(S, dtype=float)
    M2 = np.zeros(S, dtype=float)
    n_valid = np.zeros(S, dtype=int)
    count_lt = np.zeros(S, dtype=int)
    count_eq = np.zeros(S, dtype=int)

    for t in tqdm(range(1, iterations + 1), desc="iterations", unit="iter",
                  leave=False, ncols=80, ascii=True, mininterval=0.5):

        # randomization
        if randomization == "features":
            perm = rng.permutation(N)
            R_perm = R[perm, :]
            Rq_perm = _q_weight(R_perm, q)
        else:
            R_perm = np.empty_like(R)
            for j in range(S):
                R_perm[:, j] = R[rng.permutation(N), j]
            Rq_perm = _q_weight(R_perm, q)

        # compute null soft-min iMPDq
        x = np.full(S, np.nan, dtype=float)
        for s in range(S):
            w_s = Rq_perm[:, s]
            present_s = w_s > 0
            if present_s.sum() < 2:
                x[s] = np.nan
                continue

            if impdq_null_numba is not None: #Numba case
                rows = np.where(present_s)[0]
                w_loc = w_s[rows]
                x[s] = impdq_null_numba(D, rows, w_loc, r)

            else: #Python case
                rows = np.where(present_s)[0]
                Dloc = D[np.ix_(rows, rows)].copy()
                np.fill_diagonal(Dloc, np.inf)
                w_loc = w_s[rows].copy()

                x[s] = _impdq_sample(
                    Dloc=Dloc,
                    w_loc=w_loc,
                    r=r,
                    diagnostics=False,
                )

        # Welford
        valid_x = np.isfinite(x) & np.isfinite(obs)
        n_valid[valid_x] += 1
        delta = x[valid_x] - mu[valid_x]
        mu[valid_x] += delta / n_valid[valid_x]
        M2[valid_x] += delta * (x[valid_x] - mu[valid_x])
        count_lt[valid_x] += x[valid_x] < obs[valid_x]
        count_eq[valid_x] += x[valid_x] == obs[valid_x]

    null_mean = np.full(S, np.nan)
    null_std = np.full(S, np.nan)
    p = np.full(S, np.nan)
    has_mean = n_valid > 0
    null_mean[has_mean] = mu[has_mean]
    has_var = n_valid > 1
    null_std[has_var] = np.sqrt(M2[has_var] / (n_valid[has_var] - 1))
    p[has_mean] = (count_lt[has_mean] + 0.5 * count_eq[has_mean]) / n_valid[has_mean]
    with np.errstate(invalid="ignore", divide="ignore"):
        ses = np.where(null_std > 0, (null_mean - obs) / null_std, np.nan)
    print('Iterations done with backend '+backend)

    return pd.DataFrame(
        {
            "iMPDq": obs,
            "NTF": ntf,
            "ENN": enn,
            "null_mean": null_mean,
            "null_std": null_std,
            "p": p,
            "ses": ses,
            "ENN_min": ennmin,
            "ENN_max": ennmax,
            "dist_scale": [dist_scale_used] * len(obs)
        },
        index=smplist,
    )


# ---------------------------------------------------------------------------
# beta-NRIq
# ---------------------------------------------------------------------------
def beta_nriq(
    obj: Union[Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1.0,
    iterations: int = 999,
    randomization: Literal["features", "abundances"] = "features",
    use_tqdm: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Computes beta-MPD_q for all sample pairs, then contrasts against a null
    generated by (a) feature label permutations ("features") or
    (b) within-sample abundance shuffles ("abundances").

    Parameters
    ----------
    obj : MicrobiomeData, dict, or compatible object
        Input data. Must provide at least an abundance table ('tab').
    distmat : pd.DataFrame
        Square distance matrix indexed/columned by feature ids.
    q : float, default=1.0
        Order of diversity weighting applied to relative abundances.
    iterations : int, default=999
        Number of random permutations of distmat.
    randomization : {'features', 'abundances'}, default='features'
        Randomization strategy. Shuffle features in the phylogenetic tree
        or relative abundance values in each sample.
    use_tqdm : bool, default=True
        Use tqdm for progress bars.
    random_state : int or np.random.Generator, optional
        Random seed or generator for reproducibility.

    Returns
    -------
    dict of pandas.DataFrame (S x S):
        'beta_MPDq' : observed beta-MPD_q
        'beta_null_mean' : mean of null beta-MPD_q
        'beta_null_std'  : std  of null beta-MPD_q
        'beta_p'         : (count(null < obs) + 0.5 * ties) / iterations
        'beta_ses'       : (null_mean - obs) / null_std

    Notes
    -----
    - Returns a dataframe with observed beta_MPDq if iterations=0, otherwise a dictionary is returned
    - A p value close to zero means that the observed MPD between samples is lower than the null expectation
    - A p value close to one means that the observed MPD between samples is higher than the null expectation
    - A positive ses means that the observed MPD between samples is lower than the null expectation
    - A negative ses means that the observed MPD between samples is higher than the null expectation
    """
    if "seed" in kwargs:
        if random_state is not None:
            raise TypeError("Specify only one of 'random_state' or 'seed'.")
        random_state = kwargs.pop("seed")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

    # --- Input & alignment ---
    tab = get_df(obj, "tab")
    if tab is None or tab.empty:
        raise ValueError("obj must contain a non-empty pandas DataFrame under key 'tab'.")

    if not set(tab.index).issubset(set(distmat.index)) or not set(tab.index).issubset(set(distmat.columns)):
        missing = sorted(list(set(tab.index) - set(distmat.index)))
        raise ValueError(
            f"distmat must include all feature ids from tab.index. "
            f"Missing count: {len(missing)} (e.g., {missing[:5]})"
        )

    smplist = tab.columns
    D = distmat.loc[tab.index, tab.index].to_numpy()  # (N x N), float
    # Relative abundances (N x S)
    R = (tab / tab.sum(axis=0)).to_numpy(dtype=float)

    # q-weighting: only positives are powered (consistent with your nriq)
    if q == 1.0:
        Rq = R
    else:
        Rq = R.copy()
        mask_pos = Rq > 0.0
        Rq[mask_pos] = np.power(Rq[mask_pos], q)

    N, S = Rq.shape

    # --- Observed beta-MPD_q for all pairs (vectorized) ---
    # obs[s,t] = sum_{i,j} Rq[i,s] * D[i,j] * Rq[j,t] / (sum_i Rq[i,s] * sum_j Rq[j,t])
    M_obs = D @ Rq                 # (N x S)
    num_obs = Rq.T @ M_obs         # (S x S)
    z = Rq.sum(axis=0)             # (S,)
    den_obs = z[:, None] * z[None, :]  # (S x S)
    with np.errstate(invalid="ignore", divide="ignore"):
        obs = num_obs / den_obs

    # --- Streaming (Welford) over null iterations ---
    mu = np.zeros((S, S), dtype=np.float64)
    M2 = np.zeros((S, S), dtype=np.float64)
    count_lt = np.zeros((S, S), dtype=np.int64)
    count_eq = np.zeros((S, S), dtype=np.int64)

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    tqdm = _get_tqdm(use_tqdm)

    if iterations < 1:
        arr = np.asarray(obs, dtype=float).copy()
        np.fill_diagonal(arr, np.nan)
        df_obs = pd.DataFrame(arr, index=smplist, columns=smplist)
        return df_obs
    if randomization not in {"features", "abundances"}:
        raise ValueError("randomization must be 'features' or 'abundances'.")

    for t in tqdm(
            range(1, iterations + 1),
            desc="iterations",
            unit="iter",
            leave=False,
            ncols=80,
            ascii=True,
            mininterval=0.5,
            position=0,
            miniters=1,
    ):
        if randomization == "features":
            perm = rng.permutation(N)
            Rq_perm = Rq[perm, :]
        else:  # "abundances"
            Rq_perm = np.empty_like(Rq)
            for j in range(S):
                Rq_perm[:, j] = Rq[rng.permutation(N), j]

        # Null beta-MPD_q (vectorized)
        M_null = D @ Rq_perm
        num_null = Rq_perm.T @ M_null

        with np.errstate(invalid="ignore", divide="ignore"):
            x = num_null / den_obs  # (S x S)

        # Welford
        delta = x - mu
        mu += delta / t
        M2 += delta * (x - mu)

        # p-index vs observed
        count_lt += (x < obs)
        count_eq += (x == obs)

    # Finalize stats
    denom_var = max(1, iterations - 1)
    null_mean = mu
    null_std = np.sqrt(np.maximum(M2 / denom_var, 0.0))
    p = (count_lt + 0.5 * count_eq) / iterations
    with np.errstate(invalid="ignore", divide="ignore"):
        ses = np.where(null_std > 0, (null_mean - obs) / null_std, np.nan)

    # Build DataFrames
    idxcols = list(smplist)
    df_obs = pd.DataFrame(obs, index=idxcols, columns=idxcols)
    df_mean = pd.DataFrame(null_mean, index=idxcols, columns=idxcols)
    df_std = pd.DataFrame(null_std, index=idxcols, columns=idxcols)
    df_p = pd.DataFrame(p, index=idxcols, columns=idxcols)
    df_ses = pd.DataFrame(ses, index=idxcols, columns=idxcols)

    for df in (df_obs, df_mean, df_std, df_p, df_ses):
        np.fill_diagonal(df.values, np.nan)

    return {
        "beta_MPDq": df_obs,
        "beta_null_mean": df_mean,
        "beta_null_std": df_std,
        "beta_p": df_p,
        "beta_ses": df_ses,
    }

# ---------------------------------------------------------------------------
# beta-NTIq
# ---------------------------------------------------------------------------
def beta_ntiq(
    obj: Union[Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1.0,
    iterations: int = 999,
    include_conspecifics: bool = False,
    randomization: Literal["features", "abundances"] = "features",
    use_tqdm: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    use_numba: bool = True,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Computes beta-MNTD_q (mean nearest-taxon distance with q-weighted abundances)
    for all sample pairs, then contrasts the observed matrix against a null
    distribution generated by randomization:

        - randomization="features": permute feature identities (rows) identically across samples
        - randomization="abundances": shuffle abundances within each sample (column-wise)

    The null distribution is aggregated online using Welford updates, yielding
    per-pair null mean, null std, tie-aware p-index, and standardized effect size.

    Parameters
    ----------
    obj : MicrobiomeData | dict | Any
        Input with at least an abundance table under key 'tab'.
    distmat : pandas.DataFrame
        Square distance matrix (features × features) whose index/columns include `tab.index`.
    q : float, default=1.0
        Diversity order used to weight relative abundances (applied only to strictly positive entries).
    iterations : int, default=999
        Number of randomization iterations used to build the null distribution.
    include_conspecifics : bool, default=False
        Determines whether conspecifics (identical features shared between samples) are allowed 
        to contribute zero-distance matches in the nearest-taxon calculation.
    randomization : {"features", "abundances"}, default="features"
        Randomization strategy for the null model:
          - "features": permute feature identities identically for all samples (tip-label permutation).
          - "abundances": shuffle abundances within each sample (column-wise permutation).
    use_tqdm : bool, default=True
        Use `tqdm` for progress bars (a lightweight stub is used if `tqdm` is unavailable).
    random_state : int | numpy.random.Generator, optional
        Random seed or Generator for reproducibility.
    use_numba : bool, optional
        If True, uses Numba path; otherwise uses pure Python implementation.

    Returns
    -------
    dict of pandas.DataFrame
        Full (samples × samples) matrices:
          - 'beta_MNTDq' : observed beta-MNTD_q
          - 'beta_null_mean'  : mean of null beta-MNTD_q
          - 'beta_null_std'   : std  of null beta-MNTD_q
          - 'beta_p'          : (count(null < observed) + 0.5 * ties) / iterations
          - 'beta_ses'        : (null_mean - observed) / null_std
        Diagonal entries are set to NaN.

    Notes
    -----
    - Returns a dataframe with observed beta_MNTDq if iterations=0, otherwise a dictionary is returned
    - A p value close to zero means that the observed MNTD between samples is lower than the null expectation
    - A p value close to one means that the observed MNTD between samples is higher than the null expectation
    - A positive ses means that the observed MNTD between samples is lower than the null expectation
    - A negative ses means that the observed MNTD between samples is higher than the null expectation

    References
    ----------
    Webb et al. (2002) American Naturalist.
    Stegen et al. (2013) ISME Journal.
    """
    if "seed" in kwargs:
        if random_state is not None:
            raise TypeError("Specify only one of 'random_state' or 'seed'.")
        random_state = kwargs.pop("seed")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

    # ---- Input & alignment ----
    tab = get_df(obj, "tab")
    if tab is None or tab.empty:
        raise ValueError("obj must contain a non-empty pandas DataFrame under key 'tab'.")
    if not set(tab.index).issubset(set(distmat.index)) or not set(tab.index).issubset(set(distmat.columns)):
        missing = sorted(list(set(tab.index) - set(distmat.index)))
        raise ValueError(
            f"distmat must include all feature ids from tab.index. Missing count: {len(missing)} (e.g., {missing[:5]})"
        )

    smplist = tab.columns
    D = distmat.loc[tab.index, tab.index].to_numpy(copy=True)  # (N x N), float
    # Relative abundances (N x S), allowing potential NaNs if a column sums to zero
    R = (tab / tab.sum(axis=0)).to_numpy(dtype=float)          # (N x S)
    Rq = _q_weight(R, q)
    N, S = R.shape

    if use_numba:
        try:
            from .accelerate_model import beta_mntdq_numba
            backend = "numba"
        except Exception:
            print('Numba failed, falling back to Python.')
            beta_mntdq_numba = None
            backend = "python"
    else:
        beta_mntdq_numba = None
        backend = "python"

    # Column totals of q-weighted abundances per sample
    z = Rq.sum(axis=0)  # (S,)

    # ---- Helper: compute full directed MNTD_q matrices in a vectorized way ----
    def _delta_col_row(D: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Delta_col (N × S) and Delta_row (N × S) with optional
        exclusion of conspecifics (i == j)."""
    
        Delta_col = np.full((N, S), np.nan, dtype=float)
        Delta_row = np.full((N, S), np.nan, dtype=float)
    
        for t in range(S):
            mask_t = B[:, t]
            if not np.any(mask_t):
                continue
    
            sub = D[:, mask_t]         # (N × k)
            if not include_conspecifics:
                # If feature i is present in sample t, set its self-distance to +inf
                diag_mask = np.zeros_like(sub, dtype=bool)
                diag_indices = np.where(mask_t)[0]
                diag_mask[diag_indices, np.arange(len(diag_indices))] = True
                sub = sub.copy()
                sub[diag_mask] = np.inf
    
            Delta_col[:, t] = sub.min(axis=1)
    
        for s in range(S):
            mask_s = B[:, s]
            if not np.any(mask_s):
                continue
    
            sub = D[mask_s, :]         # (k × N)
            if not include_conspecifics:
                diag_indices = np.where(mask_s)[0]
                sub = sub.copy()
                sub[np.arange(len(diag_indices)), diag_indices] = np.inf
    
            Delta_row[:, s] = sub.min(axis=0)
    
        return Delta_col, Delta_row

    def _beta_mntdq_full(D: np.ndarray, Rq: np.ndarray) -> np.ndarray:
        """Observed (or null) full-matrix beta-MNTD_q from D, R (presence), and Rq (q-weights)."""
        B = Rq > 0.0  # presence/absence for each sample
        Delta_col, Delta_row = _delta_col_row(D, B)

        # Clean directed matrices
        Delta_col = np.nan_to_num(Delta_col, nan=0.0, posinf=0.0, neginf=0.0)
        Delta_row = np.nan_to_num(Delta_row, nan=0.0, posinf=0.0, neginf=0.0)

        # Numerators for directed terms
        A_num = Rq.T @ Delta_col                # (S x S)
        B_num = (Rq.T @ Delta_row).T            # (S x S) via transpose for (t → s)

        # Denominators (broadcast)
        with np.errstate(invalid="ignore", divide="ignore"):
            A = A_num / z[:, None]
            Bdir = B_num / z[None, :]
            beta = 0.5 * (A + Bdir)
        return beta

    # ---- Observed beta-MNTD_q ----
    obs = _beta_mntdq_full(D, Rq)  # (S x S)

    # ---- Streaming (Welford) over null iterations ----
    mu = np.zeros((S, S), dtype=np.float64)
    M2 = np.zeros((S, S), dtype=np.float64)
    count_lt = np.zeros((S, S), dtype=np.int64)
    count_eq = np.zeros((S, S), dtype=np.int64)

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    tqdm = _get_tqdm(use_tqdm)

    if iterations < 1:
        arr = np.asarray(obs, dtype=float).copy()
        np.fill_diagonal(arr, np.nan)
        df_obs = pd.DataFrame(arr, index=smplist, columns=smplist)
        return df_obs
    if randomization not in {"features", "abundances"}:
        raise ValueError("randomization must be 'features' or 'abundances'.")

    for t in tqdm(
            range(1, iterations + 1),
            desc="iterations",
            unit="iter",
            leave=False,
            ncols=80,
            ascii=True,
            mininterval=0.5,
            position=0,
            miniters=1,
    ):

        if randomization == "features":
            perm = rng.permutation(N)
            Rq_perm = np.ascontiguousarray(Rq[perm, :])
        
        else:
            Rq_perm = np.empty_like(Rq)
            for j in range(S):
                perm = rng.permutation(N)
                Rq_perm[:, j] = Rq[perm, j]
            Rq_perm = np.ascontiguousarray(Rq_perm)
        
        if beta_mntdq_numba is not None:
            x = beta_mntdq_numba(
                D,
                Rq_perm,
                include_conspecifics,
            )
        else:
            x = _beta_mntdq_full(D, Rq_perm)

        # Welford updates
        delta = x - mu
        mu += delta / t
        M2 += delta * (x - mu)

        # p-index bookkeeping
        count_lt += (x < obs)
        count_eq += (x == obs)

    # Finalize stats
    denom_var = max(1, iterations - 1)
    null_mean = mu
    null_std = np.sqrt(np.maximum(M2 / denom_var, 0.0))
    p = (count_lt + 0.5 * count_eq) / iterations
    with np.errstate(invalid="ignore", divide="ignore"):
        ses = np.where(null_std > 0, (null_mean - obs) / null_std, np.nan)

    # Build DataFrames
    idxcols = list(smplist)
    df_obs = pd.DataFrame(obs, index=idxcols, columns=idxcols)
    df_mean = pd.DataFrame(null_mean, index=idxcols, columns=idxcols)
    df_std = pd.DataFrame(null_std, index=idxcols, columns=idxcols)
    df_p = pd.DataFrame(p, index=idxcols, columns=idxcols)
    df_ses = pd.DataFrame(ses, index=idxcols, columns=idxcols)

    # Set diagonals to NaN for all outputs (consistent with prior beta-* functions)
    for df in (df_obs, df_mean, df_std, df_p, df_ses):
        np.fill_diagonal(df.to_numpy(), np.nan)
    print('Iterations done with backend '+backend)

    return {
        "beta_MNTDq": df_obs,
        "beta_null_mean": df_mean,
        "beta_null_std": df_std,
        "beta_p": df_p,
        "beta_ses": df_ses,
    }

# ---------------------------------------------------------------------------
# beta-iNRIq
# ---------------------------------------------------------------------------
def beta_inriq(
    obj: Union[Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1.0,
    locality: float = 1.0,
    dist_scale: float | str = "auto",
    iterations: int = 999,
    include_conspecifics: bool = True,
    randomization: Literal["features", "abundances"] = "features",
    use_tqdm: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    use_numba: bool = False,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Interpolated β-net relatedness index via a stabilized exponential soft‑minimum, 
    based on interpolated, q-weighted mean phylogenetic distance (β‑iMPD_q) .

    For each directed side (source sample s → target sample t):
      - Use a kernel K_ij(t) ∝ exp(-r * D_ij) over the *allowed* i→j pairs, with target weights w_j^q.
      - Row-wise stabilization: subtract the row maximum in log-space to avoid under/overflow.
      - The directed distance for a source feature i is the kernel-weighted mean of D_ij toward t.
      - If a row’s kernel mass is zero, fall back to the *true* nearest neighbor within the allowed set.

    Endpoints (given a *fixed* conspecific policy):
      • locality = 0   → uniform kernel over allowed pairs ⇒ MPD-like baseline
           - With include_conspecifics=True: equals β‑NRI_q (MPD_q) at r = 0.
           - With include_conspecifics=False: MPD-like baseline that excludes conspecifics.
      • locality → ∞ → hard nearest-neighbor limit within the same allowed pairs
           - With include_conspecifics=False: equals β‑NTI_q (MNTD_q).

    Parameters
    ----------
    obj : dict-like with 'tab' (N taxa × S samples)
        Input data. Must provide at least an abundance table under key 'tab'.
    distmat : pandas.DataFrame (N × N), symmetric
        Pairwise distance matrix whose index/columns include tab.index.
    q : float, default 1.0
        Hill exponent applied to strictly positive relative abundances (zeros remain zero).
    locality : float, default 1.0
        Controls the “locality” of the phylogenetic kernel on a standardized scale
        from MPD-like to nearest-neighbour-like behavior; Locality=0 means uniform kernel
        (fully MPD-like behaviour); locality=1 means intermediate behaviours; and
        locality=>2 means nearest-taxon focus. 
    iterations : int, default 999
        Null iterations (Welford streaming). If < 1, returns only the observed β matrix (DataFrame).
    include_conspecifics : bool, default=True
        Determines whether conspecifics (identical features shared between samples) are allowed 
        to contribute zero-distance matches in the nearest-taxon calculation.
        • True: r=0 equals β‑NRI_q; r→∞ gives an nearest taxon limit that includes conspecifics.
        • False: r=0 is an β‑MPD-like baseline that excludes conspecifics; r→∞ gives an nearest taxon limit that excludes conspecifics.
    randomization : {"features","abundances"}, default "features"
        Null strategy. "features" permutes feature identities (permutes both R and Rq coherently);
        "abundances" shuffles abundances within each sample (column-wise).
    use_tqdm : bool, default True
        Use tqdm for progress bars (falls back to a lightweight stub if unavailable).
    random_state : int or numpy.random.Generator, optional
        Seed or Generator for reproducibility.
    use_numba : bool, optional
        If True, uses Numba path; otherwise uses pure Python implementation.

    Returns
    -------
    dict of pandas.DataFrame (S × S):
        'beta_iMPDq' : observed interpolated β distance
        'NTF'        : symmetric S×S nearest taxon focus index; 0 = beta_MPD‑like, 1 = NT‑like.
        'null_mean'  : mean of null
        'null_std'   : std of null
        'p'          : tie-aware p-index = (count(null < obs) + 0.5 * ties) / iterations
        'ses'        : (null_mean - obs) / null_std

    Notes
    -----
    - A p value close to zero means that the observed iMPDq between samples is lower than the null expectation
    - A p value close to one means that the observed iMPDq between samples is higher than the null expectation
    - A positive ses means that the observed iMPDq between samples is lower than the null expectation
    - A negative ses means that the observed iMPDq between samples is higher than the null expectation
    - Diagonals of all output matrices are set to NaN.
    """
    if "seed" in kwargs:
        if random_state is not None:
            raise TypeError("Specify only one of 'random_state' or 'seed'.")
        random_state = kwargs.pop("seed")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

    # ---- Extract abundance table ----
    tab = get_df(obj, "tab")
    if tab is None or tab.empty:
        raise ValueError("'tab' must be provided in the input.")
    smplist = tab.columns

    # Align distances
    missing_index = sorted(set(tab.index) - set(distmat.index))
    missing_columns = sorted(set(tab.index) - set(distmat.columns))
    if missing_index or missing_columns:
        raise ValueError(
            "distmat must contain all taxa from tab.index in both rows and columns. "
            f"Missing from index: {len(missing_index)} "
            f"(e.g., {missing_index[:5]}). "
            f"Missing from columns: {len(missing_columns)} "
            f"(e.g., {missing_columns[:5]})."
        )

    # Fix D and R
    D = distmat.loc[tab.index, tab.index].to_numpy(dtype=float, copy=True) # (N x N)
    R = (tab / tab.sum(axis=0)).fillna(0).to_numpy(float)       # (N x S)
    N, S = R.shape

    if D.shape[0] != D.shape[1]:
        raise ValueError("distmat must be square.")
    if np.any(~np.isfinite(np.diag(D))):
        raise ValueError("distmat diagonal must be finite.")
    if np.any(D[np.isfinite(D)] < 0):
        raise ValueError("distmat must not contain negative distances.")

    # q-weighting
    Rq = _q_weight(R, q)

    #Calculate distance sensitivity parameter, r
    if locality < 0:
        raise ValueError("locality must be non-negative.")
    dpos = D[np.isfinite(D) & (D > 0)]
    if dpos.size == 0:
        dist_scale_used = np.nan
        r = 0.0
    else:
        if dist_scale == "auto":
            dist_scale_used = float(np.median(dpos))
        
            if not np.isfinite(dist_scale_used) or dist_scale_used <= 0:
                raise ValueError(
                    "Unable to determine a valid automatic distance scale."
                )
        else:
            try:
                dist_scale_used = float(dist_scale)
            except (TypeError, ValueError):
                raise ValueError("dist_scale must be 'auto' or a positive finite number.")
            if not np.isfinite(dist_scale_used) or dist_scale_used <= 0:
                raise ValueError("dist_scale must be 'auto' or a positive finite number.")
        r = np.expm1(locality) / dist_scale_used
        if not np.isfinite(r):
            raise ValueError(
                "Kernel sharpness is not finite. "
                "Try a smaller locality or a larger dist_scale."
            )

    if use_numba:
        try:
            from .accelerate_model import directed_beta_only_numba
            backend = "numba"
        except Exception:
            print('Numba failed, falling back to Python.')
            directed_beta_only_numba = None
            backend = "python"
    else:
        directed_beta_only_numba = None
        backend = "python"

    def _weighted_row_average_to_samples(
        row_values: np.ndarray,
        Rq_used: np.ndarray,
    ) -> np.ndarray:
        """
        Aggregate row-level values to source samples using q-weighted source weights.
    
        row_values : array, shape (N,)
            One value per source taxon row.
        Rq_used : array, shape (N, S)
            q-weighted relative abundances.
    
        Returns
        -------
        out : array, shape (S,)
            Weighted average for each source sample.
        """
        valid = np.isfinite(row_values)
        if not np.any(valid):
            return np.full(Rq_used.shape[1], np.nan, dtype=float)
        vals = np.where(valid, row_values, 0.0)
        W = np.where(valid[:, None], Rq_used, 0.0)
        denom = W.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (W.T @ vals) / denom
        out[denom <= 0] = np.nan
        return out

    def _directed_metrics(
        R_used: np.ndarray,
        Rq_used: np.ndarray,
        diagnostics: bool = True,
    ):
        """
        Compute directed beta-iMPDq and, optionally, directed ENN, NTF,
        ENNmin, and ENNmax.
    
        Matrix convention:
            rows    = source samples
            columns = target samples
    
        For each directed comparison s -> t:
            source taxa are weighted by Rq_used[:, s]
            target taxa are weighted by Rq_used[:, t]
        """
    
        S_local = R_used.shape[1]
        N_local = R_used.shape[0]
    
        A_dir = np.full((S_local, S_local), np.nan, dtype=float)
    
        if diagnostics:
            ENN_dir = np.full((S_local, S_local), np.nan, dtype=float)
            NTF_dir = np.full((S_local, S_local), np.nan, dtype=float)
            ENNmin_dir = np.full((S_local, S_local), np.nan, dtype=float)
            ENNmax_dir = np.full((S_local, S_local), np.nan, dtype=float)
        else:
            ENN_dir = None
            NTF_dir = None
            ENNmin_dir = None
            ENNmax_dir = None
    
        for t in range(S_local):

            # Target taxa present in sample t
            mt = R_used[:, t] > 0.0
    
            if not np.any(mt):
                continue
    
            idx_t = np.where(mt)[0]
            w_target = Rq_used[mt, t].astype(float, copy=False)
    
            if np.sum(w_target > 0.0) == 0.0:
                continue
    
            # Distances from all source taxa to target taxa
            Dt = D[:, mt].astype(float, copy=True)
    
            if not include_conspecifics:
                # Exclude exact taxon matches i == j for target-present taxa
                Dt[idx_t, np.arange(idx_t.size)] = np.inf
    
            finite = np.isfinite(Dt)
            target = finite & (w_target[None, :] > 0.0)
    
            # --------------------------------------------------
            # Kernel matrix
            # --------------------------------------------------
            if r == 0.0 or r < 1e-12:
                A = np.where(target, 1.0, 0.0)
            else:
                A = np.zeros_like(Dt, dtype=float)
                good_rows = target.any(axis=1)
            
                if np.any(good_rows):
                    X = np.full_like(Dt, -np.inf, dtype=float)
                    X[target] = -r * Dt[target]
            
                    row_max = np.max(X[good_rows, :], axis=1, keepdims=True)
            
                    A[good_rows, :] = np.exp(X[good_rows, :] - row_max)
                    A[~np.isfinite(A)] = 0.0
                    A[~target] = 0.0
            
            # Apply target weights
            Aw = A * w_target[None, :]
            denom = Aw.sum(axis=1)
            
            # --------------------------------------------------
            # Directed beta-iMPDq row values
            # --------------------------------------------------
            Dt_safe = np.where(finite, Dt, 0.0)
            numer = (A * Dt_safe) @ w_target
            
            row_vals = np.full(N_local, np.nan, dtype=float)
            valid_rows = denom > 0.0
            
            with np.errstate(invalid="ignore", divide="ignore"):
                row_vals[valid_rows] = numer[valid_rows] / denom[valid_rows]
            
            A_dir[:, t] = _weighted_row_average_to_samples(
                row_vals,
                Rq_used,
            )
            
            if not diagnostics:
                continue
    
            # --------------------------------------------------
            # ENN under current kernel
            # --------------------------------------------------
            P = np.divide(
                Aw,
                denom[:, None],
                out=np.zeros_like(Aw, dtype=float),
                where=denom[:, None] > 0.0,
            )
    
            plogp = np.zeros_like(P, dtype=float)
            pos = P > 0.0
            plogp[pos] = P[pos] * np.log(P[pos])
    
            enn_rows = np.full(N_local, np.nan, dtype=float)
            valid_kernel = denom > 0.0
    
            enn_rows[valid_kernel] = np.exp(
                -np.sum(plogp[valid_kernel, :], axis=1)
            )
    
            # --------------------------------------------------
            # ENNmax: uniform-kernel limit with target q-weights
            # --------------------------------------------------
            W0 = np.where(target, w_target[None, :], 0.0)
            denom0 = W0.sum(axis=1)
    
            P0 = np.divide(
                W0,
                denom0[:, None],
                out=np.zeros_like(W0, dtype=float),
                where=denom0[:, None] > 0.0,
            )
    
            p0logp0 = np.zeros_like(P0, dtype=float)
            pos0 = P0 > 0.0
            p0logp0[pos0] = P0[pos0] * np.log(P0[pos0])
    
            ennmax_rows = np.full(N_local, np.nan, dtype=float)
            valid0 = denom0 > 0.0
    
            ennmax_rows[valid0] = np.exp(
                -np.sum(p0logp0[valid0, :], axis=1)
            )
    
            # --------------------------------------------------
            # ENNmin: nearest-neighbor limit
            # --------------------------------------------------
            Dvalid = np.where(target, Dt, np.inf)
            dmin = np.min(Dvalid, axis=1, keepdims=True)
    
            nearest = (
                target
                & np.isclose(
                    Dvalid,
                    dmin,
                    rtol=1e-10,
                    atol=1e-12,
                )
            )
    
            Wmin = np.where(nearest, w_target[None, :], 0.0)
            denom_min = Wmin.sum(axis=1)
    
            Pmin = np.divide(
                Wmin,
                denom_min[:, None],
                out=np.zeros_like(Wmin, dtype=float),
                where=denom_min[:, None] > 0.0,
            )
    
            pminlogpmin = np.zeros_like(Pmin, dtype=float)
            posmin = Pmin > 0.0
            pminlogpmin[posmin] = Pmin[posmin] * np.log(Pmin[posmin])
    
            ennmin_rows = np.full(N_local, np.nan, dtype=float)
            valid_min = denom_min > 0.0
    
            ennmin_rows[valid_min] = np.exp(
                -np.sum(pminlogpmin[valid_min, :], axis=1)
            )
    
            # --------------------------------------------------
            # NTF from ENN interpolation
            # --------------------------------------------------
            ntf_rows = np.full(N_local, np.nan, dtype=float)
            denom_range = ennmax_rows - ennmin_rows
    
            valid_ntf = (
                np.isfinite(enn_rows)
                & np.isfinite(ennmax_rows)
                & np.isfinite(ennmin_rows)
                & (denom_range > 1e-12)
            )
    
            ntf_rows[valid_ntf] = np.clip(
                (ennmax_rows[valid_ntf] - enn_rows[valid_ntf])
                / denom_range[valid_ntf],
                0.0,
                1.0,
            )
    
            # --------------------------------------------------
            # Aggregate ENN, ENNmin, ENNmax, and NTF over source samples
            # --------------------------------------------------
            A_dir[:, t] = _weighted_row_average_to_samples(
                row_vals,
                Rq_used,
            )
            ENN_dir[:, t] = _weighted_row_average_to_samples(
                enn_rows,
                Rq_used,
            )
            ENNmin_dir[:, t] = _weighted_row_average_to_samples(
                ennmin_rows,
                Rq_used,
            )
            ENNmax_dir[:, t] = _weighted_row_average_to_samples(
                ennmax_rows,
                Rq_used,
            )
            NTF_dir[:, t] = _weighted_row_average_to_samples(
                ntf_rows,
                Rq_used,
            )
    
        if diagnostics:
            return A_dir, ENN_dir, NTF_dir, ENNmin_dir, ENNmax_dir
    
        return A_dir

    # ---- Observed Beta ----
    A_obs, ENN_dir_obs, NTF_dir_obs, ENNmin_dir_obs, ENNmax_dir_obs = (
        _directed_metrics(R, Rq, diagnostics=True)
    )
    
    beta_obs = 0.5 * (A_obs + A_obs.T)
    enn_obs = 0.5 * (ENN_dir_obs + ENN_dir_obs.T)
    ntf_obs = 0.5 * (NTF_dir_obs + NTF_dir_obs.T)
    ennmin_obs = 0.5 * (ENNmin_dir_obs + ENNmin_dir_obs.T)
    ennmax_obs = 0.5 * (ENNmax_dir_obs + ENNmax_dir_obs.T)
    
    # Convert to DataFrames and set diagonals to NaN
    def _to_df(mat):
        arr = np.asarray(mat, dtype=float).copy()
        np.fill_diagonal(arr, np.nan)
        return pd.DataFrame(arr, index=smplist, columns=smplist)
    
    df_obs = _to_df(beta_obs)
    df_enn = _to_df(enn_obs)
    df_ntf = _to_df(ntf_obs)
    df_ennmin = _to_df(ennmin_obs)
    df_ennmax = _to_df(ennmax_obs)
    
    if iterations < 1:
        return {
            "beta_iMPDq": df_obs,
            "beta_ENN": df_enn,
            "beta_NTF": df_ntf,
            "beta_ENN_min": df_ennmin,
            "beta_ENN_max": df_ennmax,
        }

    # ---- Null model (Welford streaming) ----
    if randomization not in {"features", "abundances"}:
        raise ValueError("randomization must be 'features' or 'abundances'.")
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    tqdm = _get_tqdm(use_tqdm)  # progress bar helper (with safe fallback)

    mu = np.zeros_like(beta_obs, dtype=np.float64)
    M2 = np.zeros_like(beta_obs, dtype=np.float64)
    clt = np.zeros_like(beta_obs, dtype=np.int64)  # count(null < obs)
    ceq = np.zeros_like(beta_obs, dtype=np.int64)  # count(null == obs)

    for t in tqdm(
        range(1, iterations + 1),
        desc="iterations",
        unit="iter",
        leave=False,
        ncols=80,
        ascii=True,
        mininterval=0.5,
        position=0,
        miniters=1,
    ):

        # randomization
        if randomization == "features":
            perm = rng.permutation(N)
            R_perm = R[perm, :]
            Rq_perm = Rq[perm, :]
        else:
            R_perm = np.empty_like(R)
            R_perm = np.empty_like(Rq)
            for j in range(S):
                perm = rng.permutation(N)
                R_perm[:, j] = R[perm, j]
                Rq_perm[:, j] = Rq[perm, j]

        if directed_beta_only_numba is not None:
            A_null = directed_beta_only_numba(
                R_perm,
                Rq_perm,
                D,
                float(r),
                include_conspecifics
            )
        else:
            A_null = _directed_metrics(
                R_perm,
                Rq_perm,
                diagnostics=False,
            )
        
        x = 0.5 * (A_null + A_null.T)

        # Welford online updates
        delta = x - mu
        mu += delta / t
        M2 += delta * (x - mu)
        clt += (x < beta_obs)
        ceq += (x == beta_obs)

    denom_var = max(1, iterations - 1)
    null_mean = mu
    null_std = np.sqrt(np.maximum(M2 / denom_var, 0.0))
    p = (clt + 0.5 * ceq) / iterations
    with np.errstate(invalid="ignore", divide="ignore"):
        ses = np.where(null_std > 0, (null_mean - beta_obs) / null_std, np.nan)

    df_mean = pd.DataFrame(null_mean, index=smplist, columns=smplist)
    df_std = pd.DataFrame(null_std, index=smplist, columns=smplist)
    df_p = pd.DataFrame(p, index=smplist, columns=smplist)
    df_ses = pd.DataFrame(ses, index=smplist, columns=smplist)

    # Diagonals to NaN (consistent with your other β functions)
    for df in (df_mean, df_std, df_p, df_ses):
        np.fill_diagonal(df.values, np.nan)
    print('Iterations done with backend '+backend)

    return {
        "beta_iMPDq": df_obs,
        "beta_ENN": df_enn,
        "beta_NTF": df_ntf,
        "beta_ENN_min": df_ennmin,
        "beta_ENN_max": df_ennmax,
        "beta_null_mean": df_mean,
        "beta_null_std": df_std,
        "beta_p": df_p,
        "beta_ses": df_ses,
    }
