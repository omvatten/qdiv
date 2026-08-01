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
            return naive_beta(t, q=q)
        if div_type == "phyl":
            return phyl_beta({"tab": t, "tree": tree}, q=q)
        if div_type == "func":
            return func_beta(t, distmat, q=q)
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
    R = (tab / tab.sum(axis=0)).to_numpy()          
    if q == 1.0:
        Rq = R
    else:
        mask = R > 0
        Rq = R.copy()
        Rq[mask] = np.power(Rq[mask], q)

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
    n = Rq.shape[1]
    mu = np.zeros(n, dtype=np.float64)
    M2 = np.zeros(n, dtype=np.float64)
    count_lt = np.zeros(n, dtype=np.int64)
    count_eq = np.zeros(n, dtype=np.int64)

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
            # permute features once and apply to all samples (permute rows of Rq)
            perm = rng.permutation(Rq.shape[0])
            Rq_perm = Rq[perm, :]
            x = _alpha_mpdq(D, Rq_perm)
        elif randomization == "abundances":
            # shuffle abundances within each sample (permute rows per column)
            # permute a view of Rq columnwise
            Rq_perm = np.empty_like(Rq)
            for j in range(n):
                Rq_perm[:, j] = Rq[rng.permutation(Rq.shape[0]), j]
            x = _alpha_mpdq(D, Rq_perm)
    
        # Welford updates
        delta = x - mu
        mu += delta / t
        M2 += delta * (x - mu)
    
        # p-index counts vs observed
        count_lt += (x < obs)
        count_eq += (x == obs)
    
    null_mean = mu
    null_std = np.sqrt(np.maximum(M2 / max(1, (iterations - 1)), 0.0))
    p = (count_lt + 0.5 * count_eq) / iterations
    ses = np.where(null_std > 0, (null_mean - obs) / null_std, np.nan)

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
    - A p value close to zero means that the observed MPNTD is lower than the null expectation
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
    # Relative abundances
    R = (tab / tab.sum(axis=0)).to_numpy(dtype=float)  # (N x S)

    # q-weighting (only for positive entries)
    if q == 1.0:
        Rq = R
    else:
        Rq = R.copy()
        mask_pos = Rq > 0
        Rq[mask_pos] = np.power(Rq[mask_pos], q)

    N, S = Rq.shape

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
    mu = np.zeros(S, dtype=np.float64)       # null_mean (per sample)
    M2 = np.zeros(S, dtype=np.float64)       # for variance
    count_lt = np.zeros(S, dtype=np.int64)   # for p-index
    count_eq = np.zeros(S, dtype=np.int64)

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
            # Permute feature identities once per iteration (relabels rows of Rq)
            perm = rng.permutation(N)
            R_perm = R[perm, :]
            if q == 1.0:
                Rq_perm = R_perm
            else:
                Rq_perm = R_perm.copy()
                posp = Rq_perm > 0.0
                Rq_perm[posp] = np.power(Rq_perm[posp], q)
            x = _mntdq_all(D, R_perm, Rq_perm)
        else:  # "abundances"
            R_perm = np.empty_like(R)
            for j in range(R.shape[1]):
                R_perm[:, j] = R[rng.permutation(R.shape[0]), j]
            if q == 1.0:
                Rq_perm = R_perm
            else:
                Rq_perm = R_perm.copy()
                posp = Rq_perm > 0.0
                Rq_perm[posp] = np.power(Rq_perm[posp], q)
            x = _mntdq_all(D, R_perm, Rq_perm)

        # --- Welford updates (vectorized) ---
        delta = x - mu
        mu += delta / t
        M2 += delta * (x - mu)

        # --- p-index bookkeeping against observed ---
        # count how often null < obs, with 0.5 for ties
        count_lt += (x < obs)
        count_eq += (x == obs)

    # Finalize stats
    null_mean = mu
    # population std estimate across iterations: sqrt(M2 / max(1, iterations-1))
    null_std = np.sqrt(np.maximum(M2 / max(1, (iterations - 1)), 0.0))
    p = (count_lt + 0.5 * count_eq) / iterations
    ses = np.where(null_std > 0, (null_mean - obs) / null_std, np.nan)

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
    iterations: int = 999,
    randomization: Literal["features", "abundances"] = "features",
    use_tqdm: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> pd.DataFrame:
    """
    Interpolated Net Relatedness Index (iNRIq) using a distance soft-min kernel.
    This metric interpolates continuously between MPDq (broad phylogenetic
    relatedness) and MNTDq (nearest-taxon relatedness). Distances are weighted
    using an exponential soft-min kernel controlled by the locality parameter.
    For each taxon i, neighbour weights are computed as
    
        p_ij ∝ exp(-r d_ij) w_j
    
    where d_ij is the pairwise phylogenetic distance, w_j is the q-weighted
    relative abundance of neighbour j, and r is a distance-sensitivity parameter
    derived from locality and the median positive distance in the distance matrix.
    The interpolated distance for taxon i is the kernel-weighted mean distance
    to all other taxa. Community-level iMPDq is then obtained as the q-weighted
    mean of these interpolated distances across taxa.
    In addition to iMPDq, two diagnostics describing kernel locality are returned:
    
    ENN
        Effective Number of Neighbours.
        Computed as the Hill number of order 2 of the kernel weight
        distribution:
    
            ENN_i = 1 / Σ(p_ij²)
    
        Low values indicate that the interpolated distance is determined
        primarily by one or a few nearest neighbours (NTI-like behaviour).
        High values indicate that many neighbours contribute appreciably
        to the distance calculation (MPD-like behaviour).
    
    NTF
        Nearest Taxon Focus.
        A normalized measure of locality:
    
            NTF_i = 1 - (ENN_i - 1) / (k_max,i - 1)
    
        where k_max,i is the number of possible neighbours for taxon i.
        NTF ranges from 0 to 1:
            NTF = 1
                Strong nearest-neighbour focus (MNTD-like).
            NTF = 0
                Broad averaging across all available neighbours (MPD-like).
    
    Parameters
    ----------
    obj : dict or MicrobiomeData
        Object containing a feature abundance table under key 'tab'.    
    distmat : pandas.DataFrame
        Symmetric pairwise phylogenetic or functional distance matrix.    
    q : float, default=1
        Hill-number order used for abundance weighting.    
    locality : float, default=1
        Controls the spatial scale of the soft-min kernel.
        locality = 0 produces broad MPD-like weighting.
        Increasing locality progressively concentrates kernel weight onto
        nearby neighbours, approaching MNTD-like behaviour.
    iterations : int, default=999
        Number of null randomizations.    
    randomization : {"features", "abundances"}, default="features"
        Null-model randomization strategy.
    
    Returns
    -------
    pandas.DataFrame
    
        Columns include:
    
        iMPDq
            Interpolated mean phylogenetic distance.
        ENN
            Effective Number of Neighbours.
        NTF
            Nearest Taxon Focus.
        null_mean
            Mean null expectation.
        null_std
            Standard deviation of null expectations.
        p
            One-sided null-model probability.
        ses
            Standardized effect size.

    Notes
    -----
    - For r = 0, the kernel is uniform; the nearest taxon focus index is therefore defined to be 0.
    - For r → ∞ the method converges to classical MNTD_q.
    - A p value close to zero means that the observed MPNTD is lower than the null expectation
    - A p value close to one means that the observed MNTD is higher than the null expectation
    - A positive ses means that the observed MNTD is lower than the null expectation
    - A negative ses means that the observed MNTD is higher than the null expectation
    """

    # ---- Extract abundance table ----
    tab = get_df(obj, "tab")
    if tab is None or tab.empty:
        raise ValueError("'tab' must be provided in the input.")
    smplist = tab.columns

    # Align distances
    if not set(tab.index).issubset(distmat.index):
        miss = sorted(list(set(tab.index) - set(distmat.index)))
        raise ValueError(f"distmat missing {len(miss)} taxa: {miss[:5]}")

    D = distmat.loc[tab.index, tab.index].to_numpy(copy=True)   # (N x N)
    R = (tab / tab.sum(axis=0)).fillna(0).to_numpy(float)       # (N x S)
    N, S = R.shape

    # q-weighting
    if q == 1.0:
        Rq = R
    else:
        Rq = R.copy()
        pos = Rq > 0
        Rq[pos] = np.power(Rq[pos], q)

    #Calculate distance sensitivity parameter
    if locality < 0:
        raise ValueError("locality must be non-negative.")

    dpos = D[np.isfinite(D) & (D > 0)]
    if dpos.size == 0:
        r = 0
    else:
        dist_scale = np.median(dpos)  # global scale of phylogenetic distances
        r = (np.exp(locality) - 1) / dist_scale   # final distance sensitivity

    # ---- row-wise soft-min operator as function ----
    def softmin_row(Drow: np.ndarray, w_target: np.ndarray) -> float:
        """
        Compute the kernel-weighted soft-min distance for one row i -> a target sample.
        Only finite distances to neighbours with positive q-weight are used.
        """
        valid = np.isfinite(Drow) & (w_target > 0)
        if not np.any(valid):
            return np.nan
    
        Df = Drow[valid]
        wf = w_target[valid]
    
        X = -r * Df
        m = np.max(X)
        a = np.exp(X - m)
    
        denom = np.dot(a, wf)
        if denom <= 0:
            return np.nan
    
        numer = np.dot(a * Df, wf)
        return float(numer / denom)

    # ---- observed values ----
    obs = np.zeros(S, float)
    enn = np.zeros(S, float)
    ntf = np.zeros(S, float)

    for s in range(S):
        w_s = Rq[:, s]
        present_s = R[:, s] > 0

        # Low richness case:
        if present_s.sum() < 2: 
            obs[s] = np.nan
            enn[s] = np.nan
            ntf[s] = np.nan
            continue
        
        # Normal case:
        # Build directed distances for each taxon i in sample s
        dvals = np.zeros(present_s.sum(), float)

        # Build D with conspecific rule for sample t = s
        # (We treat per-sample inriq as a special case of 1-sample-target)
        Dt = D.copy()
        rows = np.where(present_s)[0]
        Dt[rows, rows] = np.inf

        # Compute kernel-softmin row values for each present i
        w_target = w_s.copy()   # target weights = own sample (self-NRI; same as MPD_q definition)
        for k, i in enumerate(rows):
            dvals[k] = softmin_row(Dt[i, :], w_target)

        # MPD/MNTD interpolation (outer weights) = weighted mean of row-level soft-min
        outer_w = w_s[rows]
        valid_d = np.isfinite(dvals) & (outer_w > 0)
        
        if np.any(valid_d):
            obs[s] = float(
                np.sum(outer_w[valid_d] * dvals[valid_d]) /
                np.sum(outer_w[valid_d])
            )
        else:
            obs[s] = np.nan


        # ------------------------------------------------------------
        # ---- Effective number of neighbours and normalized NTF ----
        Dloc = Dt[np.ix_(rows, rows)]          # (k x k), diagonal already inf
        finite = np.isfinite(Dloc)
        
        w_local = w_s[rows]                    # q-weighted target abundances
        enn_rows = np.full(len(rows), np.nan, dtype=float)
        ntf_rows = np.full(len(rows), np.nan, dtype=float)
        
        for kk in range(len(rows)):
            valid = finite[kk] & (w_local > 0)
        
            if not np.any(valid):
                continue
        
            d = Dloc[kk, valid]
            w = w_local[valid]
        
            # Stabilized weighted soft-min probabilities
            X = -r * d
            m = np.max(X)
            a = np.exp(X - m)
        
            denom = np.sum(a * w)
            if denom <= 0:
                continue
        
            p = (a * w) / denom
        
            # Effective number of neighbours actually used by the kernel
            p_safe = p[p > 0]
            ENN_i = np.exp(-np.sum(p_safe * np.log(p_safe)))
            enn_rows[kk] = ENN_i
            
            # Richness-normalized nearest taxon focus
            m_i = np.sum(valid)
            
            if m_i > 1:
                ntf_i = 1.0 - ((ENN_i - 1.0) / (m_i - 1.0))
                ntf_rows[kk] = np.clip(ntf_i, 0.0, 1.0)
            else:
                ntf_rows[kk] = np.nan

        # Sample-level summaries, using the same outer q-weights as iMPDq
        outer_w = w_s[rows]
        valid_enn = np.isfinite(enn_rows) & (outer_w > 0)
        valid_ntf = np.isfinite(ntf_rows) & (outer_w > 0)
        
        if np.any(valid_enn):
            enn[s] = float(
                np.sum(outer_w[valid_enn] * enn_rows[valid_enn]) /
                np.sum(outer_w[valid_enn])
            )
        else:
            enn[s] = np.nan
        
        if np.any(valid_ntf):
            ntf[s] = float(
                np.sum(outer_w[valid_ntf] * ntf_rows[valid_ntf]) /
                np.sum(outer_w[valid_ntf])
            )
        else:
            ntf[s] = np.nan
        
        
    # ---- null distribution ----
    if iterations < 1:
        return pd.DataFrame(
            {
                "iMPDq": obs,
                "ENN": enn,
                "NTF": ntf,
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
            if q == 1.0:
                Rq_perm = R_perm
            else:
                Rq_perm = R_perm.copy()
                posp = Rq_perm > 0
                Rq_perm[posp] = np.power(Rq_perm[posp], q)
        else:
            R_perm = np.empty_like(R)
            for j in range(S):
                R_perm[:, j] = R[rng.permutation(N), j]
            if q == 1.0:
                Rq_perm = R_perm
            else:
                Rq_perm = R_perm.copy()
                posp = Rq_perm > 0
                Rq_perm[posp] = np.power(Rq_perm[posp], q)

        # compute null soft-min iMPDq
        x = np.zeros(S, float)
        for s in range(S):
            w_s = Rq_perm[:, s]
            present_s = w_s > 0
            if present_s.sum() < 2:
                x[s] = np.nan
                continue

            # conspecific rule
            Dt = D.copy()
            idx = np.where(present_s)[0]
            Dt[idx, idx] = np.inf

            rows = np.where(present_s)[0]
            dvals = np.zeros(len(rows), float)
            for k, i in enumerate(rows):
                dvals[k] = softmin_row(Dt[i, :], w_s)

            outer_w = w_s[rows]
            valid_d = np.isfinite(dvals) & (outer_w > 0)
            
            if np.any(valid_d):
                x[s] = float(
                    np.sum(outer_w[valid_d] * dvals[valid_d]) /
                    np.sum(outer_w[valid_d])
                )
            else:
                x[s] = np.nan

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

    return pd.DataFrame(
        {
            "iMPDq": obs,
            "ENN": enn,
            "NTF": ntf,
            "null_mean": null_mean,
            "null_std": null_std,
            "p": p,
            "ses": ses,
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
        'null_mean' : mean of null beta-MPD_q
        'null_std'  : std  of null beta-MPD_q
        'p'         : (count(null < obs) + 0.5 * ties) / iterations
        'ses'       : (null_mean - obs) / null_std

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
            # Permute feature identities once; apply to all samples
            perm = rng.permutation(N)
            Rq_perm = Rq[perm, :]
        else:  # "abundances"
            # Shuffle abundances independently within each sample (permute rows per column)
            Rq_perm = np.empty_like(Rq)
            for j in range(S):
                Rq_perm[:, j] = Rq[rng.permutation(N), j]

        # Null beta-MPD_q (vectorized)
        M_null = D @ Rq_perm
        num_null = Rq_perm.T @ M_null
        z_null = Rq_perm.sum(axis=0)
        den_null = z_null[:, None] * z_null[None, :]

        with np.errstate(invalid="ignore", divide="ignore"):
            x = num_null / den_null  # (S x S)

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
        "null_mean": df_mean,
        "null_std": df_std,
        "p": df_p,
        "ses": df_ses,
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

    Returns
    -------
    dict of pandas.DataFrame
        Full (samples × samples) matrices:
          - 'beta_MNTDq' : observed beta-MNTD_q
          - 'null_mean'  : mean of null beta-MNTD_q
          - 'null_std'   : std  of null beta-MNTD_q
          - 'p'          : (count(null < observed) + 0.5 * ties) / iterations
          - 'ses'        : (null_mean - observed) / null_std
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
    N, S = R.shape

    # q-weighting on positives only (consistent with nriq)
    if q == 1.0:
        Rq = R.copy()
    else:
        Rq = R.copy()
        pos = Rq > 0.0
        Rq[pos] = np.power(Rq[pos], q)

    # Column totals of q-weighted abundances per sample
    z = Rq.sum(axis=0)  # (S,)
    # Where z == 0, we will later produce NaN divisions

    # ---- Helper: compute full directed MNTD_q matrices in a vectorized way ----
    # Given presence mask B (N x S, boolean) and q-weights Rq (N x S),
    # we need two N×S "nearest-to-set" mats:
    #   Delta_col[:, t] = min over j in sample t (D[:, j])
    #   Delta_row[:, s] = min over i in sample s (D[i, :])
    # Then
    #   A = (Rq.T @ Delta_col) / z[:, None]            # s → t
    #   B = ((Rq.T @ Delta_row).T) / z[None, :]        # t → s
    # beta_MNTD_q = 0.5 * (A + B)

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

    def _beta_mntdq_full(D: np.ndarray, R: np.ndarray, Rq: np.ndarray) -> np.ndarray:
        """Observed (or null) full-matrix beta-MNTD_q from D, R (presence), and Rq (q-weights)."""
        B = R > 0.0  # presence/absence for each sample
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
    obs = _beta_mntdq_full(D, R, Rq)  # (S x S)

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
            # Permute feature identities identically across samples
            perm = rng.permutation(N)
            R_perm = R[perm, :]
        else:  # "abundances"
            # Shuffle abundances within each sample (column-wise)
            R_perm = np.empty_like(R)
            for j in range(S):
                R_perm[:, j] = R[rng.permutation(N), j]

        # Recompute q-weights from the permuted R (keeps presence mask consistent with R)
        if q == 1.0:
            Rq_perm = R_perm
        else:
            Rq_perm = R_perm.copy()
            posp = Rq_perm > 0.0
            Rq_perm[posp] = np.power(Rq_perm[posp], q)

        x = _beta_mntdq_full(D, R_perm, Rq_perm)  # (S x S) null draw

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

    return {
        "beta_MNTDq": df_obs,
        "null_mean": df_mean,
        "null_std": df_std,
        "p": df_p,
        "ses": df_ses,
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
    iterations: int = 999,
    include_conspecifics: bool = True,
    randomization: Literal["features", "abundances"] = "features",
    use_tqdm: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
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
    D = distmat.loc[tab.index, tab.index].to_numpy(copy=True)  # (N × N), symmetric
    R = (tab / tab.sum(axis=0)).to_numpy(dtype=float)          # (N × S), relative abundances
    N, S = R.shape

    # q-weighting on positives only
    if q == 1.0:
        Rq = R
    else:
        Rq = R.copy()
        pos = Rq > 0.0
        Rq[pos] = np.power(Rq[pos], q)

    #Calculate distance sensitivity parameter
    dpos = D[D > 0]
    if dpos.size == 0:
        r = 0
    else:
        dist_scale = np.median(dpos)  # global scale of phylogenetic distances
        r = 5.0 * locality / dist_scale            # final distance sensitivity

    def _directed_focus(
        R_used: np.ndarray,
        Rq_used: np.ndarray,
        include_conspecifics_for_focus: bool = False,
    ) -> np.ndarray:
        """
        Column t contains the directed Nearest‑Taxon Focus (NTF) F_{s→t} for all sources s.
        NTF ∈ [0,1]: 0 = MPD‑like (uniform kernel), 1 = kernel mass fully on ε‑nearest block.
        - Uses the same stabilized kernel and ε‑nearest mask as in `inriq`.
        - Conspecific handling is controlled explicitly for focus.
        """
        S_local = R_used.shape[1]
        F_dir = np.full((S_local, S_local), np.nan, dtype=float)
        z_local = Rq_used.sum(axis=0)  # (S,)
    
        # Fast uniform branch for r ≈ 0 to avoid any 0*inf and be explicit
        if r == 0 or r < 1e-12:
            for t in range(S_local):
                mt = R_used[:, t] > 0.0
                if not np.any(mt):
                    continue
    
                Dt = D[:, mt].astype(float, copy=True)  # (N × k)
                # Conspecific policy for focus:
                if not include_conspecifics_for_focus:
                    idx_j = np.where(mt)[0]
                    Dt[idx_j, np.arange(idx_j.size)] = np.inf
    
                finite = np.isfinite(Dt)
                good_rows = finite.any(axis=1)
                # Uniform probabilities over finite entries
                row_counts = finite.sum(axis=1, keepdims=True)
                with np.errstate(invalid="ignore", divide="ignore"):
                    P = np.divide(
                        finite, row_counts,
                        out=np.zeros_like(Dt, dtype=float),
                        where=row_counts > 0
                    )
    
                # ε‑nearest block on raw distances
                minD = np.min(np.where(finite, Dt, np.inf), axis=1)  # (N,)
                # Keep your 5% band; switch to +eps if you prefer an absolute tolerance
                closest = np.zeros_like(Dt, dtype=bool)
                thr = minD[good_rows] * 1.05
                closest[good_rows, :] = finite[good_rows, :] & (Dt[good_rows, :] <= thr[:, None])
    
                # Row mass on the nearest block, then abundance‑weight across rows
                L_rows = (P * closest).sum(axis=1)                  # (N,)
                L_rows[~good_rows] = 0.0
    
                M = Rq_used.T @ L_rows                               # (S,)
                with np.errstate(invalid="ignore", divide="ignore"):
                    F_dir[:, t] = M / z_local
            return F_dir
    
        # Stabilized exponential branch for r > 0
        for t in range(S_local):
            mt = R_used[:, t] > 0.0
            if not np.any(mt):
                continue
    
            Dt = D[:, mt].astype(float, copy=True)  # (N × k)
            # Conspecific policy for focus:
            if not include_conspecifics_for_focus:
                idx_j = np.where(mt)[0]
                Dt[idx_j, np.arange(idx_j.size)] = np.inf
    
            # ---- Robust stabilized kernel (no -r * inf) ----
            finite = np.isfinite(Dt)                                  # allowed entries
            X = np.full_like(Dt, -np.inf, dtype=float)
            X[finite] = -r * Dt[finite]                               # only finite entries participate
            m = np.max(X, axis=1, keepdims=True)                      # (N,1); -inf on fully-bad rows
    
            A = np.zeros_like(Dt, dtype=float)
            good_rows = finite.any(axis=1)                            # rows with ≥1 finite neighbor
            A[good_rows, :] = np.exp(X[good_rows, :] - m[good_rows, :])
    
            row_sum = A.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                P = np.divide(A, row_sum, out=np.zeros_like(A), where=row_sum > 0)
    
            # ε‑nearest block on raw distances
            minD = np.min(np.where(finite, Dt, np.inf), axis=1)       # (N,)
            closest = np.zeros_like(Dt, dtype=bool)
            thr = minD[good_rows] * 1.05
            closest[good_rows, :] = finite[good_rows, :] & (Dt[good_rows, :] <= thr[:, None])
    
            L_rows = (P * closest).sum(axis=1)                        # (N,)
            L_rows[~good_rows] = 0.0
    
            M = Rq_used.T @ L_rows                                    # (S,)
            with np.errstate(invalid="ignore", divide="ignore"):
                F_dir[:, t] = M / z_local                             # (S,)
        return F_dir

    # ---------- Core directed operator with stabilized soft‑min ----------
    def _directed_A(R_used: np.ndarray, Rq_used: np.ndarray) -> np.ndarray:
        """
        Build A_dir (S × S), where column t is the source-weighted average of
        row-wise soft-min distances toward target t, with robust handling of rows
        that have no finite neighbors after masking.
        """
        # --- FAST PATH: r = 0 → uniform kernel (MPD-like baseline) ---
        if r == 0 or r < 1e-12:
            # Compute β-MPDq baseline directly without kernel
            z = Rq_used.sum(axis=0)              # (S,)
            M = D @ Rq_used                      # (N × S)
            num = Rq_used.T @ M                  # (S × S)
            with np.errstate(invalid="ignore", divide="ignore"):
                A_dir = num / (z[:, None] * z[None, :])
            return A_dir

        A_dir = np.full((S, S), np.nan, dtype=float)
        z_local = Rq_used.sum(axis=0)  # (S,)
    
        for t in range(S):
            mt = R_used[:, t] > 0.0       # target presence mask
            if not np.any(mt):
                continue
    
            # Distance slice i(source) -> j(target)
            Dt = D[:, mt].astype(float, copy=True)  # (N × k)
            if not include_conspecifics:
                # Disallow i==j matches by setting to +inf along aligned positions
                idx_j = np.where(mt)[0]
                Dt[idx_j, np.arange(idx_j.size)] = np.inf
    
            # Finite mask and guard rows with no finite neighbors
            finite = np.isfinite(Dt)                 # (N × k) boolean
            good_rows = finite.any(axis=1)           # (N,)
            bad_rows = ~good_rows
    
            # Stabilized kernel: a_ij = exp(-r*D_ij - m_i); only for good rows
            X = -r * Dt                               # (N × k)
            m = np.full(N, -np.inf, dtype=float)      # row-wise max over finite X
            if np.any(good_rows):
                m[good_rows] = np.nanmax(
                    np.where(finite[good_rows, :], X[good_rows, :], -np.inf),
                    axis=1
                )
    
            A = np.zeros_like(Dt, dtype=float)        # (N × k)
            if np.any(good_rows):
                A[good_rows, :] = np.exp(X[good_rows, :] - m[good_rows, None])
                # Remove non-finite entries explicitly
                A[~finite] = 0.0
    
            # Row-wise weighted mean of ORIGINAL distances with target weights
            wp_t = Rq_used[mt, t]                    # (k,)
            row_sum = A @ wp_t                       # (N,)
            # Avoid 0*inf -> NaN by zeroing Dt where not finite
            Dt_safe = np.where(finite, Dt, 0.0)      # (N × k)
            numer = (A * Dt_safe) @ wp_t             # (N,)
    
            row_vals = np.full(N, np.nan, dtype=float)
            valid = row_sum > 0
            with np.errstate(invalid="ignore", divide="ignore"):
                row_vals[valid] = numer[valid] / row_sum[valid]
    
            # Fallback for rows with no allowed neighbor: true nearest taxon within allowed set
            if np.any(bad_rows):
                nn = np.min(np.where(finite, Dt, np.inf), axis=1)    # (N,)
                row_vals[bad_rows] = nn[bad_rows]
    
            # Aggregate across source features with source weights (Rq_used)
            # Rq_used.T: (S × N); row_vals: (N,) -> (S,)
            M = Rq_used.T @ row_vals
            with np.errstate(invalid="ignore", divide="ignore"):
                A_dir[:, t] = M / z_local   # (S,)
    
        return A_dir

    # ---- Observed β ----
    A_obs = _directed_A(R, Rq)
    F_dir_obs = _directed_focus(R, Rq)
    beta_obs = 0.5 * (A_obs + A_obs.T)
    focus_obs = 0.5 * (F_dir_obs + F_dir_obs.T)

    arr = np.asarray(beta_obs, dtype=float).copy()
    np.fill_diagonal(arr, np.nan)
    df_obs = pd.DataFrame(arr, index=smplist, columns=smplist)
    arr = np.asarray(focus_obs, dtype=float).copy()
    np.fill_diagonal(arr, np.nan)
    df_focus = pd.DataFrame(arr, index=smplist, columns=smplist)

    if iterations < 1:
        return {'beta_iMPDq': df_obs, 'NTF': df_focus}

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
        if randomization == "features":
            # Permute feature identities (coherent permutation for both R and Rq)
            perm = rng.permutation(N)
            R_perm = R[perm, :]
            if q == 1.0:
                Rq_perm = R_perm
            else:
                Rq_perm = R_perm.copy()
                posp = Rq_perm > 0.0
                Rq_perm[posp] = np.power(Rq_perm[posp], q)
        else:  # "abundances"
            # Shuffle abundances within each sample
            R_perm = np.zeros_like(R)
            for j in range(S):
                R_perm[:, j] = R[rng.permutation(N), j]
            if q == 1.0:
                Rq_perm = R_perm
            else:
                Rq_perm = R_perm.copy()
                posp = Rq_perm > 0.0
                Rq_perm[posp] = np.power(Rq_perm[posp], q)

        A_null = _directed_A(R_perm, Rq_perm)
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
    df_std  = pd.DataFrame(null_std,  index=smplist, columns=smplist)
    df_p    = pd.DataFrame(p,         index=smplist, columns=smplist)
    df_ses  = pd.DataFrame(ses,       index=smplist, columns=smplist)

    # Diagonals to NaN (consistent with your other β functions)
    for df in (df_mean, df_std, df_p, df_ses):
        np.fill_diagonal(df.values, np.nan)

    return {
        "beta_iMPDq": df_obs,
        "NTF": df_focus,
        "null_mean": df_mean,
        "null_std": df_std,
        "p": df_p,
        "ses": df_ses,
    }
