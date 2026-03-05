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

    # --- Precompute per-group selection probabilities and sample stats
    features = tab.index.tolist()

    group_specs = []
    for smp_list in groups:
        subtab = tab[smp_list]
        # selection probs
        if randomization == "abundance":
            abund_series = subtab.sum(axis=1).astype(float)            # (N,)
            abund_total = float(abund_series.sum())
            if abund_total <= 0:
                sel_p = np.full(len(features), 1.0 / len(features), dtype=float)
            else:
                sel_p = (abund_series / abund_total).to_numpy()
            # store for within-selected allocation too
            group_specs.append(("abundance", smp_list, sel_p, abund_series))
        else:  # "frequency"
            sub_bin = (subtab > 0).astype(np.int8)
            freq_counts = sub_bin.sum(axis=1).to_numpy(dtype=np.int64)  # (N,)
            if int(freq_counts.sum()) == 0:
                sel_p = np.full(len(features), 1.0 / len(features), dtype=float)
            else:
                sel_p = freq_counts / int(freq_counts.sum())
            # for allocation we still use abundances (fallback uniform if zero)
            abund_series = subtab.sum(axis=1).astype(float)
            group_specs.append(("frequency", smp_list, sel_p, abund_series))

    # per-sample richness and read totals (preserved)
    richness_vec = (tab > 0).sum(axis=0).to_numpy(dtype=np.int64)   # (S,)
    reads_vec = tab.sum(axis=0).to_numpy(dtype=np.int64)            # (S,)
    smp_index = {smp: i for i, smp in enumerate(tab.columns)}

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

    # --- Iteration loop
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
        # fresh randomized table (zeros)
        rtab = pd.DataFrame(0, index=tab.index, columns=tab.columns, dtype=np.int64)

        # randomize within each group
        for mode, smp_list, sel_p, abund_series in group_specs:
            # shared pre-objects
            features_arr = np.array(features, dtype=object)
            smp_cols = smp_list

            for smp in smp_cols:
                sidx = smp_index[smp]
                richness = int(richness_vec[sidx])
                reads = int(reads_vec[sidx])
                if richness <= 0 or reads <= 0:
                    continue

                # 1) draw a set of taxa of size 'richness' without replacement
                rows = rng.choice(features_arr, size=richness, replace=False, p=sel_p)

                # mark presence (1) for selected taxa
                rtab.loc[rows, smp] = 1

                # 2) allocate extra reads to match total counts
                extra = reads - richness
                if extra > 0:
                    # probabilities within the selected set proportional to group abundance
                    sub_abund = abund_series.loc[rows].to_numpy(dtype=float)
                    sub_total = float(sub_abund.sum())
                    if sub_total > 0:
                        sub_p = sub_abund / sub_total
                    else:
                        sub_p = np.full(len(rows), 1.0 / len(rows), dtype=float)
                    # sample with replacement and accumulate counts
                    draws = rng.choice(rows, size=extra, replace=True, p=sub_p)
                    uniq, cnts = np.unique(draws, return_counts=True)
                    rtab.loc[uniq, smp] += cnts

        # compute null beta for this iteration
        null_beta = _beta_for_table(rtab)
        x = null_beta.to_numpy()

        # --- Welford updates
        delta = x - mu
        mu += delta / t
        M2 += delta * (x - mu)

        # --- p-index bookkeeping (Raup–Crick)
        count_lt += (x < obs_arr)
        count_eq += (x == obs_arr)

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
        df_obs = pd.DataFrame(obs, index=smplist, columns=smplist)
        np.fill_diagonal(df_obs.to_numpy(), np.nan)
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
        df_obs = pd.DataFrame(obs, index=smplist, columns=smplist)
        np.fill_diagonal(df_obs.to_numpy(), np.nan)
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
