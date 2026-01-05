import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Literal
from ..diversity import bray, jaccard, naive_beta, phyl_beta, func_beta
from ..utils import get_df

def _get_tqdm(use_tqdm: bool):
    """
    Internal helper that returns tqdm if available and requested; otherwise provides
    a minimal stub compatible with tqdm's API.
    """
    if use_tqdm:
        try:
            from tqdm.auto import tqdm  # type: ignore
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

# ---------------------------------------------------------------------------
# RCQ: Null comparisons for beta-diversity
# ---------------------------------------------------------------------------
def rcq(
    obj: Union[Dict[str, Any], Any],
    *,
    constrain_by: Optional[str] = None,
    randomization: Literal["frequency", "abundance", "weighting"] = "frequency",
    weigh_by: Optional[str] = None,
    weight: float = 1.0,
    iterations: int = 999,
    div_type: Literal["naive", "phyl", "func"] = "naive",
    distmat: Union[pd.DataFrame, None] = None,
    q: float = 1.0,
    compare_by: Optional[str] = None,
    show_progress: bool = True,
    use_tqdm: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Compute Raup–Crick style null comparisons for beta-diversity.

    This function randomizes the abundance table while preserving sample richness and read counts,
    then compares observed dissimilarities against a null expectation. It accepts either a
    MicrobiomeData object or a dict with at least a 'tab' DataFrame.

    Parameters
    ----------
    obj : MicrobiomeData, dict, or compatible object
        Input data. Must provide at least an abundance table ('tab').
        Optionally, may include 'meta' (sample metadata), 'tree' (for phylogenetic measures), etc.
    constrain_by : str, default=None
        Column in metadata to constrain randomization within categories. 'None' randomizes across all samples.
    randomization : {'abundance', 'frequency', 'weighting'}, default="frequency"
        Randomization strategy.
    weigh_by : str, default=None
        Metadata column for weighting (only if randomization="weighting").
    weight : float, default=1.0
        Weight for the lowest-richness category (only if randomization="weighting").
    iterations : int, default=999
        Number of randomizations (increase for more stable null distributions).
    div_type : {'Jaccard', 'Bray', 'naive', 'phyl', 'func'}, default="naive"
        Dissimilarity index to use.
    distmat : pd.DataFrame, default=None
        Functional/trait distance matrix (required for div_type="func").
    q : float, default=1.0
        Diversity order for Hill-number-based indices.
    compare_by : str, default=None
        If not None, averages pairwise comparisons between metadata categories.
    show_progress : bool, default=True
        Whether to show progress bars/messages.
    use_tqdm : bool, default=True
        Whether to use tqdm for progress bars.
    random_state : int or np.random.Generator, optional
        Random seed or generator for reproducibility.

    Returns
    -------
    dict or None
        If compare_by == None, returns:
            - "div_type": str
            - "obs_d": observed dissimilarities (DataFrame)
            - "p": Raup–Crick probability (DataFrame)
            - "null_mean": mean null dissimilarity (DataFrame)
            - "null_std": std null dissimilarity (DataFrame)
            - "ses": standardized effect size (DataFrame)
        If compare_by != None, returns the same statistics averaged per category pairs.
        Returns None if div_type and inputs are inconsistent.

    Notes
    -----
    - Accepts both dict and MicrobiomeData input, using get_df for robust extraction.
    - For functional or phylogenetic indices, requires 'distmat' or 'tree' as appropriate.
    - For speed and reproducibility, uses numpy.random.Generator internally.
    - Probability index counts cases where observed dissimilarity exceeds the null;
      ties contribute 0.5; then normalized by ``iterations``.

    References
    ----------
    - Chase, J.M. (2011) *Ecology Letters*.
    - Stegen, J.C. et al. (2013) *ISME Journal*.
    - Modin, O. et al. (2020) *Microbiome*.
    """
    # Extract tables
    tab = get_df(obj, "tab")
    if tab is None:
        raise ValueError("'tab' is needed in input.")
    tab = tab.copy()        
    
    meta = None
    tree = None
    if hasattr(obj, "meta") or (isinstance(obj, dict) and "meta" in obj):
        try:
            meta = get_df(obj, "meta")
        except Exception:
            meta = None
    if div_type == "phyl":
        tree = get_df(obj, "tree")
    if div_type == "func" and not isinstance(distmat, pd.DataFrame):
        raise ValueError("div_type='func' requires a pandas DataFrame 'distmat'.")
    if div_type == "func":
        distmat = distmat.loc[tab.index, tab.index].copy()

    # Prepare the input dict for downstream code
    input_dict = {"tab": tab}
    if meta is not None:
        input_dict["meta"] = meta
    if tree is not None:
        input_dict["tree"] = tree

    # --- RNG and tqdm -------------------------------------------------------
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    tqdm = _get_tqdm(use_tqdm)

    def _print(msg: str, end: str = "") -> None:
        if show_progress and not use_tqdm:
            print(msg, end=end)

    if iterations < 1:
        raise ValueError("iterations must be >= 1.")

    if randomization not in {"abundance", "frequency", "weighting"}:
        raise ValueError("randomization must be one of {'abundance','frequency','weighting'}.")

    if randomization == "weighting" and (weigh_by is None or meta is None or weigh_by not in meta.columns):
        raise ValueError("randomization='weighting' requires a metadata DataFrame and a valid 'weigh_by' column.")

    if constrain_by is not None and (meta is None or constrain_by not in meta.columns):
        raise ValueError("constrain_by requires a metadata DataFrame containing the specified column.")

    if compare_by is not None and (meta is None or compare_by not in meta.columns):
        raise ValueError("compare_by requires a metadata DataFrame containing the specified column.")

    # --- helper: weighting --------------------------------------------------
    def _weighting_series(wtab: pd.DataFrame, wmeta: pd.DataFrame) -> pd.Series:
        """
        Down-weight the category with lowest richness and return summed abundances.
        """
        cats = pd.unique(wmeta[weigh_by])
        # richness per category = number of SVs present at least once across samples in that category
        richness_by_cat = {}
        for cat in cats:
            samples_group = wmeta.index[wmeta[weigh_by] == cat].tolist()
            sub = wtab[samples_group]
            present = (sub > 0).any(axis=1)  # boolean per SV
            richness_by_cat[cat] = int(present.sum())

        # apply weight to min-richness category (only if there is a difference)
        if len(richness_by_cat) > 1:
            min_cat = min(richness_by_cat, key=richness_by_cat.get)
            # if all equal, no weighting effect
            if richness_by_cat[min_cat] < max(richness_by_cat.values):
                min_samples = wmeta.index[wmeta[weigh_by] == min_cat].tolist()
                wtab[min_samples] = wtab[min_samples] * float(weight)

        return wtab.sum(axis=1)

    # --- randomization core -------------------------------------------------
    def _randomize_tabs() -> List[pd.DataFrame]:
        """Return a list of randomized tables with preserved richness and reads."""
        # Subtab list based on constraint
        if constrain_by is None:
            subtablist: List[pd.DataFrame] = [tab.copy()]
            submeta: List[Optional[pd.DataFrame]] = [meta.copy() if meta is not None else None]
        else:
            subtablist, submeta = [], []
            for cat in pd.unique(meta[constrain_by]):
                subsamples = meta.index[meta[constrain_by] == cat]
                subtablist.append(tab[subsamples])
                submeta.append(meta.loc[subsamples] if meta is not None else None)

        # pre-allocate output tabs
        randomized: List[pd.DataFrame] = [
            pd.DataFrame(0, index=tab.index, columns=tab.columns, dtype=np.int64)
            for _ in range(iterations)
        ]

        # progress
        iter_bar = None
        if show_progress:
            iter_bar = tqdm(
                total=len(subtablist) * iterations,
                desc="Randomizing tables",
                unit="it",
                leave=False,
            )

        SV_index = tab.index
        SV_index_list = SV_index.tolist()

        for sub_i, subtab in enumerate(subtablist):
            wmeta = submeta[sub_i] if submeta[sub_i] is not None else pd.DataFrame(index=subtab.columns)

            # abundances as probabilities
            if randomization == "weighting":
                abund_series = _weighting_series(subtab.copy(), wmeta)
            else:
                abund_series = subtab.sum(axis=1)
            abund_total = float(abund_series.sum())
            if abund_total <= 0:
                raise ValueError("Abundance series is empty; cannot randomize.")

            abund_p = (abund_series / abund_total).to_numpy()

            # frequencies as probabilities
            subtab_bin = (subtab > 0).astype(np.int8)
            freq_counts = subtab_bin.sum(axis=1).to_numpy(dtype=np.int64)
            freq_total = int(freq_counts.sum())
            if freq_total == 0:
                # fall back to uniform to avoid division by zero
                freq_p = np.ones_like(freq_counts, dtype=float) / len(freq_counts)
            else:
                freq_p = freq_counts / freq_total

            smplist = subtab.columns.tolist()
            richnesslist = subtab_bin.sum(axis=0).to_numpy(dtype=np.int64)
            readslist = subtab.sum(axis=0).to_numpy(dtype=np.int64)

            for i in range(iterations):
                # tqdm update
                if iter_bar is not None:
                    iter_bar.update(1)

                # draw for each sample
                for cnr, smp in enumerate(smplist):
                    richness = int(richnesslist[cnr])
                    reads = int(readslist[cnr])

                    if richness == 0:
                        continue

                    #Randomly select features matching richness based on randomization scheme
                    if randomization in {"abundance", "weighting"}:
                        rows = rng.choice(SV_index_list, size=richness, replace=False, p=abund_p)
                    else:  # frequency
                        rows = rng.choice(SV_index_list, size=richness, replace=False, p=freq_p)

                    randomized[i].loc[rows, smp] = 1 #Set count to 1 for selected features (rows)

                    # additional draws to match read counts
                    extra_draws = reads - richness
                    if extra_draws > 0:
                        # probabilities proportional to abundance within selected rows
                        sub_abund = abund_series.loc[rows].to_numpy()
                        sub_total = float(sub_abund.sum())
                        if sub_total > 0:
                            sub_p = sub_abund / sub_total
                        else:
                            sub_p = np.full_like(sub_abund, 1.0 / len(sub_abund), dtype=float)

                        randomchoice = rng.choice(rows, size=extra_draws, replace=True, p=sub_p)
                        # count unique occurrences
                        unique_rows, counts = np.unique(randomchoice, return_counts=True)
                        randomized[i].loc[unique_rows, smp] += counts

        if iter_bar is not None:
            iter_bar.close()
        return randomized

    # --- compute observed beta diversity -----------------------------------
    if div_type == "Bray":
        betadiv = bray(tab)
    elif div_type == "Jaccard":
        betadiv = jaccard(tab)
    elif div_type == "naive":
        betadiv = naive_beta(tab, q=q)
    elif div_type == "phyl":
        betadiv = phyl_beta(input_dict, q=q)
    elif div_type == "func":
        betadiv = func_beta(tab, distmat, q=q)  # type: ignore
    else:
        raise ValueError("Unsupported div_type. Choose among {'Jaccard','Bray','naive','phyl','func'}.")

    # --- randomize and compare ---------------------------------------------
    randomtabs = _randomize_tabs()

    n_samples = len(tab.columns)
    random_beta_all = np.zeros((n_samples, n_samples, iterations), dtype=np.float64)
    RC_tab = pd.DataFrame(0.0, index=tab.columns, columns=tab.columns)

    comp_bar = None
    if show_progress:
        comp_bar = _get_tqdm(use_tqdm)(
            total=iterations,
            desc="Comparing beta diversity",
            unit="iter",
            leave=False,
        )

    for i in range(iterations):
        rtab = randomtabs[i]
        if div_type == "Bray":
            randombeta = bray(rtab)
        elif div_type == "Jaccard":
            randombeta = jaccard(rtab)
        elif div_type == "naive":
            randombeta = naive_beta(rtab, q=q)
        elif div_type == "phyl":
            randombeta = phyl_beta({'tab':rtab, 'tree': input_dict["tree"]}, q=q)
        elif div_type == "func":
            randombeta = func_beta(rtab, distmat, q=q)
        else:
            raise RuntimeError("Inconsistent div_type during randomization comparison.")

        random_beta_all[:, :, i] = randombeta.to_numpy()

        # Raup–Crick counting
        mask_gt = betadiv > randombeta
        RC_tab[mask_gt] = RC_tab[mask_gt] + 1
        mask_eq = betadiv == randombeta
        RC_tab[mask_eq] = RC_tab[mask_eq] + 0.5

        if comp_bar is not None:
            comp_bar.update(1)

    if comp_bar is not None:
        comp_bar.close()

    # finalize matrices
    RC_tab = RC_tab.div(iterations)
    obs_beta = betadiv.to_numpy()
    null_mean_arr = random_beta_all.mean(axis=2)
    null_std_arr = random_beta_all.std(axis=2)

    # safe SES computation
    ses_df_arr = np.full_like(null_mean_arr, np.nan, dtype=np.float64)
    valid = null_std_arr > 0
    ses_df_arr[valid] = (null_mean_arr[valid] - obs_beta[valid]) / null_std_arr[valid]

    null_mean = pd.DataFrame(null_mean_arr, index=RC_tab.index, columns=RC_tab.columns)
    null_std = pd.DataFrame(null_std_arr, index=RC_tab.index, columns=RC_tab.columns)
    ses_df = pd.DataFrame(ses_df_arr, index=RC_tab.index, columns=RC_tab.columns)

    out: Dict[str, pd.DataFrame] = {}

    if compare_by is None:
        out["div_type"] = div_type + "_q=" + str(q)
        out["obs_d"] = betadiv
        out["p"] = RC_tab
        out["null_mean"] = null_mean
        out["null_std"] = null_std
        out["ses"] = ses_df
    else:
        # category-level averaging
        indexlist = RC_tab.index.tolist()
        cats = pd.unique(meta[compare_by])  # type: ignore
        out_RCavg = pd.DataFrame(0.0, index=cats, columns=cats)
        out_RCstd = pd.DataFrame(0.0, index=cats, columns=cats)
        out_nullavg = pd.DataFrame(0.0, index=cats, columns=cats)
        out_nullstd = pd.DataFrame(0.0, index=cats, columns=cats)
        out_sesavg = pd.DataFrame(0.0, index=cats, columns=cats)
        out_sesstd = pd.DataFrame(0.0, index=cats, columns=cats)
        out_obsavg = pd.DataFrame(0.0, index=cats, columns=cats)
        out_obsstd = pd.DataFrame(0.0, index=cats, columns=cats)

        # only evaluate upper triangle and mirror
        for c1_idx in range(len(cats) - 1):
            c1 = cats[c1_idx]
            s1list = meta.index[meta[compare_by] == c1]  # type: ignore
            for c2_idx in range(c1_idx + 1, len(cats)):
                c2 = cats[c2_idx]
                s2list = meta.index[meta[compare_by] == c2]  # type: ignore

                RC_list: List[float] = []
                null_list: List[np.ndarray] = []
                ses_list: List[float] = []
                obs_list: List[float] = []

                for s1 in s1list:
                    s1pos = indexlist.index(s1)
                    for s2 in s2list:
                        s2pos = indexlist.index(s2)
                        RC_list.append(float(RC_tab.loc[s1, s2]))
                        null_list.append(random_beta_all[s1pos, s2pos, :])
                        ses_list.append(float(ses_df.loc[s1, s2]))
                        obs_list.append(float(betadiv.loc[s1, s2]))

                # mean/std
                out_RCavg.loc[c1, c2] = np.mean(RC_list) if len(RC_list) else np.nan
                out_nullavg.loc[c1, c2] = np.mean(null_list) if len(null_list) else np.nan
                out_sesavg.loc[c1, c2] = np.mean(ses_list) if len(ses_list) else np.nan
                out_obsavg.loc[c1, c2] = np.mean(obs_list) if len(obs_list) else np.nan

                out_RCstd.loc[c1, c2] = np.std(RC_list) if len(RC_list) else np.nan
                out_nullstd.loc[c1, c2] = np.std(null_list) if len(null_list) else np.nan
                out_sesstd.loc[c1, c2] = np.std(ses_list) if len(ses_list) else np.nan
                out_obsstd.loc[c1, c2] = np.std(obs_list) if len(obs_list) else np.nan

                # symmetry fill
                out_RCavg.loc[c2, c1] = out_RCavg.loc[c1, c2]
                out_nullavg.loc[c2, c1] = out_nullavg.loc[c1, c2]
                out_sesavg.loc[c2, c1] = out_sesavg.loc[c1, c2]
                out_obsavg.loc[c2, c1] = out_obsavg.loc[c1, c2]

                out_RCstd.loc[c2, c1] = out_RCstd.loc[c1, c2]
                out_nullstd.loc[c2, c1] = out_nullstd.loc[c1, c2]
                out_sesstd.loc[c2, c1] = out_sesstd.loc[c1, c2]
                out_obsstd.loc[c2, c1] = out_obsstd.loc[c1, c2]

        out["div_type"] = div_type
        out["obs_d_mean"] = out_obsavg
        out["obs_d_std"] = out_obsstd
        out["p_mean"] = out_RCavg
        out["p_std"] = out_RCstd
        out["null_mean"] = out_nullavg
        out["null_std"] = out_nullstd
        out["ses_mean"] = out_sesavg
        out["ses_std"] = out_sesstd

    return out


# ---------------------------------------------------------------------------
# NRIq
# ---------------------------------------------------------------------------
def nriq(
    obj: Union[Dict[str, Any], Any],
    distmat: pd.DataFrame,
    *,
    q: float = 1.0,
    iterations: int = 999,
    randomization: Literal["features", "abundances"] = "features",
    show_progress: bool = True,
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
    show_progress : bool, default=True
        Show progress bar/messages.
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
        - 'p'
        - 'ses'

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
    distmat = distmat.loc[tab.index, tab.index]

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    tqdm = _get_tqdm(use_tqdm)

    def get_dmean(ra_series: pd.Series, dm: pd.DataFrame) -> float:
        ras = ra_series[ra_series > 0]
        if ras.empty:
            return np.nan
        pp = np.outer(ras.to_numpy(), ras.to_numpy())
        pp[pp > 0] = pp[pp > 0] ** q
        sum_denom = float(pp.sum())
        dm_sub = dm.loc[ras.index, ras.index].to_numpy()
        mean_d = float((pp * dm_sub).sum()) / sum_denom if sum_denom > 0 else np.nan
        return mean_d

    ra = tab / tab.sum()
    smplist = ra.columns
    output = pd.DataFrame(np.nan, index=smplist, columns=["MPDq", "null_mean", "null_std", "p", "ses"])

    # observed
    for smp in smplist:
        output.loc[smp, "MPDq"] = get_dmean(ra[smp], distmat)

    # null permutations
    darr = np.empty((len(smplist), iterations), dtype=np.float64)
    bar = tqdm(total=iterations, desc="NRIq null permutations", unit="iter", leave=False) if show_progress else None

    for i in range(iterations):
        if randomization == "features":
            # Shuffle the labels of the distance matrix (tip label permutation)
            svlist = distmat.index.tolist()
            rng.shuffle(svlist)
            dm_random = pd.DataFrame(distmat.to_numpy(), index=svlist, columns=svlist)
            for j, smp in enumerate(smplist):
                darr[j, i] = get_dmean(ra[smp], dm_random)
        elif randomization == "abundances":
            # Shuffle abundances within each sample (column)
            ra_shuffled = ra.copy()
            for col in ra_shuffled.columns:
                ra_shuffled[col] = rng.permutation(ra_shuffled[col].values)
            for j, smp in enumerate(smplist):
                darr[j, i] = get_dmean(ra_shuffled[smp], distmat)
        else:
            raise ValueError("randomization must be 'features' or 'abundances'")
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()

    # stats
    for j, smp in enumerate(smplist):
        null_mean = float(darr[j, :].mean())
        null_std = float(darr[j, :].std())
        output.loc[smp, "null_mean"] = null_mean
        output.loc[smp, "null_std"] = null_std
        obs = float(output.loc[smp, "MPDq"])
        p_index = (len(darr[j, :][darr[j, :] < obs]) + 0.5 * len(darr[j, :][darr[j, :] == obs])) / len(darr[j, :])
        output.loc[smp, "p"] = p_index
        if null_std > 0:
            output.loc[smp, "ses"] = (null_mean - obs) / null_std

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
    show_progress: bool = True,
    use_tqdm: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> pd.DataFrame:
    """
    Nearest Taxon Index (NTI) with q-weighting of relative abundances.

    Computes MNTD_q (mean nearest taxon distance with q-weighted abundances),
    contrasts against a null distribution obtained by label permutations of
    the distance matrix or by shuffling abundances within each sample.

    Parameters
    ----------
    obj : MicrobiomeData, dict, or compatible object
        Input data. Must provide at least an abundance table ('tab').
    distmat : pd.DataFrame
        Square distance matrix indexed/columned by feature ids.
    q : float, default=1.0
        Order of diversity weighting applied to relative abundances.
    iterations : int, default=999
        Number of random permutations.
    randomization : {'features', 'abundances'}, default='features'
        Randomization strategy. Shuffle features in the phylogenetic tree
        or relative abundance values in each sample.
    show_progress : bool, default=True
        Show progress bar/messages.
    use_tqdm : bool, default=True
        Use tqdm for progress bars.
    random_state : int or np.random.Generator, optional
        Random seed or generator for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Columns: 'MNTDq', 'null_mean', 'null_std', 'p', 'ses'.

    References
    ----------
    Webb et al. (2002) *American Naturalist*.
    """
    tab = get_df(obj, "tab")
    if tab is None or tab.empty:
        raise ValueError("obj must contain a non-empty pandas DataFrame under key 'tab'.")

    if not set(tab.index).issubset(set(distmat.index)) or not set(tab.index).issubset(set(distmat.columns)):
        missing = sorted(list(set(tab.index) - set(distmat.index)))
        raise ValueError(
            f"distmat must include all feature ids from tab.index. Missing count: {len(missing)} (e.g., {missing[:5]})"
        )
    distmat = distmat.loc[tab.index, tab.index]

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    tqdm = _get_tqdm(use_tqdm)

    def get_dmin(ra_series: pd.Series, dm: pd.DataFrame) -> float:
        ras = ra_series[ra_series > 0]
        if ras.empty:
            return np.nan
        dm_sub = dm.loc[ras.index, ras.index]
        # nearest neighbor distance per SV (exclude 0 distances)
        dmin = dm_sub.where(dm_sub > 0).min(axis=1)
        ras_q = ras.pow(q)
        denom = float(ras_q.sum())
        if denom == 0:
            return np.nan
        return float((ras_q * dmin).sum() / denom)

    ra = tab / tab.sum()
    smplist = ra.columns
    output = pd.DataFrame(np.nan, index=smplist, columns=["MNTDq", "null_mean", "null_std", "p", "ses"])

    # observed
    for smp in smplist:
        output.loc[smp, "MNTDq"] = get_dmin(ra[smp], distmat)

    # null permutations
    darr = np.empty((len(smplist), iterations), dtype=np.float64)
    bar = tqdm(total=iterations, desc="NTIq null permutations", unit="iter", leave=False) if show_progress else None

    for i in range(iterations):
        if randomization == "features":
            svlist = distmat.index.tolist()
            rng.shuffle(svlist)
            dm_random = pd.DataFrame(distmat.to_numpy(), index=svlist, columns=svlist)
            for j, smp in enumerate(smplist):
                darr[j, i] = get_dmin(ra[smp], dm_random)
        elif randomization == "abundances":
            ra_shuffled = ra.copy()
            for col in ra_shuffled.columns:
                ra_shuffled[col] = rng.permutation(ra_shuffled[col].values)
            for j, smp in enumerate(smplist):
                darr[j, i] = get_dmin(ra_shuffled[smp], distmat)
        else:
            raise ValueError("randomization must be 'features' or 'abundances'")
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()

    # stats
    for j, smp in enumerate(smplist):
        null_mean = float(darr[j, :].mean())
        null_std = float(darr[j, :].std())
        output.loc[smp, "null_mean"] = null_mean
        output.loc[smp, "null_std"] = null_std
        obs = float(output.loc[smp, "MNTDq"])
        p_index = (np.sum(darr[j, :] < obs) + 0.5 * np.sum(darr[j, :] == obs)) / len(darr[j, :])
        output.loc[smp, "p"] = p_index
        if null_std > 0:
            output.loc[smp, "ses"] = (null_mean - obs) / null_std

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
    show_progress: bool = True,
    use_tqdm: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Pairwise beta-NRI (q-weighted) between samples.

    Computes beta-MPD_q between sample pairs and contrasts against a null
    distribution obtained by label permutations of the distance matrix or by
    shuffling abundances within each sample.

    Parameters
    ----------
    obj : MicrobiomeData, dict, or compatible object
        Input data. Must provide at least an abundance table ('tab').
    distmat : pd.DataFrame
        Square distance matrix indexed/columned by feature ids.
    q : float, default=1.0
        Order of diversity weighting applied to relative abundances.
    iterations : int, default=999
        Number of random permutations.
    randomization : {'features', 'abundances'}, default='features'
        Randomization strategy. Shuffle features in the phylogenetic tree
        or relative abundance values in each sample.
    show_progress : bool, default=True
        Show progress bar/messages.
    use_tqdm : bool, default=True
        Use tqdm for progress bars.
    random_state : int or np.random.Generator, optional
        Random seed or generator for reproducibility.

    Returns
    -------
    dict
        DataFrames (sample x sample): 'beta_MPDq', 'null_mean', 'null_std',
        'p', and 'ses'.

    References
    ----------
    Webb et al. (2002); Stegen et al. (2013).
    """
    tab = get_df(obj, "tab")
    if tab is None or tab.empty:
        raise ValueError("obj must contain a non-empty pandas DataFrame under key 'tab'.")

    if not set(tab.index).issubset(set(distmat.index)) or not set(tab.index).issubset(set(distmat.columns)):
        missing = sorted(list(set(tab.index) - set(distmat.index)))
        raise ValueError(
            f"distmat must include all feature ids from tab.index. Missing count: {len(missing)} (e.g., {missing[:5]})"
        )
    distmat = distmat.loc[tab.index, tab.index]

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    tqdm = _get_tqdm(use_tqdm)

    def get_bdmean(ra_pair: pd.DataFrame, dm: pd.DataFrame) -> float:
        rap = ra_pair[ra_pair.sum(axis=1) > 0]
        if rap.empty:
            return np.nan
        smp1, smp2 = rap.columns.tolist()
        ra1 = rap[smp1].to_numpy()
        ra2 = rap[smp2].to_numpy()
        pp = np.outer(ra1, ra2)
        pp[pp > 0] = pp[pp > 0] ** q
        sum_denom = float(pp.sum())
        dm_sub = dm.loc[rap.index, rap.index].to_numpy()
        mean_d = float((pp * dm_sub).sum()) / sum_denom if sum_denom > 0 else np.nan
        return mean_d

    ra = tab / tab.sum()
    smplist = ra.columns

    outputMPD = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputNRI = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputAvg = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputStd = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputP = pd.DataFrame(np.nan, index=smplist, columns=smplist)

    total_pairs = (len(smplist) * (len(smplist) - 1)) // 2
    bar = tqdm(total=total_pairs, desc="beta-NRIq pairs", unit="pair", leave=False) if show_progress else None

    # upper triangle iteration
    for i in range(len(smplist) - 1):
        smp1 = smplist[i]
        for j in range(i + 1, len(smplist)):
            smp2 = smplist[j]
            ra_sub = ra[[smp1, smp2]].copy()
            obs_val = get_bdmean(ra_sub, distmat)
            outputMPD.loc[smp1, smp2] = obs_val
            outputMPD.loc[smp2, smp1] = obs_val

            darr = np.empty(iterations, dtype=np.float64)
            for x in range(iterations):
                if randomization == "features":
                    svlist = distmat.index.tolist()
                    rng.shuffle(svlist)
                    dm_random = pd.DataFrame(distmat.to_numpy(), index=svlist, columns=svlist)
                    darr[x] = get_bdmean(ra_sub, dm_random)
                elif randomization == "abundances":
                    ra_shuffled = ra_sub.copy()
                    for col in ra_shuffled.columns:
                        ra_shuffled[col] = rng.permutation(ra_shuffled[col].values)
                    darr[x] = get_bdmean(ra_shuffled, distmat)
                else:
                    raise ValueError("randomization must be 'features' or 'abundances'")

            pval = (np.sum(darr < obs_val) + 0.5 * np.sum(darr == obs_val)) / len(darr)
            outputP.loc[smp1, smp2] = pval
            outputP.loc[smp2, smp1] = pval

            dstd = float(darr.std())
            if dstd > 0:
                bNTI_val = (float(darr.mean()) - obs_val) / dstd
                outputNRI.loc[smp1, smp2] = bNTI_val
                outputNRI.loc[smp2, smp1] = bNTI_val

            mean_val = float(darr.mean())
            outputAvg.loc[smp1, smp2] = mean_val
            outputAvg.loc[smp2, smp1] = mean_val
            outputStd.loc[smp1, smp2] = dstd
            outputStd.loc[smp2, smp1] = dstd

            if bar is not None:
                bar.update(1)

    if bar is not None:
        bar.close()

    return {
        "beta_MPDq": outputMPD,
        "null_mean": outputAvg,
        "null_std": outputStd,
        "p": outputP,
        "ses": outputNRI,
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
    randomization: Literal["features", "abundances"] = "features",
    show_progress: bool = True,
    use_tqdm: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Pairwise beta-NTI (Stegen et al., 2013) with q-weighting of abundances.

    Computes beta-MNTD_q between sample pairs, contrasts against a null
    distribution obtained by label permutations of the distance matrix or by
    shuffling abundances within each sample, and returns per-pair statistics.

    Parameters
    ----------
    obj : MicrobiomeData, dict, or compatible object
        Input data. Must provide at least an abundance table ('tab').
    distmat : pd.DataFrame
        Square distance matrix indexed/columned by feature ids.
    q : float, default=1.0
        Order of diversity weighting applied to relative abundances.
    iterations : int, default=999
        Number of random permutations.
    randomization : {'features', 'abundances'}, default='features'
        Randomization strategy. Shuffle features in the phylogenetic tree
        or relative abundance values in each sample.
    show_progress : bool, default=True
        Show progress bar/messages.
    use_tqdm : bool, default=True
        Use tqdm for progress bars.
    random_state : int or np.random.Generator, optional
        Random seed or generator for reproducibility.

    Returns
    -------
    dict
        DataFrames (sample x sample): 'beta_MNTDq', 'null_mean', 'null_std',
        'p', and 'ses'.

    References
    ----------
    Stegen et al. (2013) *ISME Journal*.
    """
    tab = get_df(obj, "tab")
    if tab is None or tab.empty:
        raise ValueError("obj must contain a non-empty pandas DataFrame under key 'tab'.")

    if not set(tab.index).issubset(set(distmat.index)) or not set(tab.index).issubset(set(distmat.columns)):
        missing = sorted(list(set(tab.index) - set(distmat.index)))
        raise ValueError(
            f"distmat must include all feature ids from tab.index. Missing count: {len(missing)} (e.g., {missing[:5]})"
        )
    distmat = distmat.loc[tab.index, tab.index]

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    tqdm = _get_tqdm(use_tqdm)

    def bMNTDq(ra_pair: pd.DataFrame, dm: pd.DataFrame) -> float:
        rap = ra_pair.copy()
        rap = rap[rap.sum(axis=1) > 0]
        if rap.empty:
            return np.nan

        rap[rap > 0] = rap[rap > 0].pow(q)
        sum_denom = rap.sum().tolist()
        smp1, smp2 = rap.columns.tolist()

        ra1 = rap[smp1][rap[smp1] > 0]
        ra2 = rap[smp2][rap[smp2] > 0]
        dm_sub = dm.loc[ra1.index, ra2.index]

        # nearest neighbor distances for each SV to the opposite sample SV set
        ra1 = ra1.mul(dm_sub.min(axis=1))
        ra2 = ra2.mul(dm_sub.min(axis=0))

        denom1 = float(sum_denom[0]) if sum_denom[0] > 0 else np.nan
        denom2 = float(sum_denom[1]) if sum_denom[1] > 0 else np.nan

        return 0.5 * (float(ra1.sum()) / denom1 + float(ra2.sum()) / denom2)

    ra = tab / tab.sum()
    smplist = ra.columns

    outputMNTD = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputNTI = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputAvg = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputStd = pd.DataFrame(np.nan, index=smplist, columns=smplist)
    outputP = pd.DataFrame(np.nan, index=smplist, columns=smplist)

    total_pairs = (len(smplist) * (len(smplist) - 1)) // 2
    bar = tqdm(total=total_pairs, desc="beta-NTIq pairs", unit="pair", leave=False) if show_progress else None

    for i in range(len(smplist) - 1):
        smp1 = smplist[i]
        for j in range(i + 1, len(smplist)):
            smp2 = smplist[j]
            ra_sub = ra[[smp1, smp2]].copy()
            obs_val = bMNTDq(ra_sub, distmat)
            outputMNTD.loc[smp1, smp2] = obs_val
            outputMNTD.loc[smp2, smp1] = obs_val

            darr = np.empty(iterations, dtype=np.float64)
            for x in range(iterations):
                if randomization == "features":
                    svlist = distmat.index.tolist()
                    rng.shuffle(svlist)
                    dm_random = pd.DataFrame(distmat.to_numpy(), index=svlist, columns=svlist)
                    darr[x] = bMNTDq(ra_sub, dm_random)
                elif randomization == "abundances":
                    ra_shuffled = ra_sub.copy()
                    for col in ra_shuffled.columns:
                        ra_shuffled[col] = rng.permutation(ra_shuffled[col].values)
                    darr[x] = bMNTDq(ra_shuffled, distmat)
                else:
                    raise ValueError("randomization must be 'features' or 'abundances'")

            pval = (np.sum(darr < obs_val) + 0.5 * np.sum(darr == obs_val)) / len(darr)
            outputP.loc[smp1, smp2] = pval
            outputP.loc[smp2, smp1] = pval

            dstd = float(darr.std())
            if dstd > 0:
                bNTI_val = (float(darr.mean()) - obs_val) / dstd
                outputNTI.loc[smp1, smp2] = bNTI_val
                outputNTI.loc[smp2, smp1] = bNTI_val

            mean_val = float(darr.mean())
            outputAvg.loc[smp1, smp2] = mean_val
            outputAvg.loc[smp2, smp1] = mean_val
            outputStd.loc[smp1, smp2] = dstd
            outputStd.loc[smp2, smp1] = dstd

            if bar is not None:
                bar.update(1)

    if bar is not None:
        bar.close()

    return {
        "beta_MNTDq": outputMNTD,
        "null_mean": outputAvg,
        "null_std": outputStd,
        "p": outputP,
        "ses": outputNTI,
    }
