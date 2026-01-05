import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Sequence, Literal, TYPE_CHECKING
import re
from ..utils import get_df

if TYPE_CHECKING:
    from ..data_object import MicrobiomeData

# -----------------------------------------------------------------------------
# Subset object based on list of OTUs/ASVs to keep
# -----------------------------------------------------------------------------
def subset_samples(
    obj: Union["MicrobiomeData", Dict[str, Any]],
    *,
    by: str = "index",
    values: Optional[Union[List[Any], Any]] = None,
    exclude: bool = False,
    keep_absent: bool = False,
    inplace: bool = False,
) -> Union["MicrobiomeData", Dict[str, Any]]:
    """
    Subset samples from a MicrobiomeData object or dictionary containing
    'meta', 'tab', 'seq', 'tax', and optionally 'tree'.

    Parameters
    ----------
    obj : MicrobiomeData or dict
        Either an instance of MicrobiomeData or a dictionary with keys:
        'tab', 'tax', 'seq', 'meta', optionally 'tree'.
    by : str, default "index"
        How to select samples:
        - "index": match sample names in meta.index
        - or a column name in meta (e.g., "Treatment")
    values : list or scalar, optional
        Values to include (or exclude if `exclude=True`).
        If None and `by != "index"`, all unique values of meta[by] are used.
    exclude : bool, default False
        If True, exclude samples that match `values` (inverse selection).
    keep_absent : bool, default False
        If False, drop features (rows) with zero counts after subsetting.
    inplace : bool, default False
        Only relevant when input is a MicrobiomeData object.
        If True, mutate and return the same object. If False, return a new object.

    Returns
    -------
    MicrobiomeData or dict
        Same type as input by default.
        - Object in ⇒ object out (mutated if inplace=True, otherwise a new instance).
        - Dict in ⇒ dict out.

    Raises
    ------
    ValueError
        If `values` is not a list (or scalar convertible to list),
        or `by` is invalid when using metadata filtering.

    Notes
    -----
    - Aligns `self.tab.columns` with `self.meta.index` after filtering.
    - Subsequent components ('seq', 'tax') are subset to remaining features.
    - 'tree' is passed through unchanged.
    """
    # --- Detect input kind without importing MicrobiomeData at module import time ---
    is_object = hasattr(obj, "tab") and hasattr(obj, "meta")

    # --- Extract components in a uniform way ---
    if is_object:
        tab = obj.tab
        tax = obj.tax
        seq = obj.seq
        meta = obj.meta
        tree = obj.tree
    else:
        tab = obj.get("tab")
        tax = obj.get("tax")
        seq = obj.get("seq")
        meta = obj.get("meta")
        tree = obj.get("tree")

    # --- Normalize values to a list, when provided ---
    if values is None and by != "index":
        # If meta exists and by is a column, use all unique values
        if meta is not None and by in meta.columns:
            values = meta[by].dropna().unique().tolist()
        else:
            values = []  # nothing to filter by
    elif values is not None and not isinstance(values, list):
        # Accept scalar and convert to list
        values = [values]

    # --- Compute selected sample names (slist) ---
    slist: Optional[pd.Index] = None
    if meta is not None:
        if by == "index":
            if values is None or len(values) == 0:
                # No values → keep all meta index
                slist = meta.index
            else:
                vals = pd.Index(values)
                slist = meta.index.difference(vals) if exclude else meta.index.intersection(vals)
        elif by in meta.columns:
            mask = meta[by].isin(values) if values else meta[by].notna()
            mask = ~mask if exclude else mask
            slist = meta.index[mask]
        else:
            raise ValueError(f"Variable '{by}' not found in meta columns")
    else:
        # No meta: fall back to tab-only selection via values if provided
        if tab is not None and values:
            vals = pd.Index(values)
            slist = tab.columns.difference(vals) if exclude else tab.columns.intersection(vals)
        else:
            slist = None  # keep all

    # --- Apply subsetting to tab / meta ---
    out_tab = tab
    out_meta = meta
    if out_tab is not None:
        if slist is not None:
            out_tab = out_tab.loc[:, out_tab.columns.intersection(slist)]
        # If meta exists, align meta to kept columns
        if meta is not None:
            if slist is None:
                # keep only samples present in tab
                out_meta = meta.loc[out_tab.columns.intersection(meta.index)]
            else:
                out_meta = meta.loc[slist]
                # ensure tab columns are aligned to filtered meta index
                out_tab = out_tab.loc[:, out_tab.columns.intersection(out_meta.index)]
    elif meta is not None and slist is not None:
        # No tab: just subset meta
        out_meta = meta.loc[slist]

    # --- Optionally drop zero-count features and align seq/tax ---
    out_seq = seq
    out_tax = tax
    if out_tab is not None and not keep_absent:
        keep_features = out_tab.sum(axis=1) > 0
        out_tab = out_tab.loc[keep_features]
        if out_seq is not None:
            out_seq = out_seq.loc[out_tab.index]
        if out_tax is not None:
            out_tax = out_tax.loc[out_tab.index]
    else:
        # keep_absent=True or no tab; still align seq/tax if possible
        if out_tab is not None:
            if out_seq is not None:
                out_seq = out_seq.loc[out_tab.index]
            if out_tax is not None:
                out_tax = out_tax.loc[out_tab.index]

    # --- Build the return value in the same type as input ---
    if is_object:
        if inplace:
            obj.tab = out_tab
            obj.meta = out_meta
            obj.seq = out_seq
            obj.tax = out_tax
            # tree is passed through unchanged
            obj._autocorrect()
            obj._validate()
            return obj
        else:
            # Create a new MicrobiomeData instance lazily to avoid import cycles
            from ..data_object import MicrobiomeData
            new_obj = MicrobiomeData(
                tab=out_tab,
                tax=out_tax,
                meta=out_meta,
                seq=out_seq,
                tree=tree,
            )
            return new_obj
    else:
        out: Dict[str, Any] = {}
        if out_tab is not None: out["tab"] = out_tab
        if out_tax is not None: out["tax"] = out_tax
        if out_seq is not None: out["seq"] = out_seq
        if out_meta is not None: out["meta"] = out_meta
        if tree is not None: out["tree"] = tree
        return out

# -----------------------------------------------------------------------------
# Subset object based on list of features to keep
# -----------------------------------------------------------------------------
def subset_features(
    obj: Union["MicrobiomeData", Dict[str, Any]],
    *,
    featurelist: Optional[List[Any]] = None,
    exclude: bool = False,
    inplace: bool = False,
) -> Union["MicrobiomeData", Dict[str, Any]]:
    """
    Subset features (OTUs/ASVs/bins/MAGs) from a MicrobiomeData object or a dictionary
    containing 'tab', 'tax', 'seq', 'tree', and 'meta'.

    Parameters
    ----------
    obj : MicrobiomeData or dict
        MicrobiomeData object or dictionary containing dataframes (tab, tax, seq, meta) and optionally a tree.
    featurelist : list
        List of feature (OTU/ASV/bin) identifiers to keep or exclude.
    exclude : bool, default False
        If True, exclude values in featurelist instead of including them.
    inplace : bool, default False
        Only relevant when input is a MicrobiomeData object.
        If True, mutate and return the same object. If False, return a new object.

    Returns
    -------
    MicrobiomeData or dict
        Filtered object or dictionary with updated 'tab', 'tax', 'seq', 'tree', and 'meta'.
    """
    # Detect if input is a MicrobiomeData object
    is_object = hasattr(obj, "tab")

    # Extract components
    if is_object:
        tab = obj.tab
        tax = obj.tax
        seq = obj.seq
        tree = obj.tree
        meta = obj.meta
    else:
        tab = obj.get("tab")
        tax = obj.get("tax")
        seq = obj.get("seq")
        tree = obj.get("tree")
        meta = obj.get("meta")

    # Validate featurelist
    if featurelist is None:
        raise ValueError("featurelist must be specified as a list")
    if not isinstance(featurelist, list):
        featurelist = list(featurelist)

    # Subset tab, tax, seq
    out_tab = tab
    out_tax = tax
    out_seq = seq

    if tab is not None:
        if exclude:
            keepix = tab.index.difference(featurelist)
            out_tab = tab.loc[keepix]
        else:
            out_tab = tab.reindex(featurelist).dropna(how="all")
    if tax is not None:
        if exclude:
            keepix = tax.index.difference(featurelist)
            out_tax = tax.loc[keepix]
        else:
            out_tax = tax.reindex(featurelist).dropna(how="all")
    if seq is not None:
        if exclude:
            keepix = seq.index.difference(featurelist)
            out_seq = seq.loc[keepix]
        else:
            out_seq = seq.reindex(featurelist).dropna(how="all")

    # Build output in the same type as input
    if is_object:
        if inplace:
            obj.tab = out_tab
            obj.seq = out_seq
            obj.tax = out_tax
            # tree is passed through unchanged
            obj._autocorrect()
            obj._validate()
            return obj

        else:
            from ..data_object import MicrobiomeData
            new_obj = MicrobiomeData(
                tab=out_tab,
                tax=out_tax,
                seq=out_seq,
                meta=meta,
                tree=tree,
            )
            return new_obj
    else:
        out: Dict[str, Any] = {}
        if out_tab is not None: out["tab"] = out_tab
        if out_tax is not None: out["tax"] = out_tax
        if out_seq is not None: out["seq"] = out_seq
        if tree is not None: out["tree"] = tree
        if meta is not None: out["meta"] = meta
        return out

# -----------------------------------------------------------------------------
# Subset object to the most abundant features
# -----------------------------------------------------------------------------
def subset_abundant(
    obj: Union["MicrobiomeData", Dict[str, Any]],
    *,
    n: int = 25,
    method: Literal["sum", "mean"] = "mean",
    exclude: bool = False,
    inplace: bool = False,
) -> Union["MicrobiomeData", Dict[str, Any]]:
    """
    Keep (or exclude) the most abundant features based on relative abundance.

    The abundance score for each feature is computed from the per-sample
    **relative abundance table** (tab / tab.sum per column), then reduced
    across samples by either 'sum' or 'mean'. Ties are broken by feature index
    order for determinism.

    Parameters
    ----------
    obj : MicrobiomeData or dict
        MicrobiomeData object or a dictionary with keys:
        'tab' (required), and optionally 'tax', 'seq', 'tree', 'meta'.
        - 'tab': DataFrame [features x samples].
        - 'tax': DataFrame [features x tax-levels].
        - 'seq': DataFrame/Series [features].
        - 'tree': kept unchanged.
        - 'meta': kept unchanged (sample metadata).
    n : int, default 25
        Number of top features to keep (or exclude if `exclude=True`).
        Values outside [0, n_features] are clamped to the valid range.
    method : {'sum','mean'}, default 'mean'
        Reduction across samples of **relative abundance** per feature.
        - 'sum'  : total relative abundance across samples
        - 'mean' : mean relative abundance across samples
    exclude : bool, default False
        If False (default), keep the top-n features.
        If True, exclude the top-n features (keep the rest).
    inplace : bool, default False
        Only relevant for MicrobiomeData input.
        If True, mutate the object and return it; otherwise, return a new object.

    Returns
    -------
    MicrobiomeData or dict
        Same type as `obj`, with 'tab','tax','seq' filtered consistently.
        'tree' and 'meta' are passed through unchanged.

    Notes
    -----
    - Relative abundances are computed as: RA = tab / tab.sum(axis=0).
      Samples with zero total are treated as zeros after division.
    - The phylogenetic tree ('tree') is left unchanged. If you want to prune it,
      do so explicitly after calling this function.
    """
    # --- Detect object-vs-dict and extract components ------------------------
    is_object = hasattr(obj, "tab")
    if is_object:
        tab = obj.tab
        tax = getattr(obj, "tax", None)
        seq = getattr(obj, "seq", None)
        tree = getattr(obj, "tree", None)
        meta = getattr(obj, "meta", None)
    else:
        tab = obj.get("tab", None)
        tax = obj.get("tax", None)
        seq = obj.get("seq", None)
        tree = obj.get("tree", None)
        meta = obj.get("meta", None)

    # --- Validate inputs ------------------------------------------------------
    if not isinstance(tab, pd.DataFrame):
        raise ValueError("'tab' must be a pandas DataFrame [features x samples].")
    if tab.shape[0] == 0 or tab.shape[1] == 0:
        raise ValueError("'tab' must have at least one feature and one sample.")
    if method not in ("sum", "mean"):
        raise ValueError("`method` must be 'sum' or 'mean'.")

    # Clamp n
    n_features = tab.shape[0]
    n = int(max(0, min(n, n_features)))

    # --- Compute relative abundances once ------------------------------------
    col_sums = tab.sum(axis=0)
    # Avoid division errors; columns with sum==0 will yield NaN -> fill with 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ra = tab.div(col_sums.where(col_sums != 0), axis=1).fillna(0.0)

    # --- Feature scores and selection ----------------------------------------
    if method == "sum":
        scores = ra.sum(axis=1)
    else:  # "mean"
        scores = ra.mean(axis=1)

    # Deterministic sort: by score desc, then by index asc
    # (pandas' sort_values is stable; we add index as tie-breaker)
    tmp = pd.DataFrame({"__score__": scores})
    tmp["__ix__"] = tmp.index
    tmp = tmp.sort_values(by=["__score__", "__ix__"], ascending=[False, True])
    top_features = tmp.index[:n]

    # --- Build keep mask ------------------------------------------------------
    if n == 0:
        keep_mask = pd.Series(True, index=tab.index) if exclude else pd.Series(False, index=tab.index)
    else:
        in_top = tab.index.isin(top_features)
        keep_mask = ~in_top if exclude else in_top

    # Final feature index to keep
    keep_idx = tab.index[keep_mask]

    # --- Subset components consistently --------------------------------------
    out_tab = tab.loc[keep_idx]
    out_tax = tax.loc[keep_idx] if isinstance(tax, pd.DataFrame) else tax
    # seq may be a DataFrame or Series; subset by index when possible
    if isinstance(seq, pd.DataFrame) or isinstance(seq, pd.Series):
        out_seq = seq.loc[seq.index.intersection(keep_idx)]
    else:
        out_seq = seq

    # tree and meta: pass through unchanged (see note)
    out_tree = tree
    out_meta = meta

    # --- Return in same type as input ----------------------------------------
    if is_object:
        if inplace:
            obj.tab = out_tab
            obj.tax = out_tax
            obj.seq = out_seq
            # leave tree and meta unchanged
            if hasattr(obj, "_autocorrect"): obj._autocorrect()
            if hasattr(obj, "_validate"): obj._validate()
            return obj
        else:
            from ..data_object import MicrobiomeData  # local import to avoid cycles
            new_obj = MicrobiomeData(
                tab=out_tab,
                tax=out_tax,
                seq=out_seq,
                meta=out_meta,
                tree=out_tree,
            )
            return new_obj
    else:
        # return a NEW dict; do not mutate original dict
        return {
            "tab": out_tab.copy(deep=False),
            "tax": out_tax.copy(deep=False) if isinstance(out_tax, pd.DataFrame) else out_tax,
            "seq": out_seq.copy(deep=False) if isinstance(out_seq, (pd.DataFrame, pd.Series)) else out_seq,
            "tree": out_tree,
            "meta": out_meta,
        }

# -------------------------------------------------------------------------
# Subset object based on text patterns in taxonomic names
# -------------------------------------------------------------------------
def subset_taxa(
    obj: Union["MicrobiomeData", Dict[str, Any]],
    *,
    subset_levels: Optional[Union[str, Sequence[str]]] = None,
    subset_patterns: Optional[Union[str, Sequence[str]]] = None,
    exclude: bool = False,
    case: bool = False,
    regex: bool = True,
    match_type: Literal["contains", "fullmatch", "startswith", "endswith"] = "contains",
    inplace: bool = False,
) -> Union["MicrobiomeData", Dict[str, Any]]:
    """
    Subset features (OTUs/ASVs/bins/MAGs) from a MicrobiomeData object or a dictionary
    based on taxonomic classification.

    Parameters
    ----------
    obj : MicrobiomeData or dict
        MicrobiomeData object or dictionary containing dataframes (tab, tax, seq, meta) and optionally a tree.
    subset_levels : str or sequence of str, optional
        Taxonomic column(s) in which to search for patterns. If None, all columns in `tax` are used.
    subset_patterns : str or sequence of str
        Text patterns to identify taxa to keep. If a single string is passed, it is used as the only pattern.
    exclude : bool, default False
        If True, return taxa that do NOT match the given patterns (i.e., complement).
    case : bool, default False
        If True, pattern matching is case-sensitive.
    regex : bool, default True
        If True, patterns are treated as regex. If False, patterns are escaped (literal match).
    match_type : {'contains','fullmatch','startswith','endswith'}, default 'contains'
        Matching behavior applied to the strings in selected columns.
    inplace : bool, default False
        Only relevant when input is a MicrobiomeData object. If True, mutate and return the same object.

    Returns
    -------
    MicrobiomeData or dict
        Filtered object or dictionary with updated 'tab', 'tax', and 'seq'. 'meta' and 'tree' are passed through.
    """
    # --- Detect if input is a MicrobiomeData-like object ---
    is_object = hasattr(obj, "tax")

    # --- Extract taxonomy table ---
    tax = get_df(obj, "tax")
    if tax is None:
        raise ValueError("Object must contain a 'tax' dataframe")

    # --- Normalize patterns ---
    if subset_patterns is None:
        raise ValueError("subset_patterns must be specified")
    if isinstance(subset_patterns, str):
        patterns: list[str] = [subset_patterns]
    else:
        patterns = list(subset_patterns)
        if len(patterns) == 0:
            raise ValueError("subset_patterns must be a non-empty string or list of strings")

    # Escape when regex=False
    if not regex:
        patterns = [re.escape(p) for p in patterns]

    # Build one union pattern for efficiency
    union = "|".join(patterns)

    # Adapt the union for starts/ends; 'fullmatch' uses Series.str.fullmatch directly
    if match_type == "startswith":
        union = rf"(?:{union})"
        # Use '^' anchor to simulate startswith via contains
        union = rf"^(?:{union})"
    elif match_type == "endswith":
        union = rf"(?:{union})$"

    # --- Normalize subset_levels ---
    if subset_levels is None:
        columns = list(tax.columns)
    elif isinstance(subset_levels, str):
        if subset_levels not in tax.columns:
            raise ValueError(f"'{subset_levels}' not found in tax columns")
        columns = [subset_levels]
    else:
        columns = list(subset_levels)
        missing = [lvl for lvl in columns if lvl not in tax.columns]
        if missing:
            raise ValueError(f"subset_levels contains invalid columns: {missing}")

    # --- Build a single boolean mask across all selected columns ---
    mask = pd.Series(False, index=tax.index)

    for col in columns:
        s = tax[col].astype(str)
        if match_type == "fullmatch":
            col_mask = s.str.fullmatch(union, case=case, na=False, regex=True)
        else:
            col_mask = s.str.contains(union, case=case, na=False, regex=True)
        mask |= col_mask  # OR across columns

    if exclude:
        mask = ~mask

    keep_idx = tax.index[mask]  # preserves original order

    # Optional: warn if nothing matched
    if len(keep_idx) == 0:
        # You may prefer to raise instead:
        # raise ValueError("No taxa matched the provided patterns/levels.")
        # Here we return an object with empty rows.
        pass

    # --- Helper for safe selection on possibly-missing keys ---
    def _take(df):
        return df.loc[keep_idx] if df is not None else None

    # --- Build output in the same type as input ---
    if is_object:
        if inplace:
            obj.tab = _take(obj.tab)
            obj.tax = _take(obj.tax)
            obj.seq = _take(obj.seq)

            if hasattr(obj, "_autocorrect"):
                obj._autocorrect()
            if hasattr(obj, "_validate"):
                obj._validate()
            return obj
        else:
            # Import locally to avoid hard dependency if not needed
            try:
                from ..data_object import MicrobiomeData  # adjust path if needed
            except Exception:
                MicrobiomeData = None  # type: ignore

            if MicrobiomeData is not None:
                new_obj = MicrobiomeData(
                    tab=_take(obj.tab),
                    tax=_take(obj.tax),
                    seq=_take(obj.seq),
                    meta=obj.meta,
                    tree=obj.tree,
                )
                if hasattr(new_obj, "_autocorrect"):
                    new_obj._autocorrect()
                if hasattr(new_obj, "_validate"):
                    new_obj._validate()
                return new_obj
            else:
                # Fallback: return a dict if class isn't importable
                return {
                    "tab": _take(getattr(obj, "tab", None)),
                    "tax": _take(getattr(obj, "tax", None)),
                    "seq": _take(getattr(obj, "seq", None)),
                    "meta": getattr(obj, "meta", None),
                    "tree": getattr(obj, "tree", None),
                }
    else:
        # dict input
        out: Dict[str, Any] = {}
        out["tab"] = _take(obj.get("tab"))
        out["tax"] = _take(obj.get("tax"))
        out["seq"] = _take(obj.get("seq"))
        out["tree"] = obj.get("tree")
        out["meta"] = obj.get("meta")
        return out

# -------------------------------------------------------------------------
# Merge samples
# -------------------------------------------------------------------------
def merge_samples(
    obj: Union["MicrobiomeData", Dict[str, Any]],
    *,
    by: Union[List[str], str],
    values: Optional[List[Any]] = None,
    method: str = "sum",
    keep_absent: bool = False,
    inplace: bool = False
) -> Union["MicrobiomeData", Dict[str, Any]]:
    """
    Merge samples based on metadata grouping.

    Parameters
    ----------
    obj : MicrobiomeData or dict
        Object or dictionary containing 'tab', 'tax', 'seq', 'meta', and optionally 'tree'.
    by : str or list
        Column(s) in metadata used for grouping samples.
    values : list, optional
        Metadata values to keep. If None, all unique values in `by` are used.
    method : {'sum', 'mean'}, default 'sum'
        Aggregation method for counts.
    keep_absent : bool, default False
        If False, remove features with zero counts after merging.
    inplace : bool, default False
        If True and obj is MicrobiomeData, mutate and return same object.

    Returns
    -------
    MicrobiomeData or dict
        Object with merged samples.
    """
    # Detect object type
    is_object = hasattr(obj, "meta")
    meta = get_df(obj, "meta")

    if meta is None or meta.empty:
        raise ValueError("Object must contain 'meta' dataframe")

    # Extract tab
    tab = get_df(obj, "tab")
    if tab is None:
        raise ValueError("Object must contain 'tab' dataframe")

    if isinstance(by, str):
        by = [by]
    by = list(set(by).intersection(meta.columns))
    if len(by) == 0:
        raise ValueError("Column(s) not found in metadata")

    if values is not None and isinstance(values, str):
        values = [values]
    if values is not None and not isinstance(values, list):
        raise ValueError("Values must be None or a list")

    if method not in ["sum", "mean"]:
        raise ValueError("method must be 'sum' or 'mean'")

    # Determine values to keep
    if values is None:
        meta_df = meta.copy()
    elif isinstance(values, list):
        keepsmp = []
        for c in by:
            for v in values:
                keepsmp = keepsmp + meta[meta[c]==v].index.tolist()
        keepsmp = set(keepsmp)
        keepix = [s for s in meta.index if s in keepsmp]
        meta_df = meta.loc[keepix]

    if meta_df.empty:
        raise ValueError("No samples match the specified values")

    if len(by) == 1:
        ix = by[0]
    else:
        ix = '-'.join(by)
        meta_df[ix] = meta_df[by].astype(str).agg('-'.join, axis=1)

    # Group metadata by ix
    meta_grouped = meta_df.groupby(ix).first()
    meta_grouped[ix] = meta_grouped.index

    # Subset tab to selected samples
    tab_filtered = tab.loc[:, meta_df.index]

    # Map each selected sample (column) to its group label
    group_labels = meta_df[ix]
    
    # Aggregate by transposing -> group rows -> aggregate -> transpose back
    if method == "sum":
        merged_tab = tab_filtered.T.groupby(group_labels).sum().T
    else:  # method == "mean"
        merged_tab = tab_filtered.T.groupby(group_labels).mean().T

    # Remove zero-count features if requested
    if not keep_absent:
        keep_features = merged_tab.index[merged_tab.sum(axis=1) > 0]
        merged_tab = merged_tab.loc[keep_features]
    else:
        keep_features = merged_tab.index

    # Helper for safe selection
    def _take(df):
        return df.loc[keep_features] if df is not None else None

    # Build output
    if is_object:
        if inplace:
            obj.tab = merged_tab
            obj.tax = _take(obj.tax)
            obj.seq = _take(obj.seq)
            obj.meta = meta_grouped
            # Tree pruning optional
            if hasattr(obj, "_autocorrect"): obj._autocorrect()
            if hasattr(obj, "_validate"): obj._validate()
            return obj
        else:
            from ..data_object import MicrobiomeData
            return MicrobiomeData(
                tab=merged_tab,
                tax=_take(obj.tax),
                seq=_take(obj.seq),
                meta=meta_grouped,
                tree=obj.tree
            )
    else:
        out: Dict[str, Any] = {
            "tab": merged_tab,
            "tax": _take(obj.get("tax")),
            "seq": _take(obj.get("seq")),
            "meta": meta_grouped,
            "tree": obj.get("tree")
        }
        return out

def _rarefy_table(
    tab: pd.DataFrame,
    depth: Union[int, str] = "min",
    seed: Optional[int] = None,
    replacement: bool = False
) -> pd.DataFrame:
    """
    Rarefy a frequency table to a fixed sequencing depth.

    Parameters
    ----------
    tab : pd.DataFrame
        Frequency table (features x samples).
    depth : int or 'min'
        Target sequencing depth per sample.
    seed : int, optional
        Random seed for reproducibility.
    replacement : bool, default False
        Sample with replacement if True.

    Returns
    -------
    pd.DataFrame
        Rarefied table with samples as columns and features as rows.
    """
    tab = tab.fillna(0).astype(int)

    # Determine depth
    if depth == "min":
        depth = int(tab.sum().min())
    else:
        depth = int(depth)
    if depth <= 0:
        raise ValueError("Depth must be a positive integer.")

    rng = np.random.default_rng(seed)

    if replacement:
        # Multinomial sampling
        nvar = len(tab.index)
        rtab = tab.copy()
        for s in tab.columns:
            total_reads = tab[s].sum()
            if total_reads < depth:
                rtab = rtab.drop(s, axis=1)
                continue
            p = tab[s] / total_reads
            choice = rng.choice(nvar, size=depth, p=p)
            rtab[s] = np.bincount(choice, minlength=nvar)
    else:
        # Without replacement
        rtab = pd.DataFrame(0, index=tab.index, columns=tab.columns)
        for s in tab.columns:
            total_reads = tab[s].sum()
            if total_reads < depth:
                rtab = rtab.drop(s, axis=1)
                continue
            smp_series = tab[s][tab[s] > 0]
            name_arr = smp_series.index.to_numpy()
            counts_arr = smp_series.to_numpy()
            ind_reads_arr = np.repeat(name_arr, counts_arr)
            rng.shuffle(ind_reads_arr)
            bins, counts = np.unique(ind_reads_arr[:depth], return_counts=True)
            rtab.loc[bins, s] = counts

    return rtab


def rarefy(
    obj: Union["MicrobiomeData", Dict[str, Any]],
    *,
    depth: Union[int, str] = "min",
    seed: Optional[int] = None,
    replacement: bool = False,
    inplace: bool = False
) -> Union["MicrobiomeData", Dict[str, Any]]:
    """
    Rarefy the abundance table in a MicrobiomeData object or dictionary.

    Parameters
    ----------
    obj : MicrobiomeData or dict
        Object containing at least 'tab' (features x samples).
    depth : int or 'min'
        Target sequencing depth per sample.
    seed : int, optional
        Random seed for reproducibility.
    replacement : bool, default False
        Sample with replacement if True.
    inplace : bool, default False
        If True and obj is MicrobiomeData, modify in place.

    Returns
    -------
    MicrobiomeData or dict
        Object with rarefied table and zero-count features removed.
    """
    # Extract tab
    is_object = hasattr(obj, "tab")
    tab = get_df(obj, "tab")
    if tab is None:
        raise ValueError("Object must contain an abundance table ('tab').")

    # Rarefy table
    rtab = _rarefy_table(tab, depth=depth, seed=seed, replacement=replacement)

    # Remove zero-count features
    keep_features = rtab.index[rtab.sum(axis=1) > 0]
    keep_samples = rtab.columns

    def _take(df):
        return df.loc[keep_features] if df is not None else None

    if is_object:
        if inplace:
            obj.tab = rtab.loc[keep_features, keep_samples]
            obj.tax = _take(obj.tax)
            obj.seq = _take(obj.seq)
            obj.meta = obj.meta.loc[keep_samples] if obj.meta is not None else None
            obj._autocorrect()
            obj._validate()
            return obj
        else:
            return type(obj)(
                tab=rtab.loc[keep_features, keep_samples],
                tax=_take(obj.tax),
                seq=_take(obj.seq),
                meta=obj.meta.loc[keep_samples] if obj.meta is not None else None,
                tree=obj.tree
            )
    else:
        robj = obj.copy()
        robj["tab"] = rtab.loc[keep_features, keep_samples]
        if robj.get("tax") is not None:
            robj["tax"] = robj["tax"].loc[keep_features]
        if robj.get("seq") is not None:
            robj["seq"] = robj["seq"].loc[keep_features]
        if robj.get("meta") is not None:
            robj["meta"] = robj["meta"].loc[keep_samples]
        return robj
