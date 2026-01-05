from __future__ import annotations
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
from pathlib import Path
from typing import Dict, List, Union, Tuple, Any, Literal
from ..utils import sort_index_by_number, rename_leaves, get_df, subset_tree
from ..data_object import MicrobiomeData

__all__ = [
    "sequence_distance_matrix",
    "tree_distance_matrix",
    "align",
    "consensus",
    "merge_objects"
]

# -----------------------------------------------------------------------------
#  Distance matrix based on sequences 
# -----------------------------------------------------------------------------
def sequence_distance_matrix(
    obj: Union["MicrobiomeData", Dict[str, Any]],
    *,
    savename: str = "SeqDistMat",
    path: str = "",
    band_width: int = 12,
    save: bool = True,
    use_numba: bool = True,
) -> Dict[str, Any]:
    """
    Compute pairwise Levenshtein distances with a parallelized, Numba-accelerated
    banded Wagner–Fischer algorithm (if use_numba=True), else pure Python fallback.

    Parameters
    ----------
    obj : MicrobiomeData or dict
        Must provide a DataFrame in `obj.seq` or `obj['seq']` with index=sequence IDs
        and a column containing the sequences (default name: 'seq').
    savename : str, optional
        Base filename for CSV outputs. Default 'SeqDistMat'.
    path : str, default ""
        Directory path (absolute or relative) where output is saved. Can be "" for CWD.
    band_width : int, optional
        Sakoe–Chiba band half-width (expanded automatically to |len1-len2|).
        Larger values increase accuracy (approach exact DP) but reduce speed.
        Default 12.
    save : bool, optional
        If True, writes two CSVs: edits and normalized.
    use_numba : bool, optional
        If True, uses Numba path; otherwise uses pure Python implementation.

    Returns
    -------
    Dict[str, Any]
        {
          'edits': pd.DataFrame,       # int distances
          'normalized': pd.DataFrame,  # float in [0, 1]
          'meta': {'backend': 'numba'|'python'}
        }
    """
    # Detect sequences DataFrame
    is_object = hasattr(obj, "seq")
    seq_df = obj.seq if is_object else obj.get("seq")

    # Validation
    if seq_df is None or not isinstance(seq_df, pd.DataFrame):
        raise ValueError("Input must contain a 'seq' DataFrame (obj.seq or obj['seq']).")
    if seq_df.index is None or len(seq_df.index) == 0:
        raise ValueError("Sequences DataFrame has empty or missing index (sequence IDs).")

    # Extract IDs and sequences (coerce to strings, NaN -> "")
    ids = [str(i) for i in seq_df.index.tolist()]
    seqs = [("" if pd.isna(v) else str(v)) for v in seq_df["seq"].tolist()]

    print('sequence_distance_matrix: Running calculations...')
    backend = "python"
    if use_numba:
        try:
            from .accelerate import compute_distance_matrix_numba
            dmat, nmat = compute_distance_matrix_numba(ids, seqs, band_width)
            backend = "numba"
        except Exception:
            # Fallback to pure-Python if numba missing or compilation fails
            dmat, nmat = _compute_distance_matrix_python(ids, seqs, band_width)
            backend = "python"
    else:
        dmat, nmat = _compute_distance_matrix_python(ids, seqs, band_width)
        backend = "python"

    # Wrap into DataFrames
    edits_df = pd.DataFrame(dmat, index=ids, columns=ids)
    norm_df = pd.DataFrame(nmat, index=ids, columns=ids)

    # Save if requested
    if save:
        file_path_edits = Path(path) / f"{savename}_edits.csv"
        edits_df.to_csv(file_path_edits)
        file_path_norm = Path(path) / f"{savename}_normalized.csv"
        norm_df.to_csv(file_path_norm)
        print("Sequence distance matrices saved.")

    print('Done!')
    return {'edits': edits_df, 'normalized': norm_df, 'meta': {'backend': backend}}


def _compute_distance_matrix_python(ids, seqs, band_width: int = 12):
    """
    Pure-Python fallback that mirrors the banded behavior with safe sentinels.
    """
    import numpy as np

    n = len(ids)
    dmat = np.zeros((n, n), dtype=np.int64)
    nmat = np.zeros((n, n), dtype=np.float64)

    def levenshtein_band_py(s1: str, s2: str, band: int) -> int:
        len1 = len(s1); len2 = len(s2)
        if s1 == s2: return 0
        if len1 == 0: return len2
        if len2 == 0: return len1
        if band < 0: band = 0
        diff = abs(len1 - len2)
        if band < diff: band = diff

        sentinel = len1 + len2 + 1
        m = np.full((len1 + 1, len2 + 1), sentinel, dtype=np.int64)
        for i in range(len1 + 1): m[i, 0] = i
        for j in range(len2 + 1): m[0, j] = j

        for i in range(1, len1 + 1):
            j_lo = 1 if i - band < 1 else i - band
            j_hi_excl = i + band + 1
            if j_hi_excl > (len2 + 1): j_hi_excl = len2 + 1
            for j in range(j_lo, j_hi_excl):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                d  = m[i - 1, j]     + 1
                ins= m[i,     j - 1] + 1
                sub= m[i - 1, j - 1] + cost
                tmp = d if d < ins else ins
                m[i, j] = sub if sub < tmp else tmp
        return int(m[len1, len2])

    lengths = [len(s) for s in seqs]
    for i in range(n):
        dmat[i, i] = 0
        nmat[i, i] = 0.0
        li = lengths[i]
        for j in range(i + 1, n):
            lj = lengths[j]
            dist = levenshtein_band_py(seqs[i], seqs[j], band_width)
            dmat[i, j] = dist
            dmat[j, i] = dist
            denom = li if li > lj else lj
            val = 0.0 if denom == 0 and dist == 0 else (1.0 if denom == 0 else dist / denom)
            nmat[i, j] = val
            nmat[j, i] = val
    return dmat, nmat

# -----------------------------------------------------------------------------
#  Distance matrix based on tree 
# -----------------------------------------------------------------------------
def tree_distance_matrix(
        obj: Union["MicrobiomeData", Dict[str, Any]],
        *,
        savename: str = "TreeDistMat",
        path: str = "",
        save: bool = True,
) -> pd.DataFrame:
    """
    Compute pairwise phylogenetic distances between leaf nodes in a tree.

    Parameters
    ----------
    obj : MicrobiomeData or dict
        Must provide a DataFrame in `obj.tree` or `obj['tree']`
    savename : str, optional
        Base filename for CSV outputs. Default 'TreeDistMat'.
    path : str, default ""
        Directory path (absolute or relative) where output is saved. Can be "" for CWD.
    save : bool, optional
        If True, writes a CSV file.

    Returns
    -------
    pandas.DataFrame
        Symmetric distance matrix with leaf node names as both rows and columns.
    """
    # Detect sequences DataFrame
    is_object = hasattr(obj, "tree")
    tree_df = obj.tree if is_object else obj.get("tree")

    # Validation
    if tree_df is None or not isinstance(tree_df, pd.DataFrame):
        raise ValueError("Input must contain a 'tree' DataFrame (obj.tree or obj['tree']).")
    if tree_df.index is None or len(tree_df.index) == 0:
        raise ValueError("Tree dataFrame has empty or missing index.")

    nodes = tree_df['nodes'].tolist()
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    parent_idx = np.full(len(nodes), -1, dtype=int)

    for idx, row in tree_df.iterrows():
        if row['parent'] in node_to_idx:
            parent_idx[idx] = node_to_idx[row['parent']]

    children = [[] for _ in range(len(nodes))]
    for i, p in enumerate(parent_idx):
        if p != -1:
            children[p].append(i)

    # Euler tour for LCA
    euler, depth, first_occ = [], [], {}
    def dfs(u, d):
        first_occ[u] = len(euler)
        euler.append(u)
        depth.append(d)
        for v in children[u]:
            dfs(v, d + 1)
            euler.append(u)
            depth.append(d)

    root = np.where(parent_idx == -1)[0][0]
    dfs(root, 0)

    # Sparse table for RMQ
    n = len(depth)
    log = np.zeros(n + 1, dtype=int)
    for i in range(2, n + 1):
        log[i] = log[i // 2] + 1
    k = log[n]
    st = np.zeros((k + 1, n), dtype=int)
    st[0] = np.arange(n)
    for j in range(1, k + 1):
        for i in range(n - (1 << j) + 1):
            left, right = st[j - 1, i], st[j - 1, i + (1 << (j - 1))]
            st[j, i] = left if depth[left] < depth[right] else right

    def rmq(l, r):
        j = log[r - l + 1]
        left, right = st[j, l], st[j, r - (1 << j) + 1]
        return left if depth[left] < depth[right] else right

    def lca(u, v):
        i, j = first_occ[u], first_occ[v]
        if i > j: i, j = j, i
        return euler[rmq(i, j)]

    dist_to_root = tree_df['dist_to_root'].to_numpy()
    non_leaves = set(tree_df['parent'])
    leaves = [name for name in tree_df['nodes'] if name not in non_leaves]
    leaf_indices = [node_to_idx[name] for name in leaves]

    m = len(leaves)
    mat = np.zeros((m, m))

    print("Computing pairwise tree distances...")
    for a in tqdm(range(m), desc="Leaves"):
        for b in range(a + 1, m):
            u_idx, v_idx = leaf_indices[a], leaf_indices[b]
            ancestor = lca(u_idx, v_idx)
            d = dist_to_root[u_idx] + dist_to_root[v_idx] - 2 * dist_to_root[ancestor]
            mat[a, b] = mat[b, a] = d

    M = pd.DataFrame(mat, index=leaves, columns=leaves)
    if save:
        file_path = Path(path) / f"{savename}.csv"
        M.to_csv(file_path)
        print("Tree distance matrix saved.")
    print('Done!')
    return M

# -----------------------------------------------------------------------------
# Harmonize feature names in two or more objects
# -----------------------------------------------------------------------------
def align(
    objlist: List[Union[Dict[str, pd.DataFrame], "MicrobiomeData"]],
    *,
    different_lengths: bool = False,
    name_type: str = 'OTU',
) -> List[Union[Dict[str, pd.DataFrame], "MicrobiomeData"]]:

    """
    Align feature names across multiple objects containing sequences ('seq').

    Works with either dicts or MicrobiomeData objects. Returns the same types as inputs.

    Parameters
    ----------
    objlist : list of dict or MicrobiomeData
        List of input objects to align. Each object must contain at least:
        - 'seq': sequence table (features x sequence)
        Optionally:
        - 'tab': abundance table (features x samples)
        - 'tax': taxonomy table
        - 'meta': sample metadata
        - 'tree': sample tree data
    different_lengths : bool, optional
        If True, allows alignment of features with different sequence lengths (substring matching, O(n^2) over unique sequences).
        Default is False.
    name_type : str, optional
        Prefix for renaming aligned features (e.g., 'OTU', 'ASV'). Default is 'OTU'.

    Returns
    -------
    aligned_objects : list of dict or MicrobiomeData
        List of aligned objects, of the same types as the input.

    Notes
    -----
    - Duplicate sequences within each object are collapsed (abundance tables summed, taxonomy taken from the first occurrence).
    - Feature names are harmonized across all objects.
    - If `different_lengths=True`, substring matching is used for alignment.
    """

    # -----------------------------
    # Helper functions (nested)
    # -----------------------------
    def _is_mb(obj: Any) -> bool:
        """Treat as MicrobiomeData if it has .seq attribute that is a DataFrame."""
        return hasattr(obj, "seq") and isinstance(getattr(obj, "seq"), pd.DataFrame)

    def _set_df(obj: Union[Dict[str, pd.DataFrame], "MicrobiomeData"], key: str, value: pd.DataFrame | None) -> None:
        """
        Uniform setter for 'seq', 'tab', or 'tax' DataFrames.
        Works for both dict-based objects and MicrobiomeData instances.
        If value is None, removes the key from dict or sets attribute to None.
        """
        if hasattr(obj, "seq") and isinstance(getattr(obj, "seq"), pd.DataFrame):
            # Treat as MicrobiomeData object
            setattr(obj, key, value)
        else:
            # Treat as dict
            if value is None:
                obj.pop(key, None)  # Safely remove key if present
            else:
                obj[key] = value

    # Prepare working list (deep copies to avoid mutating inputs)
    objs: List[Union[Dict[str, pd.DataFrame], "MicrobiomeData"]] = []
    types: List[str] = []
    for obj in objlist:
        if _is_mb(obj):
            types.append("mb")
            objs.append(copy.deepcopy(obj))
        elif isinstance(obj, dict):
            types.append("dict")
            objs.append(copy.deepcopy(obj))
        else:
            raise TypeError("Each item must be a dict or MicrobiomeData with a 'seq' DataFrame.")

    # Validate and normalize 'seq' in all objects
    for k, obj in enumerate(objs):
        seq = get_df(obj, "seq")
        if seq is None or not isinstance(seq, pd.DataFrame):
            raise ValueError(f"Object {k+1} is missing a 'seq' DataFrame.")

    # Collapse duplicate sequences per object and keep info about old names for each object
    oname2seq_list: List[Dict[str, str]] = []
    seq2oname_list: List[Dict[str, str]] = []
    for k, obj in enumerate(objs):
        seq = get_df(obj, "seq").copy()
        tab = get_df(obj, "tab")
        tax = get_df(obj, "tax")
        s2n = seq["seq"].to_dict()
        oname2seq_list.append(s2n)
        inverted = {v: k for k, v in s2n.items()}
        seq2oname_list.append(inverted)

        # Keep the first occurrence's original ID as 'Newname'
        seq["Newname"] = seq.index.astype(str)

        # Attach 'seq' strings to tab/tax (by index alignment)
        if tab is not None:
            tab = tab.copy()
            tab = tab.join(seq[["seq"]], how="left")
        if tax is not None:
            tax = tax.copy()
            tax = tax.join(seq[["seq"]], how="left")

        # Collapse duplicates: group by the 'seq' string (canonical sequence)
        seq_group = seq.groupby("seq", sort=False).first()
        seq_group["seq"] = seq_group.index

        if tab is not None:
            tab_group = tab.groupby("seq", sort=False).sum(numeric_only=True)
            tab_group["Newname"] = seq_group["Newname"]
            tab_group = tab_group.set_index("Newname")
            _set_df(obj, "tab", tab_group)

        if tax is not None:
            tax_group = tax.groupby("seq", sort=False).first()
            tax_group["Newname"] = seq_group["Newname"]
            tax_group = tax_group.set_index("Newname")
            _set_df(obj, "tax", tax_group)

        # Final 'seq' index = Newname; keep 'seq' as the canonical sequence string
        seq_out = seq_group.copy()
        seq_out = seq_out.set_index("Newname")
        _set_df(obj, "seq", seq_out)

    # Build sequence -> canonical name across all objects and keep info about old and new names for each object
    oname2nname_list: List[Dict[str, str]] = []
    first_seq_list = get_df(objs[0], "seq")["seq"].astype(str).tolist()
    seq2nname_dict: Dict[str, str] = {}   # sequence string -> SV name
    nname2seq_dict: Dict[str, str] = {}  # SV name -> canonical (longest) sequence
    counter = 0

    checkold = seq2oname_list[0]
    old2new = {}
    for s in first_seq_list:
        counter += 1
        nname = f"{name_type}{counter}"
        seq2nname_dict[s] = nname #sequence to newname 
        nname2seq_dict[nname] = s #newname to sequence
        old2new[checkold[s]] = nname #Map old name to newname 
    oname2nname_list.append(old2new)

    print(f"Aligning features in {len(objs)} objects: 1.. ", end="")
    running_sequences = first_seq_list[:]  # running set of known sequences

    for i in range(1, len(objs)):
        print(f"{i+1}.. ", end="")
        this_list = get_df(objs[i], "seq")["seq"].astype(str).tolist()
        new_candidates: List[str] = []

        checkold = seq2oname_list[i]
        old2new = {}

        for s_check in this_list:
            if s_check in seq2nname_dict:
                old2new[checkold[s_check]] = seq2nname_dict[s_check]
                continue  # already known

            matched = False
            if different_lengths:
                # Substring match: check against existing set (O(n))
                for s_known in running_sequences:
                    if (s_check in s_known) or (s_known in s_check):
                        nname = seq2nname_dict[s_known]
                        seq2nname_dict[s_check] = nname
                        old2new[checkold[s_check]] = nname
                        # Keep the longest sequence as canonical for the SV
                        if len(s_check) > len(nname2seq_dict[nname]):
                            nname2seq_dict[nname] = s_check
                        matched = True
                        break

            if not matched:
                counter += 1
                nname = f"{name_type}{counter}"
                seq2nname_dict[s_check] = nname
                nname2seq_dict[nname] = s_check
                old2new[checkold[s_check]] = nname
                new_candidates.append(s_check)

        running_sequences.extend(new_candidates)
        oname2nname_list.append(old2new)

    # Rewrite all objects with new feature names and canonical sequences
    print(f"\nChanging feature names in {len(objs)} objects: ", end="")
    for i, obj in enumerate(objs):
        print(f"{i+1}.. ", end="")
        seq = get_df(obj, "seq").copy()
        tab = get_df(obj, "tab")
        tax = get_df(obj, "tax")

        # Map each row’s sequence string to new SV
        seq["newSV"] = seq["seq"].map(seq2nname_dict).astype(str)
        # Replace sequence strings by the canonical (longest) sequence for that SV
        seq["seq"] = seq["newSV"].map(nname2seq_dict).astype(str)

        # Propagate newSV to tab/tax via index alignment
        if tab is not None:
            tab = tab.copy()
            tab["newSV"] = pd.Series(seq["newSV"], index=seq.index)
        if tax is not None:
            tax = tax.copy()
            tax["newSV"] = pd.Series(seq["newSV"], index=seq.index)

        # Group by newSV
        seq_out = seq.groupby("newSV", sort=False).first()
        seq_out.index.name = None
        _set_df(obj, "seq", seq_out)

        if tab is not None:
            tab_out = tab.groupby("newSV", sort=False).sum(numeric_only=True)
            tab_out.index.name = None
            _set_df(obj, "tab", tab_out)
        if tax is not None:
            tax_out = tax.groupby("newSV", sort=False).first()
            tax_out.index.name = None
            _set_df(obj, "tax", tax_out)

        tree = get_df(obj, "tree")
        if tree is not None:
            tree_out = rename_leaves(tree, oname2nname_list[i])
            _set_df(obj, "tree", tree_out)

    print("\nDone with align")
    return objs

# -----------------------------------------------------------------------------
# Make consensus object
# -----------------------------------------------------------------------------
def consensus(
    objlist: List[Union[Dict[str, pd.DataFrame], "MicrobiomeData"]],
    *,
    keep_object: Union[str, int] = 'best',
    already_aligned: bool = False,
    different_lengths: bool = False,
    name_type: str = 'OTU',
    keep_cutoff: float = 0.2,
    only_return_seq: bool = False,
    return_type: Literal["auto", "dict", "microbiome"] = "auto",
) -> Tuple[Union[Dict[str, pd.DataFrame], "MicrobiomeData", pd.DataFrame], Dict[str, Any]]:

    """
    Build a consensus object based on features found in all input objects.

    This function aligns features (e.g., ASVs/OTUs) across multiple microbiome data objects,
    identifies features shared by all, and constructs a consensus abundance table, sequences,
    taxonomy, and metadata. Optionally, features with high abundance in any object are retained
    even if not shared. The result can be returned as a dictionary or as a MicrobiomeData object.

    Parameters
    ----------
    objlist : list of dict or MicrobiomeData
        List of input objects to merge. Each object must contain at least:
        - 'tab' : pd.DataFrame (abundance table, features x samples)
        - 'seq' : pd.DataFrame (sequences, indexed by feature IDs)
        Optionally:
        - 'tax' : pd.DataFrame (taxonomy annotations)
        - 'meta': pd.DataFrame (sample metadata)
        Objects can be either plain dicts or MicrobiomeData instances.

    keep_object : {'best', int}, default 'best'
        Determines which input object to use as the template for consensus:
        - 'best': the object with the largest fraction of reads mapped to shared features.
        - int: index of the object to use (0 = first, 1 = second, etc.).

    already_aligned : bool, default False
        If True, assumes that features are already aligned across objects.
        If False, runs the alignment step.

    different_lengths : bool, default False
        If True, allows alignment of features with different sequence lengths (substring matching).

    name_type : str, default 'OTU'
        Prefix for renaming consensus features (e.g., "OTU1", "OTU2", ...).

    keep_cutoff : float, default 0.2
        Relative abundance cutoff (%) for retaining features that are not shared by all objects,
        but are highly abundant in at least one object.

    only_return_seq : bool, default False
        If True, only returns a DataFrame of shared sequences (plus the info dictionary).
        No consensus object is constructed.

    return_type : {'auto', 'dict', 'microbiome'}, default 'auto'
        Determines the type of object returned (unless only_return_seq is True):
        - 'microbiome': always return a MicrobiomeData object (except when only_return_seq=True).
        - 'dict': always return a dictionary (legacy behavior).
        - 'auto': return a MicrobiomeData object if any input was a MicrobiomeData; otherwise, return a dict.

    Returns
    -------
    cons_obj : dict or MicrobiomeData or pd.DataFrame
        The consensus object containing:
        - 'tab': abundance table (features x samples)
        - 'seq': sequence table (features x sequence)
        - 'tax': taxonomy table (optional)
        - 'meta': metadata table (optional)
        If `return_type='microbiome'` or `'auto'` (with any MicrobiomeData input), returns a MicrobiomeData object.
        If `return_type='dict'`, returns a dictionary.
        If `only_return_seq=True`, returns a DataFrame of shared sequences.

    info : dict
        Dictionary with summary statistics about consensus construction, including:
        - 'kept_object_index': index of the selected template object
        - 'all_objects': per-object statistics (consensus abundance, lost reads/features)
        - 'selected_object': statistics for the selected object

    Notes
    -----
    - The consensus object does not include a phylogenetic tree, even if present in the inputs.
    - Feature indices are re-ordered by average abundance and renamed using `name_type`.
    - If `only_return_seq` is True, only the shared sequences DataFrame and info are returned.
    - The function automatically aligns features unless `already_aligned` is True.

    Examples
    --------
    >>> cons_obj, info = consensus([obj1, obj2], keep_object='best')
    >>> print(type(cons_obj))
    <class 'MicrobiomeData'>
    >>> cons_obj.info()
    >>> print(info)

    >>> # To get a dict instead of MicrobiomeData:
    >>> cons_dict, info = consensus([obj1, obj2], return_type='dict')

    >>> # To get only the shared sequences:
    >>> seq_df, info = consensus([obj1, obj2], only_return_seq=True)
    """

    # ---- helpers ----
    def _is_mb(obj: Any) -> bool:
        return hasattr(obj, "seq") and isinstance(getattr(obj, "seq"), pd.DataFrame)

    # Align sequences if needed
    print("Running consensus...")
    if already_aligned:
        aligned_objects = copy.deepcopy(objlist)
    else:
        aligned_objects = align(
            objlist,
            different_lengths=different_lengths,
            name_type=name_type,
        )

    info_str = "### consensus ###\n" #This will hold information about the process

    # Find features common to all objects
    incommonSVs = set(get_df(aligned_objects[0], "tab").index)
    for obj in aligned_objects[1:]:
        incommonSVs &= set(get_df(obj, "tab").index)
    incommonSVs = list(incommonSVs)

    info_str = info_str + "There are " + str(len(incommonSVs)) + " features in common between all objects.\n\n"

    # Relative abundance stats
    ra_in_tab, ra_sample_max, ra_sample_ind_max = [], [], []
    for i, obj in enumerate(aligned_objects):
        tab_all = get_df(obj, "tab")
        tab_incommon = tab_all.loc[incommonSVs]
        ra_in_tab.append(100 * tab_incommon.sum().sum() / tab_all.sum().sum())
        tab_notincommon = tab_all.loc[~tab_all.index.isin(incommonSVs)]
        ra_of_notincommon = 100 * tab_notincommon.div(tab_all.sum(), axis=1)
        ra_sample_max.append(ra_of_notincommon.sum(axis=1).max())
        ra_sample_ind_max.append(ra_of_notincommon.max().max())
        
        #Add relative abundance info the information string
        info_str = info_str + "In obj." + str(i) + ", the consensus features account for:\n"
        info_str = info_str + " " + str(round(100*len(incommonSVs)/len(tab_all), 2)) + " % of the features\n"
        info_str = info_str + " " + str(round(ra_in_tab[-1], 2)) + " % of the total abundance counts\n"
        per_sample = 100 * tab_incommon.sum() / tab_all.sum()
        info_str = info_str + " " + str(round(per_sample.min(), 2)) + " - " + str(round(per_sample.max(), 2)) + " % of the abundance counts per sample\n\n"

    # Choose which object to keep
    if keep_object == 'best':
        ra_max_pos = ra_in_tab.index(max(ra_in_tab))
    elif isinstance(keep_object, int) and 0 <= keep_object < len(aligned_objects):
        ra_max_pos = keep_object
    else:
        raise ValueError("keep_object must be 'best' or a valid integer index.")

    # Note which object is kept
    info_str = info_str + "The object with position "  + str(ra_max_pos) + " in the input list is retained as the consensus object.\n"

    # Keep abundant features even if not shared
    tab_all = get_df(aligned_objects[ra_max_pos], "tab")
    ra = 100 * tab_all.div(tab_all.sum(), axis=1)
    ra['max'] = ra.max(axis=1)
    keep_extra = ra[ra['max'] > keep_cutoff].index.tolist()
    incommonSVs = list(set(incommonSVs + keep_extra))

    # Write information about the choosen consensus object
    info_str = info_str + "This object also includes "  + str(len(keep_extra)) + " features with a relative abundance above the keep_cutoff threshold.\n"
    info_str = info_str + "All in all, the consensus object retains " + str(round(100*len(incommonSVs)/len(tab_all), 2)) + " % of its original features,\n"
    rakeep = 100 * tab_all.loc[incommonSVs].sum().sum() / tab_all.sum().sum()
    info_str = info_str + " which represent " + str(round(rakeep, 2)) + " % of the total abundance counts,\n"
    per_sample = 100 * tab_all.loc[incommonSVs].sum() / tab_all.sum()
    info_str = info_str + " and " + str(round(per_sample.min(), 2)) + " - " + str(round(per_sample.max(), 2)) + " % of the abundance counts per sample.\n"
    ra_notincommon = 100 * tab_all / tab_all.sum()
    ra_notincommon = ra_notincommon[~ra_notincommon.index.isin(incommonSVs)]
    ra_notincommon = ra_notincommon.max(axis=1)
    ra_notincommon = ra_notincommon.sort_values(ascending=False)
    info_str = info_str + "The single most abundant feature that was removed had a relative abundance of to "
    info_str = info_str + str(round(ra_notincommon.min(), 2)) + " - " + str(round(ra_notincommon.max(), 2))
    info_str = info_str + " % in the samples. "
    tax = get_df(aligned_objects[ra_max_pos], "tax")
    if tax is not None:
        sv = ra_notincommon.index.tolist()[0]
        taxlabel = ";".join(tax.loc[sv].astype(str).tolist())
        if len(taxlabel) > 4:
            info_str = info_str + "The taxonomic classification of this feature was " + taxlabel
    info_str = info_str + "\n--------------\n"

    # If only sequences requested
    if only_return_seq:
        seq = get_df(aligned_objects[0], "seq").loc[incommonSVs]
        return seq, info_str

    # Build consensus "dict" first
    cons_obj: Dict[str, pd.DataFrame] = {}
    cons_obj["tab"] = tab_all.loc[incommonSVs]
    cons_obj["seq"] = get_df(aligned_objects[ra_max_pos], "seq").loc[incommonSVs]
    meta_df = get_df(aligned_objects[ra_max_pos], "meta")
    if meta_df is not None:
        cons_obj["meta"] = meta_df
    if tax is not None:
        cons_obj["tax"] = tax.loc[incommonSVs]
    tree = get_df(aligned_objects[ra_max_pos], "tree")
    if tree is not None:
        cons_tree = subset_tree(tree, incommonSVs)
        cons_obj["tree"] = cons_tree

    # Reorder by average abundance and rename to name_type + rank
    sort_df = cons_obj["tab"].copy()
    sort_df["avg"] = sort_df.mean(axis=1)
    correct_order_svlist = sort_df.sort_values(by="avg", ascending=False).index.tolist()
    newindex_dict = {sv: f"{name_type}{i+1}" for i, sv in enumerate(correct_order_svlist)}
    for key in ["tab", "seq", "tax"]:
        if key in cons_obj:
            cons_obj[key] = cons_obj[key].loc[correct_order_svlist].rename(index=newindex_dict)
    if tree is not None:
        cons_obj["tree"] = rename_leaves(cons_obj["tree"], newindex_dict)

    print("Done with consensus.")

    # ---- Decide return type and wrap if needed ----
    want_mb = False
    if return_type == "microbiome":
        want_mb = True
    elif return_type == "dict":
        want_mb = False
    else:  # auto
        want_mb = any(_is_mb(o) for o in objlist)

    if want_mb:
        mb = MicrobiomeData(
            tab=cons_obj.get("tab"),
            tax=cons_obj.get("tax"),
            meta=cons_obj.get("meta"),
            seq=cons_obj.get("seq"),
            tree=cons_obj.get("tree")
        )
        return mb, info_str

    return cons_obj, info_str

# -----------------------------------------------------------------------------
# Merge objects and keep all features
# -----------------------------------------------------------------------------
def merge_objects(
    objlist: List[Union[Dict[str, pd.DataFrame], "MicrobiomeData"]],
    *,
    already_aligned: bool = False,
    different_lengths: bool = False,
    name_type: str = 'OTU',
    return_type: Literal["auto", "dict", "microbiome"] = "auto",
) -> Union[Dict[str, pd.DataFrame], "MicrobiomeData"]:
    """
    Merge multiple microbiome objects and retain all features (OTUs/ASVs/etc).

    This function aligns features across all input objects (unless `already_aligned=True`),
    concatenates abundance tables (columns are sample names; they are suffix-annotated
    per input object to avoid collisions), and concatenates sequences and taxonomy tables
    while removing duplicates. Feature rows are re-ordered by total abundance and renamed
    using `name_type` + rank (e.g., "OTU1", "OTU2", ...).

    Parameters
    ----------
    objlist : list of dict or MicrobiomeData
        List of input objects to merge. Each item must contain:  
        
        - 'tab' : pd.DataFrame (features × samples abundance table)
        - 'seq' : pd.DataFrame (sequences, indexed by feature IDs)
        
        Optionally may contain:  
        
        - 'tax' : pd.DataFrame (taxonomy annotations
        - 'meta': pd.DataFrame (sample metadata, indexed by sample names)
        
        Items can be plain dicts or MicrobiomeData instances.
    already_aligned : bool, default False
        If True, assumes `align` has already been applied to the objects and feature
        names/sequences are harmonized. If False, calls `align(...)`.

    different_lengths : bool, default False
        If True, allows alignment of features with different sequence lengths (substring matching).
        Passed to `align` when alignment is performed.

    name_type : str, default 'OTU'
        Prefix for renaming merged features in descending order of total abundance
        (e.g., "OTU1", "OTU2", ...).

    return_type : {'auto', 'dict', 'microbiome'}, default 'auto'
        Determines the type of object returned:
        
        - 'microbiome': return a MicrobiomeData object.
        - 'dict' : return a dict.
        - 'auto' : if any input in `objlist` is a MicrobiomeData, return MicrobiomeData, otherwise return dict.

    Returns
    -------
    out : dict or MicrobiomeData
        The merged object containing:
            
        - 'tab' : merged abundance table
        - 'seq' : merged sequence table
        - 'tax' : merged taxonomy table (if present)
        - 'meta': merged metadata table (if present)
        
        Return type depends on `return_type` (see above).

    Notes
    -----
    - Sample names in the merged 'tab' and 'meta' are suffixed with `_i` where `i` is
      the index of the source object in `objlist`, to avoid collisions.
    - Features are sorted by total abundance (sum across all samples) and then renamed
      using `name_type` and their new rank.
    - Sequences/taxonomy are de-duplicated with `drop_duplicates()` after concatenation.
    - The merged output does **not** include a phylogenetic tree.

    Examples
    --------
    >>> out = merge_objects([obj1, obj2])
    """
    # ---- helpers ----
    def _is_mb(obj: Any) -> bool:
        return hasattr(obj, "seq") and isinstance(getattr(obj, "seq"), pd.DataFrame)

    def _get_df(obj: Union[Dict[str, pd.DataFrame], "MicrobiomeData"], key: str) -> pd.DataFrame | None:
        return getattr(obj, key, None) if _is_mb(obj) else obj.get(key)

    if not objlist:
        raise ValueError("objlist must contain at least one object.")

    print("Running merge_objects...")

    # Align sequences if needed
    if already_aligned:
        aligned_objects = copy.deepcopy(objlist)
    else:
        aligned_objects = align(
            objlist,
            different_lengths=different_lengths,
            name_type=name_type,
        )

    # Collect DataFrames, rename samples to avoid collisions
    tablist: List[pd.DataFrame] = []
    seqlist: List[pd.DataFrame] = []
    taxlist: List[pd.DataFrame] = []
    metalist: List[pd.DataFrame] = []

    for i, obj in enumerate(aligned_objects):
        tab = _get_df(obj, "tab")
        seq = _get_df(obj, "seq")
        tax = _get_df(obj, "tax")
        meta = _get_df(obj, "meta")

        if tab is not None:
            renamed_tab = tab.copy()
            renamed_tab.columns = [f"{col}_{i}" for col in renamed_tab.columns]
            tablist.append(renamed_tab)

        if seq is not None:
            seqlist.append(seq)

        if tax is not None:
            taxlist.append(tax)

        if meta is not None:
            renamed_meta = meta.copy()
            renamed_meta.index = [f"{idx}_{i}" for idx in renamed_meta.index]
            metalist.append(renamed_meta)

    # Merge DataFrames
    tab_joined = pd.concat(tablist, axis=1, join='outer').fillna(0) if tablist else None
    seq_joined = pd.concat(seqlist, axis=0, join='outer') if seqlist else None
    tax_joined = pd.concat(taxlist, axis=0, join='outer') if taxlist else None
    meta_joined = pd.concat(metalist, axis=0, join='outer') if metalist else None

    # ---- DEDUPLICATE BY INDEX (critical) ----
    # If the same feature index appears multiple times (with differing rows),
    # collapse by index: keep the first non-null row for each column.
    def _collapse_by_index(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None:
            return None
        return df.groupby(level=0).first()

    seq_joined = _collapse_by_index(seq_joined)
    tax_joined = _collapse_by_index(tax_joined)

    # Order features by total abundance and rename
    if tab_joined is not None:
        tab_joined['sum'] = tab_joined.sum(axis=1)
        tab_joined = tab_joined.sort_values(by='sum', ascending=False)
        newnames = {ix: f"{name_type}{i+1}" for i, ix in enumerate(tab_joined.index)}
        tab_joined = tab_joined.drop('sum', axis=1).rename(index=newnames)
        tab_joined = sort_index_by_number(tab_joined)

    if seq_joined is not None:
        seq_joined = seq_joined.rename(index=newnames) if tab_joined is not None else seq_joined
        seq_joined = sort_index_by_number(seq_joined)

    if tax_joined is not None:
        tax_joined = tax_joined.rename(index=newnames) if tab_joined is not None else tax_joined
        tax_joined = sort_index_by_number(tax_joined)

    # Build dict first
    out_dict: Dict[str, pd.DataFrame] = {}
    if tab_joined is not None: out_dict['tab'] = tab_joined
    if seq_joined is not None: out_dict['seq'] = seq_joined
    if tax_joined is not None: out_dict['tax'] = tax_joined
    if meta_joined is not None: out_dict['meta'] = meta_joined

    # Decide return type
    want_mb = (
        True if return_type == "microbiome"
        else False if return_type == "dict"
        else any(_is_mb(o) for o in objlist)  # auto
    )

    if want_mb:
        from ..data_object import MicrobiomeData
        mb = MicrobiomeData(
            tab=out_dict.get("tab"),
            tax=out_dict.get("tax"),
            meta=out_dict.get("meta"),
            seq=out_dict.get("seq"),
            tree=None,
        )
        # __init__ of MicrobiomeData already calls _autocorrect and _validate
        print("Done with merge_objects (returned MicrobiomeData).")
        return mb

    print("Done with merge_objects (returned dict).")
    return out_dict
