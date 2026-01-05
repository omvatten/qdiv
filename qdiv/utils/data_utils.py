import pandas as pd
import numpy as np
import math
import re
from typing import Union, Any, Dict, TYPE_CHECKING
from .phylo_utils import rename_leaves

if TYPE_CHECKING:
    from ..data_object import MicrobiomeData


def sort_index_by_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort DataFrame index in natural alphanumeric order.
    Handles mixed text/numeric chunks and avoids TypeError.
    """
    def alphanum_key(s: str):
        parts = re.split(r'(\d+)', s)
        key = []
        for p in parts:
            if not p:
                continue
            if p.isdigit():
                key.append((0, int(p)))  # numeric parts
            else:
                key.append((1, p.lower()))  # text parts
        return key

    sorted_index = sorted(df.index, key=alphanum_key)
    return df.loc[sorted_index]

def get_df(
    tab_like: Union[pd.DataFrame, Dict[str, Any], Any],
    attr: str = "tab",
) -> pd.DataFrame:
    """
    Convert various input types into a dataframe
    """
    if isinstance(tab_like, pd.DataFrame):
        df = tab_like.copy(deep=False)

    elif hasattr(tab_like, attr):
        val = getattr(tab_like, attr)
        if val is None:
            df = None
        elif isinstance(val, pd.DataFrame):
            df = val.copy(deep=False)
        else:
            raise TypeError(
                f"Attribute '{attr}' exists but is not a pandas DataFrame "
                f"(got {type(val).__name__})."
            )

    elif isinstance(tab_like, dict) and attr in tab_like:
        val = tab_like[attr]
        if val is None:
            df = None
        elif isinstance(val, pd.DataFrame):
            df = val.copy(deep=False)
        else:
            raise TypeError(
                f"Dict key '{attr}' exists but is not a pandas DataFrame "
                f"(got {type(val).__name__})."
            )

    elif isinstance(tab_like, dict) and attr not in tab_like:
        df = None

    else:
        raise TypeError(
            f"Unsupported `tab_like` type: {type(tab_like).__name__}. "
            "Provide a pandas DataFrame, an object with a DataFrame attribute, "
            "or a dict-of-dicts."
        )

    # Basic non-empty validation
    if df is not None:
        if df.shape[0] == 0 and df.shape[1] == 0:
            raise ValueError(f"DataFrame {attr} is empty.")

    if attr == "tab" and df is not None:
        non_numeric = [c for c, dt in df.dtypes.items()
                       if not pd.api.types.is_numeric_dtype(dt)]
        if non_numeric:
            has_seq = df[non_numeric].applymap(lambda x: isinstance(x, (list, tuple, np.ndarray))).any().any()
            if has_seq:
                bad_cols = [c for c in non_numeric
                            if df[c].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any()]
                examples = {c: df[c].loc[df[c].apply(lambda x: isinstance(x, (list, tuple, np.ndarray)))].head(3).tolist()
                            for c in bad_cols[:3]}
                raise TypeError(
                    f"'tab' contains sequence values (list/ndarray) in columns {bad_cols}. "
                    f"Cells must be scalar counts/abundances. Examples: {examples}"
                )
            df = df.apply(pd.to_numeric, errors="raise")
        df = df.astype(float)

    return df

def beta2dist(
    beta: Union[float, pd.Series, pd.DataFrame],
    q: float = 1,
    N: float = 2,
    div_type: str = 'naive',
    viewpoint: str = 'local'
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Convert beta diversity values into distance values.

    This function implements the transformations described in 
    Chao et al. (2014) for converting pairwise beta-diversity 
    measures into distances under different diversity orders (q),
    diversity types, and viewpoints.

    Parameters
    ----------
    beta : float, pandas.Series, or pandas.DataFrame
        Beta-diversity value(s). Must be positive.
    q : float, default=1
        Hill number order. q = 1 corresponds to the Shannon case.
    N : int or float, default=2
        Number of communities being compared (pairwise = 2).
    div_type : {'naive', 'phyl', 'func'}, default='naive'
        Type of beta diversity:
        - 'naive' : all features equally distinct
        - 'phyl'  : phylogenetic
        - 'func'  : functional
    viewpoint : {'local', 'regional'}, default='local'
        Perspective used in the transformation.

    Returns
    -------
    float, pandas.Series, or pandas.DataFrame
        Distance value(s) corresponding to the input beta values.

    Notes
    -----
    - All formulas follow the definitions in Chao et al. (2014).
    - Values of beta ≤ 0 are not transformed and remain unchanged.
    - Supports scalar, Series, and DataFrame inputs.

    Raises
    ------
    ValueError
        If invalid `divType`, `viewpoint`, or non-positive `N` is provided.
    """
    # --- Input validation ----------------------------------------------------
    valid_types = {'naive', 'phyl', 'func'}
    valid_viewpoints = {'local', 'regional'}

    if div_type not in valid_types:
        raise ValueError(f"divType must be one of {valid_types}, got '{div_type}'.")

    if viewpoint not in valid_viewpoints:
        raise ValueError(f"viewpoint must be one of {valid_viewpoints}, got '{viewpoint}'.")

    if N <= 0:
        raise ValueError("N must be positive.")

    # Convert input to a pandas object for unified handling
    is_scalar = np.isscalar(beta)

    if is_scalar:
        beta_arr = pd.Series([beta], dtype=float)
    elif isinstance(beta, (pd.Series, pd.DataFrame)):
        beta_arr = beta.astype(float)
    else:
        raise TypeError("beta must be a float, pandas Series, or pandas DataFrame.")

    # Mask for valid values
    mask = beta_arr > 0
    dist = beta_arr.copy()

    # --- Core transformation logic ------------------------------------------
    if q == 1:
        # Shannon case
        if div_type in ['naive', 'phyl']:
            dist[mask] = (np.log(beta_arr[mask]) / math.log(N))
        else:  # func
            dist[mask] = (np.log(beta_arr[mask]) / (2 * math.log(N)))

    else:
        # q != 1
        if div_type in ['naive', 'phyl'] and viewpoint == 'local':
            dist[mask] = 1 - (N**(1 - q) - beta_arr[mask]**(1 - q)) / (N**(1 - q) - 1)

        elif div_type == 'func' and viewpoint == 'local':
            dist[mask] = 1 - (N**(2 * (1 - q)) - beta_arr[mask]**(1 - q)) / (N**(2 * (1 - q)) - 1)

        elif div_type in ['naive', 'phyl'] and viewpoint == 'regional':
            dist[mask] = 1 - ((1 / beta_arr[mask])**(1 - q) - (1 / N)**(1 - q)) / (1 - (1 / N)**(1 - q))

        elif div_type == 'func' and viewpoint == 'regional':
            dist[mask] = 1 - ((1 / beta_arr[mask])**(1 - q) - (1 / N)**(2 * (1 - q))) / (1 - (1 / N)**(2 * (1 - q)))

    # --- Return in original format ------------------------------------------
    if is_scalar:
        return float(dist.iloc[0])
    return dist

def rao(
    tab: Union[pd.Series, pd.DataFrame],
    distmat: pd.DataFrame
) -> Union[float, pd.Series]:
    """
    Compute Rao's quadratic entropy.

    Rao's Q is defined as:

        Q = Σᵢ Σⱼ (dᵢⱼ * pᵢ * pⱼ)

    where pᵢ and pⱼ are relative abundances and dᵢⱼ is the dissimilarity
    between species i and j. This function is used in Chiu et al.'s
    functional diversity framework.

    Parameters
    ----------
    tab : pandas.Series or pandas.DataFrame
        Abundance data.  
        - If Series: a single community (index = species).  
        - If DataFrame: multiple communities (columns = samples).
    distmat : pandas.DataFrame
        Square species-by-species distance matrix. Must contain all species
        present in `tab`.

    Returns
    -------
    float or pandas.Series
        - If `tab` is a Series: returns a single Rao Q value.
        - If `tab` is a DataFrame: returns a Series of Q values per sample.

    Raises
    ------
    ValueError
        If species in `tab` are missing from `distmat`.
    TypeError
        If `tab` is not a Series or DataFrame.
    """
    # --- Validate input ------------------------------------------------------
    if not isinstance(tab, (pd.Series, pd.DataFrame)):
        raise TypeError("tab must be a pandas Series or DataFrame.")

    # Relative abundances
    ra = tab / tab.sum()

    # Ensure distmat contains exactly the species in tab
    species = list(ra.index)
    missing = set(species) - set(distmat.index)
    if missing:
        raise ValueError(f"distmat is missing species: {missing}")

    # Align distance matrix
    distmat = distmat.loc[species, species]

    # --- Case 1: multiple samples (DataFrame) --------------------------------
    if isinstance(tab, pd.DataFrame):
        out = pd.Series(0.0, index=ra.columns)

        for sample in ra.columns:
            p = ra[sample].values
            p_outer = np.outer(p, p)  # p_i * p_j
            rao_matrix = p_outer * distmat.values
            out[sample] = rao_matrix.sum()

        return out

    # --- Case 2: single sample (Series) --------------------------------------
    p = ra.values
    p_outer = np.outer(p, p)
    rao_matrix = p_outer * distmat.values
    return float(rao_matrix.sum())

def rename_features(
    obj: Union[pd.DataFrame, Dict[str, Any], Any],
    name_type: str = 'OTU',
    inplace: bool = False,
) -> Union["MicrobiomeData", Dict[str, Any]]:

    """
    Rename feature identifiers (row indices) in microbiome-related data structures
    based on their relative abundance or taxonomic order.

    This function supports two input types:
    - A `MicrobiomeData` object (with components: tab, tax, seq, meta, tree)
    - A dictionary containing any subset of these components.

    The renaming assigns new feature names in the format `{name_type}{i}`, where `i`
    is the rank of the feature after sorting:
    - By mean relative abundance if `tab` (abundance table) is present.
    - By taxonomic order if `tax` is present and `tab` is absent.
    - By sequence order if `seq` is present and both `tab` and `tax` are absent.

    Parameters
    ----------
    obj : Union[pd.DataFrame, Dict[str, Any], MicrobiomeData]
        The input object or dictionary containing microbiome data components.
    name_type : str, default='OTU'
        Prefix for new feature names (e.g., 'OTU', 'ASV').
    inplace : bool, default=False
        If True and `obj` is a MicrobiomeData object, modify it in place.
        Otherwise, return a new object or dictionary.

    Returns
    -------
    Union[MicrobiomeData, Dict[str, Any]]
        A new MicrobiomeData object or dictionary with renamed features,
        unless `inplace=True`, in which case the original object is updated.

    Notes
    -----
    - If a phylogenetic tree (`tree`) is present, leaf names are also updated.
    - Metadata (`meta`) is passed through unchanged.
    - Sorting priority: tab > tax > seq.
    - Requires `MicrobiomeData` class for object reconstruction when `inplace=False`.

    Examples
    --------
    >>> # Rename features in a MicrobiomeData object
    >>> new_obj = rename_features(mb_data, name_type='ASV')
    >>> # Rename features in a dictionary
    >>> renamed = rename_features({'tab': tab_df, 'tax': tax_df}, name_type='OTU')
    """
    # --- Detect input kind without importing MicrobiomeData at module import time ---
    is_object = hasattr(obj, "tab") or hasattr(obj, "seq") or hasattr(obj, "tax") or hasattr(obj, "tree")

    # --- Extract components in a uniform way ---
    if is_object:
        tab = obj.tab
        tax = obj.tax
        seq = obj.seq
        meta = obj.meta
        tree = obj.tree
    elif isinstance(obj, dict):
        tab = obj.get("tab")
        tax = obj.get("tax")
        seq = obj.get("seq")
        meta = obj.get("meta")
        tree = obj.get("tree")

    old2new = {}
    if tab is not None:
        mean_ra = tab / tab.sum()
        mean_ra = mean_ra.mean(axis=1)
        mean_ra = mean_ra.sort_values(ascending=False)
        old2new = {ix: name_type + str(i+1) for i, ix in enumerate(mean_ra.index)}
    elif tax is not None:
        print(tax)
        sorted_tax = tax.astype(str).sort_values(by=tax.columns.tolist())
        print(sorted_tax)
        old2new = {ix: name_type + str(i+1) for i, ix in enumerate(sorted_tax.index)}
    elif seq is not None:
        old2new = {ix: name_type + str(i+1) for i, ix in enumerate(seq.index)}

    out_tab = tab.rename(index=old2new) if tab is not None else None
    out_tax = tax.rename(index=old2new) if tax is not None else None
    out_seq = seq.rename(index=old2new) if seq is not None else None

    if tree is not None and len(old2new) > 0:
        out_tree = rename_leaves(tree, old2new)
    else:
        out_tree = None

    # --- Build the return value in the same type as input ---
    if is_object:
        if inplace:
            obj.tab = out_tab
            obj.seq = out_seq
            obj.tax = out_tax
            obj.tree = out_tree
            # meta is passed through unchanged
            obj._autocorrect()
            obj._validate()
            return obj
        else:
            # Create a new MicrobiomeData instance lazily to avoid import cycles
            from ..data_object import MicrobiomeData
            new_obj = MicrobiomeData(
                tab=out_tab,
                tax=out_tax,
                meta=meta,
                seq=out_seq,
                tree=out_tree,
            )
            return new_obj
    else:
        out: Dict[str, Any] = {}
        if out_tab is not None: out["tab"] = out_tab
        if out_tax is not None: out["tax"] = out_tax
        if out_seq is not None: out["seq"] = out_seq
        if meta is not None: out["meta"] = meta
        if tree is not None: out["tree"] = out_tree
        return out

def tax_prefix(
    obj: Union[pd.DataFrame, Dict[str, Any], Any],
    add: bool = True,
    inplace: bool = False,
    custom_prefix: Dict[str, str] = None
) -> Union["MicrobiomeData", Dict[str, Any]]:
    """
    Add or remove prefix (e.g. d__, p__) to taxonomic classifications.

    Parameters
    ----------
    add : bool, default=True
        If True, add prefix. If False, remove prefix.
    inplace : bool, default=False
        If True, modify object in place.
    custom_prefix : dict, default=None
        A dictionary with taxonomic levels as keys and prefix as values.

    Returns
    -------
    MicrobiomeData
        The updated object. If `inplace=True`, returns self; otherwise, a new instance.
    """
    prefix_dict = {
        "domain": "d__",
        "kingdom": "k__",
        "phylum": "p__",
        "class": "c__",
        "order": "o__",
        "family": "f__",
        "genus": "g__",
        "species": "s__",
        }
    is_object = hasattr(obj, "tax")

    # --- Extract components in a uniform way ---
    if is_object:
        tab = obj.tab
        tax = obj.tax
        seq = obj.seq
        meta = obj.meta
        tree = obj.tree
    elif isinstance(obj, dict):
        tab = obj.get("tab")
        tax = obj.get("tax")
        seq = obj.get("seq")
        meta = obj.get("meta")
        tree = obj.get("tree")

    if tax is None:
        raise ValueError('tax is missing.')
    elif isinstance(tax, pd.DataFrame):
        out_tax = tax.copy()
        for c in out_tax.columns:
            if add:
                #Check prefix to add
                if custom_prefix is not None:
                    prefix = custom_prefix[c]
                elif c.lower() in prefix_dict:
                    prefix = prefix_dict[c.lower()]
                else:
                    prefix = c[0].lower() + '__'
                
                out_tax.loc[(out_tax[c].notna())&(~out_tax[c].str.contains("__"))&(out_tax[c].str.len()>1), c] = prefix + out_tax.loc[(out_tax[c].notna())&(~out_tax[c].str.contains("__"))&(out_tax[c].str.len()>1), c]
            else:
                out_tax.loc[(out_tax[c].notna())&(out_tax[c].str.contains("__")), c] = out_tax.loc[(out_tax[c].notna())&(out_tax[c].str.contains("__")), c].str.split("__").str[1]

    # --- Build the return value in the same type as input ---
    if is_object:
        if inplace:
            obj.tab = tab
            obj.seq = seq
            obj.tax = out_tax
            obj.tree = tree
            # meta is passed through unchanged
            obj._autocorrect()
            obj._validate()
            return obj
        else:
            # Create a new MicrobiomeData instance lazily to avoid import cycles
            from ..data_object import MicrobiomeData
            new_obj = MicrobiomeData(
                tab=tab,
                tax=out_tax,
                meta=meta,
                seq=seq,
                tree=tree,
            )
            return new_obj
    else:
        out: Dict[str, Any] = {}
        if tab is not None: out["tab"] = tab
        out["tax"] = out_tax
        if seq is not None: out["seq"] = seq
        if meta is not None: out["meta"] = meta
        if tree is not None: out["tree"] = tree
        return out

