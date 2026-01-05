import pandas as pd
import numpy as np
from typing import List, Optional, Union, Set, Iterable
import ast

def parse_newick(
    filename: str
) -> pd.DataFrame:
    """
    Parse a Newick tree file and return a DataFrame describing its nodes.

    This function reads a Newick-formatted tree (with branch lengths) and
    extracts:

    - node names (internal and terminal)
    - leaves (tip descendants for each node)
    - branch lengths
    - parent relationships
    - distance from each node to the root

    The output DataFrame contains one row per node with the following columns:

    - `nodes` : str  
        Node name (internal nodes are auto-named as `in1`, `in2`, ...)
    - `leaves` : list of str  
        List of descendant tip labels for that node.
    - `branchL` : float  
        Branch length leading to the node.
    - `parent` : str or None  
        Parent node name.
    - `dist_to_root` : float  
        Cumulative branch length from the root to the node.

    Parameters
    ----------
    filename : str
        Path to a Newick tree file. The tree must include branch lengths.

    Returns
    -------
    pandas.DataFrame
        A DataFrame describing all nodes in the tree.

    Raises
    ------
    FileNotFoundError
        If the file cannot be opened.
    ValueError
        If the Newick string contains no branch lengths.
    """

    # --- Read file -----------------------------------------------------------
    try:
        with open(filename, "r") as f:
            newick = "".join(line.strip() for line in f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filename}")

    if ":" not in newick:
        raise ValueError("Newick file contains no branch lengths (':' not found).")

    # Remove outer parentheses and/or final ;
    if newick.startswith("("):
        newick = newick[1:].rstrip(");")
    elif newick[-1] == ';':
        newick = newick.rstrip(";")

    # --- Parsing state -------------------------------------------------------
    intnodecounter = 0
    intnodenames: List[str] = []
    temp_endnodes: List[List[str]] = []

    nodelist: List[str] = []
    asvlist: List[List[str]] = []
    BLlist: List[float] = []
    parentlist: List[Optional[str]] = []

    # --- Parse Newick string -------------------------------------------------
    for i, char in enumerate(newick):

        if char == "(":
            temp_endnodes.append([])
            intnodecounter += 1
            intnodenames.append(f"in{intnodecounter}")

        elif char == ":":
            # Determine node name (terminal or internal)
            for j in range(i, -2, -1):
                if newick[j] in [",", "("] or j == -1:
                    # Terminal node
                    name = newick[j + 1:i]
                    nodelist.append(name)
                    asvlist.append([name])

                    # Add to all open internal nodes
                    for sub in temp_endnodes:
                        sub.append(name)

                    parent = intnodenames[-1] if intnodenames else "Root"
                    parentlist.append(parent)
                    break

                elif newick[j] == ")":
                    # Internal node
                    name = intnodenames[-1]
                    nodelist.append(name)
                    asvlist.append(temp_endnodes[-1])

                    intnodenames.pop()
                    temp_endnodes.pop()

                    parent = intnodenames[-1] if intnodenames else "Root"
                    parentlist.append(parent)
                    break

            # Extract branch length
            for j in range(i + 1, len(newick)):
                if newick[j] in [",", ")"]:
                    branchlen = float(newick[i + 1:j])
                    break
                elif j == len(newick) - 1:
                    branchlen = float(newick[i + 1:j + 1])

            BLlist.append(branchlen)

    # --- Build DataFrame -----------------------------------------------------
    df = pd.DataFrame({
        "nodes": nodelist,
        "leaves": asvlist,
        "branchL": BLlist,
        "parent": parentlist
    })

    # Add root if missing
    if "Root" not in df["nodes"].values:
        df = pd.concat([
            df,
            pd.DataFrame({
                "nodes": ["Root"],
                "leaves": [[]],
                "branchL": [0.0],
                "parent": [None]
            })
        ], ignore_index=True)

    # --- Compute distance to root -------------------------------------------
    nodes = df["nodes"].tolist()
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    parent_idx = np.full(len(nodes), -1, dtype=int)
    for idx, row in df.iterrows():
        if row["parent"] in node_to_idx:
            parent_idx[idx] = node_to_idx[row["parent"]]

    branch_len = df["branchL"].to_numpy(float)
    dist_to_root = np.zeros(len(nodes), float)

    roots = np.where(parent_idx == -1)[0]
    stack = list(roots)

    while stack:
        p = stack.pop()
        children = np.where(parent_idx == p)[0]
        for c in children:
            dist_to_root[c] = dist_to_root[p] + branch_len[c]
            stack.append(c)

    df["dist_to_root"] = dist_to_root
    return df


def parse_leaves(leaves):
    # Try to parse as a Python literal
    if isinstance(leaves, str):
        try:
            parsed = ast.literal_eval(leaves)
            # Accept list/tuple/set/ndarray; coerce to list
            if isinstance(parsed, (list, tuple, set, np.ndarray)):
                seq = parsed.tolist() if isinstance(parsed, np.ndarray) else list(parsed)
            else:
                # Not a sequence → treat as single string item
                seq = [str(parsed)]
        except Exception:
            # Fallback: CSV-like split, manual cleaning
            cleaned = (
                leaves.replace("[", "")
                .replace("]", "")
                .replace('"', "")
                .replace("'", "")
            )
            seq = [x for x in (t.strip() for t in cleaned.split(",")) if x]
    elif isinstance(leaves, (list, tuple, set, np.ndarray)):
        seq = leaves.tolist() if isinstance(leaves, np.ndarray) else list(leaves)
    elif pd.isna(leaves):
        seq = []
    else:
        seq = [leaves]

    # Keep only non-empty strings
    return {x for x in seq if isinstance(x, str) and x.strip() != ""}

def subset_tree(
    tree: pd.DataFrame,
    featurelist: Union[List[str], Set[str], Iterable[str]],
) -> pd.DataFrame:
    """
    Subset a tree DataFrame to branches whose leaves intersect with a given feature set,
    while always retaining the root branch.

    Parameters
    ----------
    tree : pandas.DataFrame
        A DataFrame representing a parsed tree structure. Must contain:
        - ``leaves`` : list-like or string
            Descendant tip labels for each node. Can be a list, tuple, set, NumPy array,
            or a string representation of these.
        - ``nodes`` : str
            Node name. The root node should be labeled as ``"Root"`` (case-insensitive).
        The DataFrame index should uniquely identify branches.

    featurelist : list of str or set of str or iterable of str
        A collection of feature names to match against the leaves of each branch.
        If a list or other iterable is provided, it will be converted to a set.
        Passing a single string is not allowed and will raise a ``TypeError``.

    Returns
    -------
    pandas.DataFrame
        A subset of the input DataFrame containing:
        - All branches whose ``leaves`` share at least one element with ``featurelist``.
        - The root branch (where ``nodes`` equals ``"Root"``).
        The returned DataFrame preserves the original column structure and includes
        only the selected rows.

    Raises
    ------
    TypeError
        If ``featurelist`` is a single string instead of a collection.
    KeyError
        If required columns (``leaves``, ``nodes``) are missing from ``tree``.

    Notes
    -----
    - Leaf parsing is delegated to the internal helper ``_parse_leaves``, which normalizes
      various formats (Python-literal strings, CSV-like strings, sequences, arrays) into
      a set of non-empty strings.
    - Intersection is computed using Python set operations for efficiency.
    - The root branch is always retained regardless of leaf content.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "nodes": ["Root", "in1", "tipA", "tipB"],
    ...     "leaves": [[], ["tipA", "tipB"], ["tipA"], ["tipB"]],
    ...     "branchL": [0.0, 0.3, 0.1, 0.2],
    ...     "parent": [None, "Root", "in1", "in1"]
    ... }, index=["r", "b1", "b2", "b3"])
    >>> subset_tree(df, ["tipA"])
       nodes    leaves  branchL parent
    r   Root        []      0.0   None
    b1   in1  [tipA, tipB]  0.3   Root
    b2  tipA     [tipA]     0.1    in1
    """

    if isinstance(featurelist, str):
        raise TypeError("featurelist must be a collection of feature names, not a single string")

    # Normalize featurelist → set[str]
    feature_set: Set[str] = featurelist if isinstance(featurelist, set) else set(featurelist)
    # Optional: filter to non-empty strings
    feature_set = {x for x in feature_set if isinstance(x, str) and x.strip()}

    sub_tree = tree.copy()
    sub_tree["leaves"] = pd.NA
    sub_tree["leaves"] = sub_tree["leaves"].astype(object)

    keep_branch = []
    for branch in tree.index:
        leaf_set = parse_leaves(tree.loc[branch, "leaves"])

        # Decide whether to keep this branch
        if (feature_set & leaf_set):
            keep_branch.append(branch)
            sub_tree.at[branch, "leaves"] = list(feature_set & leaf_set)
        else:
            node_val = tree.loc[branch, "nodes"]
            if isinstance(node_val, str) and node_val.lower() == "root":
                keep_branch.append(branch)

    out = sub_tree.loc[keep_branch].copy()
    out = out.reset_index(drop=True)
    return out

def rename_leaves(
    tree: pd.DataFrame,
    leaf_dict: dict,
) -> pd.DataFrame:
    """
    Rename leaves in a tree based on a dictionary with old names as key and 
    new names as values.

    Parameters
    ----------
    tree : pandas.DataFrame
        A DataFrame representing a parsed tree structure. Must contain:
        - ``leaves`` : list-like or string
            Descendant tip labels for each node. Can be a list, tuple, set, NumPy array,
            or a string representation of these.
        - ``nodes`` : str
            Node name. The root node should be labeled as ``"Root"`` (case-insensitive).
        The DataFrame index should uniquely identify branches.

    leaf_dict : dictionary
        Should have old names as keys and new names as values.

    Returns
    -------
    pandas.DataFrame
        A new tree dataframe with leaves renamed.

    Notes
    -----
    - Leaf parsing is delegated to the internal helper ``_parse_leaves``, which normalizes
      various formats (Python-literal strings, CSV-like strings, sequences, arrays) into
      a set of non-empty strings.
    - Intersection is computed using Python set operations for efficiency.
    - The root branch is always retained regardless of leaf content.
    """

    if not isinstance(leaf_dict, dict):
        raise TypeError("leaf_dict must be a dictionary")

    sub_tree = tree.copy()
    sub_tree["nodes"] = pd.NA
    sub_tree["nodes"] = sub_tree["nodes"].astype(object)
    sub_tree["leaves"] = pd.NA
    sub_tree["leaves"] = sub_tree["leaves"].astype(object)
    sub_tree["parent"] = pd.NA
    sub_tree["parent"] = sub_tree["parent"].astype(object)

    for branch in tree.index:
        if tree.at[branch, 'nodes'] in leaf_dict:
            sub_tree.at[branch, 'nodes'] = leaf_dict[tree.at[branch, 'nodes']]
        else:
            sub_tree.at[branch, 'nodes'] = tree.at[branch, 'nodes']

        if tree.at[branch, 'parent'] in leaf_dict:
            sub_tree.at[branch, 'parent'] = leaf_dict[tree.at[branch, 'parent']]
        else:
            sub_tree.at[branch, 'parent'] = tree.at[branch, 'parent']

        leaf_set = parse_leaves(tree.loc[branch, "leaves"])
        new_leaf_list = []
        for leaf in leaf_set:
            if leaf in leaf_dict:
                new_leaf_list.append(leaf_dict[leaf])
            else:
                new_leaf_list.append(leaf)
        sub_tree.at[branch, "leaves"] = new_leaf_list

    return sub_tree

def to_newick(
    df: pd.DataFrame,
    label_internal: bool = True,
    include_root_name: bool = False,
    order: str = "as_is",  # "as_is" preserves DataFrame order
) -> str:
    """
    Convert a DataFrame produced by `parse_newick` back to a Newick string.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns: 'nodes', 'branchL', 'parent'
        (others like 'leaves', 'dist_to_root' are not required for reconstruction).
    label_internal : bool
        Include internal node names (e.g., in1, in2) in the output.
    include_root_name : bool
        Include root node name after the top-level parentheses.
    order : {"as_is"}
        Child ordering strategy. "as_is" uses the order children appear in the DataFrame.

    Returns
    -------
    str
        Newick string terminated with ';'
    """

    required = {"nodes", "branchL", "parent"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {', '.join(sorted(missing))}")

    # Normalize types
    nodes = df["nodes"].astype(str).tolist()
    bl_map = dict(zip(df["nodes"].astype(str), df["branchL"].astype(float)))

    # Build children map
    children = {n: [] for n in nodes}
    seen_as_child = set()
    for row in df.itertuples(index=False):
        node = str(row.nodes)
        parent = None if pd.isna(row.parent) else str(row.parent)
        if parent is not None and parent in children:
            children[parent].append(node)
            seen_as_child.add(node)

    # Detect root: prefer explicit "Root"; else node that never appears as a child
    if "Root" in children:
        root = "Root"
    else:
        root_candidates = [n for n in nodes if n not in seen_as_child]
        if len(root_candidates) != 1:
            raise ValueError(f"Could not uniquely determine root (candidates={root_candidates})")
        root = root_candidates[0]

    # Tips are nodes with no children
    tips = {n for n, ch in children.items() if len(ch) == 0}

    # Quoting helper for labels that need it (spaces, punctuation used by Newick)
    def quote_label(name: str) -> str:
        if name is None:
            return ""
        needs_quote = any(ch in name for ch in "():,; \t")
        if needs_quote:
            # Escape single quotes by doubling them
            return "'" + name.replace("'", "''") + "'"
        return name

    # Recursive emitter: returns the Newick fragment for `node`
    def emit(node: str) -> str:
        if node in tips:
            label = quote_label(node)
            bl = bl_map.get(node, None)
            bl_str = f":{bl:.10g}" if bl is not None else ""  # format floats compactly
            return f"{label}{bl_str}"
        else:
            parts = [emit(ch) for ch in children[node]]
            inner = ",".join(parts)

            # Decide whether to show a name at this internal node
            show_name = (node != root and label_internal) or (node == root and include_root_name)
            name = quote_label(node) if show_name else ""

            # Branch length on the edge from parent→node:
            # Newick puts it after the node’s parentheses+label.
            bl = bl_map.get(node, None)
            # By convention, do not append a branch length to the overall root
            bl_str = "" if node == root or bl is None else f":{bl:.10g}"

            return f"({inner}){name}{bl_str}"

    newick = emit(root)
    # Ensure it ends with ";"
    if not newick.endswith(";"):
        newick = newick + ";"
    return newick

