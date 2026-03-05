"""
Functions for phylogenetic trees:
    - parse_newick : returns tree dict
    - tree_to_dataframe : converts tree dict to dataframe
    - dataframe_to_tree : converts dataframe to tree dict
    - subset_tree : subsets tree dict to list of leaf nodes
    - subset_tree_df : subsets dataframe quickly, useful for diversity calculations
    - tree_to_newick : converts tree dict to newick
    - reroot_midpoint : roots a tree dict at midpoint
    - parse_leaves : get a set of leaves descending from a node
    - rename_leaves : rename leaves in dataframe
    - ra_to_branches : get dataframe with each branch and the ra for each sample
    - compute_Tmean : get Tmean for a tree dataframe and featurelist
"""
import pandas as pd
import numpy as np
import ast

__all__ = [
    "parse_newick",
    "tree_to_dataframe",
    "dataframe_to_tree",
    "subset_tree",
    "subset_tree_df",
    "tree_to_newick",
    "reroot_midpoint",
    "parse_leaves",
    "rename_leaves",
    "ra_to_branches",
    "compute_Tmean"
]

AUTO_INTERNAL_PREFIX = "in"

def parse_newick(newick: str):
    """
    Newick parser. Returns a dictionary
    """
    s = newick
    n = len(s)
    i = 0

    # Fast helpers
    def skip_ws(j):
        while j < n and s[j].isspace():
            j += 1
        return j

    def read_until_delim(j):
        """Read a label token until one of '():,;' or whitespace."""
        start = j
        while j < n and s[j] not in '():,;':
            j += 1
        # trim trailing spaces from the chunk
        k = j
        while k > start and s[k-1].isspace():
            k -= 1
        return s[start:k], j

    def read_number(j):
        """Read a float after ':' quickly."""
        start = j
        # include signs, digits, dot, exponent
        while j < n and (s[j].isdigit() or s[j] in '+-.eE'):
            j += 1
        # strip trailing spaces
        k = j
        while k > start and s[k-1].isspace():
            k -= 1
        return float(s[start:k]), j

    # Cheap numeric check for bootstrap
    def looks_numeric(tok: str) -> bool:
        if not tok:
            return False
        c = tok[0]
        if c in '+-.' or c.isdigit():
            # fast path: try float only when plausible
            try:
                float(tok)
                return True
            except Exception:
                return False
        return False

    auto_counter = 1

    # Parsing state: stack holds current internal nodes awaiting children
    stack = []
    root = None

    i = skip_ws(i)
    expecting_child = False  # whether we expect a new child after comma or '('

    while i < n:
        ch = s[i]

        if ch == '(':
            # Start new internal node frame
            node = {"name": None, "length": None, "children": [], "parent": None}
            if stack:
                # attach to parent immediately
                parent = stack[-1]
                node["parent"] = parent
                parent["children"].append(node)
            stack.append(node)
            if root is None:
                root = node
            i += 1
            i = skip_ws(i)
            expecting_child = True

        elif ch == ',':
            # Next sibling
            i += 1
            i = skip_ws(i)
            expecting_child = True

        elif ch == ')':
            # Close current internal node; may have label and/or length
            i += 1
            i = skip_ws(i)
            label = None

            # Optional label (could be bootstrap)
            if i < n and s[i] not in ':,);':
                label, i = read_until_delim(i)
                i = skip_ws(i)
                if looks_numeric(label):
                    label = None  # treat as bootstrap

            # Optional branch length
            length = None
            if i < n and s[i] == ':':
                i += 1
                i = skip_ws(i)
                length, i = read_number(i)
                i = skip_ws(i)

            node = stack.pop()
            node["length"] = length
            if label is None:
                node["name"] = f"{AUTO_INTERNAL_PREFIX}{auto_counter}"
                auto_counter += 1
            else:
                node["name"] = label

            expecting_child = False

        elif ch == ';':
            # End of tree
            break

        else:
            # Leaf: read label [name][ :length ]
            if not expecting_child and not stack:
                # tolerate stray whitespace
                if ch.isspace():
                    i += 1
                    continue
                raise ValueError("Unexpected token while not inside a subtree")

            name, i = read_until_delim(i)
            i = skip_ws(i)

            length = None
            if i < n and s[i] == ':':
                i += 1
                i = skip_ws(i)
                length, i = read_number(i)
                i = skip_ws(i)

            leaf = {"name": name, "length": length, "children": [], "parent": None}
            parent = stack[-1] if stack else None
            if parent is None:
                # Degenerate single-node tree without parentheses
                root = leaf
            else:
                leaf["parent"] = parent
                parent["children"].append(leaf)

            expecting_child = False

    # Optional trailing ';'
    # Ensure root exists
    if root is None:
        raise ValueError("Empty or invalid Newick string")

    # Root length should be None
    root["length"] = None
    return root

def tree_to_dataframe(tree):
    """
    Convert the dictionary tree structure produced by parse_newick()
    into a DataFrame with columns:
        nodes, leaves, branchL, parent, dist_to_root
    """

    nodes = []
    parents = []
    branch_lengths = []
    leaves = []

    # --- Traversal to collect nodes ---
    def collect(node, parent_name):
        name = node["name"]
        length = node["length"]
        children = node["children"]

        nodes.append(name)
        parents.append(parent_name)
        branch_lengths.append(0.0 if parent_name is None else (length or 0.0))

        if not children:  # leaf
            leaves.append({name})
        else:
            leaves.append(set())   # internal, will fill later

        for c in children:
            collect(c, name)

    collect(tree, parent_name=None)

    df = pd.DataFrame({
        "nodes": nodes,
        "parent": parents,
        "branchL": branch_lengths,
        "leaves": leaves,
    })

    # --- Compute leaf sets bottom-up ---
    children_map = {}
    for node, parent in zip(df["nodes"], df["parent"]):
        if parent is not None:
            children_map.setdefault(parent, []).append(node)

    # Postorder: process children before parent
    order = list(reversed(df.index.tolist()))
    leaf_map = {n: set(df.loc[df["nodes"] == n, "leaves"].values[0]) for n in df["nodes"]}

    for idx in order:
        n = df.at[idx, "nodes"]
        if n in children_map:
            merged = set()
            for c in children_map[n]:
                merged |= leaf_map[c]
            leaf_map[n] = merged

    df["leaves"] = df["nodes"].map(leaf_map)

    # --- Distance to root ---
    name_to_idx = {n: i for i, n in enumerate(df["nodes"])}
    parent_idx = np.full(len(df), -1, dtype=int)

    for i, (n, p) in enumerate(zip(df["nodes"], df["parent"])):
        if p in name_to_idx:
            parent_idx[i] = name_to_idx[p]

    branch_len = df["branchL"].to_numpy(float)
    dist = np.zeros(len(df))

    roots = np.where(parent_idx == -1)[0]
    stack = list(roots)

    while stack:
        p = stack.pop()
        children = np.where(parent_idx == p)[0]
        for c in children:
            dist[c] = dist[p] + branch_len[c]
            stack.append(c)

    df["dist_to_root"] = dist
    return df


def dataframe_to_tree(df):
    """
    Convert a tree DataFrame (nodes, parent, branchL, leaves, dist_to_root)
    back into the dictionary tree structure.

    Returns:
        A tree dict:
        {
            "name": str,
            "length": float or None,
            "children": [...],
            "parent": None
        }
    """

    df = df.copy()

    # --- Normalize nodes column ---
    df["nodes"] = df["nodes"].apply(
        lambda x: None if pd.isna(x) else str(x).strip()
    )

    # Collect nodes AFTER normalization
    all_nodes = set(df["nodes"])

    # --- Normalize parent values ---
    def norm_parent(p):
        if p is None or pd.isna(p):
            return None
        s = str(p).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        return s

    df["parent"] = df["parent"].apply(norm_parent)


    # --- Identify roots ---
    roots = df.loc[df["parent"].isna(), "nodes"].tolist()

    # --- If zero or multiple roots → synthetic root ---
    if len(roots) != 1:
        synthetic = "__synthetic_root__"
        while synthetic in all_nodes:
            synthetic += "_x"

        # Add root
        df = pd.concat([
            df,
            pd.DataFrame({
                "nodes": [synthetic],
                "parent": [None],
                "branchL": [None],
                "leaves": [set()],
                "dist_to_root": [0.0],
            })
        ], ignore_index=True)

        # Attach *existing* roots under synthetic
        for r in roots:
            df.loc[df["nodes"] == r, "parent"] = synthetic

        roots = [synthetic]
        all_nodes.add(synthetic)

    # --- Build recursive node dict ---
    rec_nodes = {}
    for _, row in df.iterrows():
        name = row["nodes"]
        parent = row["parent"]
        length = row["branchL"] if parent is not None else None

        rec_nodes[name] = {
            "name": name,
            "length": length,
            "children": [],
            "parent": None,
        }

    # --- Attach children safely ---
    for _, row in df.iterrows():
        n = row["nodes"]
        p = row["parent"]
        if p is not None and p in rec_nodes:
            rec_nodes[p]["children"].append(rec_nodes[n])
            rec_nodes[n]["parent"] = rec_nodes[p]

    return rec_nodes[roots[0]]

# Subset tree works on dictionary
def subset_tree(tree, keep_leaves):
    """
    Return a pruned version of the tree,
    keeping only branches that contain at least one leaf in keep_leaves.
    Leaves not in keep_leaves are removed.
    Internal nodes with no remaining children are pruned away.
    """

    keep_leaves = set(keep_leaves)

    def prune(node):
        # Leaf?
        if not node["children"]:
            return node if node["name"] in keep_leaves else None

        # Internal node: prune children
        pruned_children = []
        for c in node["children"]:
            child = prune(c)
            if child is not None:
                pruned_children.append(child)

        if not pruned_children:
            return None  # no surviving children → remove node

        # Build a new node
        return {
            "name": node["name"],
            "length": node["length"],
            "children": pruned_children,
            "parent": None   # parent links will be refreshed later
        }

    pruned = prune(tree)

    # Root special case: if nothing kept → return a root-only empty tree
    if pruned is None:
        return {
            "name": tree["name"],
            "length": None,
            "children": [],
            "parent": None,
        }

    # Fix parents after pruning
    def fix_parents(node, parent=None):
        node["parent"] = parent
        for c in node["children"]:
            fix_parents(c, node)

    fix_parents(pruned, None)

    return pruned

#Tree to newick works on tree dictionary
def tree_to_newick(node, *, precision=6):
    """
    Convert a recursive tree structure back into a Newick-formatted string.

    Args:
        node: dict with keys:
              - "name": label (str)
              - "length": branch length (float or None)
              - "children": list of child nodes
        precision: number of digits after decimal for branch lengths

    Returns:
        A Newick string ending with a semicolon.
    """

    # Format branch length
    def fmt_length(length):
        if length is None:
            return ""
        return f":{length:.{precision}f}"

    # Recursively traverse children
    def to_newick(n):
        if not n["children"]:  # leaf
            return n["name"] + fmt_length(n["length"])

        # internal node
        children_newick = ",".join(to_newick(c) for c in n["children"])
        lbl = n["name"] if n["name"] is not None else ""
        return f"({children_newick}){lbl}{fmt_length(n['length'])}"

    return to_newick(node) + ";"

def collapse_single_child_nodes(node):
    """
    Collapse internal nodes that have exactly one child.
    Branch lengths are added (parent length + child length).

    Returns the collapsed node (root may change).
    """

    # First collapse children (postorder)
    for i, child in enumerate(node["children"]):
        node["children"][i] = collapse_single_child_nodes(child)

    # Now collapse this node if it has exactly one child
    while len(node["children"]) == 1 and node["parent"] is not None:
        child = node["children"][0]

        # Combine branch lengths
        new_length = 0.0
        if node["length"] is not None:
            new_length += node["length"]
        if child["length"] is not None:
            new_length += child["length"]

        # Replace this node with its child
        child["length"] = new_length
        child["parent"] = node["parent"]

        # Reattach child in parent's list
        parent = node["parent"]
        for idx, sib in enumerate(parent["children"]):
            if sib is node:
                parent["children"][idx] = child
                break

        # Continue collapsing upward
        node = child

    return node

def _reroot_at_node(root, target_name: str):
    """
    Re-root a recursive tree at the node with name == target_name.

    Tree node structure (as in parse_newick_recursive / tree_to_dataframe):
        {
          "name": str,
          "length": float or None,   # distance from parent to this node
          "children": [ ... ],
          "parent": <node or None>
        }

    Notes
    -----
    • Mutates the tree in place and returns the new root node.
    • Edge lengths are preserved: after re-rooting, every undirected edge
      retains the same numeric length, just oriented to the new parent.
    • The new root's 'length' is set to None.

    Raises
    ------
    ValueError: if target_name is not found or the tree is invalid.
    """

    # ---- helpers -----------------------------------------------------------
    def find_node(n, name):
        stack = [n]
        seen = set()
        while stack:
            x = stack.pop()
            if id(x) in seen:
                continue
            seen.add(id(x))
            if x["name"] == name:
                return x
            stack.extend(x["children"])
        return None

    def path_to_root(n):
        path = []
        seen = set()
        cur = n
        while cur is not None:
            if id(cur) in seen:
                raise ValueError("Cycle detected while walking to root.")
            seen.add(id(cur))
            path.append(cur)
            cur = cur["parent"]
        return path  # [target, ..., old_root]

    # ---- validate single-root invariant -----------------------------------
    def count_roots(n):
        # Count nodes with parent == None
        stack, seen = [n], set()
        roots = set()
        while stack:
            x = stack.pop()
            if id(x) in seen:
                continue
            seen.add(id(x))
            if x["parent"] is None:
                roots.add(id(x))
            stack.extend(x["children"])
        return len(roots)

    # ---- main --------------------------------------------------------------
    # Locate target
    target = find_node(root, target_name)
    if target is None:
        raise ValueError(f"Target node '{target_name}' not found in tree.")

    # Already rooted here?
    if target["parent"] is None:
        return target  # nothing to do

    # Path from target up to old root
    path = path_to_root(target)            # [target, ..., old_root]
    edge_lengths = [node["length"] for node in path[:-1]]  # edge i is between path[i] and path[i+1]

    # Flip orientation along the path:
    # for each (child = path[i], parent = path[i+1]):
    for i in range(len(path) - 1):
        child = path[i]
        parent = path[i + 1]

        # 1) detach child from parent.children
        parent["children"] = [c for c in parent["children"] if c is not child]

        # 2) attach parent as a child of child (will reassign parent pointers below)
        child["children"].append(parent)

    # Rebuild parent pointers along the path (new orientation)
    path[0]["parent"] = None  # new root
    for j in range(1, len(path)):
        path[j]["parent"] = path[j - 1]

    # Assign corrected branch lengths:
    # new root has no parent -> length None
    path[0]["length"] = None
    for j in range(1, len(path)):
        path[j]["length"] = edge_lengths[j - 1]

    # Optional sanity check: still exactly one root?
    # (skip for speed if you wish)
    if count_roots(path[0]) != 1:
        raise ValueError("Re-rooting invariant violated: the tree has multiple roots.")

    return path[0]


def reroot_midpoint(root, *, name_hint="in_midroot", tol=1e-12):
    """
    Midpoint re-root a recursive tree.

    Node structure (as before):
        {
          "name": str,
          "length": float or None,  # distance from parent to this node
          "children": [ ... ],
          "parent": <node or None>
        }

    Returns
    -------
    new_root : dict
        The new root node (with parent=None and length=None).

    Notes
    -----
    - Works for any positive branch lengths (zero lengths tolerated).
    - Inserts a new internal node if the midpoint lies within an edge.
    - Uses reroot_at_node(...) to flip orientation along the chosen path.
    """

    # ----------------- helpers -----------------
    def collect_all_nodes(n):
        out, seen = [], set()
        stack = [n]
        while stack:
            x = stack.pop()
            if id(x) in seen:
                continue
            seen.add(id(x))
            out.append(x)
            if x["parent"] is not None:
                stack.append(x["parent"])
            stack.extend(x["children"])
        return out

    def is_leaf(n):
        return len(n["children"]) == 0

    def edge_length(a, b):
        """Undirected length for edge (a,b) where one is the other's parent."""
        if b is a["parent"]:
            return 0.0 if a["length"] is None else float(a["length"])
        if a is b["parent"]:
            return 0.0 if b["length"] is None else float(b["length"])
        raise ValueError("Nodes are not adjacent in the tree")

    def neighbors(n):
        out = list(n["children"])
        if n["parent"] is not None:
            out.append(n["parent"])
        return out

    def farthest_leaf_from(start):
        """
        Return (leaf_node, dist_map, prev_map).
        Since this is a tree (no cycles, unique paths), a single DFS/BFS works.
        """
        start_id = id(start)
        dist = {start_id: 0.0}
        prev = {start_id: None}
        byid = {start_id: start}

        stack = [start]
        seen = set()

        while stack:
            u = stack.pop()
            uid = id(u)
            if uid in seen:
                continue
            seen.add(uid)
            for v in neighbors(u):
                vid = id(v)
                if vid in dist:  # already discovered via the unique path
                    continue
                w = edge_length(u, v)
                dist[vid] = dist[uid] + w
                prev[vid] = uid
                byid[vid] = v
                stack.append(v)

        # among leaves, pick the farthest
        far_leaf = None
        far_d = -1.0
        for node in byid.values():
            if is_leaf(node):
                d = dist[id(node)]
                if d > far_d:
                    far_d = d
                    far_leaf = node
        return far_leaf, dist, prev

    def path_between(a, b, prev_from_a):
        """Return the node list along the unique path a->...->b using prev map from 'a'."""
        # Build a map id->node (needed to walk 'prev')
        byid = {}
        stack = [a]
        seen = set()
        while stack:
            x = stack.pop()
            if id(x) in seen:
                continue
            seen.add(id(x))
            byid[id(x)] = x
            if x["parent"] is not None:
                stack.append(x["parent"])
            stack.extend(x["children"])

        # Walk from b back to a via prev map, then reverse
        path_ids = []
        cur_id = id(b)
        while cur_id is not None:
            path_ids.append(cur_id)
            if cur_id == id(a):
                break
            cur_id = prev_from_a.get(cur_id, None)
        if not path_ids or path_ids[-1] != id(a):
            # In rare cases where our quick byid build didn't reach all nodes,
            # rebuild via a full traversal from 'a'.
            # (Should not happen for proper trees.)
            raise ValueError("Failed to reconstruct path between nodes.")
        path_ids.reverse()
        return [byid[i] for i in path_ids]

    def unique_name(root_node, base="in_midroot"):
        """Generate a unique internal name not present in the tree."""
        used = {n["name"] for n in collect_all_nodes(root_node)}
        if base not in used:
            return base
        k = 1
        while f"{base}{k}" in used:
            k += 1
        return f"{base}{k}"

    def insert_node_on_edge(u, v, r_from_u, name):
        """
        Insert a new node M at distance r_from_u from node u along the edge (u, v).
        Adjusts parent/child relations and branch lengths accordingly.
        Returns the newly created node M.
        """
        # Determine orientation and current edge length
        if v is u["parent"]:
            # Edge stored on u.length
            w = 0.0 if u["length"] is None else float(u["length"])
            r = float(r_from_u)
            if not (0.0 < r < w):
                raise ValueError("Split distance must be within the edge (u,parent)")
            # Create M
            M = {"name": name, "length": w - r, "children": [u], "parent": v}
            # Update v.children: replace u with M
            for i, ch in enumerate(v["children"]):
                if ch is u:
                    v["children"][i] = M
                    break
            # Update u
            u["parent"] = M
            u["length"] = r
            return M

        elif u is v["parent"]:
            # Edge stored on v.length
            w = 0.0 if v["length"] is None else float(v["length"])
            r = float(r_from_u)
            if not (0.0 < r < w):
                raise ValueError("Split distance must be within the edge (parent,v)")
            # Create M
            M = {"name": name, "length": r, "children": [v], "parent": u}
            # Update u.children: replace v with M
            for i, ch in enumerate(u["children"]):
                if ch is v:
                    u["children"][i] = M
                    break
            # Update v
            v["parent"] = M
            v["length"] = w - r
            return M

        else:
            raise ValueError("u and v are not adjacent when inserting a split node.")

    # ----------------- main logic -----------------
    # Degenerate cases
    all_nodes = collect_all_nodes(root)
    if len(all_nodes) == 1:
        # Single node tree -> already 'rooted'
        root["length"] = None
        root["parent"] = None
        return root

    # Pick an arbitrary leaf
    arb_leaf = next((n for n in all_nodes if is_leaf(n)), None)
    if arb_leaf is None:
        # Polytomy with no explicit leaves (shouldn't happen); treat any node as leaf
        arb_leaf = all_nodes[0]

    # First sweep: farthest leaf from arbitrary leaf
    L1, _, _ = farthest_leaf_from(arb_leaf)

    # Second sweep: farthest leaf from L1 (one end of the diameter)
    L2, dist_from_L1, prev_from_L1 = farthest_leaf_from(L1)
    D = dist_from_L1[id(L2)]
    if D <= tol:
        # All-zero or trivial-length tree -> keep as is
        return root

    # Recover the diameter path (sequence of nodes)
    diam_path = path_between(L1, L2, prev_from_L1)

    # Walk along the path to locate the midpoint
    target = D / 2.0
    acc = 0.0
    for i in range(len(diam_path) - 1):
        a = diam_path[i]
        b = diam_path[i + 1]
        w = edge_length(a, b)
        if acc + w + tol < target:
            acc += w
            continue

        # Where is the midpoint relative to edge (a,b)?
        d_from_a = target - acc  # 0 <= d_from_a <= w
        if d_from_a <= tol:
            # Midpoint at (or extremely close to) node 'a'
            return _reroot_at_node(root, a["name"])

        if (w - d_from_a) <= tol:
            # Midpoint at (or extremely close to) node 'b'
            return _reroot_at_node(root, b["name"])

        # Otherwise, the midpoint is inside the edge (a,b):
        # Insert a new node at distance d_from_a from 'a'
        mid_name = unique_name(root, base=name_hint)
        M = insert_node_on_edge(a, b, d_from_a, mid_name)
        # Now reroot at that newly inserted node
        return _reroot_at_node(root, M["name"])

    # Fallback (should not occur): reroot at L1
    return _reroot_at_node(root, L1["name"])

#Get the set of leaves from a dataframe node
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

# Rename leaves work on dataframe
def rename_leaves(
    df: pd.DataFrame,
    leaf_dict: dict,
    *,
    allow_partial: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Fast renamer for leaf labels in a DataFrame-based tree created by tree_to_dataframe().
    Preserves your original semantics:
      • Only *tips* are renamed in 'nodes'
      • 'parent' values are also remapped if they happen to be leaf names (rare)
      • Each row's 'leaves' set is renamed accordingly
      • Validates collisions and (optionally) missing mapping keys

    Expected columns: 'nodes' (object), 'parent' (object), 'branchL' (float), 'leaves' (set of str)

    Parameters
    ----------
    df : pd.DataFrame
    leaf_dict : dict {old_leaf_name -> new_leaf_name}
    allow_partial : bool
        If False, raise if some mapping keys are not present among leaves.
    inplace : bool
        If True mutate df, else return a copy.

    Returns
    -------
    pd.DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(leaf_dict, dict):
        raise TypeError("leaf_dict must be a dict {old->new}")

    required_cols = {"nodes", "parent", "branchL", "leaves"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Tree DataFrame missing columns: {sorted(missing)}")

    T = df if inplace else df.copy(deep=True)

    # --- 1) Ensure 'leaves' column contains sets (fast-path if already true) ---
    # tree_to_dataframe() populates sets, so this is typically a no-op.
    # Only repair rows whose value isn't a set to avoid O(n) conversions.
    if not all(isinstance(x, set) for x in T["leaves"].values):
        T["leaves"] = T["leaves"].apply(
            lambda x: x if isinstance(x, set)
            else (set(x) if isinstance(x, (list, tuple)) else ({x} if isinstance(x, str) else set()))
        )

    # --- 2) Gather *all leaves* (union of sets) in one pass for validation ---
    # Use an iterative union to avoid building a giant intermediate list
    all_leaves = set()
    for s in T["leaves"].values:
        all_leaves |= s

    # --- 3) Validation: presence & collisions ---
    if not allow_partial:
        missing_keys = set(leaf_dict.keys()) - all_leaves
        if missing_keys:
            raise ValueError(f"Some mapping keys are not present among leaves: {sorted(missing_keys)}")

    # Prevent mapping two different old leaves to the same new name
    reverse = {}
    for old, new in leaf_dict.items():
        if new in reverse and reverse[new] != old:
            raise ValueError(f"Mapping would collide: '{old}' and '{reverse[new]}' -> '{new}'")
        reverse[new] = old

    # Determine which node names are *tips*: they are leaves that are not a parent anywhere
    parents = set(p for p in T["parent"].dropna().values)
    # Tip test: in all_leaves and not a parent
    # Build a mask for tips in 'nodes'
    # We avoid astype(str): 'nodes' are strings already in data produced by tree_to_dataframe()
    nodes_vals = T["nodes"].values
    is_tip_mask = [(n in all_leaves) and (n not in parents) for n in nodes_vals]

    # Check for post-rename duplicates among tip names (projected)
    projected = set()
    for n, is_tip in zip(nodes_vals, is_tip_mask):
        if not is_tip:
            continue
        new_name = leaf_dict.get(n, n)
        if new_name in projected:
            raise ValueError(f"Renaming would create duplicate tip name '{new_name}'")
        projected.add(new_name)

    # --- 4) Apply renaming ---
    # 4a) 'nodes' for tips only
    # Build a Series for minimal assignment
    if any(is_tip_mask):
        idx = [i for i, m in enumerate(is_tip_mask) if m]
        # Map only where needed to avoid touching internal nodes
        T_nodes = T["nodes"].values.copy()
        for i in idx:
            old = T_nodes[i]
            T_nodes[i] = leaf_dict.get(old, old)
        T["nodes"] = T_nodes

    # 4b) 'parent' column: if a parent name equals a leaf being renamed, remap it
    # (rare in rooted trees but harmless)
    if not T["parent"].isna().all():
        T_parent = T["parent"].values.copy()
        for i, p in enumerate(T_parent):
            if p is not None and p in leaf_dict:
                T_parent[i] = leaf_dict[p]
        T["parent"] = T_parent

    # 4c) Rename items inside each row's 'leaves' set
    # Apply in-place to avoid creating new set objects if possible
    def _rename_set(s: set) -> set:
        if not s:
            return s
        # Fast path: detect if any element is in mapping; if not, return original set
        if not (s & set(leaf_dict)):
            return s
        # Else, rebuild a new set with mapped names
        return {leaf_dict.get(x, x) for x in s}

    # Since 'leaves' holds sets, we replace only rows that change to keep data movement minimal
    new_leaves = []
    changed_any = False
    for s in T["leaves"].values:
        new_s = _rename_set(s)
        new_leaves.append(new_s)
        changed_any |= (new_s is not s)  # bool OR

    if changed_any:
        T["leaves"] = new_leaves

    return T

#Subset a tree dataframe for diversity functions
def subset_tree_df(tree_df: pd.DataFrame, keep_leaves) -> pd.DataFrame:
    keep = set(keep_leaves)
    T = tree_df.copy()
    # ensure sets (tree_to_dataframe already returns sets)
    if not all(isinstance(x, set) for x in T["leaves"].values):
        T["leaves"] = T["leaves"].apply(lambda x: set(x) if not isinstance(x, set) else x)
    mask = T["leaves"].apply(lambda s: len(s & keep) > 0)
    T = T.loc[mask].copy()
    T["leaves"] = T["leaves"].apply(lambda s: s & keep)
    return T

# Get ra for each sample and each branch
def ra_to_branches(ra: pd.DataFrame, tree_df: pd.DataFrame) -> pd.DataFrame:
    """Return tree2 = (branches × samples) relative-abundance table."""
    n_branches, n_samples = tree_df.shape[0], ra.shape[1]
    A = np.zeros((n_branches, n_samples), dtype=float)
    # Pre-take numpy view of RA in same sample order
    ra_vals = ra.to_numpy(dtype=float, copy=False)
    # Map each LEAF (row in RA) to branch rows and add its vector
    leaf_list = ra.index.to_list()
    leaf_pos = {leaf: i for i, leaf in enumerate(leaf_list)}

    idx_map = {}
    for row_idx, s in enumerate(tree_df["leaves"].values):
        for leaf in s:
            idx_map.setdefault(leaf, []).append(row_idx)
    # convert to arrays for vectorized adds
    for k, v in idx_map.items():
        idx_map[k] = np.asarray(v, dtype=np.int32)

    for leaf, rows in idx_map.items():
        pos = leaf_pos.get(leaf)
        if pos is None:     # leaf not in abundance table
            continue
        A[rows, :] += ra_vals[pos, :]
    return pd.DataFrame(A, index=tree_df.index, columns=ra.columns)

def compute_Tmean(tree_df: pd.DataFrame, features: list[str]) -> float:
    # keep only leaf rows (tips) present in features
    tips = tree_df[~tree_df["nodes"].astype(str).str.startswith("in")]
    tips = tips.set_index("nodes")
    in_common = list(set(features).intersection(tips.index))
    if len(in_common) < len(features):
        missing = set(features) - set(in_common)
        raise ValueError(f"Not all features present in tree: {sorted(list(missing))[:5]} ...")
    return float(tips.loc[in_common, "dist_to_root"].sum()) / len(in_common)

def ladderize_tree_df(df, *, right=True):
    """
    Ladderize a DataFrame-based tree (as produced by tree_to_dataframe).

    right=True  → larger clade lower in plotting (right-heavy)
    right=False → smaller clade lower
    """

    df = df.copy()

    # Ensure leaves are sets
    if not all(isinstance(x, set) for x in df["leaves"]):
        df["leaves"] = df["leaves"].apply(lambda x: set(x) if not isinstance(x, set) else x)

    # Build children map
    children = {}
    for n, p in zip(df["nodes"], df["parent"]):
        if p is not None:
            children.setdefault(p, []).append(n)

    # Subtree size directly from leaves
    size = dict(zip(df["nodes"], df["leaves"].apply(len)))

    # DFS rebuild in ladderized order
    roots = df.loc[df["parent"].isna(), "nodes"].tolist()
    if len(roots) != 1:
        raise ValueError("Tree must have exactly one root")
    root = roots[0]

    ordered = []

    def dfs(n):
        ordered.append(n)
        kids = children.get(n, [])
        if kids:
            # sort by size
            kids_sorted = sorted(
                kids,
                key=lambda c: size[c],
                reverse=right,  # right=True → large clade last → right-heavy
            )
            for c in kids_sorted:
                dfs(c)

    dfs(root)

    # Reorder the DataFrame rows in the ladderized DFS order
    df["__order"] = pd.Categorical(df["nodes"], ordered, ordered=True)
    df = df.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    return df
