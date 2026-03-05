import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple, Callable, Sequence, List, Dict

# -----------------------------------------------------------------------------
# Simulate community structure
# -----------------------------------------------------------------------------
def simulate_community(
    size: int = 100,
    communities: int = 1,
    *,
    sigma: float = 1.0,
    mean: float = 0.0,
    c_prefix: str = 'Comm',
    species_prefix: str = 'OTU',
    sort: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> pd.DataFrame:
    """
    Simulate communities with log-normal species abundance distributions.

    Parameters
    ----------
    size : int, default=100
        Number of species in each community.
    communities : int, default=1
        Number of communities to simulate.
    sigma : float, default=1.0
        Standard deviation of the log-normal distribution.
        Low values yield more even communities; high values yield more dominance.
    mean : float, default=0.0
        Mean of the log-normal distribution.
    c_prefix : str, default='Comm'
        Prefix for community (column) names.
    species_prefix : str, default='OTU'
        Prefix for species (row) names.
    sort : bool, default=True
        If True, sort species from most to least abundant.
    random_state : int, np.random.Generator, or None
        Random seed or generator for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Simulated abundance table (species x communities).
        Rows = species, columns = communities.
    """
    rng = np.random.default_rng(random_state)
    comm_names = [f"{c_prefix}{i+1}" for i in range(communities)]
    species_names = [f"{species_prefix}{i+1}" for i in range(size)]
    data = np.zeros((size, communities))

    for j in range(communities):
        abund = rng.lognormal(mean=mean, sigma=sigma, size=size)
        if sort:# Optionally sort for more realistic rank-abundance curves
            abund = np.sort(abund)[::-1]
        abund = abund / abund.sum()
        data[:, j] = abund

    df = pd.DataFrame(data, index=species_names, columns=comm_names)
    return df

# -----------------------------------------------------------------------------
# Simulate samples collected from communities with known structures
# -----------------------------------------------------------------------------
def community_sample(
    community: pd.DataFrame,
    n: int = 10000,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> Optional[pd.DataFrame]:
    """
    Draw a multinomial sample from a community abundance distribution.

    Parameters
    ----------
    community : pd.DataFrame
        DataFrame of species abundances (species x communities).
        Each column is a community; values are relative or absolute abundances.
    n : int, default=10000
        Number of individuals to sample per community.
    random_state : int, np.random.Generator, or None
        Random seed or generator for reproducibility.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame of sampled counts (species x communities).
        Returns None if input is not a DataFrame.
    """
    if not isinstance(community, pd.DataFrame):
        return None

    rng = np.random.default_rng(random_state)
    smp_out = pd.DataFrame(0, index=community.index, columns=community.columns)

    for c in community.columns:
        abund = community[c].values.astype(float)
        # Normalize to probabilities if not already
        if abund.sum() == 0:
            continue
        probs = abund / abund.sum()
        counts = rng.multinomial(n, probs)
        smp_out[c] = counts

    return smp_out

# -----------------------------------------------------------------------------
# Simulate community assembly
# -----------------------------------------------------------------------------
def simulate_assembly(
    community: pd.DataFrame,
    immigrants: pd.DataFrame,
    fitness: pd.DataFrame,
    selection: float = 1.0,
    dispersal: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-3,
    noise_level: float = 0.0,
    interdependence: Optional[pd.DataFrame] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, int, bool]:
    """
    Simulate community assembly from local and immigrant pools, with optional stochastic noise
    and species interdependence (competition/facilitation).

    Parameters
    ----------
    community : pd.DataFrame
        Initial community abundance table (species x communities).
    immigrants : pd.DataFrame
        Immigrant pool abundance table (same shape as community).
    fitness : pd.DataFrame
        Fitness values for each species in each community (same shape).
    selection : float, default=1.0
        Relative weight of selection (fitness) in assembly.
    dispersal : float, default=1.0
        Relative weight of dispersal (immigration) in assembly.
    max_iter : int, default=1000
        Maximum number of assembly iterations.
    tol : float, default=1e-3
        Convergence tolerance (sum of squared changes).
    noise_level : float, default=0.0
        Standard deviation of Gaussian noise added to abundance updates (as a fraction of abundance).
    interdependence : pd.DataFrame or None
        Square matrix (species x species) of interaction coefficients.
        Positive values = facilitation, negative = competition.
        Diagonal is typically zero or negative (self-limitation).
    random_state : int, np.random.Generator, or None
        Random seed or generator for reproducibility.
    verbose : bool, default=False
        If True, print progress at each iteration.

    Returns
    -------
    final_community : pandas.DataFrame
        Assembled community abundance table (species x communities).
    n_iter : int
        Number of iterations performed.
    converged : bool
        True if convergence was reached within max_iter, False otherwise.
    """
    # Input validation
    for df, name in zip([community, immigrants, fitness], ['community', 'immigrants', 'fitness']):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input '{name}' must be a pandas DataFrame.")
        if df.shape != community.shape or not (df.index.equals(community.index) and df.columns.equals(community.columns)):
            raise ValueError(f"Input '{name}' must have the same shape, index, and columns as 'community'.")

    if interdependence is not None:
        if not isinstance(interdependence, pd.DataFrame):
            raise ValueError("interdependence must be a pandas DataFrame or None.")
        if interdependence.shape[0] != interdependence.shape[1]:
            raise ValueError("interdependence matrix must be square.")
        if not all(interdependence.index == community.index) or not all(interdependence.columns == community.index):
            raise ValueError("interdependence matrix must have same species as community (index and columns).")

    rng = np.random.default_rng(random_state)
    c_out = community.copy().astype(float)
    mech = np.array([selection, dispersal], dtype=float)
    mech = mech / mech.sum()
    converged = False

    for counter in range(1, max_iter + 1):
        old_c_out = c_out.copy()
        # Assembly update: selection and dispersal
        frac_fitness = c_out * 0.1 * mech[0] * fitness
        frac_immigrants = c_out * 0.1 * mech[1] * immigrants
        update = frac_fitness + frac_immigrants

        # Add interdependence effect (competition/facilitation)
        if interdependence is not None:
            # For each community (column), sum effects from all other species
            for col in update.columns:
                # effect = interdependence @ abundances (matrix multiplication)
                effect = interdependence.values @ c_out[col].values
                update[col] += effect

        # Add Gaussian noise (as a fraction of abundance)
        if noise_level > 0:
            noise = rng.normal(loc=0.0, scale=noise_level, size=update.shape) * update
            update = update + noise
            update[update < 0] = 0  # Prevent negative abundances

        # Normalize each community to sum to 100 (relative abundance)
        c_out = update.div(update.sum(axis=0), axis=1) * 100

        # Compute squared change for convergence
        change = (c_out - old_c_out).pow(2)
        total_change = change.values.sum()
        if verbose:
            print(f"Iteration {counter}: total squared change = {total_change:.6f}")
        if total_change < tol:
            converged = True
            break

    return c_out, counter, converged

# -----------------------------------------------------------------------------
# Generate matrix of pairwise interdependences between species
# -----------------------------------------------------------------------------
def generate_interdependence_matrix(
    n_species: int,
    interaction_strength: float = 1.0,
    positive_fraction: float = 0.5,
    symmetric: bool = False,
    diagonal: float = 0.0,
    species_prefix: str = 'OTU',
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> pd.DataFrame:
    """
    Generate a random species interdependence (interaction) matrix.

    Parameters
    ----------
    n_species : int
        Number of species (matrix will be n_species x n_species).
    interaction_strength : float, default=1.0
        Maximum absolute value for interaction coefficients.
    positive_fraction : float, default=0.5
        Fraction of off-diagonal interactions that are positive (facilitative).
        The rest will be negative (competitive).
    symmetric : bool, default=False
        If True, matrix will be symmetric (A_ij = A_ji).
    diagonal : float, default=0.0
        Value to set on the diagonal (e.g., 0 for no self-interaction, -1 for self-limitation).
    species_prefix : str, default='OTU'
        Prefix for species (row) names.
    random_state : int, np.random.Generator, or None
        Random seed or generator for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Interaction matrix (species x species).
    """
    rng = np.random.default_rng(random_state)
    # Randomly assign positive or negative sign to each off-diagonal element
    signs = rng.choice(
        [-1, 1],
        size=(n_species, n_species),
        p=[1 - positive_fraction, positive_fraction]
    )
    # Random strengths
    strengths = rng.uniform(0, interaction_strength, size=(n_species, n_species))
    matrix = signs * strengths
    # Set diagonal
    np.fill_diagonal(matrix, diagonal)
    # Symmetrize if requested
    if symmetric:
        matrix = (matrix + matrix.T) / 2
    species_names = [f"{species_prefix}{i+1}" for i in range(n_species)]
    return pd.DataFrame(matrix, index=species_names, columns=species_names)


# -----------------------------------------------------------------------------
# Generate tree dataframe
# -----------------------------------------------------------------------------
def make_block_tree_df(
    k_per_level: Sequence[int],
    *,
    branch_length: float | Sequence[float] | Callable[[int, str, int], float] = 1.0,
    root_name: str = "Root",
    leaf_prefix: str = "OTU",
    internal_prefix: str = "in",
) -> pd.DataFrame:
    """
    Generate a block tree where branching factor varies with depth.
    Compatible with phylo_utils (nodes, parent, branchL, leaves, dist_to_root).

    Parameters
    ----------
    k_per_level : sequence of int
        k_per_level[level] = number of children created at this depth.
        Length of k_per_level = total depth.
    branch_length :
        float                → same length everywhere
        sequence[len=depth]  → branch_length[level]
        callable(level, parent_name, child_index) → full control
    """
    depth = len(k_per_level)

    # ---- Normalize BL rule --------------------------------------------------
    if isinstance(branch_length, (int, float)):
        def resolve_bl(level, parent, j):
            return float(branch_length)

    elif isinstance(branch_length, (list, tuple, np.ndarray)):
        if len(branch_length) != depth:
            raise ValueError("branch_length must have length equal to depth")
        def resolve_bl(level, parent, j):
            return float(branch_length[level])

    elif callable(branch_length):
        def resolve_bl(level, parent, j):
            return float(branch_length(level, parent, j))

    else:
        raise TypeError("branch_length must be float, sequence, or callable")

    # ---- Storage ------------------------------------------------------------
    nodes = []
    parents = []
    branchL = []
    dist = []
    children_map = {}
    dist_map = {root_name: 0.0}

    internal_counter = 0
    leaf_counter = 0

    # queue entries: (node_name, level)
    queue = [(root_name, 0)]
    parents_map = {root_name: None}

    # Track order to preserve reproducible child ordering
    while queue:
        name, level = queue.pop(0)

        # Record this row
        p = parents_map[name]
        if p is None:
            nodes.append(name)
            parents.append(None)
            branchL.append(0.0)
            dist.append(0.0)
        else:
            # find j: index of this node among parent's children
            j = children_map[p].index(name)
            L = resolve_bl(level-1, p, j)
            branchL.append(float(L))
            d = dist_map[p] + float(L)
            dist_map[name] = d
            nodes.append(name)
            parents.append(p)
            dist.append(d)

        # Expand children only if below final level
        if level < depth:
            k = k_per_level[level]
            kids = []
            for j in range(k):
                if level == depth - 1:
                    # next level would be leaves
                    leaf_counter += 1
                    child = f"{leaf_prefix}{leaf_counter}"
                else:
                    internal_counter += 1
                    child = f"{internal_prefix}{internal_counter}"

                parents_map[child] = name
                kids.append(child)
                queue.append((child, level + 1))

            children_map[name] = kids
        else:
            children_map[name] = []

    # ---- Compute leaves sets bottom-up -------------------------------------
    all_nodes = nodes.copy()
    leaves_col = [set() for _ in all_nodes]
    idx_of = {n: i for i, n in enumerate(all_nodes)}

    # initialize tips
    for n in all_nodes:
        if len(children_map[n]) == 0:
            leaves_col[idx_of[n]] = {n}

    changed = True
    while changed:
        changed = False
        for n, kids in children_map.items():
            if not kids:
                continue
            if all(leaves_col[idx_of[c]] for c in kids):
                merged = set()
                for c in kids:
                    merged |= leaves_col[idx_of[c]]
                if merged != leaves_col[idx_of[n]]:
                    leaves_col[idx_of[n]] = merged
                    changed = True

    # ---- DataFrame ----------------------------------------------------------
    return pd.DataFrame(
        {
            "nodes": nodes,
            "leaves": leaves_col,
            "branchL": branchL,
            "parent": parents,
            "dist_to_root": dist,
        }
    ).reset_index(drop=True)


def make_beta_splitting_tree_df(
    n_leaves: int,
    beta: float,
    *,
    branch_length: float | Sequence[float] | Callable[[int, str, int], float] | str = "ultrametric",
    root_name: str = "Root",
    leaf_prefix: str = "OTU",
    internal_prefix: str = "in",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Aldous β-splitting binary tree, returned as a DataFrame compatible with your phylo utils.

    Columns:
        nodes (str), leaves (set[str]), branchL (float), parent (str|None), dist_to_root (float)

    Parameters
    ----------
    n_leaves : int
        Number of tips (>= 1).
    beta : float
        β parameter; requires beta > -1 so Beta(β+1, β+1) is defined.
        Larger β → more balanced; β -> -1+ → more comb-like.
    branch_length :
        "ultrametric" (default) or:
        - float: fixed length for all edges.
        - sequence[len = max_level+1] indexed by *parent level* (0=root).
        - callable(level, parent_name, child_index)->float for full control.
    """

    if n_leaves < 1:
        raise ValueError("n_leaves must be >= 1")
    if beta <= -1:
        raise ValueError("beta must be > -1 (so Beta(β+1, β+1) is valid).")

    rng = np.random.default_rng(random_state)

    # ----- Build topology ----------------------------------------------------
    tips = [f"{leaf_prefix}{i+1}" for i in range(n_leaves)]
    rng.shuffle(tips)

    children: Dict[str, List[str]] = {}
    parent: Dict[str, Optional[str]] = {}
    internal_counter = 0

    def new_internal() -> str:
        nonlocal internal_counter
        internal_counter += 1
        return f"{internal_prefix}{internal_counter}"

    def build(subtips: List[str]) -> str:
        """Return node name for clade defined by 'subtips'."""
        m = len(subtips)
        if m == 1:
            name = subtips[0]
            children[name] = []
            parent.setdefault(name, None)
            return name

        # Sample split proportion and an integer split with no empty side
        p = rng.beta(beta + 1.0, beta + 1.0)

        if m == 2:
            L = 1  # only valid non-empty split
        else:
            # Ensure L in [1, m-1]
            L = int(rng.binomial(m - 2, p)) + 1

        left, right = subtips[:L], subtips[L:]
        assert 1 <= len(left) <= m - 1
        assert 1 <= len(right) <= m - 1

        node = new_internal()
        a = build(left)
        b = build(right)
        children[node] = [a, b]
        parent[a] = node
        parent[b] = node
        parent.setdefault(node, None)
        return node

    # Assemble a single root (named as requested)
    if n_leaves == 1:
        root = tips[0]
        children[root] = []
        parent[root] = None
    else:
        first = build(tips)
        # Rename the first internal to 'Root' to avoid a single-child super-root
        # (so the tree stays strictly binary from the top)
        if first.startswith(internal_prefix):
            root = root_name
            # rebind: move first's children under root
            kids = children[first]
            children[root] = kids
            parent[root] = None
            for k in kids:
                parent[k] = root
            # remove the old internal name from maps
            del children[first]
            parent.pop(first, None)
        else:
            # (Shouldn't happen; 'first' will be internal when n_leaves>1)
            root = root_name
            children[root] = [first]
            parent[first] = root
            parent[root] = None

    # ----- Compute leaves sets (post-order unions) ---------------------------
    all_nodes = set(parent.keys()) | set(children.keys())
    leaves_map: Dict[str, set] = {n: set() for n in all_nodes}
    for n in list(all_nodes):
        if len(children.get(n, [])) == 0:
            leaves_map[n] = {n}

    changed = True
    while changed:
        changed = False
        for n, kids in list(children.items()):
            if not kids:
                continue
            if all(leaves_map[k] for k in kids):
                newset = set()
                for k in kids:
                    newset |= leaves_map[k]
                if newset != leaves_map[n]:
                    leaves_map[n] = newset
                    changed = True

    # ----- Assign branch lengths & distances ---------------------------------
    nodes_df: List[str] = []
    parents_df: List[Optional[str]] = []
    branchL_df: List[float] = []
    dist_df: List[float] = []

    if isinstance(branch_length, str) and branch_length.lower() == "ultrametric":
        # Compute raw heights: leaf=0; internal=max(child heights)+1
        height_raw: Dict[str, float] = {}

        def post_height(n: str) -> float:
            kids = children.get(n, [])
            if not kids:
                h = 0.0
            else:
                h = max(post_height(k) for k in kids) + 1.0
            height_raw[n] = h
            return h

        H_root = post_height(root)
        height = {n: (h / H_root if H_root > 0 else 0.0) for n, h in height_raw.items()}

        # BFS traversal to fill arrays
        queue: List[Tuple[str, Optional[str], int]] = [(root, None, 0)]
        while queue:
            n, p, lvl = queue.pop(0)
            if p is None:
                bl = 0.0
                dist = 1.0 - height[n]
            else:
                bl = max(0.0, height[p] - height[n])
                dist = 1.0 - height[n]

            nodes_df.append(n)
            parents_df.append(p)
            branchL_df.append(float(bl))
            dist_df.append(float(dist))

            for j, c in enumerate(children.get(n, [])):
                queue.append((c, n, lvl + 1))

    else:
        # Non-ultrametric: resolve length by fixed/per-level/callable, accumulate distance.
        def resolve_bl(level: int, parent_name: str, child_index: int) -> float:
            if isinstance(branch_length, (int, float)):
                return float(branch_length)
            elif callable(branch_length):
                return float(branch_length(level, parent_name, child_index))
            elif isinstance(branch_length, (list, tuple, np.ndarray)):
                if level >= len(branch_length):
                    raise ValueError(f"branch_length sequence too short for level {level}")
                return float(branch_length[level])
            else:
                raise TypeError("branch_length must be 'ultrametric', float, sequence, or callable")

        queue: List[Tuple[str, Optional[str], int, float]] = [(root, None, 0, 0.0)]
        while queue:
            n, p, lvl, dist_here = queue.pop(0)
            if p is None:
                bl = 0.0
                dist_now = 0.0
            else:
                j = children[p].index(n)
                L = resolve_bl(lvl - 1, p, j)
                bl = max(0.0, float(L))
                dist_now = dist_here + bl

            nodes_df.append(n)
            parents_df.append(p)
            branchL_df.append(float(bl))
            dist_df.append(float(dist_now))

            for c in children.get(n, []):
                queue.append((c, n, lvl + 1, dist_now))

    # ----- Assemble DataFrame -------------------------------------------------
    df = pd.DataFrame({
        "nodes": nodes_df,
        "leaves": [leaves_map[n] for n in nodes_df],
        "branchL": branchL_df,
        "parent": parents_df,             # None for root
        "dist_to_root": dist_df,
    }).reset_index(drop=True)

    return df

