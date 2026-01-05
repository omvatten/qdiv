import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple

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
