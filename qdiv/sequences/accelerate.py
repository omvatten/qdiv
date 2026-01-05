import numpy as np
try:
    from numba import njit, prange
except Exception as e:
    raise RuntimeError(
        "Numba acceleration requested but 'numba' is not available. "
        "Install with `pip install numba` or use the pure-Python path."
    ) from e

@njit(cache=True)
def _levenshtein_band(s1: str, s2: str, band: int) -> np.int64:
    """
    Banded Wagner–Fischer Levenshtein distance with safe band boundaries.
    Fills the DP matrix with a large sentinel so out-of-band neighbor reads are valid.
    """
    len1 = len(s1)
    len2 = len(s2)

    # Quick exits
    if s1 == s2:
        return np.int64(0)
    if len1 == 0:
        return np.int64(len2)
    if len2 == 0:
        return np.int64(len1)

    # Ensure band connects start->end
    if band < 0:
        band = 0
    diff = abs(len1 - len2)
    if band < diff:
        band = diff

    # Sentinel larger than any possible edit distance
    sentinel = np.int64(len1 + len2 + 1)

    # DP matrix filled with sentinel
    m = np.full((len1 + 1, len2 + 1), sentinel, dtype=np.int64)

    # Initialize borders
    for i in range(len1 + 1):
        m[i, 0] = i
    for j in range(len2 + 1):
        m[0, j] = j

    # Fill only within the band; neighbors outside band read 'sentinel'
    for i in range(1, len1 + 1):
        j_lo = 1 if i - band < 1 else i - band
        j_hi_excl = i + band + 1
        if j_hi_excl > (len2 + 1):
            j_hi_excl = len2 + 1
        for j in range(j_lo, j_hi_excl):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            # deletion, insertion, substitution
            d  = m[i - 1, j]     + 1
            ins= m[i,     j - 1] + 1
            sub= m[i - 1, j - 1] + cost
            # min of three
            tmp = d if d < ins else ins
            m[i, j] = sub if sub < tmp else tmp

    return m[len1, len2]

@njit(parallel=True, cache=True)
def _compute_matrices_parallel(seqs, band, lengths):
    """
    Compute full pairwise distance and normalized matrices in parallel.
    """
    n = len(seqs)
    dists = np.zeros((n, n), dtype=np.int64)     # int64 for safety
    norms = np.zeros((n, n), dtype=np.float64)
    for i in prange(n):
        dists[i, i] = 0
        norms[i, i] = 0.0
        li = lengths[i]
        for j in range(i + 1, n):
            lj = lengths[j]
            dist = _levenshtein_band(seqs[i], seqs[j], band)
            dists[i, j] = dist
            dists[j, i] = dist
            denom = li if li > lj else lj
            if denom == 0:
                # both empty -> dist==0; one empty -> dist>0 -> normalized 1.0
                val = 0.0 if dist == 0 else 1.0
            else:
                val = dist / denom
            norms[i, j] = val
            norms[j, i] = val
    return dists, norms

def compute_distance_matrix_numba(ids, seqs, band_width: int = 12):
    """
    Public wrapper: prepare inputs, call Numba kernels, and return numpy matrices.
    """
    # Coerce sequences to strings and guard against None/NaN
    clean_seqs = [("" if (s is None) else str(s)) for s in seqs]
    # Precompute lengths (typed array)
    lengths = np.array([len(s) for s in clean_seqs], dtype=np.int64)
    dmat, nmat = _compute_matrices_parallel(clean_seqs, int(band_width), lengths)
    return dmat, nmat


