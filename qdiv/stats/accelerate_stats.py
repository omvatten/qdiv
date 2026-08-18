import numpy as np
try:
    from numba import njit
except Exception as e:
    raise RuntimeError(
        "Numba acceleration requested but 'numba' is not available. "
        "Install with `pip install numba` or use the pure-Python path."
    ) from e

@njit(cache=True)
def _deduplicate_positions(pos, seen, unique):
    """
    Deduplicate integer positions while preserving first occurrence order.

    Parameters
    ----------
    pos : 1D int array
        Possibly duplicated sample positions.
    seen : 1D int array
        Work array of length n_samples, initialized to 0.
    unique : 1D int array
        Work array with length at least len(pos).

    Returns
    -------
    m : int
        Number of unique positions written to unique[:m].
    """
    m = 0

    for i in range(pos.size):
        p = pos[i]
        if seen[p] == 0:
            seen[p] = 1
            unique[m] = p
            m += 1

    # reset marks
    for i in range(m):
        seen[unique[i]] = 0

    return m


@njit(cache=True)
def _within_sum_count(D, pos):
    """
    Sum upper-triangle distances among unique positions in pos.
    """
    n_samples = D.shape[0]

    seen = np.zeros(n_samples, dtype=np.uint8)
    unique = np.empty(pos.size, dtype=np.int64)

    m = _deduplicate_positions(pos, seen, unique)

    if m < 2:
        return np.nan, 0

    s = 0.0
    cnt = 0

    for i in range(m - 1):
        pi = unique[i]
        for j in range(i + 1, m):
            pj = unique[j]
            val = D[pi, pj]
            if np.isfinite(val):
                s += val
                cnt += 1

    if cnt == 0:
        return np.nan, 0

    return s, cnt


@njit(cache=True)
def _between_sum_count(D, pos1, pos2):
    """
    Sum rectangular between-block distances after separately deduplicating
    pos1 and pos2.
    """
    n_samples = D.shape[0]

    seen1 = np.zeros(n_samples, dtype=np.uint8)
    seen2 = np.zeros(n_samples, dtype=np.uint8)

    unique1 = np.empty(pos1.size, dtype=np.int64)
    unique2 = np.empty(pos2.size, dtype=np.int64)

    m1 = _deduplicate_positions(pos1, seen1, unique1)
    m2 = _deduplicate_positions(pos2, seen2, unique2)

    if m1 == 0 or m2 == 0:
        return np.nan, 0

    s = 0.0
    cnt = 0

    for i in range(m1):
        pi = unique1[i]
        for j in range(m2):
            pj = unique2[j]
            val = D[pi, pj]
            if np.isfinite(val):
                s += val
                cnt += 1

    if cnt == 0:
        return np.nan, 0

    return s, cnt


def weighted_mean_distance_numba(D, blocks, *, within: bool):
    total_sum = 0.0
    total_count = 0

    if within:
        for a, _ in blocks:
            s, cnt = _within_sum_count(D, np.asarray(a, dtype=np.int64))
            if cnt > 0 and np.isfinite(s):
                total_sum += s
                total_count += cnt
    else:
        for a, b in blocks:
            s, cnt = _between_sum_count(
                D,
                np.asarray(a, dtype=np.int64),
                np.asarray(b, dtype=np.int64),
            )
            if cnt > 0 and np.isfinite(s):
                total_sum += s
                total_count += cnt

    if total_count == 0:
        return np.nan, 0

    return total_sum / total_count, total_count
