import numpy as np
try:
    from numba import njit
except Exception as e:
    raise RuntimeError(
        "Numba acceleration requested but 'numba' is not available. "
        "Install with `pip install numba` or use the pure-Python path."
    ) from e

@njit(cache=False)
def impdq_null_numba(D, rows, w_loc, r):
    """
    Fast null-model version of iMPDq.
    """
    k = len(w_loc)
    weighted_sum = 0.0
    weight_total = 0.0

    for i in range(k):
        wi = w_loc[i]

        # Rows with zero focal weight do not contribute to final iMPDq
        if wi <= 0.0:
            continue

        # --------------------------------------------------
        # First pass: row-wise maximum of -r * distance
        # --------------------------------------------------
        row_max = -np.inf
        found = False

        tax_i = rows[i]
        for j in range(k):

            # Exclude conspecific/self comparison explicitly
            if j == i:
                continue
            wj = w_loc[j]
            if wj <= 0.0:
                continue
            
            tax_j = rows[j]
            dij = D[tax_i, tax_j]

            x = -r * dij
            if x > row_max:
                row_max = x
            found = True
        if not found:
            continue

        # --------------------------------------------------
        # Second pass: soft-min numerator and denominator
        # --------------------------------------------------
        denom = 0.0
        numer = 0.0

        for j in range(k):
            if j == i:
                continue
            wj = w_loc[j]
            if wj <= 0.0:
                continue

            tax_j = rows[j]
            dij = D[tax_i, tax_j]
            z = (-r * dij) - row_max

            a = np.exp(z)
            aw = a * wj
            denom += aw
            numer += aw * dij

        if denom > 0.0 and np.isfinite(denom) and np.isfinite(numer):
            dval = numer / denom
            if np.isfinite(dval):
                weighted_sum += wi * dval
                weight_total += wi

    if weight_total == 0.0:
        return np.nan

    if not np.isfinite(weighted_sum):
        return np.nan

    return weighted_sum / weight_total

