import numpy as np
try:
    from numba import njit, prange
except Exception as e:
    raise RuntimeError(
        "Numba acceleration requested but 'numba' is not available. "
        "Install with `pip install numba` or use the pure-Python path."
    ) from e

# Accelerator for nriq
@njit(cache=True, parallel=True)
def mpdq_null_numba(D, Rq_used):
    """
    Compute MPDq for all samples.

    Parameters
    ----------
    D : float64[:, :]
        Pairwise distance matrix, shape (N, N).
    Rq_used : float64[:, :]
        q-weighted relative abundance matrix, shape (N, S).

    Returns
    -------
    out : float64[:]
        MPDq value for each sample, shape (S,).
    """
    N, S = Rq_used.shape
    out = np.empty(S, dtype=np.float64)
    out[:] = np.nan

    for s in prange(S):
        # --------------------------------------------------
        # Collect taxa present in sample s with positive q-weight
        # --------------------------------------------------
        present = np.empty(N, dtype=np.int64)
        k = 0
        for i in range(N):
            if Rq_used[i, s] > 0.0:
                present[k] = i
                k += 1
        if k < 2:
            continue

        # --------------------------------------------------
        # q-weighted mean pairwise distance
        # --------------------------------------------------
        numer = 0.0
        denom = 0.0
        for ii in range(k - 1):
            tax_i = present[ii]
            wi = Rq_used[tax_i, s]

            for jj in range(ii + 1, k):
                tax_j = present[jj]
                wj = Rq_used[tax_j, s]

                dij = D[tax_i, tax_j]

                if np.isfinite(dij):
                    wij = wi * wj
                    numer += wij * dij
                    denom += wij

        if denom > 0.0:
            out[s] = numer / denom

    return out

# Accelerator for ntiq
@njit(cache=True, parallel=True)
def mntdq_null_numba(D, R_used, Rq_used):
    """
    Compute MNTDq for all samples.

    Parameters
    ----------
    D : float64[:, :]
        Distance matrix (N x N)
    R_used : float64[:, :]
        Relative abundance matrix (N x S)
    Rq_used : float64[:, :]
        q-weighted relative abundance matrix (N x S)

    Returns
    -------
    out : float64[:]
        MNTDq for each sample
    """

    N, S = R_used.shape
    out = np.empty(S, dtype=np.float64)
    out[:] = np.nan

    for s in prange(S):
        # --------------------------------------------------
        # Collect present taxa
        # --------------------------------------------------
        present = np.empty(N, dtype=np.int64)
        k = 0
        for i in range(N):
            if R_used[i, s] > 0.0:
                present[k] = i
                k += 1
        if k < 2:
            continue

        # --------------------------------------------------
        # Weighted MNTD
        # --------------------------------------------------
        numer = 0.0
        denom = 0.0
        for ii in range(k):
            tax_i = present[ii]
            dmin = np.inf

            # nearest neighbour search
            for jj in range(k):
                if ii == jj:
                    continue
                tax_j = present[jj]
                dij = D[tax_i, tax_j]
                if dij < dmin:
                    dmin = dij

            if np.isfinite(dmin):
                wi = Rq_used[tax_i, s]
                if wi > 0.0:
                    numer += wi * dmin
                    denom += wi

        if denom > 0.0:
            out[s] = numer / denom

    return out

# Accelerator for inriq
@njit(cache=True, parallel=True)
def impdq_null_numba(D, rows, w_loc, r):
    k = len(w_loc)

    # Thread-safe accumulation
    weighted_sum_i = np.zeros(k, dtype=np.float64)
    weight_total_i = np.zeros(k, dtype=np.float64)

    active = np.where(w_loc > 0.0)[0]
    na = len(active)
    for ii in prange(na):
        i = active[ii]
        wi = w_loc[i]
        row_max = -np.inf
        found = False
        tax_i = rows[i]

        # ---------------------
        # Pass 1: row maximum
        # ---------------------
        for jj in range(na):
            j = active[jj]
            if j == i:
                continue
            wj = w_loc[j]
            tax_j = rows[j]
            dij = D[tax_i, tax_j]
            x = -r * dij
            if x > row_max:
                row_max = x
            found = True
        if not found:
            continue

        # ---------------------
        # Pass 2: soft minimum
        # ---------------------
        numer = 0.0
        denom = 0.0
        for jj in range(na):
            j = active[jj]
            if j == i:
                continue
            wj = w_loc[j]
            tax_j = rows[j]
            dij = D[tax_i, tax_j]
            a = np.exp((-r * dij) - row_max)
            aw = a * wj
            denom += aw
            numer += aw * dij

        if denom > 0.0:
            dval = numer / denom
            if np.isfinite(dval):
                weighted_sum_i[i] = wi * dval
                weight_total_i[i] = wi

    weighted_sum = weighted_sum_i.sum()
    weight_total = weight_total_i.sum()

    if weight_total == 0.0:
        return np.nan
    if not np.isfinite(weighted_sum):
        return np.nan

    return weighted_sum / weight_total

# Accelerator for beta_ntiq
@njit(cache=True, parallel=True)
def beta_mntdq_numba(D, Rq, include_conspecifics):
    """
    Compute symmetric beta-MNTDq matrix.

    Matrix convention
    -----------------
    A_dir[s, t] = directed MNTDq from source sample s to target sample t
        A_dir[s, t] =
            sum_i w_i,s * min_j D[i, j] / sum_i w_i,s
    where i are taxa present in source sample s and j are taxa present
    in target sample t.

    The returned matrix is:
        beta[s, t] = 0.5 * (A_dir[s, t] + A_dir[t, s])

    Parameters
    ----------
    D : float64[:, :]
        Pairwise distance matrix, shape (N, N).
    Rq : float64[:, :]
        q-weighted abundance matrix, shape (N, S).
        Positive entries define presence.
    include_conspecifics : bool
        If False, taxa are not allowed to match themselves.

    Returns
    -------
    beta : float64[:, :]
        Symmetric beta-MNTDq matrix, shape (S, S).
    """
    N, S = Rq.shape

    # --------------------------------------------------
    # Present taxa per sample
    # --------------------------------------------------
    present = np.empty((S, N), dtype=np.int64)
    counts = np.zeros(S, dtype=np.int64)
    z = np.zeros(S, dtype=np.float64)
    for s in range(S):
        k = 0
        total = 0.0

        for i in range(N):
            wi = Rq[i, s]
            if wi > 0.0:
                present[s, k] = i
                k += 1
                total += wi

        counts[s] = k
        z[s] = total

    # --------------------------------------------------
    # Directed matrix
    # --------------------------------------------------
    A_dir = np.empty((S, S), dtype=np.float64)
    for p in prange(S * S):
        s = p // S
        t = p - s * S
        if z[s] <= 0.0:
            A_dir[s, t] = np.nan
            continue
        ks = counts[s]
        kt = counts[t]

        # if target has no taxa, directed numerator is zero.
        if ks == 0:
            A_dir[s, t] = np.nan
            continue
        if kt == 0:
            A_dir[s, t] = 0.0
            continue

        numer = 0.0
        for ii in range(ks):
            tax_i = present[s, ii]
            wi = Rq[tax_i, s]
            dmin = np.inf
            found = False
            for jj in range(kt):
                tax_j = present[t, jj]
                if not include_conspecifics and tax_i == tax_j:
                    continue
                dij = D[tax_i, tax_j]
                if np.isfinite(dij):
                    if dij < dmin:
                        dmin = dij
                    found = True

            # Converts +inf to 0.0 using
            # nan_to_num(posinf=0.0). Therefore, if no valid neighbor is
            # found, this source taxon contributes zero to the numerator
            # but its weight remains in the denominator z[s].
            if found:
                numer += wi * dmin

        A_dir[s, t] = numer / z[s]

    # --------------------------------------------------
    # Symmetrize
    # --------------------------------------------------
    beta = np.empty((S, S), dtype=np.float64)
    for p in prange(S * S):
        s = p // S
        t = p - s * S
        a = A_dir[s, t]
        b = A_dir[t, s]
        beta[s, t] = 0.5 * (a + b)

    return beta


# Accelerator for beta_inriq
@njit(cache=True, parallel=True)
def directed_beta_only_numba(
    R_used,
    Rq_used,
    D,
    r,
    include_conspecifics,
):
    """
    Directed beta-iMPDq only.

    Matrix convention:
        output rows    = source samples
        output columns = target samples
    For each directed comparison s -> t:
        source taxa are weighted by Rq_used[:, s]
        target taxa are weighted by Rq_used[:, t]
    """
    N, S = R_used.shape
    A_dir = np.empty((S, S), dtype=np.float64)
    for s in range(S):
        for t in range(S):
            A_dir[s, t] = np.nan
    use_uniform = r == 0.0 or r < 1e-12

    # Precompute target indices for each sample.
    # target_idx[t, k] gives the kth taxon present in target sample t.
    target_idx = np.empty((S, N), dtype=np.int64)
    target_count = np.zeros(S, dtype=np.int64)
    for t in range(S):
        c = 0
        for j in range(N):
            if R_used[j, t] > 0.0 and Rq_used[j, t] > 0.0:
                target_idx[t, c] = j
                c += 1
        target_count[t] = c

    # Each target column is independent, so parallelize over target samples.
    for t in prange(S):
        nt = target_count[t]
        if nt == 0:
            continue

        # Accumulators over source taxa for each source sample.
        # beta(s -> t) = sum_i Rq[i, s] * row_value[i] / sum_i Rq[i, s],
        # restricted to source taxa i with valid row_value.
        num_s = np.zeros(S, dtype=np.float64)
        den_s = np.zeros(S, dtype=np.float64)
        for i in range(N):
            row_value = np.nan
            if use_uniform:
                # Uniform-kernel limit:
                # row_value = sum_j w_j * D_ij / sum_j w_j
                numer = 0.0
                denom = 0.0

                for k in range(nt):
                    j = target_idx[t, k]
                    if not include_conspecifics and i == j:
                        continue
                    dij = D[i, j]
                    if np.isfinite(dij):
                        wj = Rq_used[j, t]
                        if wj > 0.0:
                            numer += wj * dij
                            denom += wj

                if denom > 0.0:
                    row_value = numer / denom

            else:
                # Stabilized exponential kernel:
                # a_ij = exp(-r * D_ij - row_max)
                row_max = -1.0e300
                has_valid = False

                for k in range(nt):
                    j = target_idx[t, k]
                    if not include_conspecifics and i == j:
                        continue

                    dij = D[i, j]

                    if np.isfinite(dij):
                        wj = Rq_used[j, t]
                        if wj > 0.0:
                            x = -r * dij
                            if x > row_max:
                                row_max = x
                            has_valid = True

                if has_valid:
                    numer = 0.0
                    denom = 0.0

                    for k in range(nt):
                        j = target_idx[t, k]

                        if not include_conspecifics and i == j:
                            continue

                        dij = D[i, j]

                        if np.isfinite(dij):
                            wj = Rq_used[j, t]

                            if wj > 0.0:
                                a = np.exp((-r * dij) - row_max)
                                aw = a * wj
                                numer += aw * dij
                                denom += aw

                    if denom > 0.0:
                        row_value = numer / denom

            # Aggregate this source-taxon row value into all source samples.
            if np.isfinite(row_value):
                for s in range(S):
                    wi = Rq_used[i, s]
                    if wi > 0.0:
                        num_s[s] += wi * row_value
                        den_s[s] += wi

        for s in range(S):
            if den_s[s] > 0.0:
                A_dir[s, t] = num_s[s] / den_s[s]
            else:
                A_dir[s, t] = np.nan

    return A_dir
