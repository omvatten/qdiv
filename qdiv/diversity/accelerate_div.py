import numpy as np
try:
    from numba import njit, prange
except Exception as e:
    raise RuntimeError(
        "Numba acceleration requested but 'numba' is not available. "
        "Install with `pip install numba` or use the pure-Python path."
    ) from e

# Accelerate naive_beta
@njit(cache=True, parallel=True)
def naive_beta_numba(ra, q):
    n_features, n_samples = ra.shape
    out = np.ones((n_samples, n_samples), dtype=np.float64)

    if q == 1.0:
        for i in prange(n_samples - 1):
            p1 = ra[:, i]
            for j in range(i + 1, n_samples):
                p2 = ra[:, j]
                H1 = 0.0
                H2 = 0.0
                Hg = 0.0
                for k in range(n_features):
                    x = p1[k]
                    y = p2[k]
                    if x > 0:
                        H1 += x * np.log(x)
                    if y > 0:
                        H2 += y * np.log(y)
                    m = 0.5 * (x + y)
                    if m > 0:
                        Hg += m * np.log(m)

                alpha = np.exp(-0.5 * H1 - 0.5 * H2)
                gamma = np.exp(-Hg)
                beta = gamma / alpha
                out[i, j] = beta
                out[j, i] = beta

    else:
        inv = 1.0 / (1.0 - q)
        for i in prange(n_samples - 1):
            p1 = ra[:, i]
            for j in range(i + 1, n_samples):
                p2 = ra[:, j]
                p1q = 0.0
                p2q = 0.0
                mq = 0.0
                for k in range(n_features):
                    x = p1[k]
                    y = p2[k]
                    if x > 0:
                        p1q += x ** q
                    if y > 0:
                        p2q += y ** q
                    m = 0.5 * (x + y)
                    if m > 0:
                        mq += m ** q
                alpha = (0.5 * p1q + 0.5 * p2q) ** inv
                gamma = mq ** inv
                beta = gamma / alpha
                out[i, j] = beta
                out[j, i] = beta

    return out

# Accelerate phyl_beta
@njit(cache=True, parallel=True)
def phyl_beta_numba(A, L, q):
    """
    Pairwise phylogenetic beta diversity from branch abundances.

    Parameters
    ----------
    A : ndarray, shape (n_branches, n_samples)
        Branch-level descendant relative abundances.
    L : ndarray, shape (n_branches,)
        Branch lengths aligned to rows of A.
    q : float
        Hill diversity order.

    Returns
    -------
    out : ndarray, shape (n_samples, n_samples)
        Raw phylogenetic beta diversity values. Diagonal is kept at 0.0
        to match the current pandas implementation.
    """
    n_branches, n_samples = A.shape
    out = np.zeros((n_samples, n_samples), dtype=np.float64)

    # Precompute T_j = sum_b L_b * A_bj
    T = np.zeros(n_samples, dtype=np.float64)

    for j in range(n_samples):
        total = 0.0
        for b in range(n_branches):
            total += L[b] * A[b, j]
        T[j] = total

    if abs(q - 1.0) < 1e-6:

        for i in prange(n_samples - 1):
            for j in range(i + 1, n_samples):

                Tgamma = 0.5 * (T[i] + T[j])

                if Tgamma <= 0.0:
                    beta_val = np.nan
                else:
                    gamma_term = 0.0
                    alpha_sum = 0.0

                    for b in range(n_branches):
                        a1 = A[b, i]
                        a2 = A[b, j]
                        g = 0.5 * (a1 + a2)
                        lb = L[b]

                        if g > 0.0:
                            gamma_term += lb * g * np.log(g)

                        if a1 > 0.0:
                            alpha_sum += lb * a1 * np.log(a1)

                        if a2 > 0.0:
                            alpha_sum += lb * a2 * np.log(a2)

                    gamma_div = np.exp(-gamma_term / Tgamma)
                    alpha_div = np.exp(-alpha_sum / (2.0 * Tgamma))

                    if alpha_div > 0.0:
                        beta_val = gamma_div / alpha_div
                    else:
                        beta_val = np.nan

                out[i, j] = beta_val
                out[j, i] = beta_val

    elif q == 0.0:

        for i in prange(n_samples - 1):
            for j in range(i + 1, n_samples):

                Tgamma = 0.5 * (T[i] + T[j])

                if Tgamma <= 0.0:
                    beta_val = np.nan
                else:
                    gamma_occ = 0.0
                    alpha_occ = 0.0

                    for b in range(n_branches):
                        a1 = A[b, i]
                        a2 = A[b, j]
                        lb = L[b]

                        if 0.5 * (a1 + a2) > 0.0:
                            gamma_occ += lb

                        if a1 > 0.0:
                            alpha_occ += lb

                        if a2 > 0.0:
                            alpha_occ += lb

                    gamma_div = gamma_occ / Tgamma
                    alpha_div = alpha_occ / (2.0 * Tgamma)

                    if alpha_div > 0.0:
                        beta_val = gamma_div / alpha_div
                    else:
                        beta_val = np.nan

                out[i, j] = beta_val
                out[j, i] = beta_val

    else:

        inv = 1.0 / (1.0 - q)

        for i in prange(n_samples - 1):
            for j in range(i + 1, n_samples):

                Tgamma = 0.5 * (T[i] + T[j])

                if Tgamma <= 0.0:
                    beta_val = np.nan
                else:
                    gamma_sum = 0.0
                    alpha_sum = 0.0

                    for b in range(n_branches):
                        a1 = A[b, i]
                        a2 = A[b, j]
                        g = 0.5 * (a1 + a2)
                        lb = L[b]

                        if g > 0.0:
                            gamma_sum += lb * (g ** q)

                        if a1 > 0.0:
                            alpha_sum += lb * (a1 ** q)

                        if a2 > 0.0:
                            alpha_sum += lb * (a2 ** q)

                    gamma_term = gamma_sum / Tgamma
                    alpha_term = 0.5 * alpha_sum / Tgamma

                    if gamma_term > 0.0 and alpha_term > 0.0:
                        gamma_div = gamma_term ** inv
                        alpha_div = alpha_term ** inv
                        beta_val = gamma_div / alpha_div
                    else:
                        beta_val = np.nan

                out[i, j] = beta_val
                out[j, i] = beta_val

    return out


@njit(cache=True, parallel=True)
def func_beta_numba(D, R, q):
    """
    Numba backend for functional pairwise beta diversity.

    Parameters
    ----------
    D : float64[:, :]
        Functional distance matrix, shape (N, N).
    R : float64[:, :]
        Relative abundance matrix, shape (N, S).
    q : float
        Diversity order.

    Returns
    -------
    out : float64[:, :]
        Pairwise beta matrix, shape (S, S). This is beta, not beta squared.
    """
    N, S = R.shape
    out = np.zeros((S, S), dtype=np.float64)

    # --------------------------------------------------
    # Precompute present taxa for each sample
    # --------------------------------------------------
    present = np.empty((S, N), dtype=np.int64)
    counts = np.zeros(S, dtype=np.int64)
    for s in range(S):
        k = 0
        for i in range(N):
            if R[i, s] > 0.0:
                present[s, k] = i
                k += 1
        counts[s] = k

    is_q1 = q == 1.0
    if is_q1:
        exponent = 0.0
    else:
        exponent = 1.0 / (2.0 * (1.0 - q))

    # --------------------------------------------------
    # Pairwise beta
    # --------------------------------------------------
    for a in prange(S - 1):
        ka = counts[a]

        for b in range(a + 1, S):
            kb = counts[b]
            if ka == 0 or kb == 0:
                out[a, b] = np.nan
                out[b, a] = np.nan
                continue

            # --------------------------------------------------
            # Build union of taxa present in either sample.
            # The present arrays are sorted because they were collected
            # by scanning taxa from 0 to N-1.
            # --------------------------------------------------
            union = np.empty(ka + kb, dtype=np.int64)
            ia = 0
            ib = 0
            ku = 0

            while ia < ka and ib < kb:
                ta = present[a, ia]
                tb = present[b, ib]

                if ta < tb:
                    union[ku] = ta
                    ku += 1
                    ia += 1

                elif tb < ta:
                    union[ku] = tb
                    ku += 1
                    ib += 1

                else:
                    union[ku] = ta
                    ku += 1
                    ia += 1
                    ib += 1

            while ia < ka:
                union[ku] = present[a, ia]
                ku += 1
                ia += 1
            while ib < kb:
                union[ku] = present[b, ib]
                ku += 1
                ib += 1

            # --------------------------------------------------
            # Rao's Q for pooled/mean community
            #
            # p_mean_i = 0.5 * (p_i,a + p_i,b)
            # Q_pooled = sum_i sum_j p_mean_i p_mean_j D_ij
            # --------------------------------------------------
            Q_pooled = 0.0
            for ui in range(ku):
                tax_i = union[ui]
                pi = 0.5 * (R[tax_i, a] + R[tax_i, b])

                for uj in range(ku):
                    tax_j = union[uj]
                    pj = 0.5 * (R[tax_j, a] + R[tax_j, b])
                    dij = D[tax_i, tax_j]
                    if np.isfinite(dij):
                        Q_pooled += pi * pj * dij

            if Q_pooled <= 0.0 or not np.isfinite(Q_pooled):
                out[a, b] = np.nan
                out[b, a] = np.nan
                continue

            inv_Q = 1.0 / Q_pooled

            # --------------------------------------------------
            # Gamma component, Dg
            # --------------------------------------------------
            if is_q1:
                gsum = 0.0
                for ui in range(ku):
                    tax_i = union[ui]
                    pi = 0.5 * (R[tax_i, a] + R[tax_i, b])
                    for uj in range(ku):
                        tax_j = union[uj]
                        pj = 0.5 * (R[tax_j, a] + R[tax_j, b])
                        outer = pi * pj
                        dij = D[tax_i, tax_j]
                        if outer > 0.0 and np.isfinite(dij):
                            gsum += outer * np.log(outer) * dij * inv_Q

                Dg = np.exp(-0.5 * gsum)

            else:
                gval = 0.0
                for ui in range(ku):
                    tax_i = union[ui]
                    pi = 0.5 * (R[tax_i, a] + R[tax_i, b])
                    for uj in range(ku):
                        tax_j = union[uj]
                        pj = 0.5 * (R[tax_j, a] + R[tax_j, b])
                        outer = pi * pj
                        dij = D[tax_i, tax_j]
                        if outer > 0.0 and np.isfinite(dij):
                            gval += (outer ** q) * dij * inv_Q

                if gval <= 0.0 or not np.isfinite(gval):
                    out[a, b] = np.nan
                    out[b, a] = np.nan
                    continue

                Dg = gval ** exponent

            # --------------------------------------------------
            # Alpha component, Da
            # --------------------------------------------------
            if is_q1:
                asum1 = 0.0
                asum2 = 0.0
                asum12 = 0.0

                # Within sample a
                for ii in range(ka):
                    tax_i = present[a, ii]
                    pi = R[tax_i, a]
                    for jj in range(ka):
                        tax_j = present[a, jj]
                        pj = R[tax_j, a]
                        outer = 0.25 * pi * pj
                        dij = D[tax_i, tax_j]
                        if outer > 0.0 and np.isfinite(dij):
                            asum1 += outer * np.log(outer) * dij * inv_Q

                # Within sample b
                for ii in range(kb):
                    tax_i = present[b, ii]
                    pi = R[tax_i, b]
                    for jj in range(kb):
                        tax_j = present[b, jj]
                        pj = R[tax_j, b]
                        outer = 0.25 * pi * pj
                        dij = D[tax_i, tax_j]
                        if outer > 0.0 and np.isfinite(dij):
                            asum2 += outer * np.log(outer) * dij * inv_Q

                # Cross sample a x b
                for ii in range(ka):
                    tax_i = present[a, ii]
                    pi = R[tax_i, a]
                    for jj in range(kb):
                        tax_j = present[b, jj]
                        pj = R[tax_j, b]
                        outer = 0.25 * pi * pj
                        dij = D[tax_i, tax_j]
                        if outer > 0.0 and np.isfinite(dij):
                            asum12 += outer * np.log(outer) * dij * inv_Q

                Da = 0.5 * np.exp(-0.5 * (asum1 + asum2 + 2.0 * asum12))

            else:
                asum1 = 0.0
                asum2 = 0.0
                asum12 = 0.0

                # Within sample a
                for ii in range(ka):
                    tax_i = present[a, ii]
                    pi = R[tax_i, a]
                    for jj in range(ka):
                        tax_j = present[a, jj]
                        pj = R[tax_j, a]
                        outer = 0.25 * pi * pj
                        dij = D[tax_i, tax_j]
                        if outer > 0.0 and np.isfinite(dij):
                            asum1 += (outer ** q) * dij * inv_Q

                # Within sample b
                for ii in range(kb):
                    tax_i = present[b, ii]
                    pi = R[tax_i, b]
                    for jj in range(kb):
                        tax_j = present[b, jj]
                        pj = R[tax_j, b]
                        outer = 0.25 * pi * pj
                        dij = D[tax_i, tax_j]
                        if outer > 0.0 and np.isfinite(dij):
                            asum2 += (outer ** q) * dij * inv_Q

                # Cross sample a x b
                for ii in range(ka):
                    tax_i = present[a, ii]
                    pi = R[tax_i, a]
                    for jj in range(kb):
                        tax_j = present[b, jj]
                        pj = R[tax_j, b]
                        outer = 0.25 * pi * pj
                        dij = D[tax_i, tax_j]
                        if outer > 0.0 and np.isfinite(dij):
                            asum12 += (outer ** q) * dij * inv_Q

                aval = asum1 + asum2 + 2.0 * asum12

                if aval <= 0.0 or not np.isfinite(aval):
                    out[a, b] = np.nan
                    out[b, a] = np.nan
                    continue

                Da = 0.5 * (aval ** exponent)

            # --------------------------------------------------
            # Beta
            # --------------------------------------------------
            if Da > 0.0 and np.isfinite(Da) and np.isfinite(Dg):
                beta_val = Dg / Da
            else:
                beta_val = np.nan

            out[a, b] = beta_val
            out[b, a] = beta_val

    return out
