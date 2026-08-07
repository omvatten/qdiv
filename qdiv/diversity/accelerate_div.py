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