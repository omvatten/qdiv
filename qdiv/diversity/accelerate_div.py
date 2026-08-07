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