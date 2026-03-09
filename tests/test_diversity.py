
import numpy as np
import pandas as pd


def test_naive_alpha_monotonic(qb):
    import qdiv
    a0 = qdiv.diversity.naive_alpha(qb, q=0)
    a1 = qdiv.diversity.naive_alpha(qb, q=1)
    a2 = qdiv.diversity.naive_alpha(qb, q=2)
    # Effective numbers should be non-increasing with q
    # Allow tiny numerical jitter
    assert (a0.values >= a1.values - 1e-9).all()
    assert (a1.values >= a2.values - 1e-9).all()


def test_naive_beta_distance_properties(dis):
    # square, symmetric, non-negative, zero diagonal
    assert isinstance(dis, pd.DataFrame)
    n, m = dis.shape
    assert n == m
    assert np.allclose(dis.values, dis.values.T, atol=1e-12)
    assert (dis.values >= -1e-12).all()
    assert np.allclose(np.diag(dis.values), 0.0, atol=1e-12)
