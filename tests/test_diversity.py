import numpy as np
import pandas as pd

def test_naive_alpha_monotonic(qb):
    import qdiv
    a0 = qdiv.diversity.naive_alpha(qb, q=0)
    a1 = qdiv.diversity.naive_alpha(qb, q=1)
    a2 = qdiv.diversity.naive_alpha(qb, q=2)
    # Effective numbers should be non-increasing with q
    assert (a0.values >= a1.values - 1e-9).all()
    assert (a1.values >= a2.values - 1e-9).all()

def test_naive_alpha_values(qtoy):
    import qdiv
    a0 = qdiv.diversity.naive_alpha(qtoy, q=0)
    a1 = qdiv.diversity.naive_alpha(qtoy, q=1)
    a2 = qdiv.diversity.naive_alpha(qtoy, q=2)

    # q=0
    assert a0.loc["S1"] == 1.0
    assert a0.loc["S2"] == 2.0
    assert a0.loc["S3"] == 4.0

    # q=1
    assert np.isclose(a1.loc["S1"], 1.0, atol=1e-9)
    assert np.isclose(a1.loc["S2"], 2.0, atol=1e-9)
    assert 2.56 < a1.loc["S3"] < 2.57

    # q=2
    assert np.isclose(a2.loc["S1"], 1.0, atol=1e-9)
    assert np.isclose(a2.loc["S2"], 2.0, atol=1e-9)
    assert 1.92 < a2.loc["S3"] < 1.93

def test_naive_beta_distance_properties(dis):
    # square, symmetric, non-negative, zero diagonal
    assert isinstance(dis, pd.DataFrame)
    n, m = dis.shape
    assert n == m
    assert np.allclose(dis.values, dis.values.T, atol=1e-12)
    assert (dis.values >= -1e-12).all()
    assert np.allclose(np.diag(dis.values), 0.0, atol=1e-12)

def test_naive_beta_values(qtoy):
    import qdiv
    b0 = qdiv.diversity.naive_beta(qtoy, q=0, viewpoint="regional")
    b1 = qdiv.diversity.naive_beta(qtoy, q=1, viewpoint="regional")
    b2 = qdiv.diversity.naive_beta(qtoy, q=2, viewpoint="regional")

    # q=0
    assert np.isclose(b0.loc["S1", "S3"], 0.75, atol=1e-9)

    # q=1
    assert 0.310 < b1.loc["S1", "S2"] < 0.315

    # q=2
    assert 0.767 < b2.loc["S1", "S3"] < 0.768



