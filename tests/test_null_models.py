import numpy as np
import pandas as pd
import qdiv

def test_rcq_runs_and_shapes(qb):
    """
    Smoke + structure test for the Hill-adapted Raup–Crick null model (RC_q).
    Uses a tiny iteration count so CI remains fast.
    """
    dis = qdiv.diversity.naive_beta(qb, q=1)

    # run the rcq model with few iterations
    out = qdiv.model.rcq(dis, qb.meta, by="feed", q=1, n_iter=12, seed=7)

    # out should be a DataFrame
    assert isinstance(out, pd.DataFrame)

    # output must index samples
    assert list(out.index) == list(qb.meta.index)

    # SES column must be present
    assert "ses" in out.columns

    # ses values must be numeric or NaN
    ses = out["ses"].astype(float)
    assert np.all(np.isfinite(ses) | np.isnan(ses))

def test_nriq_runs_and_shapes(qb):
    """
    Smoke + structure test for NRI_q (Net Relatedness Index with Hill weights).
    """
    # NRI uses phylogeny; ensure obj has a tree (example datasets do)
    out = qdiv.model.nriq(qb, q=1, n_iter=12, seed=11)

    # output format: DataFrame indexed by sample
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == list(qb.meta.index)

    # should contain at least the SES (standardized effect size)
    assert "ses" in out.columns
    ses = out["ses"].astype(float)
    assert np.all(np.isfinite(ses) | np.isnan(ses))

def test_null_models_repeatability(qb):
    """
    Random-seed regression test:
    Same seed => same output.
    Different seed => different output often.
    """
    out1 = qdiv.model.rcq(qb, qb.meta, by="feed", q=1, n_iter=12, seed=5)
    out2 = qdiv.model.rcq(qb, qb.meta, by="feed", q=1, n_iter=12, seed=5)

    # Exactly equal because same seed
    assert out1.equals(out2)

    # Different seed should usually produce differences
    out3 = qdiv.model.rcq(qb, qb.meta, by="feed", q=1, n_iter=12, seed=99)
    assert not out1.equals(out3)
