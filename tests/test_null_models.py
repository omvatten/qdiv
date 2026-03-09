import numpy as np
import pandas as pd
import qdiv

def test_rcq_runs_and_shapes(qb):
    """
    Smoke + structure test for the Hill-adapted Raup–Crick null model (RC_q).
    Uses a tiny iteration count so CI remains fast.
    """
    # run the rcq model with few iterations
    out = qdiv.model.rcq(qb, q=1, iterations=12, random_state=7)

    # out should be a Dict
    assert isinstance(out, dict)

    # output must index samples
    assert list(out["p"].index) == list(qb.meta.index)

    # SES column must be present
    assert "ses" in out.keys()


def test_null_models_repeatability(qb):
    """
    Random-seed regression test:
    Same seed => same output.
    Different seed => different output often.
    """
    out1 = qdiv.model.rcq(qb, q=1, iterations=12, random_state=5)
    out2 = qdiv.model.rcq(qb, q=1, iterations=12, random_state=5)

    # Exactly equal because same seed
    assert out1["ses"].equals(out2["ses"])

    # Different seed should usually produce differences
    out3 = qdiv.model.rcq(qb, q=1, iterations=12, random_state=99)
    assert not out1["ses"].equals(out3["ses"])
