
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def test_pcoa_lingoes_on_distance_readonly():
    import qdiv
    vals = np.array([[0.0, 0.3, 0.4],
                     [0.3, 0.0, 0.5],
                     [0.4, 0.5, 0.0]], dtype=float)
    dis = pd.DataFrame(vals, index=['s1','s2','s3'], columns=['s1','s2','s3'])
    # Run PCoA
    res = qdiv.stats.pcoa_lingoes(dis)
    # Basic structure checks
    assert 'site_scores' in res and 'eigenvalues' in res
    U = res['site_scores']
    lam = res['eigenvalues']
    assert isinstance(U, pd.DataFrame)
    assert len(lam) == U.shape[1]
    # Eigenvalues should be non-negative (Lingoes correction)
    assert (np.asarray(lam) >= -1e-10).all()


def test_dbrda_basic(qb, dis):
    import qdiv
    # Use a single factor known to exist in the example metadata
    res = qdiv.stats.dbrda(dis, qb, by='feed', perm_n=59, perm_seed=7)
    # Required keys
    for k in ['site_scores','biplot_scores','eigenvalues','explained_ratio',
              'total_inertia','constrained_inertia','unconstrained_inertia',
              'F_global','p_global']:
        assert k in res
    # Shapes and ranges
    U = res['site_scores']
    lam = res['eigenvalues']
    ratio = res['explained_ratio']
    assert U.shape[0] == dis.shape[0]
    assert len(lam) == len(ratio)
    assert float(np.sum(ratio)) <= 1.0 + 1e-8


def test_plot_ordination_accepts_distance_and_dict(qb, dis):
    import qdiv
    # A) Distance matrix path (PCoA computed internally)
    fig1, ax1, _, _ = qdiv.plot.ordination(dis, qb, color_by='feed', show_legend=False)
    plt.close(fig1)

    # B) Pre-computed dict path (db-RDA)
    rda = qdiv.stats.dbrda(dis, qb, by='feed', perm_n=19, perm_seed=1)
    fig2, ax2, _, _ = qdiv.plot.ordination(rda, qb, color_by='feed', show_legend=False)
    plt.close(fig2)
