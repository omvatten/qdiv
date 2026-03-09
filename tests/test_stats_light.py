
import numpy as np


def test_permanova_and_mantel_smoke(qb, dis):
    import qdiv
    # PERMANOVA should return result with a p-value field
    perma = qdiv.stats.permanova(dis, qb, by='feed', permutations=59, seed=11)
    # tolerate various return schemas: dict-like with 'p' or 'p-value'
    p = perma.get('p', perma.get('p-value', perma.get('pvalue', None)))
    assert p is not None and 0.0 <= float(p) <= 1.0

    # Mantel against itself should be near-perfect correlation
    mantel = qdiv.stats.mantel(dis, dis, permutations=59, seed=7)
    r = float(mantel.get('r', mantel.get('statistic', 0.0)))
    assert 0.9 <= r <= 1.0
