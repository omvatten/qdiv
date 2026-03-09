def test_permanova_and_mantel_smoke(qb, dis):
    import qdiv

    perma = qdiv.stats.permanova(dis, qb, by='feed', permutations=59, seed=11)

    # Basic structure
    assert isinstance(perma, dict)
    assert 'by' in perma and 'table' in perma
    table = perma['table']
    terms = perma['by']
    assert len(terms) >= 1
    for term in terms:
        # p can be NaN in valid edge cases; only assert bounds if present
        p = table.loc[term, 'p']
        if p == p:  # not NaN
            assert 0.0 <= float(p) <= 1.0

    # Mantel against itself should be near-perfect correlation
    mantel = qdiv.stats.mantel(dis, dis, permutations=59, random_state=7)
    r = float(mantel[0]) if isinstance(mantel, list) else float(mantel.get('r', mantel.get('statistic', 0.0)))
    assert 0.9 <= r <= 1.0