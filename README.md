# qdiv: Microbial Diversity Analysis Toolkit

**qdiv** is a Python toolkit for microbial diversity analysis, built around the robust and intuitive Hill number framework.

---

## Features
- Data Management: Load and validate data from multiple formats (CSV/TSV, FASTA, Newick). Functions for subsetting, rarefaction, merging, renaming, tree pruning, and more.

- Diversity analysis: Compute multiple alpha and beta diversity metrics based on the Hill number framework.

- Statistical methods: Integrate diversity metrics using the Hill number framework into null models (Raup–Crick, NRI, NTI), ordinations (PCoA, db-RDA), PERMANOVA, and Mantel tests.

- Visualization: Heatmaps, ordinations, and diversity plots, and more. 

---

## Installation
```bash
pip install qdiv
```

## Optional acceleration (Numba)
Some functions in the sequence_comparisons module support compilation via Numba. Installing qdiv with the optional accelerate extra 
enables faster sequence distance computations. 

```bash
pip install qdiv[accelerate]
```

---

## Documentation
Full documentation is available at: [https://qdiv.readthedocs.io](https://qdiv.readthedocs.io)

---

## Licence
From version 4.0.0, qdiv is licensed under the https://opensource.org/licenses/BSD-3-Clause.
Previous versions were released under the GNU General Public License v3.0 (GPL-3.0).


## Contributing
Contributions are welcome!
Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for details on:

- How to report bugs
- How to request new features
- How to submit pull requests


## Support
If you need help with qdiv:

- Browse existing issues
- If needed, open a support issue. See [SUPPORT.md](SUPPORT.md).


## Code of Conduct
To ensure a welcoming environment, this project follows the standards outlined in the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). All contributors are expected to uphold this code.
