---
title: 'qdiv: A Python package for microbial ecology analysis using the Hill number framework'
tags:
  - Python
  - microbial ecology
  - diversity
authors:
  - name: Oskar Modin
    orcid: 0000-0002-9232-6096
    affiliation: 1
affiliations:
 - name: Chalmers University of Technology, Department of Architecture and Civil Engineering, Division of Water Environment Technology, Gothenburg, Sweden
   index: 1
date: 15 March 2026
bibliography: paper.bib
---

# Summary

Microorganisms are ubiquitous and play essential roles in biogeochemical cycles, human health, and engineered systems. 
Microbial ecology research seeks to understand the structure and function of microbial communities. 
Common approaches, such as amplicon sequencing and metagenomics, 
generate tabular data that track the relative abundances of hundreds or thousands of species or genes across time or space. 
To explore how communities assemble and respond to environmental factors, microbial diversity is quantified using various metrics, 
some incorporating phylogenetic and functional relationships.

The Hill number framework, also known as effective numbers, provides a unified and intuitive methodology for assessing diversity and changes in community composition. 
However, few tools systematically implement this framework across alpha and beta diversity, phylogenetic metrics, multivariate statistics, and null models. 
The Python package `qdiv` fills this gap by applying the Hill number framework to a broad range of ecological metrics, 
offering a streamlined and versatile tool for investigating patterns of community assembly.


# Statement of need

`qdiv` is a Python package designed for microbial ecology. 
It was developed to facilitate the management and analysis of data generated from amplicon sequencing and metagenomics within a Python programming environment. 
Specifically, `qdiv` implements the Hill number framework [@chao2014] for assessing diversity across a broad range of ecological metrics and analyses. 
This framework enables systematic evaluation of how relative abundances influence diversity measures.

Traditionally, Hill numbers [@hill1973] are applied to quantify alpha diversity, which is the diversity of taxa within a single community. 
However, the framework can also be applied to beta diversity [@jost2007], which measures differences in composition between communities, 
and extended to metrics that incorporate phylogenetic and functional relationships among taxa [@chiu2014; @chiuchao2014].

Many ecological statistical tests and null models, such as Raup-Crick [@raup1979], the Net Relatedness Index, and the Nearest-Taxon Index [@webb2002], 
have used presence-absence data or relied on a single relative abundance-based index (e.g. Bray-Curtis). 
`qdiv` fills a critical gap by enabling the implementation of the Hill number framework for analysing diversity and microbial community assembly with a range of statistical methods, all within Python.


# Statement of the field

A variety of software tools support microbial community analysis, particularly within the R programming environment.
Packages such as vegan [vegan2026], phyloseq [@mcmurdie2013], microbiome [@lahti2017], and ampvis2 [@andersen2018] provide comprehensive workflows 
for diversity calculations, ordination, and phylogenetic analysis, but they primarily rely on classical indices rather than a unified diversity framework. 
Several R packages implement the Hill number framework, including hilldiv [@alberdi2019], hillR [@li2018], vegetarian [@vegetarian2019], 
and iNEXT [@inext2024]. However, to our knowledge there is no Python package that implements the Hill number framework in a unified way 
for diversity metrics and statistical workflows. The `qdiv` package fills this gap by offering a systematic Python implementation of 
Hill‑based alpha and beta diversity, phylogenetic diversity, and associated statistical methods.

Other commonly used tools include QIIME [@bolyen2019], an end‑to‑end microbiome analysis platform typically 
used through a command‑line interface, which limits flexible, script‑based workflows in Python.
The Python library scikit‑bio [@aton2025] provides data structures and algorithms for omics analyses, 
including sequence handling, phylogenetic tree operations, and ecological statistics, but it does not center around, or generalize, the Hill number framework.


# Software design

`qdiv` has modular architecture centered around the core MicrobiomeData class. 
This class serves as a container for all essential components, including relative abundance tables, 
taxonomic annotations, sequences, phylogenetic trees, and associated metadata.
The motivation for this design is to provide users with a single, coherent object that keeps data, metadata, and analysis context together, 
reducing the risk of mismatched inputs.

A key design decision concerns how much functionality should reside inside the MicrobiomeData class versus in separate subpackages. 
Methods related to data wrangling, such as subsetting, merging, and saving or loading data, are implemented as class methods 
because they directly modify or manage the internal state of the dataset. 
Keeping these operations inside the class provides users with a simple interface.

In contrast, more specialized analyses such as diversity calculations, statistical tests, and visualizations, 
are implemented in dedicated subpackages that operate on MicrobiomeData instances but do not alter them. 
This separation follows the principle of keeping data management and analytical logic independent. 

Another important trade‑off relates to how much general-purpose data manipulation the package should include. 
The primary goal of qdiv is to implement the Hill number framework and extend it to statistical methods 
for which it has not previously been applied, such as null models.
For convenience, qdiv includes a limited set of helper functions for tasks such as simple phylogenetic tree manipulations. 
However, for advanced tree operations, users are encouraged to rely on specialized libraries such as ete3 [@huerta2016].

The `qdiv` package leverages the widely used Python libraries pandas, numpy, and matplotlib, 
to ensure efficiency, interoperability, and familiarity for users already working in the Python data‑science ecosystem. 
A deliberate effort was made to keep dependencies to a minimum, to reduce installation friction and decrease the risk of version conflicts.


# Research impact statement

The `qdiv` package was first introduced by [@modin2020] to enable Hill-based beta diversity calculations and extend the Raup-Crick null model to such metrics. 
Since then, it has been adopted by other researchers, e.g., [@alberdi2021; @nikolova2021], and used by our own research group, e.g., [@abadikhah2024]. 
It has been downloaded 835 times from PyPI (excluding mirrors) from 2025-09-13 to 2026-03-12. 
Version 4 introduces the MicrobiomeData class for streamlined data management, along with new statistical and visualization capabilities.


# AI usage disclosure

Microsoft 365 Copilot powered by the GPT-5 chat model was used during the development of `qdiv` version 4.0.0 and higher. 
Copilot was used to revise existing functions, draft a new function for distance-based redundancy analysis, and revise sections of this manuscript. 
All design decisions, code implementations, and the content of the manuscript remain the sole responsibility of the author.


# Acknowledgements

Research projects that have supported the continued development of `qdiv` were funded by the Swedish Research Council VR (2023-03908), 
the Swedish Research Council FORMAS (2018-00622, 2024-01814), and the NovoNordisk Foundation (NNF24OC0093678).


# References