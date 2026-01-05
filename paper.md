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
date: 05 January 2026
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
Specifically, qdiv implements the Hill number framework [@Chao:2014] for assessing diversity across a broad range of ecological metrics and analyses. 
This framework enables systematic evaluation of how relative abundances influence diversity measures.

Traditionally, Hill numbers are applied to quantify alpha diversity, which is the diversity of taxa within a single community. 
However, the framework can also be applied to beta diversity, which measures differences in composition between communities, 
and extended to metrics that incorporate phylogenetic and functional relationships among taxa.

Many ecological statistical tests and null models, such as Raup-Crick [@Raup:1979], the Net Relatedness Index, and the Nearest-Taxon Index [@Webb2002], 
have used presence-absence data or relied on a single relative abundance-based index (typically Bray-Curtis). 
`qdiv` fills a critical gap by enabling the implementation of the Hill number framework for analysing diversity and microbial community assembly with a range of statistical methods.
It provides a streamlined workflow for data management, diversity analysis, and visualization -- all within a Python ecosystem.


# Software design

`qdiv` has modular architecture centered around the core MicrobiomeData class. 
This class serves as a container for all essential components, including relative abundance tables, 
taxonomic annotations, sequences, phylogenetic trees, and associated metadata. 
By loading these elements into a single object, qdiv enables streamlined data handling and analysis.
A comprehensive set of class methods supports convenient data management tasks such as merging datasets, 
subsetting samples, renaming features, and exporting files for downstream analyses. 
Functions for diversity calculations, statistical testing, and visualization are organized into dedicated subpackages that operate directly on MicrobiomeData instances.
Results can be returned as publication-ready figures or as structured data tables for further processing in other software environments. 
The package leverages the widely used Python libraries pandas, numpy, and matplotlib, to ensure efficiency, flexibility, and compatibility with existing data science workflows.


# Research impact statement

qdiv was first introduced by [@Modin2020] to enable Hill-based beta diversity calculations and extend the Raup-Crick null model to such metrics. 
Since then, it has been adopted by other researchers, e.g., [@Alberdi2021; @Nikolova2021], and used by our own research group, e.g., [@Abadikhah2024]. 
It has been downloaded over 1800 times from PyPI in the past six months. 
Version 4 introduces the MicrobiomeData class for streamlined data management, along with new statistical and visualization capabilities.


# AI usage disclosure

Microsoft Copilot was used to revise functions, to draft a new function for distance-based redundancy analysis, and to improve docstrings in version 4.0.0 of qdiv, 
as well as to refine the text of this manuscript.


# Acknowledgements

Research projects that have supported the continued development of qdiv were funded by the Swedish Research Council VR (2023-03908), 
the Swedish Research Council FORMAS (2018-00622, 2024-01814), and the NovoNordisk Foundation (NNF24OC0093678).

# References