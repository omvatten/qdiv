qdiv package
============

Overview
--------
The `qdiv` package provides tools for microbial diversity analysis built
around the Hill number framework. Functionality is organized into
specialized subpackages for diversity, statistics, plotting, modeling,
and sequence handling, unified by the `MicrobiomeData` container class.

Core subpackages
----------------
.. toctree::
   :maxdepth: 2

   qdiv.diversity
   qdiv.stats
   qdiv.plot
   qdiv.model
   qdiv.sequences

Main data container
-------------------
.. autoclass:: qdiv.MicrobiomeData
   :members:
   :undoc-members:
   :show-inheritance:
