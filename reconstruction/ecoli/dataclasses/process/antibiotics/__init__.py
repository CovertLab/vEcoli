"""
Reference data attached to ``sim_data.process.antibiotics.*`` for
conditional antibiotic-related simulation processes (see
``ecoli/processes/antibiotics/``).

Unlike the sibling modules in ``reconstruction/ecoli/dataclasses/process/``,
which back universal whole-cell processes that ParCa fits, modules under
this namespace carry **static reference data** that conditional processes
need at simulation time. ParCa does not fit anything against this data;
it just threads it through ``raw_data`` onto ``sim_data`` so downstream
code (cell-wall lattice plots, MIC analyses, etc.) gets it from the
single bundle pinned at ParCa time rather than re-reaching into flat
files.
"""

from .cell_wall import CellWall

__all__ = ["CellWall"]
