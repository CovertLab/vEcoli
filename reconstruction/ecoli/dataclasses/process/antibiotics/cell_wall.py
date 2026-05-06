"""
Cell-wall reference data carried through to ``sim_data.process.antibiotics.cell_wall``.

Exposes the murein peptidoglycan strand-length distribution (Obermann
& Höltje 1994) and the fitted strand-termination probability
(``strand_term_p``) used by ``ecoli/processes/antibiotics/cell_wall.py``
and ``ecoli/processes/antibiotics/pbp_binding.py``. The fit runs at
ParCa time so that downstream runtime processes receive
``strand_term_p`` from sim_data through the standard
``LoadSimData.get_<process>_config`` pathway, rather than reaching
back into ``ecoli.library.parameters.param_store`` (which itself
loaded the source CSV at module-import time before this migration).
"""

import pandas as pd

from ecoli.library.cell_wall.column_sampler import fit_strand_term_p


# Mean strand length of the >30 bin used by ``fit_strand_term_p``.
# Literature constant from Vollmer, W., Blanot, D., & De Pedro, M. A. (2008);
# also kept in ``ecoli.library.parameters`` for the other cell_wall consumers.
_UPPER_MEAN = 45


class CellWall(object):
    def __init__(self, raw_data, sim_data):
        # List of {"Strain": str, "Length": str, "Percent": str} rows from
        # ``_load_csv``; consumers convert to a DataFrame as needed.
        # Source: Obermann, W., & Höltje, J. (1994).
        self.strand_length_distribution = list(
            raw_data.cell_wall.murein_strand_length_distribution
        )

        # Fit the strand-termination probability from the distribution.
        # Computed once here so runtime processes can pull it off sim_data.
        df = pd.DataFrame(self.strand_length_distribution)
        df["Percent"] = df["Percent"].astype(float)
        self.strand_term_p = fit_strand_term_p(df, _UPPER_MEAN)
