from typing import Any, TYPE_CHECKING
import numpy as np
from scipy.stats import truncnorm

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli

def sample_kcat(kcat_val):
    a = 0
    b = np.inf
    kcat_loc = kcat_val
    kcat_scale = np.sqrt(kcat_val)
    a_transformed, b_transformed = (a - kcat_loc) / kcat_scale, (b - kcat_loc) / kcat_scale
    rv_kcat = truncnorm(a_transformed, b_transformed, loc=kcat_loc, scale=kcat_scale)
    kcat_new = rv_kcat.rvs()

    return kcat_new

def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Modify sim_data to replace kcat value for all reactions with kinetic constraints

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary of the following format::

            {
                # rxn_id: str,
                "count": int,
            }

    Returns:
        Simulation data with the following attributes modified::

            sim_data.process.metabolism
    """
    count = int(params["count"])

    kcat_rxn_all = sim_data.process.metabolism.kinetic_constraint_reactions

    for rxn_idx in range(len(kcat_rxn_all)):

        kcat_val = sim_data.process.metabolism._kcats[rxn_idx][1]
        kcat_new = sample_kcat(kcat_val)
        for k in range(3):

            sim_data.process.metabolism._kcats[rxn_idx][k] = kcat_new


    return sim_data
