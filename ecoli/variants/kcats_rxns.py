from typing import Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Modify sim_data to specify kcat value for a given rxn id

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary of the following format::

            {
                # rxn_id: str,
                "kcat_val": float,
            }

    Returns:
        Simulation data with the following attributes modified::

            sim_data.process.metabolism
    """


    rxn_id = str(params["rxn_id"])
    kcat_val = float(params["kcat_val"])

    kcat_rxn_idx = list(sim_data.process.metabolism.kinetic_constraint_reactions).index(rxn_id)

    for k in range(3):

        sim_data.process.metabolism._kcats[kcat_rxn_idx][k] = kcat_val


    return sim_data
