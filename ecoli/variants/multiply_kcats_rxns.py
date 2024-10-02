from typing import Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Modify sim_data to multiply kcat value for a given rxn id

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary of the following format::

            {
                # rxn_id: str,
                "kcat_multiplier": float,
            }

    Returns:
        Simulation data with the following attributes modified::

            sim_data.process.metabolism
    """


    rxn_id = str(params["rxn_id"])
    kcat_mpl = float(params["kcat_multiplier"])

    kcat_rxn_idx = list(sim_data.process.metabolism.kinetic_constraint_reactions).index(rxn_id)

    sim_data.process.metabolism._kcats[kcat_rxn_idx] = sim_data.process.metabolism._kcats[kcat_rxn_idx] * kcat_mpl


    return sim_data
