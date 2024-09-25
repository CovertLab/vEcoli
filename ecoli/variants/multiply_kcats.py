from typing import Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Modify sim_data to multiply kcat value for a metabolic enzyme specified by user with id

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary of the following format::

            {
                # enzyme_id: str,
                "kcat_multiplier": float,
            }

    Returns:
        Simulation data with the following attributes modified::

            sim_data.process.metabolism
    """


    compiled_enzymes = eval("lambda e: {}".format(sim_data.process.metabolism._enzymes))
    kcats_labels = np.array(compiled_enzymes(sim_data.process.metabolism.kinetic_constraint_enzymes))

    enzyme_id = str(params["enzyme_id"])
    kcat_mpl = float(params["kcat_multiplier"])

    kcat_rxns_idxs = np.where(kcats_labels == enzyme_id)[0]

    for rxn_idx in kcat_rxns_idxs:
        sim_data.process.metabolism._kcats[rxn_idx] = sim_data.process.metabolism._kcats[rxn_idx] * kcat_mpl

    return sim_data
