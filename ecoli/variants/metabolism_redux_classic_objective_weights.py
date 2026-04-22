from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Modify sim_data to set alternate objective weights for
    metabolim_redux_classic.

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary of the following format::

            {
                "secretion": float,
                "efficiency": float,
                "kinetics": float,
                "diversity": float,
                "homeostatic": float
            }

    Returns:
        Simulation data with the following attributes modified::

            sim_data.process.metabolism.redux_classic_objective_weights
    """
    # Load the objective weights into sim_data
    print("Applying variant objective weights")
    sim_data.process.metabolism.redux_classic_objective_weights = {
        "secretion": params["secretion"],
        "efficiency": params["efficiency"],
        "kinetics": params["kinetics"],
        "diversity": params["diversity"],
        "homeostatic": params["homeostatic"],
    }

    return sim_data
