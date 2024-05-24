from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, list[Any]]
) -> "SimulationDataEcoli":
    """
    All variants must define an ``apply_variant`` function that takes
    the same two arguments. Copy this file when creating your own variant.

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary of the following format::

            {
                # Document types of input parameters here
                "param_1": int,
                "param_2": float,
            }

    Returns:
        Simulation data with the following attributes modified::

            # Document sim_data changes here 
            param_1: Set to params["param_1"]
            param_2: Set to params["param_2"]

    """
    sim_data.param_1 = params["param_1"]
    sim_data.param_2 = params["param_2"]
    return sim_data
