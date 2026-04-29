from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Modify sim_data to set a fractional scale on kinetic targets for
    metabolism_redux_classic.

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary of the following format::

            {
                "fraction_kinetic_target": float
            }

    Returns:
        Simulation data with the following attributes modified::

            sim_data.process.metabolism.redux_classic_fraction_kinetic_target
    """
    metabolism = sim_data.process.metabolism

    metabolism.redux_classic_fraction_kinetic_target = params.get(
        "fraction_kinetic_target", metabolism.redux_classic_fraction_kinetic_target
    )

    return sim_data
