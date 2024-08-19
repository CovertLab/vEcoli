from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Modify sim_data to environmental condition from condition_defs.tsv.

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary of the following format::

            {
                # Environmental condition: "basal", "with_aa", "acetate",
                # "succinate", "no_oxygen"
                "condition": str,
            }

    Returns:
        Simulation data with the following attributes modified::

            sim_data.condition
            sim_data.external_state.current_timeline_id
    """
    # Set media condition
    sim_data.condition = params["condition"]
    sim_data.external_state.current_timeline_id = params["condition"]
    sim_data.external_state.saved_timelines[params["condition"]] = [
        (0, sim_data.conditions[params["condition"]]["nutrients"])
    ]

    return sim_data
