from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Uniformly scale the homeostatic metabolite target concentrations by a
    constant factor. A scale of 1.0 is a no-op; 0.90 means every target is
    multiplied by 0.9 before metabolism_redux builds its homeostatic objective.

    Args:
        sim_data: Simulation data to modify
        params: {"scale": float}

    Returns:
        sim_data with ``sim_data.homeostatic_target_scale`` set.
    """
    sim_data.homeostatic_target_scale = float(params["scale"])  # type: ignore[attr-defined]
    return sim_data
