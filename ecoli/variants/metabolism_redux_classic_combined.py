from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Modify sim_data to set alternate objective weights AND a fractional
    scale on kinetic targets for metabolism_redux_classic. Combines the
    behavior of metabolism_redux_classic_objective_weights and
    metabolism_redux_classic_fraction_kinetic_target so that a Cartesian
    product of the two parameter spaces can be swept in a single config.

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary of the following format::

            {
                "weights": {
                    "secretion": float,
                    "efficiency": float,
                    "kinetics": float,
                    "diversity": float,
                    "homeostatic": float
                },
                "fraction_kinetic_target": float
            }

        Either key may be omitted; the corresponding sim_data attribute(s)
        will then retain their default values.

    Returns:
        Simulation data with the following attributes modified::

            sim_data.process.metabolism.redux_classic_secretion_weight
            sim_data.process.metabolism.redux_classic_efficiency_weight
            sim_data.process.metabolism.redux_classic_kinetics_weight
            sim_data.process.metabolism.redux_classic_diversity_weight
            sim_data.process.metabolism.redux_classic_homeostatic_weight
            sim_data.process.metabolism.redux_classic_fraction_kinetic_target
    """
    metabolism = sim_data.process.metabolism

    weights = params.get("weights", {})
    metabolism.redux_classic_secretion_weight = weights.get(
        "secretion", metabolism.redux_classic_secretion_weight
    )
    metabolism.redux_classic_efficiency_weight = weights.get(
        "efficiency", metabolism.redux_classic_efficiency_weight
    )
    metabolism.redux_classic_kinetics_weight = weights.get(
        "kinetics", metabolism.redux_classic_kinetics_weight
    )
    metabolism.redux_classic_diversity_weight = weights.get(
        "diversity", metabolism.redux_classic_diversity_weight
    )
    metabolism.redux_classic_homeostatic_weight = weights.get(
        "homeostatic", metabolism.redux_classic_homeostatic_weight
    )

    metabolism.redux_classic_fraction_kinetic_target = params.get(
        "fraction_kinetic_target",
        metabolism.redux_classic_fraction_kinetic_target,
    )

    return sim_data
