from typing import Any, TYPE_CHECKING

from wholecell.utils import units
from ecoli.variants.condition import apply_variant as condition_variant

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Constrain the uptake rate of arbitrary environmental molecules and
    optionally set the environmental condition.

    Accepts two forms of uptake specification (can be combined; explicit
    ``uptake_constraints`` entries take precedence over flat params):

    **Form 1 - explicit molecule to rate dict:**

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary of the following format::

            {
                # Required (Form 1): mapping from environment molecule ID (no
                # location tag, e.g. "GLC") to forced uptake rate in
                # mmol/gDW/hr. Also accepts exchange IDs (e.g. "GLC[p]").
                # A rate of 0 forces zero uptake.
                "uptake_constraints": dict[str, float],

                # Optional: environmental condition (e.g. "basal", "with_aa",
                # "acetate", "succinate", "no_oxygen").
                "condition": str,
            }

    **Form 2 - flat rate params with automatic carbon source lookup:**

        Requires "condition" to be set so the correct carbon source molecule
        can be resolved from the media for that condition.

        params::

            {
                "condition": str,

                # Forced uptake rate for the condition's carbon source
                # molecule, in mmol/gDW/hr.
                "carbon_source_rate": float,

                # Forced uptake rate for O2 (OXYGEN-MOLECULE[p]),
                # in mmol/gDW/hr.
                "o2_rate": float,

                # Optional explicit overrides merged on top of the above;
                # takes precedence over carbon_source_rate / o2_rate for
                # any overlapping molecules.
                "uptake_constraints": dict[str, float],
            }

    Returns:
        Simulation data with the following attributes modified::

            sim_data.external_state.custom_uptake_constraints
            sim_data.condition                          (only if "condition" given)
            sim_data.external_state.current_timeline_id (only if "condition" given)
    """
    if "condition" in params:
        sim_data = condition_variant(sim_data, {"condition": params["condition"]})

    exchange_state = sim_data.external_state
    custom_constraints: dict = {}

    # Form 2: flat carbon_source_rate -- resolve the carbon source molecule for
    # the active condition and apply the forced rate.
    if "carbon_source_rate" in params:
        condition = params.get("condition", sim_data.condition)
        nutrients_label = sim_data.conditions[condition]["nutrients"]
        media = exchange_state.saved_media[nutrients_label]
        for cs_exchange_id in exchange_state.carbon_sources:
            cs_env_id = exchange_state.exchange_to_env_map.get(cs_exchange_id)
            if cs_env_id and cs_env_id in media:
                custom_constraints[cs_exchange_id] = params["carbon_source_rate"] * (
                    units.mmol / units.g / units.h
                )

    # Form 2: flat o2_rate -- apply forced O2 uptake rate.
    if "o2_rate" in params:
        custom_constraints["OXYGEN-MOLECULE[p]"] = params["o2_rate"] * (
            units.mmol / units.g / units.h
        )

    # Form 1: explicit uptake_constraints dict (merged last, takes precedence
    # over carbon_source_rate / o2_rate for any overlapping molecules).
    for mol_id, rate in params.get("uptake_constraints", {}).items():
        # Accept both environment IDs (no tag) and exchange IDs (with tag).
        if mol_id in exchange_state.env_to_exchange_map:
            exchange_id = exchange_state.env_to_exchange_map[mol_id]
        else:
            exchange_id = mol_id
        custom_constraints[exchange_id] = rate * (units.mmol / units.g / units.h)

    exchange_state.custom_uptake_constraints = custom_constraints
    return sim_data
