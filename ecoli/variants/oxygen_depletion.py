import math
from typing import Any, TYPE_CHECKING

from ecoli.variants.condition import apply_variant as condition_variant

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli

_ALLOWED_SHAPES = ("linear", "exponential")
_ALLOWED_OXYGEN_SCALING_MODES = ("binary", "continuous")


def _compute_oxygen_uptake_k_half(ramp: dict[str, Any]) -> float:
    """
    Half-saturation O2 concentration for the continuous carbon-uptake Hill
    interpolation (used by :py:meth:`~reconstruction.ecoli.dataclasses.state.external_state.ExternalState._continuous_oxygen_carbon_bound`
    when ``OXYGEN_SCALING_MODE == "continuous"``), set so the Hill
    transition's midpoint falls at this ramp's *temporal* midpoint instead
    of at whatever fixed default happened to be tuned for a different ramp.
    """
    start_time = ramp["start_time"]
    end_time = ramp["end_time"]
    initial_o2_conc = ramp["initial_o2_conc"]
    final_o2_conc = ramp["final_o2_conc"]
    shape = ramp.get("shape", "linear")

    if shape == "exponential":
        tau = ramp.get("tau", 1.0)
        fraction_at_midpoint = 1 - math.exp(-(end_time - start_time) / (2 * tau))
    else:
        fraction_at_midpoint = 0.5

    return initial_o2_conc + fraction_at_midpoint * (final_o2_conc - initial_o2_conc)


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Sets a base media condition (as in :py:mod:`ecoli.variants.condition`)
    and configures a continuous oxygen depletion ramp to be applied at
    runtime by :py:class:`~ecoli.processes.environment.oxygen_ramp.OxygenRamp`.
    That process must be added to the composite separately (e.g. via
    ``"add_processes": ["oxygen-ramp"]``) for the ramp to take effect.

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary of the following format::

            {
                # Base environmental condition to start from, e.g. "basal"
                "condition": str,
                "ramp": {
                    # Simulation time (s) at which the ramp begins/ends
                    "start_time": float,
                    "end_time": float,
                    # Oxygen concentration (mM) at start_time/end_time
                    "initial_o2_conc": float,
                    "final_o2_conc": float,
                    # "linear" or "exponential" (default "linear")
                    "shape": Optional(str),
                    # Decay time constant (s), only used if shape == "exponential"
                    "tau": Optional(float),
                },
                # How the FBA carbon-uptake bound responds to O2 concentration:
                # "binary" (default, matches pre-variant behavior) or
                # "continuous". See OXYGEN_SCALING_MODE_DEFAULT in
                # reconstruction/ecoli/dataclasses/state/external_state.py.
                "oxygen_scaling_mode": Optional(str),
            }

    Returns:
        Simulation data with the following attributes modified::

            sim_data.condition
            sim_data.external_state.current_timeline_id
            sim_data.external_state.oxygen_ramp
            sim_data.external_state.oxygen_scaling_mode
            sim_data.external_state.oxygen_uptake_k_half
    """
    sim_data = condition_variant(sim_data, params)

    ramp = params["ramp"]
    if ramp["end_time"] < ramp["start_time"]:
        raise ValueError("oxygen_depletion variant: end_time must be >= start_time")
    if ramp["initial_o2_conc"] < 0 or ramp["final_o2_conc"] < 0:
        raise ValueError(
            "oxygen_depletion variant: O2 concentrations must be non-negative"
        )
    shape = ramp.get("shape", "linear")
    if shape not in _ALLOWED_SHAPES:
        raise ValueError(f"oxygen_depletion variant: unknown ramp shape {shape!r}")

    oxygen_scaling_mode = params.get("oxygen_scaling_mode", "binary")
    if oxygen_scaling_mode not in _ALLOWED_OXYGEN_SCALING_MODES:
        raise ValueError(
            "oxygen_depletion variant: unknown oxygen_scaling_mode "
            f"{oxygen_scaling_mode!r}"
        )

    sim_data.external_state.oxygen_ramp = dict(ramp)  # type: ignore[attr-defined]
    sim_data.external_state.oxygen_scaling_mode = oxygen_scaling_mode  # type: ignore[attr-defined]
    sim_data.external_state.oxygen_uptake_k_half = _compute_oxygen_uptake_k_half(  # type: ignore[attr-defined]
        ramp
    )

    return sim_data
