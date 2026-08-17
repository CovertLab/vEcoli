import numpy as np
from ecoli.processes.registries import topology_registry
from vivarium.core.process import Step
from vivarium.library.units import units

NAME = "oxygen-ramp"
TOPOLOGY = {
    "boundary": ("boundary",),
    "global_time": ("global_time",),
}
topology_registry.register(NAME, TOPOLOGY)


class OxygenRamp(Step):
    """
    Continuously depletes (or restores) environmental oxygen concentration
    as an explicit function of simulation time, instead of the discrete,
    one-shot media-switch jump used by :py:class:`~ecoli.processes.environment.media_update.MediaUpdate`.

    Only meant to be added to a composite alongside the ``oxygen_depletion``
    variant (:py:mod:`ecoli.variants.oxygen_depletion`), which populates
    ``sim_data.external_state.oxygen_ramp`` with the parameters below. If
    added without that variant, defaults to holding oxygen at 0 mM.
    """

    name = NAME
    topology = TOPOLOGY
    defaults = {
        "time_step": 1,
        # NOTE: boundary.external is keyed by *untagged* molecule ids (as in
        # MediaUpdate's saved_media / ExchangeData's environment_molecules) --
        # not the location-tagged exchange id used internally by
        # exchange_data_from_concentrations. Using a tagged id here would
        # silently write to an orphaned store entry nothing else reads.
        "oxygen_id": "OXYGEN-MOLECULE",
        "start_time": 0.0,
        "end_time": 0.0,
        "initial_o2_conc": 0.0,
        "final_o2_conc": 0.0,
        # "linear" or "exponential". Exponential (first-order depletion,
        # d[O2]/dt = -k[O2]) is the more biologically natural default for a
        # well-mixed culture consuming a resource proportional to its own
        # concentration; linear is kept available as a simpler alternative.
        "shape": "linear",
        "tau": 1.0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        if self.parameters["end_time"] < self.parameters["start_time"]:
            raise ValueError("oxygen-ramp end_time must be >= start_time")
        if self.parameters["shape"] not in ("linear", "exponential"):
            raise ValueError(f"Unknown oxygen-ramp shape: {self.parameters['shape']!r}")
        self.oxygen_id = self.parameters["oxygen_id"]

    def ports_schema(self):
        return {
            "boundary": {"external": {self.oxygen_id: {"_default": 0 * units.mM}}},
            "global_time": {"_default": 0.0},
        }

    def next_update(self, timestep, states):
        target_conc = self._ramp_concentration(states["global_time"])
        # Set the absolute value directly (rather than computing a delta to
        # accumulate) since the pre-ramp aerobic concentration may be
        # Infinity, and Infinity arithmetic (inf + -inf = nan) makes a
        # diff-based update unsafe here.
        return {
            "boundary": {
                "external": {self.oxygen_id: {"_value": target_conc, "_updater": "set"}}
            }
        }

    def _ramp_concentration(self, time):
        start_time = self.parameters["start_time"]
        end_time = self.parameters["end_time"]
        initial_conc = self.parameters["initial_o2_conc"]
        final_conc = self.parameters["final_o2_conc"]

        if time <= start_time:
            fraction = 0.0
        elif time >= end_time:
            fraction = 1.0
        elif self.parameters["shape"] == "linear":
            fraction = (time - start_time) / (end_time - start_time)
        else:
            tau = self.parameters["tau"]
            fraction = 1 - np.exp(-(time - start_time) / tau)
            fraction = min(fraction, 1.0)

        return (initial_conc + fraction * (final_conc - initial_conc)) * units.mM
