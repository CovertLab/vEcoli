import numpy as np
from vivarium.library.units import units
from vivarium.core.process import Step
from vivarium.core.engine import Engine
from vivarium.plots.simulation_output import plot_variables
from vivarium.processes.timeline import TimelineProcess
from vivarium.core.emitter import timeseries_from_data

from ecoli.library.lattice_utils import AVOGADRO
from ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts

CONV_UNITS = 1 / units.mM


class ConcToCounts(Step):
    """Convert concentrations to counts (requires Pint units)"""

    name = "conc_to_counts"
    defaults = {
        "molecules_to_convert": ["antibiotic"],
        "initial_state": {
            "antibiotic": {"conc": 0 * units.mM, "volume": 0 * units.fL},
            "counts": {"antibiotic": 0},
        },
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.molecule_idx = None

    def ports_schema(self):
        schema = {
            molecule: {
                "conc": {"_default": 0.0 * units.mM, "_divider": "set", "_emit": True},
                "volume": {
                    "_default": 0 * units.fL,
                    "_divider": "split",
                },
            }
            for molecule in self.parameters["molecules_to_convert"]
        }
        schema["bulk"] = numpy_schema("bulk")
        return schema

    def initial_state(self, config=None):
        config = config or {}
        parameters = self.parameters
        parameters.update(config)

        initial_state = {
            molecule: {"conc": 0 * units.mM, "volume": 0 * units.fL}
            for molecule in self.parameters["molecules_to_convert"]
        }
        # No deep_merge: keys in `parameters` that are not in
        # `initial_state` are ignored
        for port, port_state in initial_state.items():
            for (
                variable,
                value,
            ) in port_state.items():
                port_state[variable] = (
                    parameters["initial_state"].get(port, {}).get(variable, value)
                )
        return initial_state

    def next_update(self, timestep, states):
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.parameters["molecules_to_convert"], states["bulk"]["id"]
            )
        update = {"bulk": []}
        for mol_idx, molecule in zip(
            self.molecule_idx, self.parameters["molecules_to_convert"]
        ):
            conc = states[molecule]["conc"]
            volume = states[molecule]["volume"]
            molar_to_counts = (AVOGADRO * volume).to(1 / conc.units)
            new_count = int((conc * molar_to_counts).magnitude)
            old_count = counts(states["bulk"], mol_idx)
            update["bulk"].append((mol_idx, new_count - old_count))

        return update


def main():
    sim_time = 10

    initial_state = {}
    initial_state["boundary"] = {"internal": {"antibiotic": 0 * units.mM}}
    initial_state["listeners"] = {"mass": {"volume": 1.352695 * units.fL}}
    initial_state["bulk"] = np.array(
        [("antibiotic", 0)], dtype=[("id", "U40"), ("count", int)]
    )

    conv_process = ConcToCounts()

    timeline = []
    for i in range(5):
        timeline.append(
            (
                i * 2,
                {
                    ("antibiotic",): initial_state["boundary"]["internal"]["antibiotic"]
                    + (i + 1) * units.uM,
                },
            )
        )
    timeline_params = {
        "time_step": 2.0,
        "timeline": timeline,
    }
    timeline_process = TimelineProcess(timeline_params)

    sim = Engine(
        processes={"conc_to_counts": conv_process, "timeline": timeline_process},
        topology={
            "conc_to_counts": {
                "antibiotic": {
                    "conc": ("boundary", "internal", "antibiotic"),
                    "volume": ("listeners", "mass", "volume"),
                },
                "bulk": ("bulk",),
            },
            "timeline": {
                "global": ("global",),  # The global time is read here
                # This port is based on the declared timeline
                "antibiotic": ("boundary", "internal", "antibiotic"),
            },
        },
        initial_state=initial_state,
    )
    sim.update(sim_time)
    timeseries_data = timeseries_from_data(sim.emitter.get_data())
    antibiotic_str_to_float = []
    for string in timeseries_data["boundary"]["internal"]["antibiotic"]:
        antibiotic_str_to_float.append(units(string).magnitude)
    timeseries_data["boundary"]["internal"]["antibiotic"] = antibiotic_str_to_float
    timeseries_data["bulk"] = {
        "antibiotic": [bulk_data[0] for bulk_data in timeseries_data["bulk"]]
    }
    print(timeseries_data)
    plot_variables(
        timeseries_data,
        [("bulk", "antibiotic"), ("boundary", "internal", "antibiotic")],
        out_dir="out",
        filename="conc_to_counts",
    )


if __name__ == "__main__":
    main()
