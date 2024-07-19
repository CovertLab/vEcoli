import numpy as np
from ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts
from ecoli.library.json_state import get_state_from_file
from ecoli.processes.bulk_timeline import BulkTimelineProcess
from vivarium.core.emitter import timeseries_from_data
from vivarium.core.engine import Engine
from vivarium.core.process import Step
from vivarium.library.units import units
from vivarium.plots.simulation_output import plot_variables

from typing import Any

# TODO: Recompute average surface area with corrected formula
# To calculate SA_AVERAGE, we calculated the average surface area of the model up until division.
SA_AVERAGE = 6.22200939450696 * units.micron**2
# To calculate CEPH_OMPC_CON_PERM and CEPH_OMPF_CON_PERM, we calculated the average counts of ompC and ompF
# in the model up until division and divided each by the average surface area to get the average concentrations
# of ompC and ompF. We then divided the corresponding cephaloridine permeability coefficients from Nikaido, 1983
# by these average concentrations to get our permeability per concentration constants for cephaloridine. Likewise, we
# divided the tetracycline permeability coefficient due to ompF (overall permeability (estimated in (Thanassi et al.,
# 1995)) subtracted by pH-gradient induced permeability (Nikaido and Pages, 2012)) by the average concentration of ompF
# to get TET_OMPF_CON_PERM.
CEPH_OMPC_CON_PERM = (
    0.003521401200296894 * 1e-5 * units.cm * units.micron * units.micron / units.sec
)
CEPH_OMPF_CON_PERM = (
    0.01195286573132685 * 1e-5 * units.cm * units.micron * units.micron / units.sec
)
TET_OMPF_CON_PERM = (
    2.2496838543752056 * 1e-9 * units.cm * units.micron * units.micron / units.sec
)

# Cephaloridine is assumed to not permeate through the outer membrane bilayer. (Nikaido, 1983)
OUTER_BILAYER_CEPH_PERM = 0 * units.cm / units.sec

# Estimated in (Thanassi et al., 1995)
OUTER_BILAYER_TET_PERM = 1 * 1e-7 * units.cm / units.sec
# Estimated in (Thanassi et al., 1995)
INNER_BILAYER_TET_PERM = 3 * 1e-6 * units.cm / units.sec


class Permeability(Step):
    name = "permeability"
    defaults: dict[str, Any] = {
        "porin_ids": [],
        "diffusing_molecules": [],
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.porin_ids = self.parameters["porin_ids"]
        self.diffusing_molecules = self.parameters["diffusing_molecules"]
        # Helper indices for Numpy arrays
        self.porin_idx = None

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "permeabilities": {
                mol_id: {
                    "_default": 1e-5 * units.cm / units.sec,
                    "_emit": True,
                    "_updater": "set",
                }
                for mol_id in self.diffusing_molecules
            },  # Different permeability for every molecule
            "surface_area": {
                "_default": 0.0  # * units.micron ** 2
            },
        }

    def next_update(self, timestep, states):
        if self.porin_idx is None:
            self.porin_idx = bulk_name_to_idx(self.porin_ids, states["bulk"]["id"])
        porins = counts(states["bulk"], self.porin_idx)
        porins = dict(zip(self.porin_ids, porins))
        surface_area = states["surface_area"]
        permeabilities = {}
        for molecule in self.diffusing_molecules:
            cell_permeability = 0
            for porin_id, permeability in self.diffusing_molecules[molecule][
                "concentration_perm"
            ].items():
                cell_permeability += (porins[porin_id] / surface_area) * permeability
            cell_permeability += self.diffusing_molecules[molecule]["bilayer_perm"]
            permeabilities[molecule] = cell_permeability
        return {"permeabilities": permeabilities}


def main():
    sim_time = 10

    initial_state = get_state_from_file(path="data/vivecoli_t2527.json")
    initial_state["boundary"] = {}
    initial_state["boundary"]["surface_area"] = SA_AVERAGE
    porin_idx_1 = np.where(initial_state["bulk"]["id"] == "CPLX0-7533[o]")[0]
    porin_idx_2 = np.where(initial_state["bulk"]["id"] == "CPLX0-7534[o]")[0]
    porin_count_1 = initial_state["bulk"]["count"][porin_idx_1]
    porin_count_2 = initial_state["bulk"]["count"][porin_idx_2]

    porin_parameters = {
        "porin_ids": ["CPLX0-7533[o]", "CPLX0-7534[o]"],
        "diffusing_molecules": {
            "cephaloridine": {
                "concentration_perm": {
                    "CPLX0-7533[o]": CEPH_OMPC_CON_PERM,
                    "CPLX0-7534[o]": CEPH_OMPF_CON_PERM,
                },
                "bilayer_perm": OUTER_BILAYER_CEPH_PERM,
            },
            "tetracycline": {
                "concentration_perm": {
                    "CPLX0-7534[o]": TET_OMPF_CON_PERM,
                },
                "bilayer_perm": OUTER_BILAYER_TET_PERM,
            },
        },
    }
    porin_process = Permeability(porin_parameters)

    timeline = {}
    for i in range(5):
        timeline[i * 2] = {
            ("bulk", "CPLX0-7533[o]"): porin_count_1 + ((i + 1) * 500),
            ("bulk", "CPLX0-7534[o]"): porin_count_2 + ((i + 1) * 500),
        }
    timeline_params = {
        "time_step": 2.0,
        "timeline": timeline,
    }
    timeline_process = BulkTimelineProcess(timeline_params)

    sim = Engine(
        processes={"porin_permeability": porin_process, "timeline": timeline_process},
        topology={
            "porin_permeability": {
                "bulk": ("bulk",),
                "permeabilities": (
                    "boundary",
                    "permeabilities",
                ),
                "surface_area": (
                    "boundary",
                    "surface_area",
                ),
            },
            "timeline": {
                "global": ("global",),
                "bulk": ("bulk",),
            },
        },
        initial_state=initial_state,
    )
    sim.update(sim_time)
    timeseries_data = timeseries_from_data(sim.emitter.get_data())
    ceph_str_to_float = []
    for string in timeseries_data["boundary"]["permeabilities"]["cephaloridine"]:
        ceph_str_to_float.append(units(string).magnitude)
    timeseries_data["boundary"]["permeabilities"]["cephaloridine"] = ceph_str_to_float
    tet_str_to_float = []
    for string in timeseries_data["boundary"]["permeabilities"]["tetracycline"]:
        tet_str_to_float.append(units(string).magnitude)
    timeseries_data["boundary"]["permeabilities"]["tetracycline"] = tet_str_to_float
    bulk_array = np.array(timeseries_data["bulk"])
    timeseries_data["bulk"] = {
        "CPLX0-7533[o]": bulk_array[:, porin_process.porin_idx[0]],
        "CPLX0-7534[o]": bulk_array[:, porin_process.porin_idx[1]],
    }
    plot_variables(
        timeseries_data,
        [
            ("bulk", "CPLX0-7533[o]"),
            ("bulk", "CPLX0-7534[o]"),
            ("boundary", "permeabilities", "cephaloridine"),
            ("boundary", "permeabilities", "tetracycline"),
        ],
        out_dir="out",
        filename="permeability_counts",
    )


if __name__ == "__main__":
    main()
