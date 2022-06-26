"""
TODO: references for parameters
"""

import os

import numpy as np

from vivarium.core.process import Process
from vivarium.core.composition import add_timeline, simulate_composite
from vivarium.library.units import units
from vivarium.plots.simulation_output import plot_variables

from ecoli.library.cell_wall.hole_detection import detect_holes
from ecoli.library.cell_wall.column_sampler import sample_column, poisson_sampler
from ecoli.library.cell_wall.lattice import calculate_lattice_size, plot_lattice
from ecoli.library.schema import bulk_schema
from ecoli.library.create_timeline import create_timeline_from_csv
from ecoli.processes.antibiotics.cephaloridine_antagonism import PBPBinding
from ecoli.processes.registries import topology_registry
from vivarium.core.composition import simulate_process
from ecoli.processes.shape import length_from_volume


# Register default topology for this process, associating it with process name
NAME = "ecoli-cell-wall"
TOPOLOGY = {
    "shape": ("cell_global",),
    "bulk_murein": ("bulk",),
    "murein_state": ("murein_state",),
    "PBP": ("bulk",),
    "wall_state": ("wall_state",),
}
topology_registry.register(NAME, TOPOLOGY)


class CellWall(Process):
    name = NAME
    topology = TOPOLOGY
    defaults = {
        # Molecules
        "murein": "CPD-12261[p]",  # two crosslinked peptidoglycan units
        "PBP": {  # penicillin-binding proteins
            "PBP1A": "CPLX0-7717[m]",  # transglycosylase-transpeptidase ~100
            "PBP1B": "CPLX0-3951[i]",  # transglycosylase-transpeptidase ~100
        },
        "strand_length_distribution": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        # Physical parameters
        "critical_radius": 20 * units.nm,
        "cell_radius": 0.5 * units.um,
        "disaccharide_length": 1.03 * units.nm,
        "crossbridge_length": 4.1
        * units.nm
        / 3,  # 4.1 in maximally stretched configuration,
        # divided by 3 because the sacculus can be stretched threefold
        "peptidoglycan_unit_area": 4 * units.nm**2,  # replace with precise
        # literature value
        # Simulation parameters
        "seed": 0,
        "timestep": 30,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.murein = self.parameters["murein"]
        self.strand_length_distribution = self.parameters["strand_length_distribution"]

        self.cell_radius = self.parameters["cell_radius"]
        self.critical_radius = self.parameters["critical_radius"]
        self.critical_area = np.pi * self.critical_radius**2
        self.circumference = 2 * np.pi * self.cell_radius

        self.peptidoglycan_unit_area = self.parameters["peptidoglycan_unit_area"]
        self.disaccharide_length = self.parameters["disaccharide_length"]
        self.crossbridge_length = self.parameters["crossbridge_length"]

        # Create pseudorandom number generator
        self.rng = np.random.default_rng(self.parameters["seed"])

    def ports_schema(self):
        schema = {
            "bulk_murein": bulk_schema([self.parameters["murein"]]),
            "murein_state": bulk_schema(
                ["incorporated_murein", "unincorporated_murein"]
            ),
            "PBP": bulk_schema(list(self.parameters["PBP"].values())),
            "shape": {"volume": {"_default": 0 * units.fL, "_emit": True}},
            "wall_state": {
                "lattice": {
                    "_default": np.array([], dtype=int),
                    "_updater": "set",
                    "_emit": True,
                },
                "lattice_rows": {"_default": 0, "_updater": "set", "_emit": True},
                "lattice_cols": {"_default": 0, "_updater": "set", "_emit": True},
                "cracked": {"_default": False, "_updater": "set", "_emit": True},
            },
        }

        return schema

    def next_update(self, timestep, states):
        # Unpack states
        volume = states["shape"]["volume"]
        lattice = np.array(states["wall_state"]["lattice"])
        unincorporated_murein = states["murein_state"]["unincorporated_murein"]
        incorporated_murein = states["murein_state"]["incorporated_murein"]
        PBPs = states["PBP"]

        update = {}

        # Get number of synthesis sites
        # TODO: get this from cephaloridine antagonism (alpha * sum(PBPs))
        n_sites = sum(PBPs.values())
        if n_sites == 0:
            return {}

        # Translate volume into length
        length = length_from_volume(volume, self.cell_radius * 2).to("micrometer")

        print(f"Cell Wall: Bulk murein = {states['bulk_murein'][self.murein]}")
        print(f"Cell Wall: Unincorporated murein = {unincorporated_murein}")
        print(f"Cell Wall: Incorporated murein = {incorporated_murein}")
        print(f"Cell Wall: Cell length = {length}")
        print(f"Cell Wall: Lattice dimensions: {lattice.shape}")

        # Calculate new lattice dimensions
        new_rows, new_columns = calculate_lattice_size(
            length,
            self.crossbridge_length,
            self.disaccharide_length,
            self.circumference,
        )

        # Update lattice to reflect new dimensions,
        # change in murein, synthesis sites
        (
            lattice,
            new_unincorporated_monomers,
            new_incorporated_monomers,
        ) = self.update_murein(
            lattice,
            4 * unincorporated_murein,
            4 * incorporated_murein,
            new_rows,
            new_columns,
            n_sites,
            self.strand_length_distribution,
        )

        update["wall_state"] = {
            "lattice": lattice,
            "lattice_rows": lattice.shape[0],
            "lattice_cols": lattice.shape[1],
        }

        # update["murein_state"] = {
        #     "unincorporated_murein": new_unincorporated_monomers // 4,
        #     "incorporated_murein": new_incorporated_monomers // 4,
        # }

        # Crack detection (cracking is irreversible)
        if (
            not states["wall_state"]["cracked"]
            and self.get_largest_defect_area(lattice) > self.critical_area
        ):
            update["wall_state"]["cracked"] = True

        # Testing (TODO: remove)
        assert new_incorporated_monomers == lattice.sum()
        assert (
            states["bulk_murein"][self.murein]
            == states["murein_state"]["unincorporated_murein"]
            + states["murein_state"]["incorporated_murein"]
        )

        return update

    def update_murein(
        self,
        lattice,
        unincorporated_monomers,
        incorporated_monomers,
        new_rows,
        new_columns,
        n_sites,
        strand_length_distribution,
    ):
        rows, columns = lattice.shape
        d_columns = new_columns - columns

        # Create new lattice
        new_lattice = np.zeros((new_rows, new_columns))

        # Sample columns for synthesis sites
        insertion_points = self.rng.choice(
            list(range(columns)), size=n_sites, replace=False
        )
        insertion_size = np.repeat(d_columns // n_sites, n_sites)

        # Add additional columns at random if necessary
        while insertion_size.sum() < d_columns:
            insertion_size[self.rng.integers(0, n_sites)] += 1

        # Get murein per column
        current_incorporated = lattice.sum()
        murein_to_allocate = incorporated_monomers - current_incorporated

        if murein_to_allocate == 0 or d_columns == 0:
            new_lattice = lattice
            total_monomers = unincorporated_monomers + incorporated_monomers
            new_incorporated_monomers = new_lattice.sum()
            new_free_monomers = total_monomers - new_incorporated_monomers
            return new_lattice, new_free_monomers, new_incorporated_monomers

        murein_per_column = murein_to_allocate / d_columns

        print(
            f"Cell Wall: Assigning {murein_to_allocate} monomers to {d_columns} columns ({murein_per_column} per column)"
        )

        # Sample insertions
        insertions = []
        for insert_size in insertion_size:
            insertions.append(
                np.array(
                    [
                        sample_column(
                            rows,
                            murein_per_column,
                            poisson_sampler(self.rng, 1),
                            self.rng,
                        )
                        for _ in range(insert_size)
                    ]
                )
            )

        # Combine insertions and old material into new lattice
        insert_i = 0
        for c in range(columns):
            if c not in insertion_points:
                new_lattice[:, c] = lattice[:, c]
            else:
                insert_size = insertion_size[insert_i]
                if insert_size > 0:
                    new_lattice[:, c : (c + insert_size)] = insertions[insert_i].T
                insert_i += 1

        total_monomers = unincorporated_monomers + incorporated_monomers
        new_incorporated_monomers = new_lattice.sum()
        new_free_monomers = total_monomers - new_incorporated_monomers
        return new_lattice, new_free_monomers, new_incorporated_monomers

    def get_largest_defect_area(self, lattice):
        hole_sizes, _ = detect_holes(
            lattice,
            critical_size=int(self.critical_area / self.peptidoglycan_unit_area),
        )
        max_size = hole_sizes.get_max()

        return max_size * self.peptidoglycan_unit_area


def test_cell_wall():
    # Create composite with timeline
    processes = {"cell_wall": CellWall({})}
    topology = {
        "cell_wall": {
            "shape": ("cell_global",),
            "bulk_murein": ("bulk",),
            "murein_state": ("murein_state",),
            "PBP": ("bulk",),
            "wall_state": ("wall_state",),
        }
    }
    add_timeline(
        processes,
        topology,
        create_timeline_from_csv(
            "data/cell_wall/test_murein_21_06_2022_17_42_11.csv",
            {
                "CPD-12261[p]": ("bulk", "CPD-12261[p]"),
                "CPLX0-7717[m]": ("bulk", "CPLX0-7717[m]"),
                "CPLX0-3951[i]": ("bulk", "CPLX0-3951[i]"),
            },
        )
        # {
        #     "timeline": [
        #         (
        #             time,
        #             {
        #                 ("bulk", "CPD-12261[p]"): int(383237 + 10 * time),
        #                 ("murein_state", "incorporated_murein"): int(
        #                     383237 + 10 * time
        #                 ),
        #                 ("cell_global", "volume"): (1 + time / 1000) * units.fL,
        #             },
        #         )
        #         for time in range(0, 10)
        #     ]
        # },
    )

    # Run experiment
    rng = np.random.default_rng(5)
    initial_lattice = rng.binomial(1, 0.75, size=(3050, 670))
    settings = {
        "return_raw_data": True,
        "total_time": 10,
        "initial_state": {
            "murein_state": {
                "incorporated_murein": initial_lattice.sum() // 4,
                "unincorporated_murein": 0,
            },
            "cell_global": {
                "volume": 1 * units.fL,
            },
            "bulk": {
                "CPD-12261[p]": initial_lattice.sum() // 4,
                "CPLX0-7717[m]": 24,
                "CPLX0-3951[i]": 0,
            },
            "wall_state": {
                "lattice_rows": initial_lattice.shape[0],
                "lattice_cols": initial_lattice.shape[1],
                "lattice": initial_lattice,
            },
        },
    }
    data = simulate_composite({"processes": processes, "topology": topology}, settings)

    # Plot output
    # plot_variables(
    #     data,
    #     variables=[
    #         ("bulk", "CPD-12261[p]"),
    #         ("murein_state", "incorporated_murein"),
    #         ("murein_state", "unincorporated_murein")
    #     ],
    #     out_dir="out/processes/cell_wall/",
    #     filename="test.png",
    # )

    for t, data_t in data.items():
        t = int(t)
        print(f"Plotting t={t}...")
        lattice = data_t["wall_state"]["lattice"]
        fig, _ = plot_lattice(np.array(lattice), on_cylinder=False)
        fig.tight_layout()
        fig.savefig(f"out/processes/cell_wall/cell_wall_t{t}.png")
        print("Done.\n")


def main():
    from vivarium.core.composer import Composite
    from vivarium.processes.timeline import TimelineProcess

    # Stub for rest of cell (increasing murein)
    cell_stub = TimelineProcess(
        {
            "time_step": 2.0,
            "timeline": [
                (time, {("bulk_murein", "CPD-12261[p]"): int(3e6 + 1000 * time)})
                for time in range(0, 10, 2)
            ],
        }
    )

    # Cell wall process
    params = {}
    cell_wall = CellWall(params)

    settings = {
        "total_time": 10,
        "initial_state": {
            "bulk_murein": {"CPD-12261[p]": int(3e6)},
            "volume": 1 * units.fL,
            "murein_state": {
                "unincorporated_murein": 0,
                "incorporated_murein": 3000000,
            },
            "wall_state": {
                "lattice_rows": 1525,
                "lattice_cols": 834,
                "lattice": np.ones((1525, 834), dtype=int),
            },
        },
    }

    # sim = Engine(
    #     processes={
    #         # 'cell_stub': cell_stub,
    #         "cell_wall": cell_wall
    #     },
    #     topology={
    #         'cell_wall': {
    #             'bulk_murein': ("bulk", 'CPD-12261[p]'),
    #             'shape' : ("shape",)
    #         },
    #         # 'cell_stub': {
    #         #     'global': ('global',),
    #         #     'bulk_murein': ("bulk", 'CPD-12261[p]'),
    #         # }
    #     },
    #     initial_state={
    #         'bulk': {
    #             'CPD-12261[p]': int(3e6),
    #         },
    #         'shape' : {
    #             'length' : 2 * units.um,
    #         }})
    #
    # sim.run_for(10)
    # data = sim.emitter.get_data()

    data = simulate_process(cell_wall, settings)
    # data = simulate_process(cell_stub, settings)
    # data = simulate_composite(test_composite, settings)
    fig = plot_variables(
        data,
        variables=[
            ("murein_state", "unincorporated_murein"),
            ("murein_state", "incorporated_murein"),
            ("wall_state", "lattice_rows"),
            ("wall_state", "lattice_cols"),
        ],
    )
    fig.tight_layout()

    os.makedirs("out/processes/cell_wall/", exist_ok=True)
    fig.savefig("out/processes/cell_wall/test.png")

    for t, lattice in enumerate(data["wall_state"]["lattice"]):
        print(f"Plotting t={t}...")
        fig, _ = plot_lattice(np.array(lattice), on_cylinder=True)
        fig.tight_layout()
        fig.savefig(f"out/processes/cell_wall/cell_wall_t{t}.png")
        print("Done.\n")


if __name__ == "__main__":
    test_cell_wall()
