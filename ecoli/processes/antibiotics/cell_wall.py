"""
TODO: references for parameters
"""

import os
from time import perf_counter

import numpy as np
from ecoli.library.cell_wall.column_sampler import geom_sampler, sample_column
from ecoli.library.cell_wall.hole_detection import detect_holes_skimage
from ecoli.library.cell_wall.lattice import (
    calculate_lattice_size,
    de_novo_lattice,
    plot_lattice,
)
from ecoli.library.create_timeline import create_timeline_from_csv
from ecoli.library.schema import bulk_schema
from ecoli.processes.registries import topology_registry
from ecoli.processes.shape import length_from_volume
from vivarium.core.composition import add_timeline, simulate_composite
from vivarium.core.process import Process
from vivarium.library.units import units

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
        "strand_term_p": 0.058,
        # Physical parameters
        "critical_radius": 20 * units.nm,
        "cell_radius": 0.5 * units.um,
        "disaccharide_length": 1.03 * units.nm,
        "crossbridge_length": 4.1
        / 3
        * units.nm,  # 4.1 in maximally stretched configuration,
        # divided by 3 because the sacculus can be stretched threefold
        "peptidoglycan_unit_area": 4 * units.nm**2,  # replace with precise
        # literature value
        # Simulation parameters
        "seed": 0,
        # "time_step": 10,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.murein = self.parameters["murein"]
        self.strand_term_p = self.parameters["strand_term_p"]

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
                ["incorporated_murein", "unincorporated_murein"], updater="set"
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
        DEBUG = True

        # Unpack states
        volume = states["shape"]["volume"]
        lattice = np.array(states["wall_state"]["lattice"])
        unincorporated_murein = states["murein_state"]["unincorporated_murein"]
        incorporated_murein = states["murein_state"]["incorporated_murein"]
        PBPs = states["PBP"]

        update = {}

        # Get number of synthesis sites
        # TODO: get this from pbp binding process (alpha * sum(PBPs))
        n_sites = sum(PBPs.values())
        if n_sites == 0:
            return {}

        # Translate volume into length
        length = length_from_volume(volume, self.cell_radius * 2).to("micrometer")

        if DEBUG:
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
            self.strand_term_p,
        )

        update["wall_state"] = {
            "lattice": lattice,
            "lattice_rows": lattice.shape[0],
            "lattice_cols": lattice.shape[1],
        }

        update["murein_state"] = {
            "unincorporated_murein": int(new_unincorporated_monomers // 4),
            "incorporated_murein": int(new_incorporated_monomers // 4),
        }

        # Crack detection (cracking is irreversible)
        if (
            not states["wall_state"]["cracked"]
            and self.get_largest_defect_area(lattice) > self.critical_area
        ):
            update["wall_state"]["cracked"] = True

        if DEBUG:
            assert new_incorporated_monomers == lattice.sum()
            assert (
                states["bulk_murein"][self.murein]
                == states["murein_state"]["unincorporated_murein"]
                + states["murein_state"]["incorporated_murein"]
            )

            try:
                self.time += timestep
            except AttributeError:
                self.time = 0

            fig, _ = plot_lattice(lattice)
            fig.savefig(f"out/processes/cell_wall/cell_wall_ecoli_t{self.time}.png")

        return update

    def update_murein(
        self,
        lattice,
        unincorporated_monomers,
        incorporated_monomers,
        new_rows,
        new_columns,
        n_sites,
        strand_term_p,
    ):
        rows, columns = lattice.shape
        d_columns = new_columns - columns

        # Create new lattice
        new_lattice = np.zeros((new_rows, new_columns), dtype=lattice.dtype)

        # Sample columns for synthesis sites
        insertion_points = self.rng.choice(
            list(range(columns)), size=min(n_sites, d_columns), replace=False
        )
        insertion_points.sort()
        insertion_size = np.repeat(d_columns // n_sites, insertion_points.size)

        # Add additional columns at random if necessary
        while insertion_size.sum() < d_columns:
            insertion_size[self.rng.integers(0, insertion_size.size)] += 1

        # Get murein per column
        current_incorporated = lattice.sum()
        murein_to_allocate = incorporated_monomers - current_incorporated

        # Stop early is there is no murein to allocate, or if the cell has not grown
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

        # Sample columns to insert
        insertions = []
        for insert_size in insertion_size:
            insertions.append(
                np.array(
                    [
                        sample_column(
                            rows,
                            murein_per_column,
                            geom_sampler(self.rng, strand_term_p),
                            self.rng,
                        )
                        for _ in range(insert_size)
                    ]
                ).T
            )

        # Combine insertions and old material into new lattice
        index_new = 0
        index_old = 0
        gaps_between_insertions = np.diff(insertion_points, prepend=0)
        for insert_i, (gap, insert_size) in enumerate(
            zip(gaps_between_insertions, insertion_size)
        ):
            # Copy from old lattice, from end of last insertion to start of next
            new_lattice[:, index_new : (index_new + gap)] = lattice[
                :, index_old : (index_old + gap)
            ]
            # Do insertion
            new_lattice[
                :, (index_new + gap) : (index_new + gap + insert_size)
            ] = insertions[insert_i]

            # update indices
            index_new += gap + insert_size
            index_old += gap

        # Copy from last insertion to end
        new_lattice[:, index_new:] = lattice[:, index_old:]

        total_monomers = unincorporated_monomers + incorporated_monomers
        new_incorporated_monomers = new_lattice.sum()
        new_free_monomers = total_monomers - new_incorporated_monomers
        return new_lattice, new_free_monomers, new_incorporated_monomers

    def get_largest_defect_area(self, lattice):
        hole_sizes, _ = detect_holes_skimage(
            lattice
            # critical_size=int(self.critical_area / self.peptidoglycan_unit_area),
        )
        max_size = hole_sizes.max()

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


if __name__ == "__main__":
    test_cell_wall()
