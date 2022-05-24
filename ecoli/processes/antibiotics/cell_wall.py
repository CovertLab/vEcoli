"""
TODO: references for parameters
"""

import os

import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from skimage.transform import resize

from vivarium.core.process import Process
from vivarium.library.units import units
from vivarium.plots.simulation_output import plot_variables

from ecoli.library.cell_wall.hole_detection import detect_holes
from ecoli.library.cell_wall.column_sampler import sample_column, poisson_sampler
from ecoli.library.schema import bulk_schema
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
    "wall_state": ("wall_state"),
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

        # Calculate new lattice dimensions
        new_rows, new_columns = self.calculate_lattice_size(length)

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

        # # Expand lattice size if necessary, depending on cell size
        # print("resizing lattice")
        # lattice, rows, columns = self.resize_lattice(
        #     length, lattice, lattice_rows, lattice_cols
        # )

        # # Cell wall construction/destruction
        # print("assigning murein")
        # lattice, new_unincorporated_murein, new_incorporated_murein = self.assign_murein(
        #     unincorporated_murein,
        #     incorporated_murein,
        #     lattice,
        #     rows,
        #     columns,
        # )
        # print(f"Lattice size: {lattice.shape}")
        # print(f"Holes: {lattice.size - lattice.sum()}")

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

    def calculate_lattice_size(self, cell_length):
        # Calculate new lattice size
        columns = int(
            cell_length / (self.crossbridge_length + self.disaccharide_length)
        )

        rows = int(self.circumference / self.disaccharide_length)

        return rows, columns

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

        # Create new lattice
        new_lattice = np.zeros((new_rows, new_columns))

        # Sample columns for synthesis sites
        insertion_points = self.rng.choice(
            list(range(columns)), size=n_sites, replace=False
        )
        insertion_size = np.repeat((new_columns - columns) // n_sites, n_sites)

        # Add additional columns at random if necessary
        while columns + insertion_size.sum() < new_columns:
            insertion_size[self.rng.integers(0, n_sites)] += 1

        # Get murein per column
        current_incorporated = lattice.sum()
        murein_to_allocate = incorporated_monomers - current_incorporated
        murein_per_column = murein_to_allocate / new_columns

        print(
            f"Cell Wall: Assigning {murein_to_allocate} monomers to {new_columns} columns ({murein_per_column} per column)"
        )

        if murein_to_allocate == 0 or columns == new_columns:
            new_lattice = lattice
            total_monomers = unincorporated_monomers + incorporated_monomers
            new_incorporated_monomers = new_lattice.sum()
            new_free_monomers = total_monomers - new_incorporated_monomers
            return new_lattice, new_free_monomers, new_incorporated_monomers

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
                    new_lattice[:, c:(c + insert_size)] = insertions[insert_i].T
                insert_i += 1

        total_monomers = unincorporated_monomers + incorporated_monomers
        new_incorporated_monomers = new_lattice.sum()
        new_free_monomers = total_monomers - new_incorporated_monomers
        return new_lattice, new_free_monomers, new_incorporated_monomers

    # def resize_lattice(self, cell_length, lattice, prev_rows=0, prev_cols=0):

    #     rows, columns = self.calculate_lattice_size(cell_length)

    #     # Fill in new positions with defects initially
    #     lattice = np.pad(
    #         lattice,
    #         ((0, max(0, rows - prev_rows)), (0, max(0, columns - prev_cols))),
    #         mode="constant",
    #         constant_values=0,
    #     )

    #     assert lattice.shape == (rows, columns)
    #     return lattice, rows, columns

    # def assign_murein(self, unincorporated_murein, incorporated_murein, lattice, rows, columns):
    #     n_incorporated = lattice.sum()
    #     n_holes = lattice.size - n_incorporated

    #     # fill holes
    #     # TODO: Replace random selection with strand extrusion
    #     #       from a length distribution
    #     fill_n = min(unincorporated_murein, n_holes)

    #     if fill_n > 0:
    #         fill_idx = self.rng.choice(
    #             np.arange(lattice.size), size=fill_n, replace=False
    #         )
    #         for idx in fill_idx:
    #             # Convert 1D index into row, column
    #             r = idx // columns
    #             c = idx - (r * columns)

    #             # fill hole
    #             lattice[r, c] = 1

    #     # add holes
    #     # TODO: Replace random selection with biased selection
    #     #       based on existing holes/stress map
    #     new_holes = lattice.sum() - incorporated_murein

    #     # choose random occupied locations
    #     if new_holes > 0:
    #         idx_occupancies = np.array(np.where(lattice))
    #         idx_new_holes = self.rng.choice(
    #             idx_occupancies.T, size=new_holes, replace=False
    #         )

    #         for hole_r, hole_c in idx_new_holes:
    #             lattice[hole_r, hole_c] = 0

    #     total_murein = unincorporated_murein + incorporated_murein
    #     new_incorporated = lattice.sum()
    #     new_free = total_murein - new_incorporated

    #     return lattice, new_free, new_incorporated

    def get_largest_defect_area(self, lattice):
        hole_sizes, _ = detect_holes(
            lattice,
            critical_size=int(self.critical_area / self.peptidoglycan_unit_area),
        )
        max_size = hole_sizes.get_max()

        return max_size * self.peptidoglycan_unit_area


def plot_lattice(lattice, on_cylinder=False):
    if not on_cylinder:
        fig, ax = plt.subplots()
        mappable = ax.imshow(lattice, interpolation="nearest")
        fig.colorbar(mappable, ax=ax)
    else:
        print("Downscaling lattice...")
        lattice = resize(
            lattice,
            (lattice.shape[0] // 10, lattice.shape[1] // 10),
            preserve_range=True,
            anti_aliasing=True,
        )
        lattice = lattice.T
        print("Done.\nDrawing on cylinder...")

        h, w = lattice.shape
        theta, z = np.linspace(0, 2 * np.pi, w), np.linspace(0, 1, h)
        THETA, Z = np.meshgrid(theta, z)
        X = np.cos(THETA)
        Y = np.sin(THETA)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        mappable = plt.cm.ScalarMappable(cmap=plt.cm.inferno)
        mappable.set_clim(0, 1)
        mappable.set_array(lattice)
        plot = ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            facecolors=mappable.cmap(lattice),
            linewidth=0,
            antialiased=False,
            alpha=0.75,
        )
        fig.colorbar(mappable)

    return fig, ax


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
    main()
