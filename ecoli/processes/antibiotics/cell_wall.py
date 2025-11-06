"""
A coarse-grained model of the cell wall, aimed at predicting
cracking (which leads irreversibly to lysis) under conditions limiting
production of crosslinked murein.

Parameters:
- Probability of terminating a murein strand (p): fitted from data in
  Obermann, W., & Höltje, J. (1994).
    Alterations of murein structure and of penicillin-binding proteins
    in minicells from Escherichia coli. Microbiology.
    https://doi.org/10.1099/13500872-140-1-79
  Vollmer, W., Blanot, D., & De Pedro, M. A. (2008).
    Peptidoglycan structure and architecture.
    FEMS Microbiology Reviews, 32(2), 149-167.
    https://doi.org/10.1111/j.1574-6976.2007.00094.x

- critical radius:
  Daly, K. E., Huang, K. C., Wingreen, N. S., & Mukhopadhyay, R. (2011).
    Mechanics of membrane bulging during cell-wall disruption in
    Gram-negative bacteria. Physical Review E, 83(4), 041922.
    https://doi.org/10.1103/PhysRevE.83.041922

- cell_radius: chosen to be consistent with value in  cell shape process

- disaccharide height:
  Vollmer, W., & Höltje, J.-V. (2004).
    The Architecture of the Murein (Peptidoglycan) in Gram-Negative
    Bacteria: Vertical Scaffold or Horizontal Layer(s)?
    Journal of Bacteriology, 186(18), 5978-5987.
    https://doi.org/10.1128/JB.186.18.5978-5987.2004

- disaccharide width, inter-strand distance:
  Turner, R. D., Mesnage, S., Hobbs, J. K., & Foster, S. J. (2018).
    Molecular imaging of glycan chains couples cell-wall polysaccharide
    architecture to bacterial cell morphology.
    Nature Communications, 9, 1263.
    https://doi.org/10.1038/s41467-018-03551-y


- max expansion:
  Koch, A. L., & Woeste, S. (1992).
    Elasticity of the sacculus of Escherichia coli.
    Journal of Bacteriology, 174(14), 4811-4819.

- peptidoglycan unit area:
  Wientjes, F. B., Woldringh, C. L., & Nanninga, N. (1991).
    Amount of peptidoglycan in cell walls of gram-negative bacteria.
    Journal of Bacteriology, 173(23), 7684-7691.
"""

import numpy as np
import warnings
from ecoli.library.cell_wall.column_sampler import (
    geom_sampler,
    sample_column,
)
from ecoli.library.cell_wall.hole_detection import detect_holes_skimage
from ecoli.library.cell_wall.lattice import (
    calculate_lattice_size,
    get_length_distributions,
)
from ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts
from ecoli.library.parameters import param_store
from ecoli.processes.registries import topology_registry
from ecoli.processes.shape import length_from_volume, surface_area_from_length
from vivarium.core.process import Process
from vivarium.library.units import units, remove_units

# Register default topology for this process, associating it with process name
NAME = "ecoli-cell-wall"
TOPOLOGY = {
    "shape": ("boundary",),
    "murein_state": ("murein_state",),
    "bulk": ("bulk",),
    "wall_state": ("wall_state",),
    "listeners": ("listeners",),
}
topology_registry.register(NAME, TOPOLOGY)


def divide_lattice(lattice):
    return np.array_split(lattice, 2, axis=1)


class CellWall(Process):
    name = NAME
    topology = TOPOLOGY
    defaults = {
        # Molecules
        "murein": "CPD-12261[p]",  # four crosslinked peptidoglycan units
        "PBP": {  # penicillin-binding proteins
            "PBP1A": "CPLX0-7717[p]",  # transglycosylase-transpeptidase
            # PBP1B has three isoforms: α (currently not produced by model),
            # β (degradation product of α, not in vivo), and γ (made by model)
            "PBP1B_alpha": "CPLX0-3951[i]",
            "PBP1B_gamma": "CPLX0-8300[c]",
        },
        # Probability of terminating a strand on the next monomer,
        # fitted from data
        "strand_term_p": param_store.get(("cell_wall", "strand_term_p")),
        # Physical parameters
        "critical_radius": param_store.get(("cell_wall", "critical_radius")),
        "cell_radius": param_store.get(("cell_wall", "cell_radius")),
        "disaccharide_height": param_store.get(("cell_wall", "disaccharide_height")),
        "disaccharide_width": param_store.get(("cell_wall", "disaccharide_width")),
        "inter_strand_distance": param_store.get(
            ("cell_wall", "inter_strand_distance")
        ),
        "max_expansion": param_store.get(("cell_wall", "max_expansion")),
        # Simulation parameters
        "seed": 0,
        "time_step": 10,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.murein = self.parameters["murein"]
        self.pbp1a = self.parameters["PBP"]["PBP1A"]
        self.pbp1b_alpha = self.parameters["PBP"]["PBP1B_alpha"]
        self.pbp1b_gamma = self.parameters["PBP"]["PBP1B_gamma"]
        self.strand_term_p = self.parameters["strand_term_p"]

        self.cell_radius = self.parameters["cell_radius"]
        self.critical_radius = self.parameters["critical_radius"]
        self.critical_area = np.pi * self.critical_radius**2
        self.circumference = 2 * np.pi * self.cell_radius

        self.disaccharide_height = self.parameters["disaccharide_height"]
        self.disaccharide_width = self.parameters["disaccharide_width"]
        self.inter_strand_distance = self.parameters["inter_strand_distance"]
        self.max_expansion = self.parameters["max_expansion"]
        self.peptidoglycan_unit_area = (
            self.inter_strand_distance + self.disaccharide_width
        ) * self.disaccharide_height

        # Create pseudorandom number generator
        self.rng = np.random.default_rng(self.parameters["seed"])

        # Helper indices for Numpy arrays
        self.pbp_ids = list(self.parameters["PBP"].values())
        self.pbp_idx = None

    def ports_schema(self):
        schema = {
            "murein_state": {
                # Divider sets to zero because the correct value is initialized
                # from the bulk store the first timestep after division.
                "incorporated_murein": {
                    "_default": 0,
                    "_emit": True,
                    "_updater": "set",
                    "_divider": "zero",
                },
                "unincorporated_murein": {
                    "_default": 0,
                    "_emit": True,
                    "_divider": "binomial_ecoli",
                },
                "shadow_murein": {
                    "_default": 0,
                    "_emit": True,
                    "_divider": "binomial_ecoli",
                },
            },
            "bulk": numpy_schema("bulk"),
            "shape": {"volume": {"_default": 0 * units.fL, "_emit": True}},
            "wall_state": {
                "lattice": {
                    "_default": None,
                    "_updater": "set",
                    "_emit": False,
                    "_divider": divide_lattice,
                },
                "lattice_rows": {
                    "_default": 0,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": "zero",
                },
                "lattice_cols": {
                    "_default": 0,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": "zero",
                },
                "extension_factor": {
                    "_default": 1,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": {"divider": "set_value", "config": {"value": 1}},
                },
                "cracked": {
                    "_default": False,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": {"divider": "set_value", "config": {"value": False}},
                },
                "attempted_shrinkage": {
                    "_default": False,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": {"divider": "set_value", "config": {"value": False}},
                },
            },
            "pbp_state": {
                "active_fraction_PBP1A": {"_default": 0.0, "_updater": "set"},
                "active_fraction_PBP1B": {"_default": 0.0, "_updater": "set"},
            },
            "listeners": {
                "porosity": {
                    "_default": 0,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": {"divider": "set_value", "config": {"value": 0}},
                },
                "hole_size_distribution": {
                    "_default": np.array([], int),
                    "_updater": "set",
                    "_emit": True,
                    "_divider": {
                        "divider": "set_value",
                        "config": {"value": np.array([], int)},
                    },
                },
                "strand_length_distribution": {
                    "_default": [],
                    "_updater": "set",
                    "_emit": True,
                    "_divider": {"divider": "set_value", "config": {"value": []}},
                },
            },
        }

        return schema

    def next_update(self, timestep, states):
        if self.pbp_idx is None:
            self.pbp_idx = bulk_name_to_idx(self.pbp_ids, states["bulk"]["id"])
        update = {}

        # Unpack states
        volume = states["shape"]["volume"]
        extension_factor = states["wall_state"]["extension_factor"]
        unincorporated_monomers = states["murein_state"]["unincorporated_murein"]
        incorporated_monomers = states["murein_state"]["incorporated_murein"]
        PBPs = dict(zip(self.pbp_ids, counts(states["bulk"], self.pbp_idx)))
        active_fraction_PBP1a = states["pbp_state"]["active_fraction_PBP1A"]
        active_fraction_PBP1b = states["pbp_state"]["active_fraction_PBP1B"]

        # Get lattice
        lattice = states["wall_state"]["lattice"]

        # When not run in an EngineProcess, this process sets the incorporated
        # murein count before MureinDivision and PBPBinding run after division
        if states["murein_state"]["incorporated_murein"] == 0:
            incorporated_monomers = np.sum(lattice)
            update["murein_state"] = {"incorporated_murein": incorporated_monomers}

        if not isinstance(lattice, np.ndarray):
            lattice = np.array(lattice)

        # Do not run process if the cell is already cracked
        if states["wall_state"]["cracked"]:
            return update

        # Get number of synthesis sites
        n_sites = int(
            remove_units(
                PBPs[self.pbp1a] * active_fraction_PBP1a
                + PBPs[self.pbp1b_alpha] * active_fraction_PBP1b
                + PBPs[self.pbp1b_gamma] * active_fraction_PBP1b
            )
        )

        # Translate volume into length,
        # Calculate new lattice dimensions
        length = length_from_volume(volume, self.cell_radius * 2).to("micrometer")
        _, new_columns = calculate_lattice_size(
            length,
            self.inter_strand_distance,
            self.disaccharide_height,
            self.disaccharide_width,
            self.circumference,
            extension_factor,
        )

        # Shrink extension factor when excess murein and PBPs are available
        d_full_columns = unincorporated_monomers // lattice.shape[0]
        d_columns = new_columns - lattice.shape[1]
        if d_full_columns > d_columns and extension_factor > 1 and n_sites > 0:
            new_columns = lattice.shape[1] + d_full_columns
            extension_factor = remove_units(
                (
                    length
                    / (
                        new_columns
                        * (self.inter_strand_distance + self.disaccharide_width)
                    )
                ).to("dimensionless")
            )
            # Minimum extension factor of 1
            if extension_factor < 1:
                _, new_columns = calculate_lattice_size(
                    length,
                    self.inter_strand_distance,
                    self.disaccharide_height,
                    self.disaccharide_width,
                    self.circumference,
                    1,
                )
                extension_factor = 1

        # Update lattice to reflect new dimensions,
        # change in murein, synthesis sites
        (
            new_lattice,
            new_unincorporated_monomers,
            new_incorporated_monomers,
            attempted_shrinkage,
        ) = self.update_murein(
            lattice,
            unincorporated_monomers,
            incorporated_monomers,
            new_columns,
            n_sites,
            self.strand_term_p,
        )

        # Crack detection (cracking is irreversible)
        hole_sizes, _ = detect_holes_skimage(new_lattice)
        max_size = hole_sizes.max() * self.peptidoglycan_unit_area * extension_factor

        # See if stretching will save from cracking
        will_crack = max_size > self.critical_area
        resting_length = lattice.shape[1] * (
            self.inter_strand_distance + self.disaccharide_width
        )
        can_stretch = (
            surface_area_from_length(length, self.cell_radius * 2)
            / surface_area_from_length(resting_length, self.cell_radius * 2)
            <= self.max_expansion
        )
        if will_crack and can_stretch:
            # stretch more and try again...
            extension_factor = remove_units(
                (
                    length
                    / (
                        lattice.shape[1]
                        * (self.inter_strand_distance + self.disaccharide_width)
                    )
                ).to("dimensionless")
            )

            _, new_columns = calculate_lattice_size(
                length,
                self.inter_strand_distance,
                self.disaccharide_height,
                self.disaccharide_width,
                self.circumference,
                extension_factor,
            )

            # Update lattice to reflect new dimensions,
            # change in murein, synthesis sites
            (
                new_lattice,
                new_unincorporated_monomers,
                new_incorporated_monomers,
                attempted_shrinkage,
            ) = self.update_murein(
                lattice,
                unincorporated_monomers,
                incorporated_monomers,
                new_columns,
                n_sites,
                self.strand_term_p,
            )

            # Crack detection (cracking is irreversible)
            hole_sizes, _ = detect_holes_skimage(new_lattice)
            max_size = (
                hole_sizes.max() * self.peptidoglycan_unit_area * extension_factor
            )

            will_crack = max_size > self.critical_area

        # Accept proposed new lattice
        lattice = new_lattice

        # Form updates
        update["wall_state"] = {
            "lattice": lattice,
            "lattice_rows": lattice.shape[0],
            "lattice_cols": lattice.shape[1],
            "extension_factor": extension_factor,
            "attempted_shrinkage": attempted_shrinkage,
        }
        update["murein_state"] = {
            "unincorporated_murein": new_unincorporated_monomers
            - unincorporated_monomers,
            "incorporated_murein": new_incorporated_monomers,
        }
        update["listeners"] = {
            "porosity": 1 - (lattice.sum() / lattice.size),
            "hole_size_distribution": np.bincount(hole_sizes),
            "strand_length_distribution": np.bincount(
                get_length_distributions(lattice)[1]
            ),
        }

        if will_crack:
            update["wall_state"]["cracked"] = True

        return update

    def update_murein(
        self,
        lattice,
        unincorporated_monomers,
        incorporated_monomers,
        new_columns,
        n_sites,
        strand_term_p,
    ):
        rows, columns = lattice.shape
        d_columns = new_columns - columns

        attempted_shrinkage = False

        # Stop early if the cell has not grown
        if d_columns == 0:
            return (
                lattice,
                unincorporated_monomers,
                incorporated_monomers,
                attempted_shrinkage,
            )

        if d_columns < 0:
            warnings.warn(
                f"Lattice shrinkage is currently not supported ({-d_columns} lost)."
            )
            attempted_shrinkage = True
            return (
                lattice,
                unincorporated_monomers,
                incorporated_monomers,
                attempted_shrinkage,
            )

        # Create new lattice
        new_lattice = np.zeros((rows, new_columns), dtype=lattice.dtype)

        # Sample columns for synthesis sites
        # First choose positions:
        n_points = min(n_sites, d_columns)
        if n_points > 0:
            insertion_points = self.rng.choice(
                list(range(columns)), size=n_points, replace=False
            )
            insertion_points.sort()

            # Assign insert sizes to positions (at least one column per position):
            insertion_size = np.ones(n_points, dtype=int)
            N_remaining = d_columns - n_points
            if N_remaining > 0:
                insertion_size += self.rng.multinomial(
                    N_remaining, np.repeat([1 / n_points], n_points)
                )

            murein_per_column = self.rng.multinomial(
                unincorporated_monomers, np.repeat([1 / d_columns], d_columns)
            )

            # Sample columns to insert. Columns to insert together are sampled
            # as a chunk (an "insertion").
            insertions = []
            columns_sampled = 0
            for insert_size in insertion_size:
                columns_to_insert = []
                for _ in range(insert_size):
                    columns_to_insert.append(
                        sample_column(
                            rows,
                            murein_per_column[columns_sampled],
                            geom_sampler(self.rng, strand_term_p),
                            self.rng,
                        )
                    )
                    columns_sampled += 1
                insertions.append(np.array(columns_to_insert).T)
        # If no active PBPs, assume empty column(s) inserted at center of wall
        else:
            insertion_points = [int(np.mean(range(columns)))]
            insertion_size = [d_columns]
            insertions = np.array(
                [np.zeros(rows, dtype=int) for _ in range(d_columns)]
            ).T

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
            new_lattice[:, (index_new + gap) : (index_new + gap + insert_size)] = (
                insertions[insert_i]
            )

            # update indices
            index_new += gap + insert_size
            index_old += gap

        # Copy from last insertion to end
        new_lattice[:, index_new:] = lattice[:, index_old:]

        total_real_monomers = unincorporated_monomers + incorporated_monomers
        new_incorporated_monomers = new_lattice.sum()
        new_unincorporated_monomers = total_real_monomers - new_incorporated_monomers
        return (
            new_lattice,
            new_unincorporated_monomers,
            new_incorporated_monomers,
            attempted_shrinkage,
        )
