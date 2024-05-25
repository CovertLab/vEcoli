"""
=========================
E. coli spatial composite
=========================

This composite is intended to run a spatial model of E. coli, where each
cellular compartment is designated as a node. These nodes are connected by
edges which specify how cellular compartments interface with each other.
Ecoli master processes could theoretically be run within each node to run
a spatially defined model of E. coli. Currently, this composite is only
set up to run a diffusion model on three nodes, the two cytosol poles and the
nucleoid, which are connected by two edges between the poles and the nucleoid.
Furthermore, since vivarium-ecoli is not yet fully migrated, this composite
initializes molecule counts from a snapshot of wcEcoli at t=0 s. This model
can be run with bulk molecules, polyribosomes, or both together. If only
polyribosomes are of interest, it is recommended to not include bulk
molecules for faster runtime.

If polyribosomes are included, one of three assumptions must be input to
define how polyribosomes' hydrodynamic radius is calculated. Each of these
assumption have significant limitations, however they likely indicate a
plausible range of polyribosomes' sizes. These assumptions are as follows:

* ``spherical``: This assumes that polyribosomes are spherical proteins
  and calculates the hydrodynamic radius from the total molecular weight
  of the mRNA molecules, attached ribosomes, and attached polypeptides.
  This is the default of the diffusion network process that is called
  here (and is the assumption for all bulk molecules); further details
  can be found in ``diffusion_network.py`` in ``vivarium-cell``.

  * :math:`r_p = 0.515*MW^{0.392}`
  * :math:`MW` = molecular weight

  Ref: Schuwirth et al., Science (2005)

* ``mrna``: This assumes that polyribosomes are solely the mRNA molecule and
  calculates the hydrodynamic radius of the mRNA molecule from the
  nucleotide count.

  * :math:`r_p = 5.5*N^{1/3}`
  * :math:`N` = number of nucleotides in mRNA

  Ref: Hyeon et al., J Chem Phys. (2006)

* ``linear``: This assumes that polyribosomes are solely the ribosomes and
  calculates the radius to be half the length of the summed sizes of
  ribosomes. This assumption does not have a reference

  * :math:`r_p = \\frac{n_ribosome * ribosome_size}{2}`

Since all molecules are treated as concentrations in the diffusion network
process, polyribosomes are bucketed into groups defined by the number of
ribosomes attached to each mRNA molecule.

This test case uses a mesh size of 50 nm, which is used by the diffusion
network process to scale diffusion constants to represent the impact that
a meshgrid formed by DNA in the nucleoid has on bulk molecule and polyribosome
diffusion.

Ref: Xiang et al., bioRxiv (2020)

Other ``vivarium-cell`` processes are also intended to be compatible with this
composite, but are unfinished or have not been incorporated. These processes
are ``growth_rate.py`` and ``spatial_geometry.py``.

"""

import argparse
import numpy as np
import math
import pickle
from wholecell.utils import units
from scipy import constants

from vivarium.core.composer import Composer
from vivarium.core.composition import simulate_composer

# processes
from ecoli.processes.spatiality.diffusion_network import DiffusionNetwork

# plots
from ecoli.plots.ecoli_spatial_plots import (
    plot_NT_availability,
    plot_nucleoid_diff,
    plot_large_molecules,
    plot_polyribosomes_diff,
    plot_molecule_characterizations,
)

from ecoli.states.wcecoli_state import get_state_from_file
from ecoli.library.schema import attrs, bulk_name_to_idx

SIM_DATA_PATH = "reconstruction/sim_data/kb/simData.cPickle"
RIBOSOME_SIZE = 21  # in nm


class EcoliSpatial(Composer):
    defaults = {
        "time_step": 2.0,
        "seed": 0,
        "sim_data_path": SIM_DATA_PATH,
        "nodes": [],
        "edges": {},
        "mesh_size": 50,  # in nm
        "radii": {},
        "temp": 310.15,  # in K
    }

    def __init__(self, config):
        super().__init__(config)
        self.seed = np.uint32(self.config["seed"] % np.iinfo(np.uint32).max)
        self.random_state = np.random.RandomState(seed=self.seed)
        self.sim_data_path = self.config["sim_data_path"]
        self.nodes = self.config["nodes"]
        self.edges = self.config["edges"]
        self.mesh_size = self.config["mesh_size"]
        self.time_step = self.config["time_step"]
        self.radii = self.config["radii"]
        self.temp = self.config["temp"]

        # load sim_data
        with open(self.sim_data_path, "rb") as sim_data_file:
            sim_data = pickle.load(sim_data_file)

        bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"]

        # molecular weight is converted to femtograms
        self.bulk_molecular_weights = {
            molecule_id: (
                sim_data.getter.get_mass(molecule_id) / constants.N_A
            ).asNumber(units.fg / units.mol)
            for molecule_id in bulk_ids
        }

        # unique molecule masses
        self.unique_masses = {}
        unique_molecular_masses = (
            sim_data.internal_state.unique_molecule.unique_molecule_masses
        )
        for id_, mass in zip(
            unique_molecular_masses["id"], unique_molecular_masses["mass"]
        ):
            self.unique_masses[id_] = sum(
                (mass / sim_data.constants.n_avogadro).asNumber(units.fg)
            )

    def initial_state(self, config):
        initial_state = get_state_from_file(
            path="data/wcecoli_t0.json",
        )

        if config["include_bulk"]:
            bulk = initial_state["bulk"].copy()
        else:
            bulk = np.empty(0, dtype=initial_state["bulk"].dtype)
            self.bulk_molecular_weights = {}
        if config["include_polyribosomes"]:
            polyribosome_assumption = config["polyribosome_assumption"]
            polyribosomes, polyribosomes_mw, polyribosomes_radii = add_polyribosomes(
                initial_state["unique"], self.unique_masses, polyribosome_assumption
            )
            bulk = np.append(
                bulk, np.array(list(polyribosomes.values()), dtype=bulk.dtype)
            )
            self.bulk_molecular_weights.update(polyribosomes_mw)
            self.radii.update(polyribosomes_radii)

        # Buckets half of cytosol labeled molecules into each pole
        cytosol_front = {
            str(bulk_mol[0]): (
                math.ceil(bulk_mol[1] / 2) if "[c]" in bulk_mol[0] else 0
            )
            for bulk_mol in bulk
        }
        cytosol_rear = {
            str(bulk_mol[0]): (
                math.floor(bulk_mol[1] / 2) if "[c]" in bulk_mol[0] else 0
            )
            for bulk_mol in bulk
        }
        nucleoid = {
            str(bulk_mol[0]): (bulk_mol[1] if "[n]" in bulk_mol[0] else 0)
            for bulk_mol in bulk
        }

        self.nodes["cytosol_front"]["molecules"] = cytosol_front
        self.nodes["nucleoid"]["molecules"] = nucleoid
        self.nodes["cytosol_rear"]["molecules"] = cytosol_rear

        initial_spatial_state = self.nodes
        return initial_spatial_state

    def generate_processes(self, config):
        diffusion_config = {
            "nodes": list(self.nodes.keys()),
            "edges": self.edges,
            "mw": self.bulk_molecular_weights,
            "mesh_size": self.mesh_size,
            "time_step": self.time_step,
            "radii": self.radii,
            "temp": self.temp,
        }
        return {"diffusion_network": DiffusionNetwork(diffusion_config)}

    def generate_topology(self, config):
        return {
            "diffusion_network": {node: (node,) for node in config["nodes"]},
        }


def add_polyribosomes(
    unique, unique_masses, polyribosome_assumption, save_output=False
):
    (is_full_transcript_rna, is_mrna, unique_index_rna, mrna_length, mrna_mass) = attrs(
        unique["RNA"],
        [
            "is_full_transcript",
            "is_mRNA",
            "unique_index",
            "transcript_length",
            "massDiff_mRNA",
        ],
    )

    # Remove ribosomes associated with RNAs that do not exist,
    # are not mRNAs, or have not been fully transcribed
    active_ribosomes = unique["active_ribosome"][
        unique["active_ribosome"]["_entryState"].astype(np.bool_)
    ]
    active_ribosomes = active_ribosomes[
        np.isin(
            active_ribosomes["mRNA_index"],
            unique_index_rna[is_mrna & is_full_transcript_rna],
        )
    ]
    mrna_index_ribosome_on_full_mrna, peptide_length_on_full_mrna = attrs(
        active_ribosomes, ["mRNA_index", "peptide_length"]
    )

    # Calculate number of ribosomes on each unique mRNA
    (n_ribosome_mrna_indices, mrna_ribo_inverse, n_ribosomes_on_full_mrna) = np.unique(
        mrna_index_ribosome_on_full_mrna, return_inverse=True, return_counts=True
    )

    # Get mRNA length and mass in same order as n_ribosomes_on_full_mrna
    mrna_mask = bulk_name_to_idx(n_ribosome_mrna_indices, unique_index_rna)
    mrna_length = mrna_length[mrna_mask]
    mrna_mass = mrna_mass[mrna_mask]
    # Calculate NT spacing + footprint of ribosomes per mRNA
    avg_NT_per_ribosome = mrna_length / n_ribosomes_on_full_mrna
    if save_output:
        plot_NT_availability(avg_NT_per_ribosome)

    # Defines buckets for unique polyribosomes to be combined into
    groups = [f"polyribosome_{i}[c]" for i in range(1, 10)] + ["polyribosome_>=10[c]"]

    # Groups polyribosomes based on number of ribosomes on mRNA
    # and calculates properties
    polyribosomes = {}
    mw = {}
    radii = {}
    # Since mass of ribosomes and polypeptides are accounted for in
    # unique molecules, ensure that bulk polyribosomes have zero mass
    zero_mass = (0,) * 9
    for i, group in enumerate(groups):
        if i < len(groups) - 1:
            group_mask = n_ribosomes_on_full_mrna == (i + 1)
        # Final group for all mRNAs with >=len(groups) ribosomes
        else:
            group_mask = n_ribosomes_on_full_mrna >= (i + 1)
        n_polyribosome_by_group = n_ribosomes_on_full_mrna[group_mask].sum()
        avg_n_ribosome_by_group = n_ribosomes_on_full_mrna[group_mask].mean()
        avg_mrna_mass = mrna_mass[group_mask].mean()
        avg_mrna_length = mrna_length[group_mask].mean()
        avg_peptide_length = (
            peptide_length_on_full_mrna[
                np.isin(mrna_ribo_inverse, np.nonzero(group_mask)[0])
            ].sum()
            / n_polyribosome_by_group
        )
        # Average MW of amino acid is 110 Da or 1.82659422e-7 fg
        mw[group] = (
            avg_mrna_mass
            + avg_n_ribosome_by_group * unique_masses["active_ribosome"]
            + avg_peptide_length * 1.82659422 * 10**-7
        )

        # Recalculates polyribosome size per input assumption
        if polyribosome_assumption == "linear":
            radii[group] = (avg_n_ribosome_by_group * RIBOSOME_SIZE) / 2
        if polyribosome_assumption == "mrna":
            radii[group] = 5.5 * avg_mrna_length**0.3333

        # Adds polyribosomes to bulk molecules and molecular weights
        polyribosomes[group] = (group, n_polyribosome_by_group) + zero_mass

    return polyribosomes, mw, radii


def test_spatial_ecoli(
    polyribosome_assumption="spherical",  # choose from 'mrna', 'linear', or 'spherical'
    total_time=10,  # in seconds
    return_data=False,
):
    ecoli_config = {
        "nodes": {
            "cytosol_front": {
                "length": 0.5,  # in um
                "volume": 0.25,  # in um^3
                "molecules": {},
            },
            "nucleoid": {
                "length": 1.0,
                "volume": 0.5,
                "molecules": {},
            },
            "cytosol_rear": {
                "length": 0.5,
                "volume": 0.25,
                "molecules": {},
            },
        },
        "edges": {
            "1": {
                "nodes": ["cytosol_front", "nucleoid"],
                "cross_sectional_area": np.pi * 0.3**2,  # in um^2
                "mesh": True,
            },
            "2": {
                "nodes": ["nucleoid", "cytosol_rear"],
                "cross_sectional_area": np.pi * 0.3**2,
                "mesh": True,
            },
        },
        "mesh_size": 50,  # in nm
        "time_step": 1,
    }

    ecoli = EcoliSpatial(ecoli_config)

    initial_config = {
        "include_bulk": False,
        "include_polyribosomes": True,
        "polyribosome_assumption": polyribosome_assumption,
    }

    settings = {
        "total_time": total_time,  # in s
        "initial_state": ecoli.initial_state(initial_config),
    }

    data = simulate_composer(ecoli, settings)

    if return_data:
        return ecoli, initial_config, data


def run_spatial_ecoli(
    polyribosome_assumption="linear",  # choose from 'mrna', 'linear', or 'spherical'
):
    ecoli, initial_config, output = test_spatial_ecoli(
        polyribosome_assumption=polyribosome_assumption,
        total_time=5 * 60,
        return_data=True,
    )
    mesh_size = ecoli.config["mesh_size"]
    nodes = ecoli.config["nodes"]
    plot_molecule_characterizations(ecoli, initial_config)
    if initial_config["include_bulk"]:
        mol_ids = ["ADHE-CPLX[c]", "CPLX0-3962[c]", "CPLX0-3953[c]"]
        plot_large_molecules(output, mol_ids, mesh_size, nodes)
    if initial_config["include_polyribosomes"]:
        filename = (
            "out/polyribosome_diffusion_" + initial_config["polyribosome_assumption"]
        )
        plot_polyribosomes_diff(output, mesh_size, nodes, filename)
    if initial_config["include_polyribosomes"] and not initial_config["include_bulk"]:
        plot_nucleoid_diff(output, nodes, initial_config["polyribosome_assumption"])


def main():
    parser = argparse.ArgumentParser(description="ecoli spatial")
    parser.add_argument(
        "-mrna", "-m", action="store_true", default=False, help="mrna assumption"
    )
    parser.add_argument(
        "-linear", "-l", action="store_true", default=False, help="linear assumption"
    )
    parser.add_argument(
        "-spherical",
        "-s",
        action="store_true",
        default=False,
        help="spherical assumption",
    )
    args = parser.parse_args()

    if args.mrna:
        run_spatial_ecoli("mrna")
    if args.linear:
        run_spatial_ecoli("linear")
    if args.spherical:
        run_spatial_ecoli("spherical")


if __name__ == "__main__":
    main()
