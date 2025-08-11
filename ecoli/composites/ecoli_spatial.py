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
import matplotlib.pyplot as plt
import pickle
from wholecell.utils import units
from scipy import constants

from vivarium.core.composer import Composer
from vivarium.core.composition import simulate_composer

# processes
from ecoli.processes.spatiality.diffusion_network import (
    compute_diffusion_constants_from_rp,
    calculate_rp_from_mw,
    DiffusionNetwork,
)

from ecoli.library.schema import attrs, bulk_name_to_idx
from ecoli.library.sim_data import LoadSimData

RIBOSOME_SIZE = 21  # in nm


class EcoliSpatial(Composer):
    defaults = {
        "time_step": 2.0,
        "seed": 0,
        "sim_data_path": "out/kb/simData.cPickle",
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
        self.load_sim_data = LoadSimData(self.sim_data_path)

        bulk_ids = self.load_sim_data.sim_data.internal_state.bulk_molecules.bulk_data[
            "id"
        ]

        # molecular weight is converted to femtograms
        self.bulk_molecular_weights = {
            molecule_id: (
                self.load_sim_data.sim_data.getter.get_mass(molecule_id) / constants.N_A
            ).asNumber(units.fg / units.mol)
            for molecule_id in bulk_ids
        }

        # unique molecule masses
        self.unique_masses = {}
        unique_molecular_masses = self.load_sim_data.sim_data.internal_state.unique_molecule.unique_molecule_masses
        for id_, mass in zip(
            unique_molecular_masses["id"], unique_molecular_masses["mass"]
        ):
            self.unique_masses[id_] = sum(
                (mass / self.load_sim_data.sim_data.constants.n_avogadro).asNumber(
                    units.fg
                )
            )

    def initial_state(self, config):
        initial_state = self.load_sim_data.generate_initial_state()

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


def check_fig_pickle(path):
    try:
        fig = pickle.load(open(path, "rb"))
    except (OSError, IOError):
        fig = plt.figure()
        pickle.dump(fig, open(path, "wb"))
    return fig


"""
These plots are all intended to be run on `ecoli_spatial.py` and can be called
from `run_spatial_ecoli()`. It is important to note that these plots are
specific to the configuration defined in `test_spatial_ecoli.py`. A few of
of these plotting functions that compare output for different polyribosome
assumptions should be run in a specific way to achieve the desired output plot.
The simulation needs to be run three times each time with a different polyribosome
assumption declared in the configuration. For intended output, start with the
spherical assumption, then the mrna assumption and lastly the linear assumption.
"""


# Plots the diffusion of a single molecule
def plot_single_molecule_diff(output, mol_id, mesh_size, nodes):
    plt.figure()
    plt.plot(
        output["time"],
        np.divide(
            output["cytosol_front"]["molecules"][mol_id],
            nodes["cytosol_front"]["volume"],
        ),
        color="#d8b365",
    )
    plt.plot(
        output["time"],
        np.divide(output["nucleoid"]["molecules"][mol_id], nodes["nucleoid"]["volume"]),
        color="#5ab4ac",
    )
    plt.xlabel("time (s)")
    plt.ylabel(r"Concentration (molecules / $\mu m^3$)")
    plt.title(f"Diffusion of {mol_id} over compartments with {mesh_size} nm mesh")
    plt.legend(["Cytosol front", "Nucleoid"])
    out_file = "out/single_molecule.png"
    plt.savefig(out_file)


# Plots the diffusion of a set of three large molecules
def plot_large_molecules(output, mol_ids, mesh_size, nodes):
    plt.figure()
    linestyle = ["solid", "dashed", "dotted"]
    legend = []
    for i, mol_id in enumerate(mol_ids):
        plt.plot(
            output["time"],
            np.divide(
                np.divide(
                    output["cytosol_front"]["molecules"][mol_id],
                    nodes["cytosol_front"]["volume"],
                ),
                output["cytosol_front"]["molecules"][mol_id][0],
            ),
            color="#d8b365",
            linestyle=linestyle[i],
        )
        plt.plot(
            output["time"],
            np.divide(
                np.divide(
                    output["nucleoid"]["molecules"][mol_id], nodes["nucleoid"]["volume"]
                ),
                output["cytosol_front"]["molecules"][mol_id][0],
            ),
            color="#5ab4ac",
            linestyle=linestyle[i],
        )
        legend.append(f"{mol_id} in pole")
        legend.append(f"{mol_id} in nucleoid")
    plt.xlabel("time (s)")
    plt.ylabel("Normalized concentration (% total concentration)")
    plt.title(
        f"Diffusion of large molecules over compartments with {mesh_size} nm mesh"
    )
    plt.legend(legend)
    out_file = "out/large_molecules.png"
    plt.savefig(out_file)


# Plots nucleoid diffusion of polyribosomes on same plot for different assumptions/runs
def plot_nucleoid_diff(output, nodes, polyribosome_assumption):
    x = np.arange(1, 11)
    total_molecules = (
        array_from(output["nucleoid"]["molecules"])[:, 0]
        + array_from(output["cytosol_front"]["molecules"])[:, 0]
        + array_from(output["cytosol_rear"]["molecules"])[:, 0]
    )
    if polyribosome_assumption == "mrna":
        fig = check_fig_pickle("out/nucleoid_diff.pickle")
        plt.plot(
            x,
            np.average(array_from(output["nucleoid"]["molecules"]), axis=1)
            / total_molecules
            * nodes["nucleoid"]["volume"]
            / nodes["cytosol_front"]["volume"],
            color="#5ab4ac",
        )
        pickle.dump(fig, open("out/nucleoid_diff.pickle", "wb"))
    elif polyribosome_assumption == "linear":
        fig = check_fig_pickle("out/nucleoid_diff.pickle")
        plt.plot(
            x,
            np.average(array_from(output["nucleoid"]["molecules"]), axis=1)
            / total_molecules
            * nodes["nucleoid"]["volume"]
            / nodes["cytosol_front"]["volume"],
            color="#018571",
        )
        pickle.dump(fig, open("out/nucleoid_diff.pickle", "wb"))
    else:  # spherical assumption
        fig = check_fig_pickle("out/nucleoid_diff.pickle")
        plt.plot(
            x,
            np.average(array_from(output["nucleoid"]["molecules"]), axis=1)
            / total_molecules
            * nodes["nucleoid"]["volume"]
            / nodes["cytosol_front"]["volume"],
            color="#d8b365",
        )
        labels = [str(val) if val < 10 else str(val) + "+" for val in np.arange(1, 11)]

        plt.xticks(np.arange(1, 11), labels)
        plt.xlabel("Number of ribosomes on polyribosome")
        plt.ylabel("Ratio of nucleoid localization to pole localization")
        handles = [
            plt.Line2D([0], [0], color=c) for c in ["#d8b365", "#5ab4ac", "#018571"]
        ]
        plt.legend(
            handles,
            [
                "Spherical protein assumption",
                "mRNA assumption",
                "Linear ribosomes assumption",
            ],
        )
        plt.title("Polyribosome nucleoid localization over 5 min")
        pickle.dump(fig, open("out/nucleoid_diff.pickle", "wb"))
    out_file = "out/nucleoid_diff.png"
    plt.savefig(out_file, dpi=300)


# Plots diffusion of polyribosomes in a pole and the nucleoid
def plot_polyribosomes_diff(output, mesh_size, nodes, filename):
    plt.figure()
    groups = [
        "polyribosome_1[c]",
        "polyribosome_2[c]",
        "polyribosome_3[c]",
        "polyribosome_4[c]",
        "polyribosome_5[c]",
        "polyribosome_6[c]",
        "polyribosome_7[c]",
        "polyribosome_8[c]",
        "polyribosome_9[c]",
        "polyribosome_>=10[c]",
    ]
    colors = [
        "#543005",
        "#8c510a",
        "#bf812d",
        "#dfc27d",
        "#f6e8c3",
        "#c7eae5",
        "#80cdc1",
        "#35978f",
        "#01665e",
        "#003c30",
    ]
    time = np.divide(output["time"], 60)
    for i, mol_id in enumerate(groups):
        plt.plot(
            time,
            np.divide(
                output["cytosol_front"]["molecules"][mol_id],
                output["cytosol_front"]["molecules"][mol_id][0],
            ),
            linestyle="dashed",
            color=colors[i],
            label=str(mol_id + " in pole"),
        )
        plt.plot(
            time,
            np.divide(
                np.divide(
                    output["nucleoid"]["molecules"][mol_id],
                    output["cytosol_front"]["molecules"][mol_id][0],
                ),
                nodes["nucleoid"]["volume"] / nodes["cytosol_front"]["volume"],
            ),
            color=colors[i],
            label=str(mol_id + " in nucleoid"),
        )
    plt.xlabel("time (min)")
    plt.ylabel("Normalized concentration (% total concentration)")
    plt.title(f"Diffusion of polyribosomes with mesh of {str(mesh_size)} nm")
    out_file = filename or "out/polyribosomes.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)


# Plots the average nucleotide availability
def plot_NT_availability(avg_NT_per_ribosome):
    """
    Note that ribosomes are allowed to overlap with each other in wcEcoli, which
    is why this is an important analysis to look at. The average footprint of
    ribosomes in E. coli on mRNA is assumed to be about 25 on average with a
    range of 15-40 NT.

        - Ref: Mohammad et al., eLife (2019)
    """
    plt.figure()
    tot = len(avg_NT_per_ribosome)
    (counts, bins) = np.histogram(avg_NT_per_ribosome, bins=np.arange(25, 3500, 50))
    plt.hist(bins[:-1], bins, weights=counts / tot, color="#5ab4ac")
    plt.xlabel("Average number of available NTs per ribosome")
    plt.ylabel("Percentage of total number of polyribosomes (%)")
    plt.title(f"Available NTs per ribosome on polyribosomes (n = {tot})")
    out_file = "out/avg_NT_per_ribosomes.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)

    plt.figure()
    tot = len(avg_NT_per_ribosome)
    (counts, bins) = np.histogram(avg_NT_per_ribosome, bins=np.arange(25, 100))
    plt.hist(bins[:-1], bins, weights=counts / tot, color="#5ab4ac")
    plt.axvline(x=15, color="k", linestyle="dashed")
    plt.axvline(x=40, color="k", linestyle="dashed")
    plt.xlabel("Average number of available NTs per ribosome")
    plt.ylabel("Percentage of total number of polyribosomes (%)")
    plt.title(f"Available NTs per ribosome on polyribosomes (n = {tot})")
    out_file = "out/avg_NT_per_ribosomes_zoom.png"
    handles = [plt.Line2D([0], [0], color=c, linestyle="dashed") for c in ["k"]]
    plt.legend(handles, ["Minimum and maximum ribosome footprints"])
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)


# Performs calculations for and calls bulk and polyribosome molecules characterization plots
def plot_molecule_characterizations(ecoli, initial_config):
    molecule_ids = ecoli.config["nodes"]["cytosol_front"]["molecules"].keys()
    mesh_size = ecoli.config["mesh_size"]
    rp = calculate_rp_from_mw(molecule_ids, ecoli.bulk_molecular_weights)
    ecoli.config["edges"]["3"] = {
        "nodes": ["cytosol_front", "cytosol_rear"],
        "cross_sectional_area": np.pi * 0.3**2,
    }
    molecule_ids = np.asarray(list(molecule_ids))
    for mol_id, r in ecoli.config["radii"].items():
        rp[np.where(molecule_ids == mol_id)[0][0]] = r
    dc = compute_diffusion_constants_from_rp(
        ecoli.config["nodes"]["cytosol_front"]["molecules"].keys(),
        rp,
        ecoli.config["mesh_size"],
        ecoli.config["edges"],
        ecoli.config["temp"],
    )

    if initial_config["include_bulk"]:
        cytosol_mask = [
            i
            for i, mol_id in enumerate(
                ecoli.config["nodes"]["cytosol_front"]["molecules"].keys()
            )
            if "[c]" in mol_id
        ]
        rp_cytosol = np.asarray(rp)[cytosol_mask]
        dc_cytosol = array_from(dc["1"])[cytosol_mask]
        dc_cytosol_no_mesh = array_from(dc["3"])[cytosol_mask]
        plot_bulk_molecules(rp_cytosol, dc_cytosol, dc_cytosol_no_mesh, mesh_size)

    if initial_config["include_polyribosomes"]:
        polyribosome_mask = [
            i
            for i, mol_id in enumerate(
                ecoli.config["nodes"]["cytosol_front"]["molecules"].keys()
            )
            if "polyribosome" in mol_id
        ]
        rp_polyribosome = np.asarray(rp)[polyribosome_mask]
        dc_polyribosome = array_from(dc["1"])[polyribosome_mask]
        dc_polyribosome_no_mesh = array_from(dc["3"])[polyribosome_mask]
        total_molecules_polyribosome = np.add(
            array_from(ecoli.config["nodes"]["cytosol_front"]["molecules"])[
                polyribosome_mask
            ],
            array_from(ecoli.config["nodes"]["cytosol_rear"]["molecules"])[
                polyribosome_mask
            ],
        )
        polyribosome_assumption = initial_config["polyribosome_assumption"]
        plot_polyribosomes(
            rp_polyribosome,
            rp_polyribosome,
            dc_polyribosome,
            dc_polyribosome_no_mesh,
            total_molecules_polyribosome,
            ecoli.config["mesh_size"],
            polyribosome_assumption,
        )


# Plots characteristics of polyribosomes: counts, sizes, and diffusion constants
def plot_polyribosomes(
    rp, radii, dc, dc_no_mesh, total_molecules, mesh_size, polyribosome_assumption
):
    x = np.arange(1, 11)
    labels = [str(val) if val < 10 else str(val) + "+" for val in np.arange(1, 11)]
    if polyribosome_assumption == "mrna":
        fig = check_fig_pickle("out/polyribosomes_sizes.pickle")
        plt.plot(x, np.multiply(radii, 2), color="#5ab4ac")
        pickle.dump(fig, open("out/polyribosomes_sizes.pickle", "wb"))
    elif polyribosome_assumption == "linear":
        fig = check_fig_pickle("out/polyribosomes_sizes.pickle")
        plt.plot(x, np.multiply(radii, 2), color="#018571")
        pickle.dump(fig, open("out/polyribosomes_sizes.pickle", "wb"))
    else:  # spherical assumption
        fig = check_fig_pickle("out/polyribosomes_sizes.pickle")
        mesh = np.full(len(x), mesh_size)
        plt.plot(x, np.multiply(rp, 2), color="#d8b365")

        plt.plot(x, mesh, linestyle="dashed", color="k")
        plt.xticks(np.arange(1, 11), labels)
        plt.xlabel("Number of ribosomes")
        plt.ylabel("Polyribosome size (nm)")
        plt.title("Sizes of polyribosomes")
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=c)
            for c in ["#d8b365", "#5ab4ac", "#018571"]
        ]
        plt.legend(
            handles,
            [
                "spherical protein assumption",
                "mRNA assumption",
                "linear ribosomes assumption",
                "mesh size",
            ],
        )
        pickle.dump(fig, open("out/polyribosomes_sizes.pickle", "wb"))

    out_file = "out/polyribosomes_sizes.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)

    fig = plt.figure()
    # Note: this value is hardcoded
    avg_num_ribosomes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11.87]
    tot = sum(total_molecules)
    total_ribosomes = np.multiply(total_molecules, avg_num_ribosomes)
    tot_rib = np.sum(total_ribosomes)
    plt.bar(x - 0.2, total_molecules / tot, width=0.4, color="#5ab4ac", align="center")
    plt.bar(
        x + 0.2, total_ribosomes / tot_rib, width=0.4, color="#d8b365", align="center"
    )
    plt.xticks(np.arange(1, 11), labels)
    plt.xlabel("Number of ribosomes")
    plt.ylabel("Percentage of total count (%)")
    plt.legend(["Polyribosomes", "70S ribosomes"])
    plt.title(f"Polyribosome counts (n = {tot})")
    out_file = "out/polyribosomes_counts.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)

    if polyribosome_assumption == "linear":
        fig = check_fig_pickle("out/polyribosomes_dc.pickle")
        plt.plot(x, dc, color="#018571")
        plt.plot(x, dc_no_mesh, color="#018571", linestyle="dashed")
        plt.legend(
            [
                "spherical protein assumption: 50 nm mesh",
                "spherical protein assumption: no mesh",
                "mRNA assumption: 50 nm mesh",
                "mRNA assumption: no mesh",
                "linear ribosomes assumption: 50 nm mesh",
                "linear ribosomes assumption: no mesh",
            ]
        )
    elif polyribosome_assumption == "mrna":
        fig = check_fig_pickle("out/polyribosomes_dc.pickle")
        plt.plot(x, dc, color="#5ab4ac")
        plt.plot(x, dc_no_mesh, color="#5ab4ac", linestyle="dashed")
    else:
        fig = check_fig_pickle("out/polyribosomes_dc.pickle")
        tot = len(dc)
        plt.plot(x, dc, color="#d8b365")
        plt.plot(x, dc_no_mesh, color="#d8b365", linestyle="dashed")
        plt.xticks(np.arange(1, 11), labels)
        plt.xlabel("Number of ribosomes")
        plt.ylabel(r"Diffusion constant ($\mu m^2 / s$)")
        plt.yscale("log")
        plt.title(f"Diffusion constants of polyribosomes (n = {tot})")
        plt.legend(
            [
                "spherical protein assumption: 50 nm mesh",
                "spherical protein assumption: no mesh",
                "mRNA assumption: 50 nm mesh",
                "mRNA assumption: no mesh",
                "linear ribosomes assumption: 50 nm mesh",
                "linear ribosomes assumption: no mesh",
            ]
        )
        pickle.dump(fig, open("out/polyribosomes_dc.pickle", "wb"))
    out_file = "out/polyribosomes_dc.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)


# Plots characteristics of bulk molecules: sizes and diffusion constants
def plot_bulk_molecules(rp, dc, dc_no_mesh, mesh_size):
    plt.figure()
    tot = len(rp)
    size = np.round(np.multiply(rp, 2))
    (counts, bins) = np.histogram(size, bins=range(int(max(size))))
    plt.hist(bins[:-1], bins, weights=counts / tot, color="#5ab4ac")
    plt.xlabel("Molecule size (nm)")
    plt.ylabel("Percentage of total number of molecules (%)")
    plt.title(f"Sizes of bulk molecules (n = {tot})")
    out_file = "out/bulk_molecules_sizes.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)

    plt.figure()
    new_size, new_dc, new_dc_no_mesh = zip(*sorted(zip(size, dc, dc_no_mesh)))
    new_size, idx = np.unique(new_size, return_index=True)
    plt.plot(new_size, np.asarray(new_dc)[idx], color="#d8b365")
    plt.plot(new_size, np.asarray(new_dc_no_mesh)[idx], color="#5ab4ac")
    plt.ylabel(r"Diffusion constant ($\mu m^2 / s$)")
    plt.yscale("log")
    plt.xlabel("Molecule size (nm)")
    plt.title(f"Diffusion constants of bulk molecules (n = {tot})")
    plt.legend([f"with {str(mesh_size)} nm mesh", "without mesh"])
    out_file = "out/bulk_molecules_dc.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)


def array_from(d):
    return np.array(list(d.values()))


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
