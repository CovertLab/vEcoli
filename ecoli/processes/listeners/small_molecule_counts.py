"""
Small Molecule Counts Listener

Records the total counts of all trackable intracellular small molecules at each
timestep, plus extracellular small molecule concentrations in the media (NOTE:
the free counts for all small molecules is accessible via the bulk container).

The tracked set is built from the model's reaction network (see below), so it
spans the cell's small molecules broadly (i.e. metabolites, amino acids,
nucleotides (NTPs/dNTPs), cofactors, and inorganic ions).

reconstruction/ecoli/flat/metabolites.tsv has ~7,600 metabolite compounds,
but most never appear as a countable species in a running simulation. This
listener tracks every small molecule that both (a) exists as a bulk molecule
(has a countable id) AND (b) participates in an encoded reaction within the model.
The tracked set is built in the get_small_molecule_counts_listener_config()
function within sim_data.py as:

    ( homeostatic concentration targets (across all saved media)
      ∪ equilibrium-reaction ligands  (equilibrium.metabolite_set)
      ∪ every species in metabolism.reaction_stoich)
      ∩  bulk molecule ids (filters by countable species)
      −  macromolecules  (proteins, RNAs, and equilibrium / TCS / complexation
        complexes that get looped in from equilibrium and TCS stoich maps and
        need to be removed as they are tracked elsewhere)

This results in roughly ~4,950 (compound + compartment combos) distinct species
tracked (~2,200 distinct compounds). Stepwise for one example run:
    union ≈ 6,160 ids  ->  ∩ bulk ≈ 5,030  ->  − ~83 macromolecules  ≈  4,950.
Note: each compound is tracked separately per compartment, so ATP[c], ATP[e],
and ATP[p] are tracked within the 4,950 as three distinct species, for example.

The exact tracked list for a given run is attached as metadata on the
totalSmallMoleculeCounts field, so the ID list can be read back by passing the
field's name to field_metadata: ``field_metadata(conn, config_sql,
"listeners__small_molecule_counts__totalSmallMoleculeCounts")``.

In the default model media conditions, only ~170 of the tracked small molecules
are present in nonzero amounts over the course of the simulation (as of June 2026);
the rest exist in the reaction network but sit at zero count in that condition.

Arrays emitted: totalSmallMoleculeCounts and environmentSmallMoleculeConcentrations.

Total counts are computed by adding back small molecules currently sequestered
in non-bulk locations:

  1. Equilibrium complexes (e.g., one-component and two-component systems bound
     to ligands): ligands are bound inside these complexes and are released
     upon dissociation. Unpacked via equilibrium.stoich_matrix_monomers().

  2. Two component system (TCS) complexes (PHOSPHO-HK, PHOSPHO-RR, etc.):
     Each phosphorylated TCS molecule carries exactly 1 Pi[c] covalently.
     ATP is consumed and ADP is released to the free pool during
     phosphorylation, so only Pi needs to be recovered here.

  3. TCS complexes that contain equilibrium complexes as subunits
     (e.g. PHOSPHO-HK-LIGAND contains the HK-LIGAND eq complex, which
     contains a small molecule ligand). When HK-LIGAND is phosphorylated, the
     ligand is no longer counted by the equilibrium unpacking (HK-LIGAND
     count dropped) but is still sequestered in the TCS complex, so it is
     recovered here separately to maintain the correct sequestration breakdown
     between equilibrium and TCS complexes.

  4. TCS and equilibrium complexes can become bound to transcription units
     on DNA, so small molecules are technically in bound transcription factors
     (TFs) as well (which are tracked in the promoters table as the bound_TF field).

Complexation complexes currently only contain protein subunits.

Extracellular small molecules are stored as concentrations (not counts) because
there is no single cell volume to convert with at the environment level.
These use a separate molecule ID list (environment_sm_ids).
"""

import numpy as np
from vivarium.core.process import Step

from ecoli.library.schema import numpy_schema, counts, bulk_name_to_idx, attrs
from ecoli.processes.registries import topology_registry

NAME = "small_molecule_counts_listener"
TOPOLOGY = {
    "listeners": ("listeners",),
    "bulk": ("bulk",),
    "promoters": ("unique", "promoter"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}
topology_registry.register(NAME, TOPOLOGY)


class SmallMoleculeCounts(Step):
    """
    Listener for total counts of intracellular small molecules,
    including small molecules within complexes and bound transcription factors
    (which can be both equilibrium complexes and TCS complexes).
    """

    name = NAME
    topology = TOPOLOGY

    defaults = {
        "bulk_molecule_ids": [],
        "sm_ids": [],
        "equilibrium_molecule_ids": [],
        "equilibrium_complex_ids": [],
        "equilibrium_stoich": [],
        "phosphorylated_tcs_mol_ids": [],
        "tcs_ligand_complex_ids": [],
        "tcs_ligand_sm_ids": [],
        "tcs_ligand_stoichs": [],
        "tf_bound_col_idxs": [],
        "tf_bound_sm_ids": [],
        "tf_bound_stoichs": [],
        "pi_id": "Pi[c]",
        "environment_sm_ids": [],
        "emit_unique": False,
        "time_step": 1,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Obtain the bulk molecule ids and tracked small molecule ids (decided
        # by get_small_molecule_counts_listener_config in sim_data.py):
        self.bulk_molecule_ids = self.parameters["bulk_molecule_ids"]
        self.sm_ids = self.parameters["sm_ids"]

        # Equilibrium complex unpacking (to obtain counts of small molecules
        # bound in equilibrium complexes as ligands):
        self.equilibrium_molecule_ids = self.parameters["equilibrium_molecule_ids"]
        self.equilibrium_complex_ids = self.parameters["equilibrium_complex_ids"]
        self.equilibrium_stoich = np.array(self.parameters["equilibrium_stoich"])

        # Build submatrix (rows in equilibrium_stoich that correspond to
        # tracked small molecules, and their mapping to the sm_ids index):
        sm_to_idx = {m: i for i, m in enumerate(self.sm_ids)}
        eq_sm_rows = []
        eq_sm_to_tracked_idx = []
        for row_idx, mol_name in enumerate(self.equilibrium_molecule_ids):
            if mol_name in sm_to_idx:
                eq_sm_rows.append(row_idx)
                eq_sm_to_tracked_idx.append(sm_to_idx[mol_name])

        self.eq_sm_rows = np.array(eq_sm_rows)
        self.eq_sm_to_tracked_idx = np.array(eq_sm_to_tracked_idx)

        if len(self.eq_sm_rows) > 0:
            self.eq_sm_stoich = self.equilibrium_stoich[self.eq_sm_rows, :]
        else:
            self.eq_sm_stoich = np.zeros(
                (0, len(self.equilibrium_complex_ids)), dtype=np.float64
            )

        # Unpack Pi molecules from TCS complexes (each phosphorylated TCS complex
        # (PHOSPHO-HK, PHOSPHO-RR, etc.) carries exactly 1 Pi[c]):
        self.phosphorylated_tcs_mol_ids = self.parameters["phosphorylated_tcs_mol_ids"]
        pi_id = self.parameters["pi_id"]
        if pi_id in sm_to_idx:
            self.pi_sm_idx = sm_to_idx[pi_id]
            self.track_tcs_pi = True
        else:
            self.pi_sm_idx = -1
            self.track_tcs_pi = False

        # Account for small molecule ligands sequestered in TCS complexes (via
        # equilibrium complex subunits):
        self.tcs_ligand_complex_ids = list(self.parameters["tcs_ligand_complex_ids"])
        tcs_ligand_sm_ids = list(self.parameters["tcs_ligand_sm_ids"])
        self.tcs_ligand_stoichs = np.array(
            self.parameters["tcs_ligand_stoichs"], dtype=np.float64
        )
        self.tcs_ligand_sm_idxs = np.array(
            [sm_to_idx[m] for m in tcs_ligand_sm_ids], dtype=np.int64
        )
        self.track_tcs_ligands = len(self.tcs_ligand_complex_ids) > 0

        # Small molecule ligands inside transcription factors bound to DNA
        # (includes equilibrium ligands and Pi[c]):
        self.tf_bound_col_idxs = np.array(
            self.parameters["tf_bound_col_idxs"], dtype=np.int64
        )
        tf_bound_sm_ids = list(self.parameters["tf_bound_sm_ids"])
        self.tf_bound_sm_idxs = np.array(
            [sm_to_idx[m] for m in tf_bound_sm_ids], dtype=np.int64
        )
        self.tf_bound_stoichs = np.array(
            self.parameters["tf_bound_stoichs"], dtype=np.float64
        )
        self.track_bound_tf_sm = len(self.tf_bound_col_idxs) > 0

        # Extracellular small molecules:
        self.environment_sm_ids = self.parameters["environment_sm_ids"]

        # Initialized bulk indices:
        self.sm_idx = None

    def ports_schema(self):
        return {
            "listeners": {
                "small_molecule_counts": {
                    "totalSmallMoleculeCounts": {
                        "_default": [],
                        "_updater": "set",
                        "_emit": True,
                        "_properties": {"metadata": self.sm_ids},
                    },
                    "environmentSmallMoleculeConcentrations": {
                        "_default": [],
                        "_updater": "set",
                        "_emit": True,
                        "_properties": {"metadata": self.environment_sm_ids},
                    },
                }
            },
            "bulk": numpy_schema("bulk"),
            "promoters": numpy_schema("promoters", emit=self.parameters["emit_unique"]),
            "global_time": {"_default": 0.0},
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def next_update(self, timestep, states):
        # Initialize bulk indices on first call:
        if self.sm_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.sm_idx = bulk_name_to_idx(self.sm_ids, bulk_ids)
            self.equilibrium_complex_idx = bulk_name_to_idx(
                self.equilibrium_complex_ids, bulk_ids
            )
            self.tcs_phosphorylated_idx = bulk_name_to_idx(
                self.phosphorylated_tcs_mol_ids, bulk_ids
            )
            if self.track_tcs_ligands:
                self.tcs_ligand_complex_idx = bulk_name_to_idx(
                    self.tcs_ligand_complex_ids, bulk_ids
                )
            self.environment_sm_idx = bulk_name_to_idx(
                self.environment_sm_ids, bulk_ids, strict=False
            )

        bulk_counts_arr = counts(states["bulk"], self.sm_idx)

        # Add up total counts, starting with free counts:
        free_sm_counts = bulk_counts_arr.copy()

        # Counts for small molecule ligands bound in equilibrium complexes:
        eq_bound = np.zeros(len(self.sm_ids), np.int64)
        if len(self.eq_sm_rows) > 0:
            eq_complex_counts = counts(states["bulk"], self.equilibrium_complex_idx)
            bound = np.dot(self.eq_sm_stoich, np.negative(eq_complex_counts))
            eq_bound[self.eq_sm_to_tracked_idx] += bound.astype(np.int64)

        # Counts of Pi[c] in TCS complexes:
        tcs_bound = np.zeros(len(self.sm_ids), np.int64)
        if self.track_tcs_pi:
            n_phosphorylated = int(
                counts(states["bulk"], self.tcs_phosphorylated_idx).sum()
            )
            tcs_bound[self.pi_sm_idx] = n_phosphorylated

        # Counts of small molecule ligands in TCS complexes:
        tcs_complex_bound = np.zeros(len(self.sm_ids), np.int64)
        if self.track_tcs_ligands:
            tcs_cplx_counts = counts(states["bulk"], self.tcs_ligand_complex_idx)
            for i in range(len(self.tcs_ligand_complex_ids)):
                tcs_complex_bound[self.tcs_ligand_sm_idxs[i]] += int(
                    tcs_cplx_counts[i] * self.tcs_ligand_stoichs[i]
                )

        # Counts of small molecules within DNA-bound transcription factors:
        bound_tf_sm = np.zeros(len(self.sm_ids), np.int64)
        if self.track_bound_tf_sm:
            (bound_TF,) = attrs(states["promoters"], ["bound_TF"])
            n_bound_per_tf = bound_TF.astype(np.int64).sum(axis=0)
            for i in range(len(self.tf_bound_col_idxs)):
                col = self.tf_bound_col_idxs[i]
                bound_tf_sm[self.tf_bound_sm_idxs[i]] += int(
                    n_bound_per_tf[col] * self.tf_bound_stoichs[i]
                )

        # Total counts :
        total_sm_counts = (
            free_sm_counts + eq_bound + tcs_bound + tcs_complex_bound + bound_tf_sm
        )

        # Extracellular concentrations (the environment bulk entries store
        # concentrations directly here):
        if len(self.environment_sm_idx) > 0:
            env_conc = counts(states["bulk"], self.environment_sm_idx).astype(
                np.float64
            )
        else:
            env_conc = np.zeros(len(self.environment_sm_ids), np.float64)

        return {
            "listeners": {
                "small_molecule_counts": {
                    "totalSmallMoleculeCounts": total_sm_counts,
                    "environmentSmallMoleculeConcentrations": env_conc,
                }
            }
        }


def test_small_molecule_counts_listener():
    from ecoli.experiments.ecoli_master_sim import EcoliSim

    sim = EcoliSim.from_file()
    sim.max_duration = 2
    sim.raw_output = False
    sim.build_ecoli()
    sim.run()
    listeners = sim.query()["agents"]["0"]["listeners"]
    assert "small_molecule_counts" in listeners
    assert isinstance(
        listeners["small_molecule_counts"]["totalSmallMoleculeCounts"][0], list
    )


if __name__ == "__main__":
    test_small_molecule_counts_listener()
