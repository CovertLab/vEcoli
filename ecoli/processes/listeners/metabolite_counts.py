"""
=========================
Metabolite Counts Listener
=========================

Tracks free and total counts of all intracellular metabolites at each
timestep, plus extracellular metabolite concentrations in the media.

Only THREE arrays are emitted: freeMetaboliteCounts, totalMetaboliteCounts, and
environmentMetaboliteConcentrations. The per-category sequestration breakdown
(eq complexes / TCS phospho / TCS complexes / bound TFs) is NOT stored -- it is
redundant, since each component is just (free complex/phospho/bound-TF count) x
stoich. Analysis scripts can recompute these from the bulk complex counts,
the unique-promoter bound_TF counts, and the shared stoich maps in
ecoli.library.metabolite_sequestration. The components are still computed here
internally because totalMetaboliteCounts needs them.

Total counts are computed by adding back metabolites currently sequestered
in non-bulk locations:

  1. Equilibrium complexes (e.g., TF-ligand, 2CS-ligand bound forms):
     Metabolite ligands are bound inside these complexes and are released
     upon dissociation. Unpacked via equilibrium.stoich_matrix_monomers().

  2. TCS phosphorylated molecules (PHOSPHO-HK, PHOSPHO-RR, etc.):
     Each phosphorylated TCS molecule carries exactly 1 Pi[c] covalently.
     ATP is consumed and ADP is released to the free pool during
     phosphorylation, so only Pi needs to be recovered here.

  3. TCS complexes that contain equilibrium complexes as subunits
     (e.g. PHOSPHO-HK-LIGAND contains the HK-LIGAND eq complex, which
     contains a metabolite LIGAND). When HK-LIGAND is phosphorylated, the
     ligand is no longer counted by the equilibrium unpacking (HK-LIGAND
     count dropped) but is still sequestered in the TCS complex, so it is
     recovered here separately.

    4. TCS and equilibrium complexes can become bound to transcription units
     on DNA, so metabolites are technically in bound transcription factors (TFs)
     as well.

Complexation complexes currently only contain protein subunits.

Four component arrays are stored so one can see exactly where deviations
from the free pool are coming from:
  - freeMetaboliteCounts
  - metabolitesInEquilibriumComplexes
  - metabolitesInTCSPhosphorylation
  - metabolitesInTCSComplexes
  - metabolitesInBoundTFs
  - totalMetaboliteCounts  (sum of the four above)

Extracellular metabolites are stored as concentrations (not counts) because
there is no single cell volume to convert with at the environment level.
These use a separate molecule ID list (environment_metabolite_ids).
"""

import numpy as np
from vivarium.core.process import Step

from ecoli.library.schema import numpy_schema, counts, bulk_name_to_idx, attrs
from ecoli.processes.registries import topology_registry

NAME = "metabolite_counts_listener"
TOPOLOGY = {
    "listeners": ("listeners",),
    "bulk": ("bulk",),
    "promoters": ("unique", "promoter"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}
topology_registry.register(NAME, TOPOLOGY)


class MetaboliteCounts(Step):
    """
    Listener for free and total counts of intracellular metabolites,
    including metabolites bound in equilibrium complexes and Pi sequestered
    in phosphorylated TCS molecules.
    """

    name = NAME
    topology = TOPOLOGY

    defaults = {
        "bulk_molecule_ids": [],
        "metabolite_ids": [],
        "equilibrium_molecule_ids": [],
        "equilibrium_complex_ids": [],
        "equilibrium_stoich": [],
        "phosphorylated_tcs_mol_ids": [],
        "tcs_ligand_complex_ids": [],
        "tcs_ligand_met_ids": [],
        "tcs_ligand_stoichs": [],
        "tf_bound_col_idxs": [],
        "tf_bound_met_ids": [],
        "tf_bound_stoichs": [],
        "pi_id": "Pi[c]",
        "environment_metabolite_ids": [],
        "emit_unique": False,
        "time_step": 1,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.bulk_molecule_ids = self.parameters["bulk_molecule_ids"]
        self.metabolite_ids = self.parameters["metabolite_ids"]

        # Equilibrium complex unpacking
        self.equilibrium_molecule_ids = self.parameters["equilibrium_molecule_ids"]
        self.equilibrium_complex_ids = self.parameters["equilibrium_complex_ids"]
        self.equilibrium_stoich = np.array(self.parameters["equilibrium_stoich"])

        # Build submatrix: rows in equilibrium_stoich that correspond to
        # tracked metabolites, and their mapping to our metabolite_ids index.
        metabolite_to_idx = {m: i for i, m in enumerate(self.metabolite_ids)}
        eq_metabolite_rows = []
        eq_metabolite_to_our_idx = []
        for row_idx, mol_name in enumerate(self.equilibrium_molecule_ids):
            if mol_name in metabolite_to_idx:
                eq_metabolite_rows.append(row_idx)
                eq_metabolite_to_our_idx.append(metabolite_to_idx[mol_name])

        self.eq_metabolite_rows = np.array(eq_metabolite_rows)
        self.eq_metabolite_to_our_idx = np.array(eq_metabolite_to_our_idx)

        if len(self.eq_metabolite_rows) > 0:
            self.eq_metabolite_stoich = self.equilibrium_stoich[
                self.eq_metabolite_rows, :
            ]
        else:
            self.eq_metabolite_stoich = np.zeros(
                (0, len(self.equilibrium_complex_ids)), dtype=np.float64
            )

        # TCS phosphorylation Pi unpacking (each phosphorylated TCS molecule
        # (PHOSPHO-HK, PHOSPHO-RR, etc.) carries exactly 1 Pi[c] covalently)
        self.phosphorylated_tcs_mol_ids = self.parameters["phosphorylated_tcs_mol_ids"]
        pi_id = self.parameters["pi_id"]
        if pi_id in metabolite_to_idx:
            self.pi_metabolite_idx = metabolite_to_idx[pi_id]
            self.track_tcs_pi = True
        else:
            self.pi_metabolite_idx = -1
            self.track_tcs_pi = False

        # TCS-complex ligand sequestration: metabolite ligands held inside
        # TCS complexes that contain equilibrium complexes as subunits.
        self.tcs_ligand_complex_ids = list(self.parameters["tcs_ligand_complex_ids"])
        tcs_ligand_met_ids = list(self.parameters["tcs_ligand_met_ids"])
        self.tcs_ligand_stoichs = np.array(
            self.parameters["tcs_ligand_stoichs"], dtype=np.float64
        )
        self.tcs_ligand_met_idxs = np.array(
            [metabolite_to_idx[m] for m in tcs_ligand_met_ids], dtype=np.int64
        )
        self.track_tcs_ligands = len(self.tcs_ligand_complex_ids) > 0

        # Metabolite ligands inside transcription factors bound to DNA (when a
        # TF that is an equilibrium complex that binds to a promoter, its
        # complex leaves the bulk pool, so its metabolite ligand
        # drops out of metabolitesInEquilibriumComplexes):
        self.tf_bound_col_idxs = np.array(
            self.parameters["tf_bound_col_idxs"], dtype=np.int64
        )
        tf_bound_met_ids = list(self.parameters["tf_bound_met_ids"])
        self.tf_bound_met_idxs = np.array(
            [metabolite_to_idx[m] for m in tf_bound_met_ids], dtype=np.int64
        )
        self.tf_bound_stoichs = np.array(
            self.parameters["tf_bound_stoichs"], dtype=np.float64
        )
        self.track_bound_tf_metabolites = len(self.tf_bound_col_idxs) > 0

        # Extracellular metabolites
        self.environment_metabolite_ids = self.parameters["environment_metabolite_ids"]

        # Initialized bulk indices (set on first call to next_update)
        self.metabolite_idx = None

    def ports_schema(self):
        return {
            "listeners": {
                "metabolite_counts": {
                    "freeMetaboliteCounts": {
                        "_default": [],
                        "_updater": "set",
                        "_emit": True,
                        "_properties": {"metadata": self.metabolite_ids},
                    },
                    "totalMetaboliteCounts": {
                        "_default": [],
                        "_updater": "set",
                        "_emit": True,
                        "_properties": {"metadata": self.metabolite_ids},
                    },
                    "environmentMetaboliteConcentrations": {
                        "_default": [],
                        "_updater": "set",
                        "_emit": True,
                        "_properties": {"metadata": self.environment_metabolite_ids},
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
        # Initialize bulk indices on first call
        if self.metabolite_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.metabolite_idx = bulk_name_to_idx(self.metabolite_ids, bulk_ids)
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
            self.environment_metabolite_idx = bulk_name_to_idx(
                self.environment_metabolite_ids, bulk_ids, strict=False
            )

        bulk_counts_arr = counts(states["bulk"], self.metabolite_idx)

        # Free counts:
        free_metabolite_counts = bulk_counts_arr.copy()

        # Counts for metabolites bound in equilibrium complexes:
        eq_bound = np.zeros(len(self.metabolite_ids), np.int64)
        if len(self.eq_metabolite_rows) > 0:
            eq_complex_counts = counts(states["bulk"], self.equilibrium_complex_idx)
            bound = np.dot(self.eq_metabolite_stoich, np.negative(eq_complex_counts))
            eq_bound[self.eq_metabolite_to_our_idx] += bound.astype(np.int64)

        # Counts of Pi in TCS complexes:
        tcs_bound = np.zeros(len(self.metabolite_ids), np.int64)
        if self.track_tcs_pi:
            n_phosphorylated = int(
                counts(states["bulk"], self.tcs_phosphorylated_idx).sum()
            )
            tcs_bound[self.pi_metabolite_idx] = n_phosphorylated

        # Counts of metabolite ligands in TCS complexes (eq complex subunits):
        tcs_complex_bound = np.zeros(len(self.metabolite_ids), np.int64)
        if self.track_tcs_ligands:
            tcs_cplx_counts = counts(states["bulk"], self.tcs_ligand_complex_idx)
            for i in range(len(self.tcs_ligand_complex_ids)):
                tcs_complex_bound[self.tcs_ligand_met_idxs[i]] += int(
                    tcs_cplx_counts[i] * self.tcs_ligand_stoichs[i]
                )

        # Counts of metabolite ligands inside DNA-bound transcription factors:
        bound_tf_metabolites = np.zeros(len(self.metabolite_ids), np.int64)
        if self.track_bound_tf_metabolites:
            (bound_TF,) = attrs(states["promoters"], ["bound_TF"])
            n_bound_per_tf = bound_TF.astype(np.int64).sum(axis=0)
            for i in range(len(self.tf_bound_col_idxs)):
                col = self.tf_bound_col_idxs[i]
                bound_tf_metabolites[self.tf_bound_met_idxs[i]] += int(
                    n_bound_per_tf[col] * self.tf_bound_stoichs[i]
                )

        # Total counts :
        total_metabolite_counts = (
            free_metabolite_counts
            + eq_bound
            + tcs_bound
            + tcs_complex_bound
            + bound_tf_metabolites
        )

        # Extracellular concentrations (the environment bulk entries store
        # concentrations directly here):
        if len(self.environment_metabolite_idx) > 0:
            env_conc = counts(states["bulk"], self.environment_metabolite_idx).astype(
                np.float64
            )
        else:
            env_conc = np.zeros(len(self.environment_metabolite_ids), np.float64)

        return {
            "listeners": {
                "metabolite_counts": {
                    "freeMetaboliteCounts": free_metabolite_counts,
                    "totalMetaboliteCounts": total_metabolite_counts,
                    "environmentMetaboliteConcentrations": env_conc,
                }
            }
        }


def test_metabolite_counts_listener():
    from ecoli.experiments.ecoli_master_sim import EcoliSim

    sim = EcoliSim.from_file()
    sim.max_duration = 2
    sim.raw_output = False
    sim.build_ecoli()
    sim.run()
    listeners = sim.query()["agents"]["0"]["listeners"]
    assert "metabolite_counts" in listeners
    assert isinstance(listeners["metabolite_counts"]["totalMetaboliteCounts"][0], list)


# uvenv ecoli/processes/listeners/metabolite_counts.py
if __name__ == "__main__":
    test_metabolite_counts_listener()
