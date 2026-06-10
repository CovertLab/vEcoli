"""
Small molecule sequestration mappings (for unpacking counts of small molecules
within other molecules they can form).

Builds the stoichiometric mappings that describe how many of each tracked
small molecule are sequestered (not in the free bulk pool) inside:

  1. Equilibrium complexes        (eq_complex_count   x stoich  -> ligand)
  2. TCS phosphorylated complexes (phospho_count      x 1       -> Pi[c])
  3. TCS ligand complexes         (tcs_complex_count  x stoich  -> ligand)
  4. DNA-bound transcription factors that are eq complexes (w/ a small molecule
    ligand) or TCS complexes (containing a Pi[c]).

This logic allows analysis scripts to rebuild the exact same mappings from
sim_data and recompute the per-category breakdown so the simulation does not
need to store listeners for values that can be computed using matricies and
other saved simulation data.

``build_sequestration_maps`` returns a plain dict of parallel lists / arrays so
it can be passed straight into the listener config or used by analyses.
"""

import numpy as np
import polars as pl
from ecoli.library.parquet_emitter import (
    named_idx,
    read_stacked_columns,
)


def build_sequestration_maps(sim_data, sm_ids):
    """Build all small-molecule-sequestration stoich mappings.

    Args:
        sim_data: the simulation data object.
        sm_ids: ordered list of tracked small molecule IDs (with compartment
            tags) that the listener/analysis indexes into.

    Returns:
        dict with keys:
          - equilibrium_molecule_ids, equilibrium_complex_ids, equilibrium_stoich
          - phosphorylated_tcs_mol_ids
          - tcs_ligand_complex_ids, tcs_ligand_sm_ids, tcs_ligand_stoichs
          - tf_bound_col_idxs, tf_bound_sm_ids, tf_bound_stoichs
          - pi_id
    """
    pi_id = "Pi[c]"
    sm_to_idx = {m: i for i, m in enumerate(sm_ids)}

    equilibrium = sim_data.process.equilibrium
    tcs = sim_data.process.two_component_system

    # Obtain phosphorylated TCS complexes -- each carries 1 Pi[c] covalently:
    phosphorylated_tcs_mol_ids = list(tcs.independent_to_dependent_molecules.values())

    eq_complex_id_set = set(equilibrium.ids_complexes)
    eq_mol_names = list(equilibrium.molecule_names)
    eq_stoich = equilibrium.stoich_matrix_monomers()

    # Obtain eq complex to small molecule mapping
    # (eq_complex -> {sm_id: stoich_per_complex}):
    eq_complex_to_sms = {}
    for col_idx, cplx_id in enumerate(equilibrium.ids_complexes):
        for row_idx, mol_name in enumerate(eq_mol_names):
            if (
                mol_name in equilibrium.metabolite_set
                and eq_stoich[row_idx, col_idx] < 0
            ):
                eq_complex_to_sms.setdefault(cplx_id, {})[mol_name] = -eq_stoich[
                    row_idx, col_idx
                ]

    # Obtain TCS complex to small molecule ligand mapping (for TCS complexes that
    # have equilibrium complexes containing small molecule ligands as substrates):
    tcs_mol_names = list(tcs.molecule_names)
    tcs_stoich = tcs.stoich_matrix()
    tcs_complex_to_ligands = {}
    for tcs_cplx in tcs.complex_to_monomer.keys():
        if tcs_cplx not in tcs_mol_names:
            continue
        cplx_row = tcs_mol_names.index(tcs_cplx)
        prod_rxn_cols = np.where(tcs_stoich[cplx_row, :] > 0)[0]
        for rxn_col in prod_rxn_cols:
            reactant_rows = np.where(tcs_stoich[:, rxn_col] < 0)[0]
            for r_row in reactant_rows:
                r_mol = tcs_mol_names[r_row]
                if r_mol in eq_complex_id_set and r_mol in eq_complex_to_sms:
                    stoich_eq_per_tcs = -tcs_stoich[r_row, rxn_col]
                    for sm_id, stoich_sm in eq_complex_to_sms[r_mol].items():
                        if sm_id not in sm_to_idx:
                            continue
                        tcs_complex_to_ligands.setdefault(tcs_cplx, {})[sm_id] = (
                            stoich_eq_per_tcs * stoich_sm
                        )

    tcs_ligand_complex_ids = []
    tcs_ligand_sm_ids = []
    tcs_ligand_stoichs = []
    for cplx_id, sm_stoichs in tcs_complex_to_ligands.items():
        for sm_id, stoich in sm_stoichs.items():
            tcs_ligand_complex_ids.append(cplx_id)
            tcs_ligand_sm_ids.append(sm_id)
            tcs_ligand_stoichs.append(float(stoich))

    # Obtain mapping of bound TFs to small molecules:
    tf_ids = sim_data.process.transcription_regulation.tf_ids
    phospho_tcs_set = set(phosphorylated_tcs_mol_ids)
    tf_bound_col_idxs = []
    tf_bound_sm_ids = []
    tf_bound_stoichs = []
    for col, tf_id in enumerate(tf_ids):
        # NOTE: this assumes the first compartment (if multiple exist) is
        # the one the TF is in:
        tf_mol = tf_id + f"[{sim_data.getter.get_compartment(tf_id)[0]}]"
        # Case 1: TF is an eq complex carrying a small molecule ligand:
        if tf_mol in eq_complex_to_sms:
            for sm_id, stoich in eq_complex_to_sms[tf_mol].items():
                if sm_id in sm_to_idx:
                    tf_bound_col_idxs.append(int(col))
                    tf_bound_sm_ids.append(sm_id)
                    tf_bound_stoichs.append(float(stoich))
        # Case 2: TF is a TCS complex -- same complex as the bulk
        # phospho form, so a bound one still carries its 1 Pi[c]:
        if tf_mol in phospho_tcs_set and pi_id in sm_to_idx:
            tf_bound_col_idxs.append(int(col))
            tf_bound_sm_ids.append(pi_id)
            tf_bound_stoichs.append(1.0)

    return {
        "equilibrium_molecule_ids": equilibrium.molecule_names,
        "equilibrium_complex_ids": equilibrium.ids_complexes,
        "equilibrium_stoich": eq_stoich,
        "phosphorylated_tcs_mol_ids": phosphorylated_tcs_mol_ids,
        "tcs_ligand_complex_ids": tcs_ligand_complex_ids,
        "tcs_ligand_sm_ids": tcs_ligand_sm_ids,
        "tcs_ligand_stoichs": tcs_ligand_stoichs,
        "tf_bound_col_idxs": tf_bound_col_idxs,
        "tf_bound_sm_ids": tf_bound_sm_ids,
        "tf_bound_stoichs": tf_bound_stoichs,
        "pi_id": pi_id,
    }


def compute_avg_sequestration(conn, history_sql, sim_data, sm_ids):
    """Recompute the time-averaged per-small-molecule sequestration breakdown.

    Compute the counts of small molecules in different molecule types
    (equilibrium complexes, TCS complexes, bound TFs) from data that is emitted
    by the sim:
      - bulk counts of eq complexes / phospho-TCS mols / TCS ligand complexes
      - per-TF bound counts (rna_synth_prob.bound_TF_indexes, the sparse list of
        bound TF indices -- the same per-promoter quantity the listener uses)
    plus the stoich maps from ``build_sequestration_maps``.

    NOTE: this function does time averaging, not generation averaging (but
    an equivalent generation-averaged version could be implemented easily by
    creating a similar function to this and averaging by cell first).

    Returns:
        dict of length-n_sm numpy arrays:
          avg_eq, avg_tcs_pi, avg_tcs_complex, avg_bound_tf
    """

    maps = build_sequestration_maps(sim_data, sm_ids)
    n_sm = len(sm_ids)
    sm_to_idx = {m: i for i, m in enumerate(sm_ids)}

    bulk_ids = list(sim_data.internal_state.bulk_molecules.bulk_data["id"])
    bname_to_idx = {n: i for i, n in enumerate(bulk_ids)}

    eq_complex_ids = list(maps["equilibrium_complex_ids"])
    phospho_ids = list(maps["phosphorylated_tcs_mol_ids"])
    tcs_lig_ids = list(maps["tcs_ligand_complex_ids"])

    # Unique list of every bulk molecule that needs to be unpacked:
    needed = list(dict.fromkeys(eq_complex_ids + phospho_ids + tcs_lig_ids))
    needed_idx = [bname_to_idx[n] for n in needed]

    columns = []
    if needed:
        columns.append(named_idx("bulk", needed, [needed_idx]))

    # Only read n_bound_TF_per_TU if needed for the small molecule IDs passed in:
    need_bound_tf = len(maps["tf_bound_col_idxs"]) > 0

    # Per-TF time-averaged bound count from rna_synth_prob.bound_TF_indexes
    # (the sparse per-timestep list of bound TF indices):
    avg_bound_per_tf = {}
    if need_bound_tf:
        bound_tf_subquery = read_stacked_columns(
            history_sql,
            ["listeners__rna_synth_prob__bound_TF_indexes AS bti"],
            order_results=False,
        )
        n_timesteps = conn.execute(
            f"SELECT count(*) FROM ({bound_tf_subquery})"
        ).fetchone()[0]
        if n_timesteps:
            tf_counts = conn.execute(
                f"""
                WITH idx AS (SELECT unnest(bti) AS tf FROM ({bound_tf_subquery}))
                SELECT tf, count(*) AS c FROM idx GROUP BY tf
                """
            ).fetchall()
            avg_bound_per_tf = {int(tf): c / n_timesteps for tf, c in tf_counts}

    # Handle case where no bulk data is needed (e.g. if no eq complexes or TCS
    # complexes with tracked ligands):
    if not columns:
        avg_bulk = {}
    else:
        raw = pl.DataFrame(
            read_stacked_columns(history_sql, columns, order_results=True, conn=conn)
        )
        # Average bulk counts:
        avg_bulk = {name: float(raw[name].mean()) for name in needed}

    # 1. Unpack equilibrium complexes:
    avg_eq = np.zeros(n_sm, np.float64)
    eq_mol_names = list(maps["equilibrium_molecule_ids"])
    eq_stoich = maps["equilibrium_stoich"]
    eq_rows, eq_sm_idxs = [], []
    for r, mol in enumerate(eq_mol_names):
        if mol in sm_to_idx:
            eq_rows.append(r)
            eq_sm_idxs.append(sm_to_idx[mol])
    if eq_rows and eq_complex_ids:
        sub = eq_stoich[np.array(eq_rows), :]  # (k, n_eq_complex)
        avg_eq_complex = np.array([avg_bulk.get(c, 0.0) for c in eq_complex_ids])
        np.add.at(avg_eq, np.array(eq_sm_idxs), sub.dot(-avg_eq_complex))

    # 2. Pi in phosphorylated TCS molecules:
    avg_tcs_pi = np.zeros(n_sm, np.float64)
    if maps["pi_id"] in sm_to_idx and phospho_ids:
        avg_tcs_pi[sm_to_idx[maps["pi_id"]]] = sum(
            avg_bulk.get(p, 0.0) for p in phospho_ids
        )

    # 3. Ligands in TCS complexes (eq-complex subunits):
    avg_tcs_complex = np.zeros(n_sm, np.float64)
    for cplx, sm, st in zip(
        maps["tcs_ligand_complex_ids"],
        maps["tcs_ligand_sm_ids"],
        maps["tcs_ligand_stoichs"],
    ):
        avg_tcs_complex[sm_to_idx[sm]] += avg_bulk.get(cplx, 0.0) * st

    # 4. Small molecules in DNA-bound TFs (ligands + Pi[c]):
    avg_bound_tf = np.zeros(n_sm, np.float64)
    for col, sm, st in zip(
        maps["tf_bound_col_idxs"],
        maps["tf_bound_sm_ids"],
        maps["tf_bound_stoichs"],
    ):
        avg_bound_tf[sm_to_idx[sm]] += avg_bound_per_tf.get(int(col), 0.0) * st

    return {
        "avg_eq": avg_eq,
        "avg_tcs_pi": avg_tcs_pi,
        "avg_tcs_complex": avg_tcs_complex,
        "avg_bound_tf": avg_bound_tf,
    }
