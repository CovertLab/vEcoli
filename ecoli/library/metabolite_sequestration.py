"""
Metabolite sequestration mappings (for metabolites the form complexes).

Builds the stoichiometric mappings that describe how many of each tracked
metabolite are sequestered (not in the free bulk pool) inside:

  1. Equilibrium complexes        (eq_complex_count   x stoich  -> ligand)
  2. TCS phosphorylated molecules (phospho_count      x 1       -> Pi[c])
  3. TCS ligand complexes         (tcs_complex_count  x stoich  -> ligand)
  4. DNA-bound transcription factors that are eq complexes (ligand) or
     phospho-response-regulators (Pi[c]).

This logic allows analysis scripts to rebuild the exact same mappings from
sim_data and recompute the per-category breakdown so the  simulation no longer
has to store listeners for values that can be computed in analyses scripts
using matricies and other saved simulation data.

``build_sequestration_maps`` returns a plain dict of parallel lists / arrays so
it can be passed straight into the listener config or consumed by analysis.
"""

import numpy as np
import polars as pl
from ecoli.library.parquet_emitter import (
    named_idx,
    read_stacked_columns,
    ndlist_to_ndarray,
)


def build_sequestration_maps(sim_data, metabolite_ids):
    """Build all metabolite-sequestration stoich mappings.

    Args:
        sim_data: the simulation data object.
        metabolite_ids: ordered list of tracked metabolite IDs (with
            compartment tags) that the listener/analysis indexes into.

    Returns:
        dict with keys:
          - equilibrium_molecule_ids, equilibrium_complex_ids, equilibrium_stoich
          - phosphorylated_tcs_mol_ids
          - tcs_ligand_complex_ids, tcs_ligand_met_ids, tcs_ligand_stoichs
          - tf_bound_col_idxs, tf_bound_met_ids, tf_bound_stoichs
          - pi_id
    """
    pi_id = "Pi[c]"
    metabolite_to_idx = {m: i for i, m in enumerate(metabolite_ids)}

    equilibrium = sim_data.process.equilibrium
    tcs = sim_data.process.two_component_system

    # Phosphorylated TCS molecules -- each carries 1 Pi[c] covalently.
    phosphorylated_tcs_mol_ids = list(tcs.independent_to_dependent_molecules.values())

    eq_complex_id_set = set(equilibrium.ids_complexes)
    eq_mol_names = list(equilibrium.molecule_names)
    eq_stoich = equilibrium.stoich_matrix_monomers()

    # eq_complex -> {met_id: stoich_per_complex}
    eq_complex_to_metabolites = {}
    for col_idx, cplx_id in enumerate(equilibrium.ids_complexes):
        for row_idx, mol_name in enumerate(eq_mol_names):
            if (
                mol_name in equilibrium.metabolite_set
                and eq_stoich[row_idx, col_idx] < 0
            ):
                eq_complex_to_metabolites.setdefault(cplx_id, {})[
                    mol_name
                ] = -eq_stoich[row_idx, col_idx]

    # TCS complexes that contain an eq complex (with a ligand) as a subunit.
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
                if r_mol in eq_complex_id_set and r_mol in eq_complex_to_metabolites:
                    stoich_eq_per_tcs = -tcs_stoich[r_row, rxn_col]
                    for met_id, stoich_met in eq_complex_to_metabolites[r_mol].items():
                        if met_id not in metabolite_to_idx:
                            continue
                        tcs_complex_to_ligands.setdefault(tcs_cplx, {})[met_id] = (
                            stoich_eq_per_tcs * stoich_met
                        )

    tcs_ligand_complex_ids = []
    tcs_ligand_met_ids = []
    tcs_ligand_stoichs = []
    for cplx_id, met_stoichs in tcs_complex_to_ligands.items():
        for met_id, stoich in met_stoichs.items():
            tcs_ligand_complex_ids.append(cplx_id)
            tcs_ligand_met_ids.append(met_id)
            tcs_ligand_stoichs.append(float(stoich))

    # Metabolites inside DNA-bound TFs.
    tf_ids = sim_data.process.transcription_regulation.tf_ids
    phospho_tcs_set = set(phosphorylated_tcs_mol_ids)
    tf_bound_col_idxs = []
    tf_bound_met_ids = []
    tf_bound_stoichs = []
    for col, tf_id in enumerate(tf_ids):
        tf_mol = tf_id + f"[{sim_data.getter.get_compartment(tf_id)[0]}]"
        # Case 1: TF is an eq complex carrying a metabolite ligand.
        if tf_mol in eq_complex_to_metabolites:
            for met_id, stoich in eq_complex_to_metabolites[tf_mol].items():
                if met_id in metabolite_to_idx:
                    tf_bound_col_idxs.append(int(col))
                    tf_bound_met_ids.append(met_id)
                    tf_bound_stoichs.append(float(stoich))
        # Case 2: TF is a TCS complex -- same complex as the bulk
        # phospho form, so a bound one still carries its 1 Pi (which left the
        # bulk pool, so metabolitesInTCSPhosphorylation no longer counts it).
        if tf_mol in phospho_tcs_set and pi_id in metabolite_to_idx:
            tf_bound_col_idxs.append(int(col))
            tf_bound_met_ids.append(pi_id)
            tf_bound_stoichs.append(1.0)

    return {
        "equilibrium_molecule_ids": equilibrium.molecule_names,
        "equilibrium_complex_ids": equilibrium.ids_complexes,
        "equilibrium_stoich": eq_stoich,
        "phosphorylated_tcs_mol_ids": phosphorylated_tcs_mol_ids,
        "tcs_ligand_complex_ids": tcs_ligand_complex_ids,
        "tcs_ligand_met_ids": tcs_ligand_met_ids,
        "tcs_ligand_stoichs": tcs_ligand_stoichs,
        "tf_bound_col_idxs": tf_bound_col_idxs,
        "tf_bound_met_ids": tf_bound_met_ids,
        "tf_bound_stoichs": tf_bound_stoichs,
        "pi_id": pi_id,
    }


def compute_avg_sequestration_old(conn, history_sql, sim_data, metabolite_ids):
    """Recompute the time-averaged per-metabolite sequestration breakdown.

    Compute the counts of metabolites in different molecule types (equilibrium
    complexes, TCS complexes, bound TFs) from data that is emitted by the sim:
      - bulk counts of eq complexes / phospho-TCS mols / TCS ligand complexes
      - per-TF bound counts (rna_synth_prob.n_bound_TF_per_TU, summed over TUs)
    plus the stoich maps from ``build_sequestration_maps``.

    Averaging is linear and the maps are linear, so this function averages the
    complex / bound-TF counts over time first, then apply the maps once.

    NOTE: this function does time averaging, not generation averaging (but
    an equivalent generation-averaged version could be implemented easily by
    creating a similar function to this).

    Returns:
        dict of length-n_metabolites numpy arrays:
          avg_eq, avg_tcs_pi, avg_tcs_complex, avg_bound_tf
    """

    maps = build_sequestration_maps(sim_data, metabolite_ids)
    n_met = len(metabolite_ids)
    met_to_idx = {m: i for i, m in enumerate(metabolite_ids)}

    bulk_ids = list(sim_data.internal_state.bulk_molecules.bulk_data["id"])
    bname_to_idx = {n: i for i, n in enumerate(bulk_ids)}

    eq_complex_ids = list(maps["equilibrium_complex_ids"])
    phospho_ids = list(maps["phosphorylated_tcs_mol_ids"])
    tcs_lig_ids = list(maps["tcs_ligand_complex_ids"])

    # Unique list of every bulk molecule whose average count needed:
    needed = list(dict.fromkeys(eq_complex_ids + phospho_ids + tcs_lig_ids))
    needed_idx = [bname_to_idx[n] for n in needed]

    columns = ["listeners__rna_synth_prob__n_bound_TF_per_TU AS nbt"]
    if needed:
        columns.append(named_idx("bulk", needed, [needed_idx]))

    raw = pl.DataFrame(
        read_stacked_columns(history_sql, columns, order_results=True, conn=conn)
    )

    avg_bulk = {name: float(raw[name].mean()) for name in needed}

    # Average per-TF bound count: (T, n_TU, n_TF) -> sum TUs -> mean time.
    nbt = ndlist_to_ndarray(raw["nbt"].to_list())
    avg_bound_per_tf = nbt.sum(axis=1).mean(axis=0)

    # 1. Equilibrium complexes
    avg_eq = np.zeros(n_met, np.float64)
    eq_mol_names = list(maps["equilibrium_molecule_ids"])
    eq_stoich = maps["equilibrium_stoich"]
    rows, our = [], []
    for r, mol in enumerate(eq_mol_names):
        if mol in met_to_idx:
            rows.append(r)
            our.append(met_to_idx[mol])
    if rows:
        sub = eq_stoich[np.array(rows), :]  # (k, n_eq_complex)
        avg_eq_complex = np.array([avg_bulk[c] for c in eq_complex_ids])
        np.add.at(avg_eq, np.array(our), sub.dot(-avg_eq_complex))

    # 2. Pi in phosphorylated TCS molecules
    avg_tcs_pi = np.zeros(n_met, np.float64)
    if maps["pi_id"] in met_to_idx and phospho_ids:
        avg_tcs_pi[met_to_idx[maps["pi_id"]]] = sum(avg_bulk[p] for p in phospho_ids)

    # 3. Ligands in TCS complexes (eq-complex subunits)
    avg_tcs_complex = np.zeros(n_met, np.float64)
    for cplx, met, st in zip(
        maps["tcs_ligand_complex_ids"],
        maps["tcs_ligand_met_ids"],
        maps["tcs_ligand_stoichs"],
    ):
        avg_tcs_complex[met_to_idx[met]] += avg_bulk[cplx] * st

    # 4. Metabolites in DNA-bound TFs
    avg_bound_tf = np.zeros(n_met, np.float64)
    for col, met, st in zip(
        maps["tf_bound_col_idxs"],
        maps["tf_bound_met_ids"],
        maps["tf_bound_stoichs"],
    ):
        avg_bound_tf[met_to_idx[met]] += avg_bound_per_tf[col] * st

    return {
        "avg_eq": avg_eq,
        "avg_tcs_pi": avg_tcs_pi,
        "avg_tcs_complex": avg_tcs_complex,
        "avg_bound_tf": avg_bound_tf,
    }


def compute_avg_sequestration(conn, history_sql, sim_data, metabolite_ids):
    """Recompute the time-averaged per-metabolite sequestration breakdown.

    Compute the counts of metabolites in differnt molecule types (equilibirum
    complexes, TCS complexes, bound TFs) from data that is emitted by the sim:
      - bulk counts of eq complexes / phospho-TCS mols / TCS ligand complexes
      - per-TF bound counts (rna_synth_prob.n_bound_TF_per_TU, summed over TUs)
    plus the stoich maps from ``build_sequestration_maps``.

    Averaging is linear and the maps are linear, so this function averages the
    complex / bound-TF counts over time first, then apply the maps once.

    NOTE: this function does time averaging, not generation averaging (but
    an equivalent generation-averaged version could be implemented easily by
    creating a similar function to this).

    Returns:
        dict of length-n_metabolites numpy arrays:
          avg_eq, avg_tcs_pi, avg_tcs_complex, avg_bound_tf
    """

    maps = build_sequestration_maps(sim_data, metabolite_ids)
    n_met = len(metabolite_ids)
    met_to_idx = {m: i for i, m in enumerate(metabolite_ids)}

    bulk_ids = list(sim_data.internal_state.bulk_molecules.bulk_data["id"])
    bname_to_idx = {n: i for i, n in enumerate(bulk_ids)}

    eq_complex_ids = list(maps["equilibrium_complex_ids"])
    phospho_ids = list(maps["phosphorylated_tcs_mol_ids"])
    tcs_lig_ids = list(maps["tcs_ligand_complex_ids"])

    # Unique list of every bulk molecule whose average count we need.
    needed = list(dict.fromkeys(eq_complex_ids + phospho_ids + tcs_lig_ids))
    needed_idx = [bname_to_idx[n] for n in needed]

    columns = []
    if needed:
        columns.append(named_idx("bulk", needed, [needed_idx]))

    # Only read n_bound_TF_per_TU if needed for the metabolite IDs passed through:
    need_bound_tf = len(maps["tf_bound_col_idxs"]) > 0

    # Compute TF averages using a single SQL query
    if need_bound_tf:
        # Get dimensions from first row
        test_query = f"""
            SELECT listeners__rna_synth_prob__n_bound_TF_per_TU
            FROM ({history_sql})
            LIMIT 1
        """
        test_result = conn.execute(test_query).fetchone()

        if test_result and test_result[0]:
            n_tu = len(test_result[0])
            n_tf = len(test_result[0][0]) if n_tu > 0 else 0

            # Build a single query that computes all TF averages at once
            # For each TF: sum across all TUs, then average across time
            tf_selects = [
                f"""AVG(
                    list_sum(
                        list_transform(
                            listeners__rna_synth_prob__n_bound_TF_per_TU,
                            tu -> tu[{tf_idx + 1}]
                        )
                    )
                ) as tf_{tf_idx}"""
                for tf_idx in range(n_tf)
            ]

            all_tf_query = f"""
                SELECT {", ".join(tf_selects)}
                FROM ({history_sql})
            """

            tf_results = conn.execute(all_tf_query).fetchone()
            avg_bound_per_tf = np.array(
                [float(x) if x is not None else 0.0 for x in tf_results]
            )
        else:
            avg_bound_per_tf = np.zeros(0)
    else:
        avg_bound_per_tf = np.zeros(0)

    # Handle case where no bulk data is needed (e.g. if no eq complexes or TCS
    # complexes with tracked ligands)
    if not columns:
        avg_bulk = {}
    else:
        raw = pl.DataFrame(
            read_stacked_columns(history_sql, columns, order_results=True, conn=conn)
        )
        # Average bulk counts
        avg_bulk = {name: float(raw[name].mean()) for name in needed}

    # 1. Equilibrium complexes
    avg_eq = np.zeros(n_met, np.float64)
    eq_mol_names = list(maps["equilibrium_molecule_ids"])
    eq_stoich = maps["equilibrium_stoich"]
    rows, our = [], []
    for r, mol in enumerate(eq_mol_names):
        if mol in met_to_idx:
            rows.append(r)
            our.append(met_to_idx[mol])
    if rows and eq_complex_ids:
        sub = eq_stoich[np.array(rows), :]  # (k, n_eq_complex)
        avg_eq_complex = np.array([avg_bulk.get(c, 0.0) for c in eq_complex_ids])
        np.add.at(avg_eq, np.array(our), sub.dot(-avg_eq_complex))

    # 2. Pi in phosphorylated TCS molecules
    avg_tcs_pi = np.zeros(n_met, np.float64)
    if maps["pi_id"] in met_to_idx and phospho_ids:
        avg_tcs_pi[met_to_idx[maps["pi_id"]]] = sum(
            avg_bulk.get(p, 0.0) for p in phospho_ids
        )

    # 3. Ligands in TCS complexes (eq-complex subunits)
    avg_tcs_complex = np.zeros(n_met, np.float64)
    for cplx, met, st in zip(
        maps["tcs_ligand_complex_ids"],
        maps["tcs_ligand_met_ids"],
        maps["tcs_ligand_stoichs"],
    ):
        avg_tcs_complex[met_to_idx[met]] += avg_bulk.get(cplx, 0.0) * st

    # 4. Metabolites in DNA-bound TFs
    avg_bound_tf = np.zeros(n_met, np.float64)
    for col, met, st in zip(
        maps["tf_bound_col_idxs"],
        maps["tf_bound_met_ids"],
        maps["tf_bound_stoichs"],
    ):
        avg_bound_tf[met_to_idx[met]] += avg_bound_per_tf[col] * st

    return {
        "avg_eq": avg_eq,
        "avg_tcs_pi": avg_tcs_pi,
        "avg_tcs_complex": avg_tcs_complex,
        "avg_bound_tf": avg_bound_tf,
    }
