"""
Promoter binding and transcription factor fitting for the ParCa pipeline.

This module contains functions for fitting transcription factor binding
probabilities and their effects on RNA synthesis. The main functions are:

- fitPromoterBoundProbability: Fits TF-promoter binding probabilities
- fitLigandConcentrations: Adjusts ligand concentrations based on fitted P
- calculatePromoterBoundProbability: Initial probability calculation from bulk counts
- calculateRnapRecruitment: Constructs basal_prob and delta_prob from fitted r values

The promoter fitting algorithm uses convex optimization (CVXPY with ECOS solver)
to find parameters that satisfy physiological constraints while matching
experimental RNA synthesis probabilities.
"""

import itertools

from cvxpy import Variable, Problem, Minimize, norm
import numpy as np
import scipy.sparse

from ecoli.library.schema import bulk_name_to_idx, counts
from wholecell.utils import units

# Parameters used in fitPromoterBoundProbability()
PROMOTER_PDIFF_THRESHOLD = 0.06  # Minimum difference between binding probabilities
PROMOTER_REG_COEFF = 1e-3  # Optimization weight for staying close to original values
PROMOTER_SCALING = 10  # Multiplied to all matrices for numerical stability
PROMOTER_NORM_TYPE = 1  # Matrix 1-norm
PROMOTER_MAX_ITERATIONS = 100
PROMOTER_CONVERGENCE_THRESHOLD = 1e-9
ECOS_0_TOLERANCE = 1e-10  # Tolerance to adjust solver output to 0


# ============================================================================
# Matrix/vector builder functions (extracted from fitPromoterBoundProbability)
# ============================================================================


def _build_vector_k(sim_data, cell_specs):
    """
    Construct vector k containing fit transcription probabilities normalized
    by gene copy number.

    Returns:
        k: Array of RNA synthesis probabilities per gene copy
        kInfo: List of dicts with condition and RNA index for each k value
    """
    k, kInfo = [], []

    for idx, (rnaId, rnaCoordinate) in enumerate(
        zip(
            sim_data.process.transcription.rna_data["id"],
            sim_data.process.transcription.rna_data["replication_coordinate"],
        )
    ):
        tfs = sim_data.relation.rna_id_to_regulating_tfs.get(rnaId, [])
        conditions = ["basal"]
        tfsWithData = []

        for tf in tfs:
            if tf not in sorted(sim_data.tf_to_active_inactive_conditions):
                continue
            conditions.append(tf + "__active")
            conditions.append(tf + "__inactive")
            tfsWithData.append(tf)

        for condition in conditions:
            if len(tfsWithData) > 0 and condition == "basal":
                continue

            tau = cell_specs[condition]["doubling_time"].asNumber(units.min)
            n_avg_copy = sim_data.process.replication.get_average_copy_number(
                tau, rnaCoordinate
            )
            prob_per_copy = (
                sim_data.process.transcription.rna_synth_prob[condition][idx]
                / n_avg_copy
            )

            k.append(prob_per_copy)
            kInfo.append({"condition": condition, "idx": idx})

    return np.array(k), kInfo


def _build_matrix_G(sim_data, pPromoterBound):
    """
    Construct matrix G containing pPromoterBound values arranged by RNA and TF.

    Each row corresponds to an RNA-condition pair.
    Each column corresponds to an RNA-TF pair or RNA-alpha.

    Returns:
        G: Matrix of promoter bound probabilities
        row_name_to_index: Dict mapping row names to indices
        col_name_to_index: Dict mapping column names to indices
    """
    gI, gJ, gV = [], [], []
    row_name_to_index, col_name_to_index = {}, {}

    for idx, rnaId in enumerate(sim_data.process.transcription.rna_data["id"]):
        rnaIdNoLoc = rnaId[:-3]

        tfs = sim_data.relation.rna_id_to_regulating_tfs.get(rnaId, [])
        conditions = ["basal"]
        tfsWithData = []

        for tf in tfs:
            if tf not in sorted(sim_data.tf_to_active_inactive_conditions):
                continue
            conditions.append(tf + "__active")
            conditions.append(tf + "__inactive")
            tfsWithData.append(tf)

        for condition in conditions:
            if len(tfsWithData) > 0 and condition == "basal":
                continue

            row_name = rnaIdNoLoc + "__" + condition
            row_name_to_index[row_name] = len(row_name_to_index)

            for tf in tfsWithData:
                col_name = rnaIdNoLoc + "__" + tf
                if col_name not in col_name_to_index:
                    col_name_to_index[col_name] = len(col_name_to_index)

                gI.append(row_name_to_index[row_name])
                gJ.append(col_name_to_index[col_name])
                gV.append(pPromoterBound[condition][tf])

            col_name = rnaIdNoLoc + "__alpha"
            if col_name not in col_name_to_index:
                col_name_to_index[col_name] = len(col_name_to_index)

            gI.append(row_name_to_index[row_name])
            gJ.append(col_name_to_index[col_name])
            gV.append(1.0)

    gI, gJ, gV = np.array(gI), np.array(gJ), np.array(gV)
    G = np.zeros((len(row_name_to_index), len(col_name_to_index)), np.float64)
    G[gI, gJ] = gV

    return G, row_name_to_index, col_name_to_index


def _build_matrix_Z(sim_data, col_name_to_index):
    """
    Construct matrix Z connecting all TF combinations to individual TFs.

    Matrix value is 1 if the TF in the column is active in the combination
    specified by the row.

    Returns:
        Z: Binary matrix of TF combinations
    """
    zI, zJ, zV = [], [], []
    row_idx = 0

    for rna_id in sim_data.process.transcription.rna_data["id"]:
        rna_id_no_loc = rna_id[:-3]

        tfs = sim_data.relation.rna_id_to_regulating_tfs[rna_id]
        tfs_with_data = []
        col_idxs = [col_name_to_index[rna_id_no_loc + "__alpha"]]

        for tf in tfs:
            if tf not in sim_data.tf_to_active_inactive_conditions:
                continue
            tfs_with_data.append(tf)
            col_idxs.append(col_name_to_index[rna_id_no_loc + "__" + tf])

        n_tfs = len(tfs_with_data)

        for n_combinations in range(n_tfs + 1):
            for combination in itertools.combinations(range(1, n_tfs + 1), n_combinations):
                zI.append(row_idx)
                zJ.append(col_idxs[0])
                zV.append(1)

                for col_idx in combination:
                    zI.append(row_idx)
                    zJ.append(col_idxs[col_idx])
                    zV.append(1)

                row_idx += 1

    zI, zJ, zV = np.array(zI), np.array(zJ), np.array(zV)
    Z = np.zeros((zI.max() + 1, zJ.max() + 1), np.float64)
    Z[zI, zJ] = zV

    return Z


def _build_matrix_T(sim_data, col_name_to_index):
    """
    Construct matrix T specifying regulation direction for each RNA-TF pair.

    Diagonal value is +1 for positive regulation, -1 for negative, 0 for alpha.

    Returns:
        T: Diagonal matrix of regulation directions
    """
    tI, tJ, tV = [], [], []
    row_idx = 0

    for rnaId in sim_data.process.transcription.rna_data["id"]:
        rnaIdNoLoc = rnaId[:-3]

        tfs = sim_data.relation.rna_id_to_regulating_tfs[rnaId]
        tfsWithData = []

        constituent_cistron_ids = [
            sim_data.process.transcription.cistron_data["id"][i]
            for i in sim_data.process.transcription.rna_id_to_cistron_indexes(rnaId)
        ]

        for tf in tfs:
            if tf not in sim_data.tf_to_active_inactive_conditions:
                continue
            tfsWithData.append(tf)

        for tf in tfsWithData:
            directions = np.array(
                [
                    sim_data.tf_to_direction[tf].get(cistron_id, 0)
                    for cistron_id in constituent_cistron_ids
                ]
            )
            consensus_direction = -1 + 2 * (directions.sum() >= 0)

            col_name = rnaIdNoLoc + "__" + tf
            tI.append(row_idx)
            tJ.append(col_name_to_index[col_name])
            tV.append(consensus_direction)
            row_idx += 1

        col_name = rnaIdNoLoc + "__alpha"
        tI.append(row_idx)
        tJ.append(col_name_to_index[col_name])
        tV.append(0)
        row_idx += 1

    tI, tJ, tV = np.array(tI), np.array(tJ), np.array(tV)
    T = np.zeros((tI.max() + 1, tJ.max() + 1), np.float64)
    T[tI, tJ] = tV

    return T


def _build_matrix_H(sim_data, col_name_to_index, pPromoterBound, r, fixedTFs, cell_specs):
    r"""
    Construct matrix H containing optimized r values for each RNA and condition.

    Returns:
        H: Matrix of r values
        pInit: Vector of initial promoter bound probabilities
        pAlphaIdxs: Indices of alpha columns
        pNotAlphaIdxs: Indices of non-alpha columns
        fixedTFIdxs: Indices of fixed TF columns
        pPromoterBoundIdxs: Dict of indices for updating pPromoterBound
        H_col_name_to_index: Column name to index mapping
    """
    rDict = dict([(col_name, value) for col_name, value in zip(col_name_to_index, r)])

    pPromoterBoundIdxs = dict([(condition, {}) for condition in pPromoterBound])
    hI, hJ, hV, pInitI, pInitV = [], [], [], [], []
    H_row_name_to_index, H_col_name_to_index = {}, {}

    for idx, rnaId in enumerate(sim_data.process.transcription.rna_data["id"]):
        rnaIdNoLoc = rnaId[:-3]

        tfs = sim_data.relation.rna_id_to_regulating_tfs[rnaId]
        conditions = ["basal"]
        tfsWithData = []

        for tf in tfs:
            if tf not in sorted(sim_data.tf_to_active_inactive_conditions):
                continue
            conditions.append(tf + "__active")
            conditions.append(tf + "__inactive")
            tfsWithData.append(tf)

        for condition in conditions:
            if len(tfsWithData) > 0 and condition == "basal":
                continue

            row_name = rnaIdNoLoc + "__" + condition
            H_row_name_to_index[row_name] = len(H_row_name_to_index)

            for tf in tfsWithData:
                col_name = tf + "__" + condition
                if col_name not in H_col_name_to_index:
                    H_col_name_to_index[col_name] = len(H_col_name_to_index)

                hI.append(H_row_name_to_index[row_name])
                hJ.append(H_col_name_to_index[col_name])

                tf_idx = bulk_name_to_idx(
                    tf + "[c]", cell_specs[condition]["bulkAverageContainer"]["id"]
                )
                if counts(cell_specs[condition]["bulkAverageContainer"], tf_idx) == 0:
                    hV.append(0)
                else:
                    hV.append(rDict[rnaIdNoLoc + "__" + tf])

                pInitI.append(H_col_name_to_index[col_name])
                pInitV.append(pPromoterBound[condition][tf])
                pPromoterBoundIdxs[condition][tf] = H_col_name_to_index[col_name]

            col_name = rnaIdNoLoc + "__alpha"
            if col_name not in H_col_name_to_index:
                H_col_name_to_index[col_name] = len(H_col_name_to_index)

            hI.append(H_row_name_to_index[row_name])
            hJ.append(H_col_name_to_index[col_name])
            hV.append(rDict[col_name])

            pInitI.append(H_col_name_to_index[col_name])
            pInitV.append(1.0)

    # Save indices for combined conditions
    for condition, tfs in sim_data.condition_active_tfs.items():
        for tf in tfs:
            col_name = f"{tf}__{tf}__active"
            pPromoterBoundIdxs[condition][tf] = H_col_name_to_index[col_name]

    for condition, tfs in sim_data.condition_inactive_tfs.items():
        for tf in tfs:
            col_name = f"{tf}__{tf}__inactive"
            pPromoterBoundIdxs[condition][tf] = H_col_name_to_index[col_name]

    pInit = np.zeros(len(set(pInitI)))
    pInit[pInitI] = pInitV

    hI, hJ, hV = np.array(hI), np.array(hJ), np.array(hV)
    Hshape = (hI.max() + 1, hJ.max() + 1)
    H = np.zeros(Hshape, np.float64)
    H[hI, hJ] = hV

    pAlphaIdxs = np.array([
        idx for col_name, idx in H_col_name_to_index.items()
        if col_name.endswith("__alpha")
    ])
    pNotAlphaIdxs = np.array([
        idx for col_name, idx in H_col_name_to_index.items()
        if not col_name.endswith("__alpha")
    ])

    fixedTFIdxs = []
    for col_name, idx in H_col_name_to_index.items():
        secondElem = col_name.split("__")[1]
        if secondElem in fixedTFs:
            fixedTFIdxs.append(idx)

    fixedTFIdxs = np.array(fixedTFIdxs, dtype=int)

    return (H, pInit, pAlphaIdxs, pNotAlphaIdxs, fixedTFIdxs,
            pPromoterBoundIdxs, H_col_name_to_index)


def _build_matrix_pdiff(sim_data, H_col_name_to_index):
    """
    Construct matrix Pdiff specifying TF-condition correspondence.

    Matrix value is +1 for TF__active, -1 for TF__inactive.

    Returns:
        Pdiff: Matrix for enforcing minimum probability difference
    """
    PdiffI, PdiffJ, PdiffV = [], [], []

    for rowIdx, tf in enumerate(sorted(sim_data.tf_to_active_inactive_conditions)):
        condition = tf + "__active"
        col_name = tf + "__" + condition
        PdiffI.append(rowIdx)
        PdiffJ.append(H_col_name_to_index[col_name])
        PdiffV.append(1)

        condition = tf + "__inactive"
        col_name = tf + "__" + condition
        PdiffI.append(rowIdx)
        PdiffJ.append(H_col_name_to_index[col_name])
        PdiffV.append(-1)

    PdiffI, PdiffJ, PdiffV = np.array(PdiffI), np.array(PdiffJ), np.array(PdiffV)
    Pdiffshape = (PdiffI.max() + 1, len(H_col_name_to_index))
    Pdiff = np.zeros(Pdiffshape, np.float64)
    Pdiff[PdiffI, PdiffJ] = PdiffV

    return Pdiff


def _update_p_promoter_bound(p, pPromoterBound, pPromoterBoundIdxs):
    """Update pPromoterBound with optimized probabilities from p vector."""
    for condition in sorted(pPromoterBoundIdxs):
        for tf in sorted(pPromoterBoundIdxs[condition]):
            pPromoterBound[condition][tf] = p[pPromoterBoundIdxs[condition][tf]]


def _update_synth_prob(sim_data, cell_specs, kInfo, k):
    """
    Update RNA synthesis probabilities with fit values.

    Multiplies per-copy probabilities by gene copy numbers and normalizes.
    """
    replication_coordinate = sim_data.process.transcription.rna_data[
        "replication_coordinate"
    ]

    for D, k_value in zip(kInfo, k):
        condition = D["condition"]
        rna_idx = D["idx"]
        rnaCoordinate = replication_coordinate[rna_idx]
        tau = cell_specs[condition]["doubling_time"].asNumber(units.min)
        n_avg_copy = sim_data.process.replication.get_average_copy_number(
            tau, rnaCoordinate
        )

        sim_data.process.transcription.rna_synth_prob[condition][rna_idx] = (
            max(0, k_value) * n_avg_copy
        )

    for condition in sim_data.process.transcription.rna_synth_prob:
        assert np.all(sim_data.process.transcription.rna_synth_prob[condition] >= 0)
        sim_data.process.transcription.rna_synth_prob[condition] /= (
            sim_data.process.transcription.rna_synth_prob[condition].sum()
        )


# ============================================================================
# Main promoter fitting functions
# ============================================================================


def fitPromoterBoundProbability(sim_data, cell_specs):
    r"""
    Fit probabilities that each TF binds to its target promoter.

    Uses convex optimization to find parameters alpha and r such that:
        v_{synth, j} = alpha_j + sum_i P_{T,i} * r_{ij}

    Modifies:
        sim_data.pPromoterBound: Fitted TF-promoter binding probabilities
        sim_data.process.transcription.rna_synth_prob: Updated synthesis probabilities
        cell_specs['basal']['r_vector']: Fitted r values
        cell_specs['basal']['r_columns']: Column name to index mapping
    """
    # Initialize pPromoterBound using mean TF and ligand concentrations
    pPromoterBound = calculatePromoterBoundProbability(sim_data, cell_specs)
    pInit0 = None
    lastNorm = np.inf

    # Identify TFs with fixed activities
    fixedTFs = []
    for tf in sim_data.tf_to_active_inactive_conditions:
        if sim_data.process.transcription_regulation.tf_to_tf_type[tf] == "2CS":
            fixedTFs.append(tf)
        if (
            sim_data.process.transcription_regulation.tf_to_tf_type[tf] == "1CS"
            and sim_data.tf_to_active_inactive_conditions[tf]["active nutrients"]
            == sim_data.tf_to_active_inactive_conditions[tf]["inactive nutrients"]
        ):
            fixedTFs.append(tf)

    # Build vector of existing fit transcription probabilities
    k, kInfo = _build_vector_k(sim_data, cell_specs)

    # Iterate optimization
    for i in range(PROMOTER_MAX_ITERATIONS):
        # Build matrices for R optimization
        G, G_row_name_to_index, G_col_name_to_index = _build_matrix_G(
            sim_data, pPromoterBound
        )
        Z = _build_matrix_Z(sim_data, G_col_name_to_index)
        T = _build_matrix_T(sim_data, G_col_name_to_index)

        # Optimize R
        R = Variable(G.shape[1])
        objective_r = Minimize(
            norm(G @ (PROMOTER_SCALING * R) - PROMOTER_SCALING * k, PROMOTER_NORM_TYPE)
        )
        constraint_r = [
            0 <= Z @ (PROMOTER_SCALING * R),
            Z @ (PROMOTER_SCALING * R) <= PROMOTER_SCALING,
            T @ (PROMOTER_SCALING * R) >= 0,
        ]

        prob_r = Problem(objective_r, constraint_r)
        prob_r.solve(solver="ECOS", max_iters=1000)

        if prob_r.status == "optimal_inaccurate":
            raise RuntimeError(
                "Solver found an optimum that is inaccurate."
                " Try increasing max_iters or adjusting tolerances."
            )
        elif prob_r.status != "optimal":
            raise RuntimeError("Solver could not find optimal value")

        r = np.array(R.value).reshape(-1)
        r[np.abs(r) < ECOS_0_TOLERANCE] = 0

        # Build matrices for P optimization
        (H, pInit, pAlphaIdxs, pNotAlphaIdxs, fixedTFIdxs,
         pPromoterBoundIdxs, H_col_name_to_index) = _build_matrix_H(
            sim_data, G_col_name_to_index, pPromoterBound, r, fixedTFs, cell_specs
        )
        pdiff = _build_matrix_pdiff(sim_data, H_col_name_to_index)

        if i == 0:
            pInit0 = pInit.copy()

        # Optimize P
        P = Variable(H.shape[1])
        D = np.zeros(H.shape[1])
        D[pAlphaIdxs] = 1
        D[fixedTFIdxs] = 1

        Drhs = pInit0.copy()
        Drhs[D != 1] = 0

        objective_p = Minimize(
            norm(H @ (PROMOTER_SCALING * P) - PROMOTER_SCALING * k, PROMOTER_NORM_TYPE)
            + PROMOTER_REG_COEFF * norm(P - pInit0, PROMOTER_NORM_TYPE)
        )

        constraint_p = [
            0 <= PROMOTER_SCALING * P,
            PROMOTER_SCALING * P <= PROMOTER_SCALING,
            np.diag(D) @ (PROMOTER_SCALING * P) == PROMOTER_SCALING * Drhs,
            pdiff @ (PROMOTER_SCALING * P) >= PROMOTER_SCALING * PROMOTER_PDIFF_THRESHOLD,
        ]

        prob_p = Problem(objective_p, constraint_p)
        prob_p.solve(solver="ECOS")

        if prob_p.status == "optimal_inaccurate":
            raise RuntimeError(
                "Solver found an optimum that is inaccurate."
                " Try increasing max_iters or adjusting tolerances."
            )
        elif prob_p.status != "optimal":
            raise RuntimeError("Solver could not find optimal value")

        p = np.array(P.value).reshape(-1)
        p[p < ECOS_0_TOLERANCE] = 0
        p[p > (1 - ECOS_0_TOLERANCE)] = 1

        _update_p_promoter_bound(p, pPromoterBound, pPromoterBoundIdxs)

        # Check convergence
        if (
            np.abs(np.linalg.norm(np.dot(H, p) - k, PROMOTER_NORM_TYPE) - lastNorm)
            < PROMOTER_CONVERGENCE_THRESHOLD
        ):
            break
        else:
            lastNorm = np.linalg.norm(np.dot(H, p) - k, PROMOTER_NORM_TYPE)

    # Update sim_data with results
    sim_data.pPromoterBound = pPromoterBound
    _update_synth_prob(sim_data, cell_specs, kInfo, np.dot(H, p))

    return r, G_col_name_to_index


def fitLigandConcentrations(sim_data, cell_specs):
    """
    Update ligand concentrations and Kd values based on fitted pPromoterBound.

    For 1CS TFs, adjusts the set concentrations of ligand metabolites and
    the Kd values of ligand-TF binding reactions to match fitted probabilities.

    Modifies:
        sim_data.process.metabolism.concentration_updates.molecule_set_amounts
        sim_data.process.equilibrium reverse rates
    """
    cellDensity = sim_data.constants.cell_density
    pPromoterBound = sim_data.pPromoterBound

    for tf in sorted(sim_data.tf_to_active_inactive_conditions):
        # Skip non-1CS TFs and those with genotypic perturbations
        if sim_data.process.transcription_regulation.tf_to_tf_type[tf] != "1CS":
            continue
        if (
            len(sim_data.tf_to_active_inactive_conditions[tf]["active genotype perturbations"]) > 0
            or len(sim_data.tf_to_active_inactive_conditions[tf]["inactive genotype perturbations"]) > 0
        ):
            continue

        activeKey = tf + "__active"
        inactiveKey = tf + "__inactive"

        boundId = sim_data.process.transcription_regulation.active_to_bound[tf]
        negativeSignal = tf != boundId

        fwdRate = sim_data.process.equilibrium.get_fwd_rate(boundId + "[c]")
        revRate = sim_data.process.equilibrium.get_rev_rate(boundId + "[c]")
        kd = revRate / fwdRate

        metabolite = sim_data.process.equilibrium.get_metabolite(boundId + "[c]")
        metaboliteCoeff = sim_data.process.equilibrium.get_metabolite_coeff(boundId + "[c]")

        metabolite_idx = bulk_name_to_idx(
            metabolite, cell_specs[activeKey]["bulkAverageContainer"]["id"]
        )
        activeCellVolume = (
            cell_specs[activeKey]["avgCellDryMassInit"]
            / cellDensity
            / sim_data.mass.cell_dry_mass_fraction
        )
        activeCountsToMolar = 1 / (sim_data.constants.n_avogadro * activeCellVolume)
        activeSignalConc = (
            activeCountsToMolar
            * counts(cell_specs[activeKey]["bulkAverageContainer"], metabolite_idx)
        ).asNumber(units.mol / units.L)

        inactiveCellVolume = (
            cell_specs[inactiveKey]["avgCellDryMassInit"]
            / cellDensity
            / sim_data.mass.cell_dry_mass_fraction
        )
        inactiveCountsToMolar = 1 / (sim_data.constants.n_avogadro * inactiveCellVolume)
        inactiveSignalConc = (
            inactiveCountsToMolar
            * counts(cell_specs[inactiveKey]["bulkAverageContainer"], metabolite_idx)
        ).asNumber(units.mol / units.L)

        p_active = pPromoterBound[activeKey][tf]
        p_inactive = pPromoterBound[inactiveKey][tf]

        if negativeSignal:
            if p_inactive == 0:
                raise ValueError(
                    "Inf ligand concentration from p_inactive = 0."
                    " Check results from fitPromoterBoundProbability and Kd values."
                )
            if 1 - p_active < 1e-9:
                kdNew = kd
            else:
                kdNew = (
                    (activeSignalConc**metaboliteCoeff) * p_active / (1 - p_active)
                ) ** (1 / metaboliteCoeff)

            sim_data.process.metabolism.concentration_updates.molecule_set_amounts[
                metabolite
            ] = (kdNew**metaboliteCoeff * (1 - p_inactive) / p_inactive) ** (
                1.0 / metaboliteCoeff
            ) * (units.mol / units.L)
        else:
            if p_active == 1:
                raise ValueError(
                    "Inf ligand concentration from p_active = 1."
                    " Check results from fitPromoterBoundProbability and Kd values."
                )
            if p_inactive < 1e-9:
                kdNew = kd
            else:
                kdNew = (
                    (inactiveSignalConc**metaboliteCoeff)
                    * (1 - p_inactive)
                    / p_inactive
                ) ** (1 / metaboliteCoeff)

            sim_data.process.metabolism.concentration_updates.molecule_set_amounts[
                metabolite
            ] = (kdNew**metaboliteCoeff * p_active / (1 - p_active)) ** (
                1.0 / metaboliteCoeff
            ) * (units.mol / units.L)

        sim_data.process.equilibrium.set_rev_rate(boundId + "[c]", kdNew * fwdRate)


def calculatePromoterBoundProbability(sim_data, cell_specs):
    """
    Calculate initial TF-promoter binding probabilities from bulk counts.

    Computes probabilities based on TF type (0CS, 1CS, 2CS) and bulk
    average concentrations of TFs and their ligands.

    Returns:
        pPromoterBound: Dict[condition][TF] -> probability
    """
    pPromoterBound = {}
    cellDensity = sim_data.constants.cell_density
    init_to_average = sim_data.mass.avg_cell_to_initial_cell_conversion_factor

    # Build regulation matrix for TF target counting
    tf_idx = {tf: i for i, tf in enumerate(sim_data.tf_to_active_inactive_conditions)}
    cistron_id_to_tu_indexes = {
        cistron_id: sim_data.process.transcription.cistron_id_to_rna_indexes(cistron_id)
        for cistron_id in sim_data.process.transcription.cistron_data["id"]
    }
    regulation_i, regulation_j, regulation_v = [], [], []
    for tf, cistrons in sim_data.tf_to_fold_change.items():
        if tf not in tf_idx:
            continue
        for cistron in cistrons:
            for tu_index in cistron_id_to_tu_indexes[cistron]:
                regulation_i.append(tf_idx[tf])
                regulation_j.append(tu_index)
                regulation_v.append(1)

    regulation = scipy.sparse.csr_matrix(
        (regulation_v, (regulation_i, regulation_j)),
        shape=(len(tf_idx), len(sim_data.process.transcription.rna_data)),
    )
    rna_coords = sim_data.process.transcription.rna_data["replication_coordinate"]

    # Get all TF IDs (not just the filtered ones in smoke mode)
    all_tf_ids = sim_data.process.transcription_regulation.tf_ids

    for conditionKey in sorted(cell_specs):
        # Initialize all TFs with 0.0 so calculate_attenuation can access them
        # even for TFs not fitted in smoke mode
        pPromoterBound[conditionKey] = {tf: 0.0 for tf in all_tf_ids}

        tau = sim_data.condition_to_doubling_time[conditionKey].asNumber(units.min)
        n_avg_copy = sim_data.process.replication.get_average_copy_number(tau, rna_coords)
        n_promoter_targets = regulation.dot(n_avg_copy)

        cellVolume = (
            cell_specs[conditionKey]["avgCellDryMassInit"]
            / cellDensity
            / sim_data.mass.cell_dry_mass_fraction
        )
        countsToMolar = 1 / (sim_data.constants.n_avogadro * cellVolume)

        # Only compute probabilities for TFs being fitted (subset in smoke mode)
        for tf in sorted(sim_data.tf_to_active_inactive_conditions):
            tfType = sim_data.process.transcription_regulation.tf_to_tf_type[tf]
            curr_tf_idx = bulk_name_to_idx(
                tf + "[c]", cell_specs[conditionKey]["bulkAverageContainer"]["id"]
            )
            tf_counts = counts(cell_specs[conditionKey]["bulkAverageContainer"], curr_tf_idx)
            tf_targets = n_promoter_targets[tf_idx[tf]]
            limited_tf_counts = min(1, tf_counts * init_to_average / tf_targets)

            if tfType == "0CS":
                pPromoterBound[conditionKey][tf] = limited_tf_counts

            elif tfType == "1CS":
                boundId = sim_data.process.transcription_regulation.active_to_bound[tf]
                kd = sim_data.process.equilibrium.get_rev_rate(
                    boundId + "[c]"
                ) / sim_data.process.equilibrium.get_fwd_rate(boundId + "[c]")

                signal = sim_data.process.equilibrium.get_metabolite(boundId + "[c]")
                signalCoeff = sim_data.process.equilibrium.get_metabolite_coeff(boundId + "[c]")

                signal_idx = bulk_name_to_idx(
                    signal, cell_specs[conditionKey]["bulkAverageContainer"]["id"]
                )
                signalConc = (
                    countsToMolar
                    * counts(cell_specs[conditionKey]["bulkAverageContainer"], signal_idx)
                ).asNumber(units.mol / units.L)
                tfConc = (countsToMolar * tf_counts).asNumber(units.mol / units.L)

                if tf == boundId:
                    if tfConc > 0:
                        pPromoterBound[conditionKey][tf] = (
                            limited_tf_counts
                            * sim_data.process.transcription_regulation.p_promoter_bound_SKd(
                                signalConc, kd, signalCoeff
                            )
                        )
                    else:
                        pPromoterBound[conditionKey][tf] = 0.0
                else:
                    if tfConc > 0:
                        pPromoterBound[conditionKey][tf] = (
                            1.0
                            - limited_tf_counts
                            * sim_data.process.transcription_regulation.p_promoter_bound_SKd(
                                signalConc, kd, signalCoeff
                            )
                        )
                    else:
                        pPromoterBound[conditionKey][tf] = 0.0

            elif tfType == "2CS":
                activeTfConc = (countsToMolar * tf_counts).asNumber(units.mol / units.L)
                inactiveTf = sim_data.process.two_component_system.active_to_inactive_tf[tf + "[c]"]
                inactive_tf_idx = bulk_name_to_idx(
                    inactiveTf, cell_specs[conditionKey]["bulkAverageContainer"]["id"]
                )
                inactiveTfConc = (
                    countsToMolar
                    * counts(cell_specs[conditionKey]["bulkAverageContainer"], inactive_tf_idx)
                ).asNumber(units.mol / units.L)

                if activeTfConc == 0 and inactiveTfConc == 0:
                    pPromoterBound[conditionKey][tf] = 0.0
                else:
                    pPromoterBound[conditionKey][tf] = (
                        limited_tf_counts * activeTfConc / (activeTfConc + inactiveTfConc)
                    )

    # Check for inconsistencies
    for condition in pPromoterBound:
        if "inactive" in condition:
            tf = condition.split("__")[0]
            active_p = pPromoterBound[f"{tf}__active"][tf]
            inactive_p = pPromoterBound[f"{tf}__inactive"][tf]

            if inactive_p >= active_p:
                print(
                    "Warning: active condition does not have higher binding"
                    f" probability than inactive condition for {tf}"
                    f" ({active_p:.3f} vs {inactive_p:.3f})."
                )

    return pPromoterBound


def calculateRnapRecruitment(sim_data, cell_specs):
    """
    Construct basal_prob vector and delta_prob matrix from fitted r values.

    basal_prob contains the basal transcription probability for each TU.
    delta_prob contains the probability changes when TFs bind.

    Returns:
        basal_prob: np.ndarray of basal transcription probabilities
        delta_prob: dict with deltaI, deltaJ, deltaV, shape
    """
    r = cell_specs["basal"]["r_vector"]
    col_names_to_index = cell_specs["basal"]["r_columns"]

    transcription = sim_data.process.transcription
    transcription_regulation = sim_data.process.transcription_regulation
    all_TUs = transcription.rna_data["id"]
    all_tfs = transcription_regulation.tf_ids

    basal_prob = np.zeros(len(all_TUs))
    deltaI, deltaJ, deltaV = [], [], []

    for rna_idx, rnaId in enumerate(all_TUs):
        rnaIdNoLoc = rnaId[:-3]

        for tf in sim_data.relation.rna_id_to_regulating_tfs.get(rnaId, []):
            if tf not in sorted(sim_data.tf_to_active_inactive_conditions):
                continue

            colName = rnaIdNoLoc + "__" + tf
            deltaI.append(rna_idx)
            deltaJ.append(all_tfs.index(tf))
            deltaV.append(r[col_names_to_index[colName]])

        colName = rnaIdNoLoc + "__alpha"
        basal_prob[rna_idx] = r[col_names_to_index[colName]]

    deltaI, deltaJ, deltaV = np.array(deltaI), np.array(deltaJ), np.array(deltaV)
    delta_shape = (len(all_TUs), len(all_tfs))

    basal_prob[basal_prob < 0] = 0

    return basal_prob, {
        "deltaI": deltaI,
        "deltaJ": deltaJ,
        "deltaV": deltaV,
        "shape": delta_shape,
    }
