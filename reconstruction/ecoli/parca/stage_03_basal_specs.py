"""
Stage 3: basal_specs — Build basal cell specifications.

Three public functions:
    extract_input(sim_data, cell_specs, **kwargs) -> BasalSpecsInput
    compute_basal_specs(inp) -> BasalSpecsOutput
    merge_output(sim_data, cell_specs, out)

This stage:
1. Runs expressionConverge to iteratively fit RNA expression and synthesis
   probabilities for the basal condition
2. Sets ppGpp-regulated expression
3. Fits Km values for endoRNase-mediated RNA decay
4. Fits growth-associated maintenance costs

NOTE: compute_basal_specs mutates sim_data_ref because downstream
sub-functions (set_ppgpp_expression, Km fitting, maintenance fitting)
depend on mass and expression values set by expressionConverge.
merge_output only writes cell_specs["basal"].
"""

import binascii
import functools
import os
import pickle

import numpy as np
import scipy.optimize

from ecoli.library.schema import bulk_name_to_idx, counts
from wholecell.utils import units

from reconstruction.ecoli.parca._shared import (
    VERBOSE,
    expressionConverge,
    rescaleMassForSolubleMetabolites,
)
from reconstruction.ecoli.parca._types import BasalSpecsInput, BasalSpecsOutput


# ============================================================================
# Extract / Merge
# ============================================================================


def extract_input(sim_data, cell_specs, **kwargs) -> BasalSpecsInput:
    """Pull configuration from kwargs and pass sim_data as mutable ref."""
    return BasalSpecsInput(
        variable_elongation_transcription=kwargs.get(
            "variable_elongation_transcription", True
        ),
        variable_elongation_translation=kwargs.get(
            "variable_elongation_translation", False
        ),
        disable_ribosome_capacity_fitting=kwargs.get(
            "disable_ribosome_capacity_fitting", False
        ),
        disable_rnapoly_capacity_fitting=kwargs.get(
            "disable_rnapoly_capacity_fitting", False
        ),
        cache_dir=kwargs.get("cache_dir"),
        sim_data_ref=sim_data,
    )


def merge_output(sim_data, cell_specs, out: BasalSpecsOutput):
    """Write computed results into cell_specs["basal"].

    sim_data mutations are already applied by compute_basal_specs
    via sim_data_ref.
    """
    cell_specs["basal"] = {
        "concDict": out.conc_dict,
        "expression": out.expression,
        "synthProb": out.synth_prob,
        "fit_cistron_expression": out.fit_cistron_expression,
        "doubling_time": out.doubling_time,
        "avgCellDryMassInit": out.avg_cell_dry_mass_init,
        "fitAvgSolubleTargetMolMass": out.fit_avg_soluble_target_mol_mass,
        "bulkContainer": out.bulk_container,
    }


# ============================================================================
# Compute
# ============================================================================


def compute_basal_specs(inp: BasalSpecsInput) -> BasalSpecsOutput:
    """Run the full basal_specs stage.

    This function mutates inp.sim_data_ref as a side effect because
    sub-functions (set_ppgpp_expression, setKm, fitMaintenanceCosts)
    depend on earlier sim_data updates.
    """
    sim_data = inp.sim_data_ref

    # --- Step 1: Build basal cell specifications via expressionConverge ---
    conc_dict = (
        sim_data.process.metabolism.concentration_updates
        .concentrations_based_on_nutrients(media_id="minimal")
    )
    expression = sim_data.process.transcription.rna_expression["basal"].copy()
    doubling_time = sim_data.condition_to_doubling_time["basal"]

    (
        expression,
        synth_prob,
        fit_cistron_expression,
        avg_cell_dry_mass_init,
        fit_avg_soluble_target_mol_mass,
        bulk_container,
        _,
    ) = expressionConverge(
        sim_data,
        expression,
        conc_dict,
        doubling_time,
        conditionKey="basal",
        variable_elongation_transcription=inp.variable_elongation_transcription,
        variable_elongation_translation=inp.variable_elongation_translation,
        disable_ribosome_capacity_fitting=inp.disable_ribosome_capacity_fitting,
        disable_rnapoly_capacity_fitting=inp.disable_rnapoly_capacity_fitting,
    )

    # --- Apply sim_data mass updates (needed by downstream functions) ---
    sim_data.mass.avg_cell_dry_mass_init = avg_cell_dry_mass_init
    sim_data.mass.avg_cell_dry_mass = (
        sim_data.mass.avg_cell_dry_mass_init
        * sim_data.mass.avg_cell_to_initial_cell_conversion_factor
    )
    sim_data.mass.avg_cell_water_mass_init = (
        sim_data.mass.avg_cell_dry_mass_init
        / sim_data.mass.cell_dry_mass_fraction
        * sim_data.mass.cell_water_mass_fraction
    )
    sim_data.mass.fitAvgSolubleTargetMolMass = fit_avg_soluble_target_mol_mass

    # --- Apply sim_data expression updates (needed by set_ppgpp_expression) ---
    sim_data.process.transcription.rna_expression["basal"][:] = expression
    sim_data.process.transcription.rna_synth_prob["basal"][:] = synth_prob
    sim_data.process.transcription.fit_cistron_expression["basal"] = (
        fit_cistron_expression
    )

    # --- Step 2: Set ppGpp-regulated expression ---
    sim_data.process.transcription.set_ppgpp_expression(sim_data)

    # --- Step 3: Fit Km values for endoRNase-mediated RNA decay ---
    Km = setKmCooperativeEndoRNonLinearRNAdecay(
        sim_data, bulk_container, inp.cache_dir
    )
    n_transcribed_rnas = len(sim_data.process.transcription.rna_data)
    sim_data.process.transcription.rna_data["Km_endoRNase"] = (
        Km[:n_transcribed_rnas]
    )
    sim_data.process.transcription.mature_rna_data["Km_endoRNase"] = (
        Km[n_transcribed_rnas:]
    )

    # --- Step 4: Fit maintenance costs ---
    fitMaintenanceCosts(sim_data, bulk_container)

    return BasalSpecsOutput(
        conc_dict=conc_dict,
        expression=expression,
        synth_prob=synth_prob,
        fit_cistron_expression=fit_cistron_expression,
        doubling_time=doubling_time,
        avg_cell_dry_mass_init=avg_cell_dry_mass_init,
        fit_avg_soluble_target_mol_mass=fit_avg_soluble_target_mol_mass,
        bulk_container=bulk_container,
    )


# ============================================================================
# Sub-functions (ported from fit_sim_data_1.py)
# ============================================================================


def _crc32(*arrays: np.ndarray, initial: int = 0) -> int:
    """Return a CRC32 checksum of the given ndarrays."""

    def crc_next(initial, array):
        shape = str(array.shape).encode()
        values = array.tobytes()
        return binascii.crc32(values, binascii.crc32(shape, initial))

    return functools.reduce(crc_next, arrays, initial)


def setKmCooperativeEndoRNonLinearRNAdecay(sim_data, bulkContainer, cache_dir):
    """
    Fit Michaelis-Menten constants for RNAs binding to endoRNases.

    Returns:
        Km values in units of M (mol/L)
    """

    def arrays_differ(a: np.ndarray, b: np.ndarray) -> bool:
        return a.shape != b.shape or not np.allclose(a, b, equal_nan=True)

    cellDensity = sim_data.constants.cell_density
    cellVolume = (
        sim_data.mass.avg_cell_dry_mass_init
        / cellDensity
        / sim_data.mass.cell_dry_mass_fraction
    )
    countsToMolar = 1 / (sim_data.constants.n_avogadro * cellVolume)

    degradable_rna_ids = np.concatenate(
        (
            sim_data.process.transcription.rna_data["id"],
            sim_data.process.transcription.mature_rna_data["id"],
        )
    )
    degradation_rates = (1 / units.s) * np.concatenate(
        (
            sim_data.process.transcription.rna_data["deg_rate"].asNumber(1 / units.s),
            sim_data.process.transcription.mature_rna_data["deg_rate"].asNumber(
                1 / units.s
            ),
        )
    )
    endoRNase_idx = bulk_name_to_idx(
        sim_data.process.rna_decay.endoRNase_ids, bulkContainer["id"]
    )
    endoRNaseConc = countsToMolar * counts(bulkContainer, endoRNase_idx)
    kcatEndoRNase = sim_data.process.rna_decay.kcats
    totalEndoRnaseCapacity = units.sum(endoRNaseConc * kcatEndoRNase)

    endoRnaseRnaIds = sim_data.molecule_groups.endoRNase_rnas
    isEndoRnase = np.array([(x in endoRnaseRnaIds) for x in degradable_rna_ids])

    degradable_rna_idx = bulk_name_to_idx(degradable_rna_ids, bulkContainer["id"])
    rna_counts = counts(bulkContainer, degradable_rna_idx)
    rna_conc = countsToMolar * rna_counts
    Km_counts = (
        (1 / degradation_rates * totalEndoRnaseCapacity) - rna_conc
    ).asNumber()
    sim_data.process.rna_decay.Km_first_order_decay = Km_counts

    # Compute derivative g(Km)
    KmQuadratic = 1 / np.power((1 / countsToMolar * Km_counts).asNumber(), 2)
    denominator = np.power(
        np.sum(rna_counts / (1 / countsToMolar * Km_counts).asNumber()), 2
    )
    numerator = (1 / countsToMolar * totalEndoRnaseCapacity).asNumber() * (
        denominator - (rna_counts / (1 / countsToMolar * Km_counts).asNumber())
    )
    gDerivative = np.abs(KmQuadratic * (1 - (numerator / denominator)))
    if VERBOSE:
        print("Max derivative (counts) = %f" % max(gDerivative))

    KmQuadratic = 1 / np.power(Km_counts, 2)
    denominator = np.power(np.sum(rna_conc.asNumber() / Km_counts), 2)
    numerator = totalEndoRnaseCapacity.asNumber() * (
        denominator - (rna_conc.asNumber() / Km_counts)
    )
    gDerivative = np.abs(KmQuadratic * (1 - (numerator / denominator)))
    if VERBOSE:
        print("Max derivative (concentration) = %f" % max(gDerivative))

    # Sensitivity analysis: alpha
    Alphas = []
    if sim_data.constants.sensitivity_analysis_alpha:
        Alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

    total_endo_rnase_capacity_mol_l_s = totalEndoRnaseCapacity.asNumber(
        units.mol / units.L / units.s
    )
    rna_conc_mol_l = rna_conc.asNumber(units.mol / units.L)
    degradation_rates_s = degradation_rates.asNumber(1 / units.s)

    for alpha in Alphas:
        if VERBOSE:
            print("Alpha = %f" % alpha)

        loss, loss_jac, res, res_aux = sim_data.process.rna_decay.km_loss_function(
            total_endo_rnase_capacity_mol_l_s,
            rna_conc_mol_l,
            degradation_rates_s,
            isEndoRnase,
            alpha,
        )
        Km_cooperative_model = np.exp(
            scipy.optimize.minimize(loss, np.log(Km_counts), jac=loss_jac).x
        )
        sim_data.process.rna_decay.sensitivity_analysis_alpha_residual[alpha] = np.sum(
            np.abs(res_aux(Km_cooperative_model))
        )

    alpha = 0.5

    # Sensitivity analysis: kcatEndoRNase
    kcatEndo = []
    if sim_data.constants.sensitivity_analysis_kcat_endo:
        kcatEndo = [0.0001, 0.001, 0.01, 0.1, 1, 10]

    for kcat in kcatEndo:
        if VERBOSE:
            print("Kcat = %f" % kcat)

        totalEndoRNcap = units.sum(endoRNaseConc * kcat)
        loss, loss_jac, res, res_aux = sim_data.process.rna_decay.km_loss_function(
            totalEndoRNcap.asNumber(units.mol / units.L),
            rna_conc_mol_l,
            degradation_rates_s,
            isEndoRnase,
            alpha,
        )
        km_counts_ini = (
            (totalEndoRNcap / degradation_rates.asNumber()) - rna_conc
        ).asNumber()
        Km_cooperative_model = np.exp(
            scipy.optimize.minimize(loss, np.log(km_counts_ini), jac=loss_jac).x
        )
        sim_data.process.rna_decay.sensitivity_analysis_kcat[kcat] = (
            Km_cooperative_model
        )
        sim_data.process.rna_decay.sensitivity_analysis_kcat_res_ini[kcat] = np.sum(
            np.abs(res_aux(km_counts_ini))
        )
        sim_data.process.rna_decay.sensitivity_analysis_kcat_res_opt[kcat] = np.sum(
            np.abs(res_aux(Km_cooperative_model))
        )

    # Loss function and derivative
    loss, loss_jac, res, res_aux = sim_data.process.rna_decay.km_loss_function(
        total_endo_rnase_capacity_mol_l_s,
        rna_conc_mol_l,
        degradation_rates_s,
        isEndoRnase,
        alpha,
    )

    # Cache handling
    needToUpdate = ""
    checksum = _crc32(Km_counts, isEndoRnase, np.array(alpha))
    km_filepath = os.path.join(cache_dir, f"parca-km-{checksum}.cPickle")

    if os.path.exists(km_filepath):
        with open(km_filepath, "rb") as f:
            Km_cache = pickle.load(f)

        Km_cooperative_model = Km_cache["Km_cooperative_model"]
        if (
            Km_counts.shape != Km_cooperative_model.shape
            or np.sum(np.abs(res_aux(Km_cooperative_model))) > 1e-15
            or arrays_differ(
                Km_cache["total_endo_rnase_capacity_mol_l_s"],
                total_endo_rnase_capacity_mol_l_s,
            )
            or arrays_differ(Km_cache["rna_conc_mol_l"], rna_conc_mol_l)
            or arrays_differ(Km_cache["degradation_rates_s"], degradation_rates_s)
        ):
            needToUpdate = "recompute"
    else:
        needToUpdate = "compute"

    if needToUpdate:
        if VERBOSE:
            print(f"Running non-linear optimization to {needToUpdate} {km_filepath}")
        sol = scipy.optimize.minimize(
            loss, np.log(Km_counts), jac=loss_jac, tol=1e-8
        )
        Km_cooperative_model = np.exp(sol.x)
        Km_cache = dict(
            Km_cooperative_model=Km_cooperative_model,
            total_endo_rnase_capacity_mol_l_s=total_endo_rnase_capacity_mol_l_s,
            rna_conc_mol_l=rna_conc_mol_l,
            degradation_rates_s=degradation_rates_s,
        )

        with open(km_filepath, "wb") as f:
            pickle.dump(Km_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if VERBOSE:
            print(
                "Not running non-linear optimization--using cached result {}".format(
                    km_filepath
                )
            )

    # Calculate log Km for loss functions
    log_Km_cooperative_model = np.log(Km_cooperative_model)
    log_Km_counts = np.log(Km_counts)

    if VERBOSE > 1:
        print("Loss function (Km inital) = %f" % np.sum(np.abs(loss(log_Km_counts))))
        print(
            "Loss function (optimized Km) = %f"
            % np.sum(np.abs(loss(log_Km_cooperative_model)))
        )
        print("Residuals (Km initial) = %f" % np.sum(np.abs(res(Km_counts))))
        print("Residuals optimized = %f" % np.sum(np.abs(res(Km_cooperative_model))))
        print(
            "EndoR residuals (Km initial) = %f"
            % np.sum(np.abs(isEndoRnase * res(Km_counts)))
        )
        print(
            "EndoR residuals optimized = %f"
            % np.sum(np.abs(isEndoRnase * res(Km_cooperative_model)))
        )
        print(
            "Residuals (scaled by Kdeg * RNAcounts) Km initial = %f"
            % np.sum(np.abs(res_aux(Km_counts)))
        )
        print(
            "Residuals (scaled by Kdeg * RNAcounts) optimized = %f"
            % np.sum(np.abs(res_aux(Km_cooperative_model)))
        )

    # Save statistics
    sim_data.process.rna_decay.stats_fit["LossKm"] = np.sum(
        np.abs(loss(log_Km_counts))
    )
    sim_data.process.rna_decay.stats_fit["LossKmOpt"] = np.sum(
        np.abs(loss(log_Km_cooperative_model))
    )
    sim_data.process.rna_decay.stats_fit["ResKm"] = np.sum(np.abs(res(Km_counts)))
    sim_data.process.rna_decay.stats_fit["ResKmOpt"] = np.sum(
        np.abs(res(Km_cooperative_model))
    )
    sim_data.process.rna_decay.stats_fit["ResEndoRNKm"] = np.sum(
        np.abs(isEndoRnase * res(Km_counts))
    )
    sim_data.process.rna_decay.stats_fit["ResEndoRNKmOpt"] = np.sum(
        np.abs(isEndoRnase * res(Km_cooperative_model))
    )
    sim_data.process.rna_decay.stats_fit["ResScaledKm"] = np.sum(
        np.abs(res_aux(Km_counts))
    )
    sim_data.process.rna_decay.stats_fit["ResScaledKmOpt"] = np.sum(
        np.abs(res_aux(Km_cooperative_model))
    )

    return units.mol / units.L * Km_cooperative_model


def fitMaintenanceCosts(sim_data, bulkContainer):
    """
    Fit growth-associated maintenance (GAM) cost.

    Modifies sim_data.constants.darkATP.
    """
    aaCounts = sim_data.process.translation.monomer_data["aa_counts"]
    protein_idx = bulk_name_to_idx(
        sim_data.process.translation.monomer_data["id"], bulkContainer["id"]
    )
    proteinCounts = counts(bulkContainer, protein_idx)
    nAvogadro = sim_data.constants.n_avogadro
    avgCellDryMassInit = sim_data.mass.avg_cell_dry_mass_init
    gtpPerTranslation = sim_data.constants.gtp_per_translation
    atp_per_charge = 2

    aaMmolPerGDCW = units.sum(
        aaCounts * np.tile(proteinCounts.reshape(-1, 1), (1, 21)), axis=0
    ) * ((1 / (units.aa * nAvogadro)) * (1 / avgCellDryMassInit))

    aasUsedOverCellCycle = units.sum(aaMmolPerGDCW)
    explicit_mmol_maintenance_per_gdcw = (
        atp_per_charge + gtpPerTranslation
    ) * aasUsedOverCellCycle

    darkATP = (
        sim_data.constants.growth_associated_maintenance
        - explicit_mmol_maintenance_per_gdcw
    )

    if darkATP.asNumber() < 0:
        raise ValueError(
            "GAM has been adjusted too low. Explicit energy accounting should not exceed GAM."
            " Consider setting darkATP to 0 if energy corrections are accurate."
        )

    sim_data.constants.darkATP = darkATP
