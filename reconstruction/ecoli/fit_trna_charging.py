"""
Optional ParCa step that sets tRNA charging from measured data and makes
charging, attenuation, amino acid supply and ppGpp consistent with each other
before the simulation starts.

Without this step every synthetase shares one rate constant (kS = 100 /s), so
charging capacity ranges from about 3x to 40x demand between amino acids, and
the ParCa calibrates attenuation as if all tRNA were charged. Genome-wide
measurements put tRNA charging in glucose minimal medium near 60%
(Avcilar-Kucukgoze et al. 2016, and note that this paper uses E. coli MC4100,
which is different from MG1655 (especially relA), but they use a 43 min
doubling time and that is pretty close to the 44 min target used in the ParCa,
and a tRNA-Phe Northern in MG1655 gives about 65% (Bullwinkle & Ibba, 2016),
so using this data for the model (which is closer to MG1655) is probably
reasonable for now). At that charging, the literature RelA constants
(Bosdriesz et al. 2015, fit alongside kS near full charging) would drive ppGpp
 far above its target, so they are refit to the ParCa's context here.

:py:func:`fit_charging_attenuation_and_ppgpp` does the following:

1. Fixed-point loop:

   a. Fit amino acid supply constants (synthesis, export, uptake) and the
      ppGpp kinetics spread from the current expected bulk counts (these
      depend on attenuation through the readthrough of attenuated operons).
   b. Per-synthetase charging rate constants are set once, before the loop,
      so each synthetase has its measured capacity (Dykeman 2025 kcat x copies
      per cell) on the model's own counts. (Optionally, one shared kS can be
      fit to a measured mean charged fraction instead; see
      DEFAULT_ANCHOR_CHARGED_FRACTION for why that is off by default.)
   c. Solve the steady-state tRNA charging and free amino acid concentrations
      for each fitted condition with the simulation's own rate laws
      (:py:func:`ecoli.processes.polypeptide_elongation.dcdt_jit` for
      charging and elongation, and the mechanistic synthesis, import and
      export functions for supply). Amino acids released by protein
      degradation are included in supply, matching the simulation with
      ``credit_recycled_aa``. Charging in the other conditions (rich) is a
      prediction of the shared kS.
   d. Recalibrate attenuation on that charged tRNA and repeat until kS and
      the charged fractions stop changing.

2. ppGpp kinetics, with RelA's maximum rate (k_RelA) fixed:

   a. The anchor condition's SpoT synthesis factor is set so SpoT alone holds
      ppGpp at the measured relA-null level (Murray and Bremer, 1996) at that
      condition's charging.
   b. KD_RelA (one scale on the per amino acid values) is fit so SpoT plus
      RelA holds the anchor condition at its target ppGpp.
   c. Every other fitted condition's SpoT factor is set from its own target
      ppGpp at its predicted charging (Murray and Bremer's principle that
      SpoT synthesis sets the difference between media, and so the value is the
      model's prediction).
"""

from typing import Any, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ecoli.library.initial_conditions import create_expected_bulk_container
from ecoli.library.schema import bulk_name_to_idx, counts
from ecoli.library.trna_charging_fit import ppgpp_rates, ribosome_elongation_rate
from wholecell.utils import units

UM = units.umol / units.L

# Conditions whose charging is solved (and used for attenuation)
DEFAULT_FIT_CONDITIONS = ("basal", "with_aa")
# Condition whose measured charging anchors kS, RelA and its SpoT factor
ANCHOR_CONDITION = "basal"
# Measured mean charged fraction in glucose minimal medium (Avcilar-Kucukgoze
# et al. 2016: "oscillates around 50-60%", MC4100, which carries relA1, but a
# tRNA-Phe Northern in MG1655 M9 gives about 65%, so assuming this data is ok
# to use now for the most part). Not used by default: with tRNA KMs
# (0.16-1.8 uM) far below tRNA levels (1-24 uM per amino acid) the
# charging rate law saturates in uncharged tRNA, so charging is either nearly
# full or collapses when capacity falls below demand. Stable intermediate
# charging would need a different rate law (e.g. inhibition by charged tRNA).
MEASURED_MINIMAL_CHARGED_FRACTION = 0.60
DEFAULT_ANCHOR_CHARGED_FRACTION: Optional[float] = None
# Condition whose growth rate sets the per-synthetase rate constants from
# Dykeman 2025 capacities (kcat x copies per cell)
SYNTHETASE_CAPACITY_CONDITIONS = ("basal", "with_aa")
# Smallest charging capacity/demand any synthetase gets in each fitted
# condition, at that condition's target amino acid pools.
#
# TODO (mia): fix this because 2.2 is not the value Dykeman 2025 used. Find a
#  better/less patchy fix here!
# 2.2 came from the 2.2-4.2x capacity/demand range computed with
# the model's counts and demand. Dykeman's own margins (S1 Tab E) are mostly
# 1.2-1.5x (Trp about 2.4x). It is a uniform design floor that compensates for
# strong charged-tRNA inhibition (KI = 3 x KM). Without it, ThrRS sits at about
# 1.2x in rich (on the r-protein operon, it rises less with growth rate than
# Dykeman's copies), leaving Thr about 65% uncharged. None turns the floor off.
MIN_SYNTHETASE_HEADROOM: Optional[float] = 2.2
# The floor applies only where the condition's target amino acid pool is at
# least this many KMs: near KM the amino acid is supply-limited, and more
# synthetase only drains the pool (seen for Trp in basal) without improving
# charging.
MIN_HEADROOM_POOL_OVER_KM = 2.0
# Inhibition of synthetases by their own charged tRNA, as a multiple of the
# uncharged tRNA KM (1 = charged and uncharged tRNA bind equally, Dykeman 2025's
# mechanistic scheme includes charged-tRNA-bound synthetase states, and for
# class I synthetases release of the charged tRNA is often rate limiting).
# With tRNA levels far above the KMs, this makes the uncharged fraction track
# demand/capacity instead of switching between nearly full charging and
# collapse. None turns it off.
# 3 puts mean basal charging near 72% (measured range about 60-80%); 1 gives
# about 52%, 10 about 84%
DEFAULT_CHARGED_TRNA_KI_OVER_KM: Optional[float] = 3.0
# relA-null ppGpp in the anchor condition: Murray and Bremer, 1996, glucose
# minimal, 24 pmol/OD460, about 46 uM with 1.9 uM per pmol/OD460
# (Ahn-Horst et al. 2022)
DEFAULT_RELA_NULL_PPGPP_UM = 46.0
MAX_LOOP_ITERATIONS = 20
# Fraction of the way each pass moves the charged fractions used for attenuation
# toward the new charging solve (1 = no damping). Damping only changes the path,
# not a converged answer.
ATTENUATION_DAMPING = 0.5
# Largest change in any amino acid's charged fraction to call the loop converged
CHARGED_FRACTION_TOLERANCE = 2e-3
# Integration time per round of the steady-state solve (s) and rounds allowed
STEADY_STATE_TIME = 2000.0
MAX_STEADY_STATE_ROUNDS = 10
STEADY_STATE_TOLERANCE = 1e-4
# Search ranges (multiples of the current value) for kS and KD_RelA
KS_RANGE = (1e-3, 10.0)
KD_RELA_RANGE = (1e-3, 1e5)


def _dry_mass_of_counts(sim_data, container) -> float:
    """
    Dry mass (fg) of a bulk container, as the simulation's mass listener sums it.
    """
    molecule_masses = sim_data.internal_state.bulk_molecules.bulk_data["mass"].asNumber(
        units.fg / units.count
    )
    water_index = sim_data.submass_name_to_index["water"]
    dry_masses = molecule_masses.sum(axis=1) - molecule_masses[:, water_index]
    return float(container["count"] @ dry_masses)


def condition_charging_state(sim_data, condition: str) -> dict[str, Any]:
    """
    Concentrations (uM), counts and rate parameters for one condition, from the
    expected bulk counts the simulation starts from.
    """
    transcription = sim_data.process.transcription
    metabolism = sim_data.process.metabolism
    translation = sim_data.process.translation
    constants = sim_data.constants
    growth = sim_data.growth_rate_parameters
    nutrients = sim_data.conditions[condition]["nutrients"]
    doubling_time = sim_data.condition_to_doubling_time[condition]

    container = create_expected_bulk_container(sim_data, condition=condition)
    ids = container["id"]

    # Volume from the dry mass of the counts present, as in the simulation
    dry_mass = _dry_mass_of_counts(sim_data, container) * units.fg
    volume = dry_mass / sim_data.mass.cell_dry_mass_fraction / constants.cell_density
    to_um = (1 / (constants.n_avogadro * volume)).asNumber(UM)

    trna_counts = counts(
        container, bulk_name_to_idx(transcription.uncharged_trna_names, ids)
    ) + counts(container, bulk_name_to_idx(transcription.charged_trna_names, ids))
    total_trna = transcription.aa_from_trna @ trna_counts * to_um
    synthetase = (
        transcription.aa_from_synthetase
        @ counts(container, bulk_name_to_idx(transcription.synthetase_names, ids))
        * to_um
    )
    rela = counts(container, bulk_name_to_idx(sim_data.molecule_ids.RelA, ids)) * to_um
    spot = counts(container, bulk_name_to_idx(sim_data.molecule_ids.SpoT, ids)) * to_um
    n_30s = counts(
        container, bulk_name_to_idx(sim_data.molecule_ids.s30_full_complex, ids)
    )
    n_50s = counts(
        container, bulk_name_to_idx(sim_data.molecule_ids.s50_full_complex, ids)
    )
    ribosome_from_subunits = (
        min(n_30s, n_50s) * growth.get_fraction_active_ribosome(doubling_time) * to_um
    )

    aa_ids = sim_data.molecule_groups.amino_acids
    conc = metabolism.concentration_updates.concentrations_based_on_nutrients(
        media_id=nutrients
    )
    aa_conc = np.array([conc[aa].asNumber(UM) for aa in aa_ids])

    # Amino acids used by translation (dilution plus degradation) and released
    # by protein degradation, per amino acid
    monomers = translation.monomer_data
    monomer_counts = counts(container, bulk_name_to_idx(monomers["id"], ids))
    deg_rate = monomers["deg_rate"].asNumber(1 / units.s)
    aa_counts = monomers["aa_counts"].asNumber()
    dilution_rate = np.log(2) / doubling_time.asNumber(units.s)
    aa_use = (monomer_counts * (dilution_rate + deg_rate)) @ aa_counts
    recycled = (monomer_counts * deg_rate) @ aa_counts * to_um

    # Same amino acids as the simulation's charging (selenocysteine excluded)
    charging_mask = np.array(
        [aa[:-3] != "L-SELENOCYSTEINE" for aa in aa_ids], dtype=bool
    )
    aa_use[~charging_mask] = 0
    f = aa_use / aa_use.sum()

    # Active ribosomes: subunit pairs times the active fraction
    ribosome = ribosome_from_subunits

    # Supply inputs, as the simulation builds them
    fwd_enzymes, rev_enzymes = metabolism.get_pathway_enzyme_counts_per_aa(
        counts(container, bulk_name_to_idx(metabolism.aa_enzymes, ids))
    )
    importers = counts(container, bulk_name_to_idx(metabolism.aa_importer_names, ids))
    exporters = counts(container, bulk_name_to_idx(metabolism.aa_exporter_names, ids))
    media = sim_data.external_state.saved_media[nutrients]
    threshold = sim_data.external_state.import_constraint_threshold
    aa_in_media = np.array(
        [
            media.get(sim_data.external_state.env_to_exchange_map[aa[:-3]][:-3], 0)
            > threshold
            for aa in aa_ids
        ]
    )

    target_ppgpp = growth.get_ppGpp_conc(doubling_time).asNumber(UM)
    max_elong_rate = growth.get_ribosome_elongation_rate_by_ppgpp(
        target_ppgpp * UM,
        constants.ribosome_elongation_rate_basal.asNumber(units.aa / units.s),
    ).asNumber(units.aa / units.s)

    params = {
        "kS": (
            transcription.synthetase_kcats.asNumber(1 / units.s)
            if getattr(transcription, "synthetase_kcats", None) is not None
            else constants.synthetase_charging_rate.asNumber(1 / units.s)
        ),
        "KMaa": transcription.aa_kms.asNumber(UM),
        "KMtf": transcription.trna_kms.asNumber(UM),
        "KI_charged_trna": (
            transcription.charged_trna_kis.asNumber(UM)
            if getattr(transcription, "charged_trna_kis", None) is not None
            else np.full(len(aa_ids), np.inf)
        ),
        "krta": constants.Kdissociation_charged_trna_ribosome.asNumber(UM),
        "krtf": constants.Kdissociation_uncharged_trna_ribosome.asNumber(UM),
        "max_elong_rate": max_elong_rate,
        "unit_conversion": metabolism.get_amino_acid_conc_conversion(UM),
        "KD_RelA": transcription.KD_RelA.asNumber(UM),
        "k_RelA": constants.k_RelA_ppGpp_synthesis.asNumber(1 / units.s),
        "k_SpoT_syn": constants.k_SpoT_ppGpp_synthesis.asNumber(1 / units.s),
        "k_SpoT_deg": constants.k_SpoT_ppGpp_degradation.asNumber(
            units.L / units.umol / units.s
        ),
        "KI_SpoT": transcription.KI_SpoT.asNumber(UM),
    }

    return {
        "condition": condition,
        "nutrients": nutrients,
        "doubling_time_min": doubling_time.asNumber(units.min),
        "dilution_rate": dilution_rate,
        "target_ppgpp": target_ppgpp,
        "to_um": to_um,
        "dry_mass": dry_mass,
        "total_trna": total_trna,
        "synthetase": synthetase,
        "rela": rela,
        "spot": spot,
        "ribosome": ribosome,
        "ribosome_from_subunits": ribosome_from_subunits,
        "aa_conc": aa_conc,
        "recycled": recycled,
        "f": f,
        "charging_mask": charging_mask,
        "fwd_enzymes": fwd_enzymes,
        "rev_enzymes": rev_enzymes,
        "importers": importers,
        "exporters": exporters,
        "aa_in_media": aa_in_media,
        "params": params,
        "aa_ids": [aa[:-3] for aa in aa_ids],
    }


def steady_state_charging(
    sim_data, state: dict[str, Any], include_recycled: bool = True
) -> dict[str, Any]:
    """
    Steady-state tRNA charging and free amino acid concentrations with the
    simulation's charging, elongation and supply rate laws, holding enzyme,
    tRNA and ribosome concentrations at their expected values.

    Integrates the same system as
    :py:func:`ecoli.processes.polypeptide_elongation.calculate_trna_charging`
    (with supply) in rounds until the state stops changing.
    """
    # Imported here to keep the ParCa module importable without the process
    from ecoli.processes.polypeptide_elongation import dcdt_jit

    metabolism = sim_data.process.metabolism
    p = state["params"]
    mask = state["charging_mask"]
    n_aas = len(mask)
    n_masked = int(mask.sum())
    to_um = state["to_um"]
    synthetase = state["synthetase"][mask]
    synthetase_rate = p["kS"]
    if np.ndim(synthetase_rate):
        synthetase_rate = np.asarray(synthetase_rate, dtype=float)[mask]
    f = state["f"][mask]
    recycled = state["recycled"] if include_recycled else np.zeros(n_aas)
    unit_conversion = p["unit_conversion"]

    def supply(aa_conc):
        aa_conc = unit_conversion * aa_conc
        synthesis = metabolism.amino_acid_synthesis(
            state["fwd_enzymes"], state["rev_enzymes"], aa_conc
        )[0]
        imported = metabolism.amino_acid_import(
            state["aa_in_media"],
            state["dry_mass"],
            aa_conc,
            state["importers"],
            True,
        )
        exported = metabolism.amino_acid_export(state["exporters"], aa_conc, True)
        return to_um * synthesis, to_um * imported, to_um * exported

    def dcdt(t, c):
        v_charging, dtrna, _ = dcdt_jit(
            t,
            c,
            n_masked,
            n_aas,
            mask,
            synthetase_rate,
            synthetase,
            p["KMaa"],
            p["KMtf"],
            f,
            p["krta"],
            p["krtf"],
            p["max_elong_rate"],
            state["ribosome"],
            False,
            np.zeros(n_masked),
            0.0,
            p["KI_charged_trna"],
        )
        aa_conc = c[2 * n_masked :]
        synthesis, imported, exported = supply(aa_conc)
        daa = np.zeros(n_aas)
        daa[mask] = (synthesis + imported - exported + recycled)[mask] - v_charging
        return np.hstack((-dtrna, dtrna, daa))

    # Start from 10% uncharged at the target amino acid concentrations
    total = state["total_trna"][mask]
    c = np.hstack((0.1 * total, 0.9 * total, state["aa_conc"].copy()))
    converged = False
    for _ in range(MAX_STEADY_STATE_ROUNDS):
        sol = solve_ivp(dcdt, [0, STEADY_STATE_TIME], c, method="BDF")
        new_c = np.fmax(sol.y[:, -1], 0)
        change = np.abs(new_c - c) / np.fmax(np.abs(c), 1e-9)
        c = new_c
        if change.max() < STEADY_STATE_TOLERANCE:
            converged = True
            break

    uncharged = np.zeros(n_aas)
    charged = np.zeros(n_aas)
    uncharged[mask] = c[:n_masked]
    charged[mask] = c[n_masked : 2 * n_masked]
    aa_conc = c[2 * n_masked :]
    total_all = uncharged + charged
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction_charged = np.where(total_all > 0, charged / total_all, 1.0)
    v_rib = ribosome_elongation_rate(
        charged[mask],
        uncharged[mask],
        state["ribosome"],
        f,
        p["krta"],
        p["krtf"],
        p["max_elong_rate"],
    )
    return {
        "uncharged": uncharged,
        "charged": charged,
        "fraction_charged": fraction_charged,
        "aa_conc": aa_conc,
        "v_rib": v_rib,
        "converged": converged,
    }


def spot_factor_for_target(state: dict[str, Any], charging: dict[str, Any]):
    """
    SpoT synthesis factor that puts ppGpp synthesis and loss (degradation plus
    dilution) in balance at the condition's target ppGpp for this charging.
    Synthesis is linear in the factor, so the factor follows directly.

    Returns:
        factor, and a dict of the rate terms at the target (uM/s)
    """
    p = state["params"]
    mask = state["charging_mask"]
    target = state["target_ppgpp"]
    v_rela, v_spot_unit, v_deg, _ = ppgpp_rates(
        charging["uncharged"][mask],
        charging["charged"][mask],
        state["ribosome"],
        state["f"][mask],
        charging["v_rib"],
        state["rela"],
        state["spot"],
        target,
        p["krta"],
        p["krtf"],
        p["max_elong_rate"],
        p["KD_RelA"][mask],
        p["k_RelA"],
        p["k_SpoT_syn"],
        p["k_SpoT_deg"],
        p["KI_SpoT"][mask],
    )
    v_rela = float(np.sum(v_rela))
    loss = v_deg + state["dilution_rate"] * target
    factor = (loss - v_rela) / v_spot_unit
    return factor, {
        "v_rela_syn": v_rela,
        "v_spot_syn_unscaled": v_spot_unit,
        "v_deg": v_deg,
        "v_dilution": state["dilution_rate"] * target,
    }


def _fit_supply_and_ppgpp_kinetics(sim_data, cell_specs):
    """
    Amino acid supply constants and ppGpp kinetics spread from the current
    expected bulk counts (same calls and order as final_adjustments).
    """
    metabolism = sim_data.process.metabolism
    basal_container = create_expected_bulk_container(sim_data)
    with_aa_container = create_expected_bulk_container(sim_data, condition="with_aa")
    metabolism.set_phenomological_supply_constants(sim_data)
    metabolism.set_mechanistic_supply_constants(
        sim_data, cell_specs, basal_container, with_aa_container
    )
    metabolism.set_mechanistic_export_constants(sim_data, cell_specs, basal_container)
    metabolism.set_mechanistic_uptake_constants(sim_data, cell_specs, with_aa_container)
    sim_data.process.transcription.set_ppgpp_kinetics_parameters(
        basal_container, sim_data.constants
    )


def set_synthetase_kcats_from_capacity(
    sim_data, conditions: tuple[str, ...] = SYNTHETASE_CAPACITY_CONDITIONS
) -> Optional[dict[str, Any]]:
    """
    Per-synthetase charging rate constants that give each synthetase at least
    the measured charging capacity (Dykeman 2025 kcat x copies per cell, copies
    interpolated to each condition's growth rate) on the model's own synthetase
    counts:

        k_eff = max over conditions of kcat_Dykeman x copies_Dykeman(mu)/copies_model(mu)

    The maximum is used because the model's synthetase counts rise less with
    growth rate than Dykeman's for some synthetases (e.g. PheRS, on an
    r-protein operon), so a rate set at one growth rate leaves them short of
    capacity at the other. Expression, and therefore synthetase counts, stay as
    fit from RNA-seq. Amino acids without data keep the shared rate constant.
    Sets sim_data.process.transcription.synthetase_kcats.
    TODO (mia): look into why this is mechanistically and see if there is a
    biological fix that can be made with a TF or something.

    Returns:
        report with the per amino acid values, or None if there is no data
    """
    transcription = sim_data.process.transcription
    data = getattr(transcription, "synthetase_kcat_data", None)
    if data is None or np.all(np.isnan(data["kcat"])):
        return None

    shared_rate = sim_data.constants.synthetase_charging_rate.asNumber(1 / units.s)
    per_condition = {}
    for condition in conditions:
        doubling_time = sim_data.condition_to_doubling_time[condition]
        growth_rate = np.log(2) / doubling_time.asNumber(units.h)
        container = create_expected_bulk_container(sim_data, condition=condition)
        model_copies = transcription.aa_from_synthetase @ counts(
            container,
            bulk_name_to_idx(transcription.synthetase_names, container["id"]),
        )
        # Make it so copies scale close to exponentially with growth rate
        # (interpolate log copies):
        measured_copies = np.array(
            [
                np.exp(np.interp(growth_rate, data["growth_rates_per_h"], np.log(row)))
                if np.all(np.isfinite(row))
                else np.nan
                for row in data["copies"]
            ]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = data["kcat"] * measured_copies / model_copies
        per_condition[condition] = {
            "growth_rate_per_h": growth_rate,
            "rate": rate,
            "copies_ratio": measured_copies
            / np.where(model_copies > 0, model_copies, np.nan),
        }

    rates = np.vstack([v["rate"] for v in per_condition.values()])
    has_data = np.all(np.isfinite(rates), axis=0)
    kcats = np.full(rates.shape[1], shared_rate)
    kcats[has_data] = rates[:, has_data].max(axis=0)

    # Headroom floor on this model's own demand in each condition
    floors = {}
    if MIN_SYNTHETASE_HEADROOM is not None:
        for condition in conditions:
            state = condition_charging_state(sim_data, condition)
            elongation = sim_data.process.translation.ribosomeElongationRateDict[
                state["nutrients"]
            ].asNumber(units.aa / units.s)
            demand = state["f"] * elongation * state["ribosome"]
            km_aa = state["params"]["KMaa"]
            saturation = state["aa_conc"] / (km_aa + state["aa_conc"])
            with np.errstate(divide="ignore", invalid="ignore"):
                floor = (
                    MIN_SYNTHETASE_HEADROOM
                    * demand
                    / (state["synthetase"] * saturation)
                )
            supply_limited = state["aa_conc"] < MIN_HEADROOM_POOL_OVER_KM * km_aa
            floor[~np.isfinite(floor) | ~state["charging_mask"] | supply_limited] = 0.0
            floors[condition] = floor
            kcats[has_data] = np.fmax(kcats[has_data], floor[has_data])
    transcription.synthetase_kcats = (1 / units.s) * kcats

    aa_ids = [aa[:-3] for aa in sim_data.molecule_groups.amino_acids]
    return {
        "conditions": list(conditions),
        "growth_rate_per_h": {
            c: v["growth_rate_per_h"] for c, v in per_condition.items()
        },
        "kcat_effective": dict(zip(aa_ids, kcats)),
        "kcat_by_condition": {
            c: dict(zip(aa_ids, v["rate"])) for c, v in per_condition.items()
        },
        "kcat_measured": dict(zip(aa_ids, data["kcat"])),
        "min_headroom": MIN_SYNTHETASE_HEADROOM,
        "headroom_floor_rate": {c: dict(zip(aa_ids, v)) for c, v in floors.items()},
        "copies_ratio_measured_over_model": {
            c: dict(zip(aa_ids, v["copies_ratio"])) for c, v in per_condition.items()
        },
    }


def _with_params(state: dict[str, Any], **changes) -> dict[str, Any]:
    """
    Copy of a condition state with some rate parameters replaced.
    """
    new_state = dict(state)
    new_state["params"] = {**state["params"], **changes}
    return new_state


def mean_charged_fraction(state: dict[str, Any], charging: dict[str, Any]) -> float:
    """
    Unweighted mean charged fraction over the amino acids in charging.
    """
    return float(charging["fraction_charged"][state["charging_mask"]].mean())


def fit_synthetase_rate(
    sim_data,
    state: dict[str, Any],
    target_charged_fraction: float,
    include_recycled: bool = True,
) -> tuple[float, str]:
    """
    Shared synthetase rate constant kS (1/s) at which this condition's
    steady-state mean charged fraction equals the target. Charging rises
    monotonically with kS.
    """
    ks_current = state["params"]["kS"]

    def mismatch(log_scale):
        charging = steady_state_charging(
            sim_data,
            _with_params(state, kS=ks_current * 10**log_scale),
            include_recycled=include_recycled,
        )
        return mean_charged_fraction(state, charging) - target_charged_fraction

    try:
        log_scale = brentq(mismatch, *np.log10(KS_RANGE), xtol=1e-3)
        return ks_current * 10**log_scale, "ok"
    except ValueError:
        return ks_current, "no_solution"


def rela_null_spot_factor(
    state: dict[str, Any], charging: dict[str, Any], rela_null_ppgpp: float
) -> tuple[float, float]:
    """
    SpoT synthesis factor at which SpoT alone (no RelA) holds ppGpp at the
    relA-null level for this charging:
    k_SpoT_syn x factor x SpoT = k_SpoT_deg x SpoT x P0 / (1 + sum(uncharged / KI_SpoT))
    + dilution x P0.

    Returns:
        factor, and the SpoT degradation inhibition term (1 + sum)
    """
    p = state["params"]
    mask = state["charging_mask"]
    inhibition = 1 + float((charging["uncharged"][mask] / p["KI_SpoT"][mask]).sum())
    loss = (
        p["k_SpoT_deg"] * state["spot"] * rela_null_ppgpp / inhibition
        + state["dilution_rate"] * rela_null_ppgpp
    )
    return loss / (p["k_SpoT_syn"] * state["spot"]), inhibition


def ki_spot_rescale(
    sim_data,
    state: dict[str, Any],
    charging: dict[str, Any],
    include_recycled: bool = True,
) -> tuple[float, float, float]:
    """
    Scale for KI_SpoT (uncharged tRNA inhibition of SpoT ppGpp degradation)
    that keeps its effect at the anchor condition's basal state the same in
    relative terms as when it was set. KI_SpoT (20 uM, "empirically determined
    for stable ppGpp") was set with one shared synthetase rate constant and no
    charged tRNA inhibition, where about 1% of tRNA is uncharged. No measured
    value exists, so it is scaled by the ratio of total uncharged tRNA now to
    total uncharged tRNA under those original charging kinetics.

    Returns:
        scale, reference uncharged total (uM), current uncharged total (uM)
    """
    mask = state["charging_mask"]
    reference_state = _with_params(
        state,
        kS=sim_data.constants.synthetase_charging_rate.asNumber(1 / units.s),
        KI_charged_trna=np.full(len(mask), np.inf),
    )
    reference = steady_state_charging(
        sim_data, reference_state, include_recycled=include_recycled
    )
    reference_total = float(reference["uncharged"][mask].sum())
    current_total = float(charging["uncharged"][mask].sum())
    return current_total / reference_total, reference_total, current_total


def fit_kd_rela_scale(
    state: dict[str, Any], charging: dict[str, Any], spot_factor: float
) -> tuple[float, str]:
    """
    Scale on KD_RelA (all amino acids) at which SpoT (with this factor) plus
    RelA holds the condition at its target ppGpp. A larger KD_RelA makes RelA
    less sensitive to ribosomes bound to uncharged tRNA, so the SpoT factor
    needed for the target rises with the scale.
    """
    kd = state["params"]["KD_RelA"]

    def mismatch(log_scale):
        needed, _ = spot_factor_for_target(
            _with_params(state, KD_RelA=kd * 10**log_scale), charging
        )
        return needed - spot_factor

    # Search away from the current value only: at much smaller KD_RelA the
    # product of per amino acid competition terms overflows
    low, high = np.log10(KD_RELA_RANGE)
    at_current = mismatch(0.0)
    if at_current == 0:
        return 1.0, "ok"
    bracket = (0.0, high) if at_current < 0 else (low, 0.0)
    try:
        log_scale = brentq(mismatch, *bracket, xtol=1e-4)
        return 10**log_scale, "ok"
    except ValueError:
        return 1.0, "no_solution"


def fit_charging_attenuation_and_ppgpp(
    sim_data,
    cell_specs,
    fit_conditions: tuple[str, ...] = DEFAULT_FIT_CONDITIONS,
    anchor_condition: str = ANCHOR_CONDITION,
    anchor_charged_fraction: Optional[float] = DEFAULT_ANCHOR_CHARGED_FRACTION,
    rela_null_ppgpp: float = DEFAULT_RELA_NULL_PPGPP_UM,
    charged_trna_ki_over_km: Optional[float] = DEFAULT_CHARGED_TRNA_KI_OVER_KM,
    include_recycled: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Set kS from measured charging, make charging, attenuation and amino acid
    supply consistent, then fit KD_RelA and the SpoT synthesis factors (see
    the module docstring). Replaces the supply and ppGpp kinetics calls at the
    end of final_adjustments.
    # TODO (mia): decide if the report is how you want to keep this long term

    Args:
        sim_data: SimulationDataEcoli object, modified in place
            (constants.synthetase_charging_rate, constants.KD_RelA_ribosome,
            transcription.KD_RelA, transcription.spot_synthesis_media_factors,
            attenuation and amino acid supply constants)
        cell_specs: cell specifications from the fitted conditions
        fit_conditions: conditions to solve charging for. Each must have
            distinct media. Attenuation in other media uses the charging of the
            solved condition with the same media, or all tRNA charged if none.
        anchor_condition: condition (in fit_conditions) whose measured
            charging sets kS and whose relA-null level and target set its SpoT
            factor and KD_RelA
        anchor_charged_fraction: measured mean charged fraction for the anchor
            condition. None keeps kS at its current value.
        rela_null_ppgpp: measured relA-null ppGpp (uM) in the anchor condition
        charged_trna_ki_over_km: synthetase inhibition constant for charged
            tRNA as a multiple of the uncharged tRNA KM (None: no inhibition)
        include_recycled: add amino acids released by protein degradation to
            supply in the charging solve (matches the simulation run with
            ``credit_recycled_aa``)
        verbose: print a summary for each iteration and condition

    Returns:
        report stored as sim_data.process.transcription.charging_fit_report
    """
    transcription = sim_data.process.transcription
    constants = sim_data.constants

    # Synthetase inhibition by charged tRNA
    if charged_trna_ki_over_km is None:
        transcription.charged_trna_kis = None
    else:
        transcription.charged_trna_kis = (
            transcription.trna_kms * charged_trna_ki_over_km
        )

    # Attenuation volume from counts, as in the simulation, before the first fit
    transcription.calculate_attenuation(
        sim_data, cell_specs, charged_fraction=None, volume_from_counts=True
    )
    _fit_supply_and_ppgpp_kinetics(sim_data, cell_specs)

    # Per-synthetase rate constants from measured capacities (after the first
    # supply fit: the headroom floor needs the supply enzyme lists)
    capacity_report = set_synthetase_kcats_from_capacity(sim_data)
    if verbose and capacity_report is not None:
        print(
            "Synthetase rate constants from Dykeman 2025 capacities (larger of"
            f" {', '.join(capacity_report['conditions'])}) (1/s): "
            + ", ".join(
                f"{aa} {k:.1f}" for aa, k in capacity_report["kcat_effective"].items()
            )
        )

    ks = constants.synthetase_charging_rate.asNumber(1 / units.s)
    ks_initial = ks
    if anchor_charged_fraction is not None and capacity_report is not None:
        raise ValueError(
            "Fitting one shared kS to measured charging would override the"
            " per-synthetase rate constants; use one or the other"
        )
    charged_fraction: Optional[dict[str, np.ndarray]] = None
    history = []
    converged = False
    ks_status = "not_fit"
    fallback: Optional[dict[str, Any]] = None
    for iteration in range(MAX_LOOP_ITERATIONS):
        states = {c: condition_charging_state(sim_data, c) for c in fit_conditions}
        ks_change = 0.0
        if anchor_charged_fraction is not None:
            new_ks, ks_status = fit_synthetase_rate(
                sim_data,
                states[anchor_condition],
                anchor_charged_fraction,
                include_recycled=include_recycled,
            )
            ks_change = abs(new_ks / ks - 1)
            ks = new_ks
            constants.synthetase_charging_rate = ks / units.s
            for state in states.values():
                state["params"]["kS"] = ks

        charging = {
            c: steady_state_charging(
                sim_data, states[c], include_recycled=include_recycled
            )
            for c in fit_conditions
        }
        new_fraction = {
            states[c]["nutrients"]: charging[c]["fraction_charged"]
            for c in fit_conditions
        }
        change = (
            np.inf
            if charged_fraction is None
            else max(
                np.abs(new_fraction[m] - charged_fraction[m]).max()
                for m in new_fraction
            )
        )
        history.append(
            {
                "iteration": iteration,
                "kS": ks,
                "kS_status": ks_status,
                "max_charged_fraction_change": change,
                **{
                    f"{c}_mean_charged": mean_charged_fraction(states[c], charging[c])
                    for c in fit_conditions
                },
            }
        )
        if verbose:
            print(
                f"Charging/attenuation loop {iteration}: kS {ks:.2f} /s"
                f" ({ks_status}), max change in charged fraction {change:.2e}; "
                + ", ".join(
                    f"{c} mean charged {history[-1][f'{c}_mean_charged']:.3f}"
                    for c in fit_conditions
                )
            )
        if change < CHARGED_FRACTION_TOLERANCE and ks_change < 1e-2:
            converged = True
            break

        # Damped update of the charging that attenuation is calibrated on
        if charged_fraction is None:
            applied = new_fraction
        else:
            applied = {
                m: charged_fraction[m]
                + ATTENUATION_DAMPING * (new_fraction[m] - charged_fraction[m])
                for m in new_fraction
            }
        try:
            transcription.calculate_attenuation(
                sim_data,
                cell_specs,
                charged_fraction=applied,
                volume_from_counts=True,
            )
            _fit_supply_and_ppgpp_kinetics(sim_data, cell_specs)
        except ValueError as error:
            # Keep the last pass whose supply fit worked: restore its attenuation
            # and supply constants, whose charging was solved above
            transcription.calculate_attenuation(
                sim_data,
                cell_specs,
                charged_fraction=charged_fraction,
                volume_from_counts=True,
            )
            _fit_supply_and_ppgpp_kinetics(sim_data, cell_specs)
            fallback = {"iteration": iteration, "error": str(error)}
            if verbose:
                print(
                    f"WARNING: charging/attenuation loop did NOT converge. Supply fit"
                    f" failed at pass {iteration + 1} ({error}); keeping pass"
                    f" {iteration}'s attenuation, supply constants and charging"
                )
            break
        charged_fraction = applied

    # KI_SpoT rescaled to the new charging level when charged tRNA inhibition is on
    ki_spot_scale, ki_reference_uncharged, ki_current_uncharged = 1.0, np.nan, np.nan
    if charged_trna_ki_over_km is not None:
        ki_spot_scale, ki_reference_uncharged, ki_current_uncharged = ki_spot_rescale(
            sim_data,
            states[anchor_condition],
            charging[anchor_condition],
            include_recycled=include_recycled,
        )
        constants.KI_SpoT_ppGpp_degradation = (
            constants.KI_SpoT_ppGpp_degradation * ki_spot_scale
        )
        transcription.KI_SpoT = transcription.KI_SpoT * ki_spot_scale
        for state in states.values():
            state["params"]["KI_SpoT"] = state["params"]["KI_SpoT"] * ki_spot_scale
        if verbose:
            print(
                f"KI_SpoT x{ki_spot_scale:.2f}: basal uncharged tRNA"
                f" {ki_current_uncharged:.1f} uM vs {ki_reference_uncharged:.1f} uM"
                " under the original charging kinetics"
            )

    # ppGpp: anchor condition sets its SpoT factor (relA-null) and KD_RelA
    factors = dict(getattr(transcription, "spot_synthesis_media_factors", {}))
    previous_factors = dict(factors)
    fit_media = {states[c]["nutrients"] for c in fit_conditions}
    anchor_state, anchor_charging = states[anchor_condition], charging[anchor_condition]
    anchor_factor, anchor_inhibition = rela_null_spot_factor(
        anchor_state, anchor_charging, rela_null_ppgpp
    )
    kd_scale, kd_status = fit_kd_rela_scale(
        anchor_state, anchor_charging, anchor_factor
    )
    constants.KD_RelA_ribosome = constants.KD_RelA_ribosome * kd_scale
    transcription.KD_RelA = transcription.KD_RelA * kd_scale
    for state in states.values():
        state["params"]["KD_RelA"] = state["params"]["KD_RelA"] * kd_scale

    anchor_media = anchor_state["nutrients"]
    previous_anchor_factor = factors.get(anchor_media, 1.0)
    factors[anchor_media] = anchor_factor
    # Media without amino acids that shared the anchor's previous factor follow it
    for media, value in previous_factors.items():
        if value == previous_anchor_factor and media not in fit_media:
            factors[media] = anchor_factor

    condition_reports = {}
    for condition in fit_conditions:
        state, result = states[condition], charging[condition]
        status = "ok"
        if condition != anchor_condition:
            factor, _ = spot_factor_for_target(state, result)
            if factor <= 0:
                # RelA alone already exceeds the target at this charging
                status = "target_below_rela_alone"
                factor = 1e-3
            factors[state["nutrients"]] = factor
        factor = factors[state["nutrients"]]
        needed, terms = spot_factor_for_target(state, result)
        mask = state["charging_mask"]
        uncharged_fraction = 1 - result["fraction_charged"]
        inhibition = 1 + float(
            (result["uncharged"][mask] / state["params"]["KI_SpoT"][mask]).sum()
        )
        condition_reports[condition] = {
            "nutrients": state["nutrients"],
            "doubling_time_min": state["doubling_time_min"],
            "target_ppgpp": state["target_ppgpp"],
            "spot_factor": factor,
            "previous_spot_factor": previous_factors.get(state["nutrients"], 1.0),
            "status": status,
            "steady_state_converged": result["converged"],
            "mean_charged": mean_charged_fraction(state, result),
            "rela_to_spot_synthesis": terms["v_rela_syn"]
            / (terms["v_spot_syn_unscaled"] * factor),
            "spot_inhibition": inhibition,
            "rate_terms": terms,
            "uncharged_fraction": dict(zip(state["aa_ids"], uncharged_fraction)),
            "aa_conc_over_target": dict(
                zip(
                    state["aa_ids"],
                    np.where(
                        state["aa_conc"] > 0,
                        result["aa_conc"] / state["aa_conc"],
                        np.nan,
                    ),
                )
            ),
            "v_rib_per_ribosome": result["v_rib"] / state["ribosome"],
        }
        if verbose:
            report = condition_reports[condition]
            top = np.argsort(-uncharged_fraction * mask)[:4]
            print(
                f"ppGpp fit {condition} ({state['nutrients']},"
                f" {state['doubling_time_min']:.0f} min): target"
                f" {state['target_ppgpp']:.1f} uM, mean charged"
                f" {report['mean_charged']:.3f}, SpoT factor"
                f" {report['previous_spot_factor']:.2f} -> {factor:.2f} ({status}),"
                f" RelA/SpoT synthesis {report['rela_to_spot_synthesis']:.2f},"
                f" SpoT inhibition {inhibition:.2f}, elongation"
                f" {report['v_rib_per_ribosome']:.1f} aa/s, most uncharged: "
                + ", ".join(
                    f"{state['aa_ids'][k]} {uncharged_fraction[k]:.2f}" for k in top
                )
            )
    transcription.spot_synthesis_media_factors = factors

    report = {
        "converged": converged,
        "fallback": fallback,
        "history": history,
        "kS_initial": ks_initial,
        "kS": ks,
        "kS_status": ks_status,
        "anchor_condition": anchor_condition,
        "anchor_charged_fraction": anchor_charged_fraction,
        "synthetase_capacity": capacity_report,
        "charged_trna_ki_over_km": charged_trna_ki_over_km,
        "rela_null_ppgpp": rela_null_ppgpp,
        "anchor_spot_inhibition": anchor_inhibition,
        "KI_SpoT_scale": ki_spot_scale,
        "KI_SpoT_reference_uncharged_uM": ki_reference_uncharged,
        "KI_SpoT_current_uncharged_uM": ki_current_uncharged,
        "KD_RelA_scale": kd_scale,
        "KD_RelA_status": kd_status,
        "KD_RelA_ribosome": constants.KD_RelA_ribosome.asNumber(UM),
        "conditions": condition_reports,
        "include_recycled": include_recycled,
    }
    if verbose:
        print(
            f"Charging fit: kS {ks_initial:.1f} -> {ks:.2f} /s ({ks_status}),"
            f" KD_RelA x{kd_scale:.3g} -> {report['KD_RelA_ribosome']:.3g} uM"
            f" ({kd_status}), anchor relA-null SpoT inhibition"
            f" {anchor_inhibition:.2f}, converged {converged}"
        )
    transcription.charging_fit_report = report
    return report
