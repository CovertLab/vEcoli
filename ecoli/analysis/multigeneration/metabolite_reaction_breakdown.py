"""
Per-reaction breakdown of how a homeostatic metabolite is produced/consumed,
distinguishing kinetically-constrained reactions from FBA-only reactions, so
one can see which reactions the homeostatic objective tunes individually and
how the unconstrained reactions divide up the remaining demand. NOTE: if a
metabolite does not have a homeostatic objective (HO) target, then this plot
may still be able to plot the metabolite as long as it has an MW entry in
metabolites.tsv (and thus an entry for holding counts in the bulk container).
If a molecule is purely an intermediate and never accumulates and has no bulk
tracking index, it may not be plottable here.


This plot includes 3 full-width context panels on top, then a grid of
per-reaction breakdowns (``reactions_per_row`` reactions per grid row, default
 4). Each reaction block is three stacked sub-panels:

Top 3 context panels (share the time axis with everything below):
  1. Every active reaction's signed contribution to the metabolite's dm/dt,
     ``S[met, r] * v_r`` (mM/s), stacked (producers up, consumers down), with the
     net (solid black) = the homeostatic accumulation rate. Uptake from the
     medium is not in ``reaction_fluxes``, so it is added from
     ``external_exchange_fluxes`` as its own "external exchange (uptake)"
     series (tagged EXCHANGE in its block).
  2. The same, converted to molecules per timestep (counts), the units the
     homeostatic objective and division-mass accounting actually work with. The
     solid black net is the actual change FBA made to the pool; the red dotted
     line is the targeted change the homeostatic objective asked for
     (``target_homeostatic_dmdt``). The net is recomputed here from the
     reaction fluxes and checked every run against the model's own
     ``estimated_homeostatic_dmdt`` listener; a warning is emitted (console +
     on the panel) if they disagree by more than rounding. Known cause: FBA's
     maintenance fluxes (growth-associated, non-growth-associated, translation
     energy) are not emitted, so ATP / ADP / Pi / H2O / H+ always warn -- their
     stacks are missing that (large) term.
  3. The metabolite's homeostatic target concentration (red dotted) with its
     actual pool concentration both pre-FBA (grey dashed, start of the step) and
     post-FBA (solid black, end of the step, = ``metabolite_counts_final`` *
     counts_to_molar). The avg deviation in the title is measured post-FBA vs
     target, i.e. the residual homeostatic error after FBA has acted. Might be
     blank if a molecule not in the HO is passed.

Per-reaction block (one per active, nonzero flux producing reaction):
  a. First subpanel: Contribution to the metabolite flux (mM/s). Kinetic
     reactions also show their kinetic target flux (dashed) and the kcat-derived
     bound band, with the raw kcat annotated. FBA-only reactions show just
     their realized (actual) contribution.
  b. Second subpanel: The same contribution in molecules per timestep but in count units.
  c. Third subpanel: HO override tracking (applying to kinetic reactions only):
     red where the FBA/homeostatic solution pushed the flux ABOVE its kinetic
     target, blue where below.

Requires a classic ``metabolism.py`` run (reads the ``enzyme_kinetics`` and
``fba_results.reaction_fluxes`` listeners).
"""

import os
import pickle
import warnings
from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import (
    field_metadata,
    named_idx,
    open_arbitrary_sim_data,
    read_stacked_columns,
)

# Emitted listener columns:
FLUX_ACTUAL = (
    "listeners__enzyme_kinetics__actual_fluxes"  # realized kinetic flux (mM/s)
)
FLUX_TARGET = "listeners__enzyme_kinetics__target_fluxes"  # kinetic target flux (mM/s)
FLUX_UPPER = (
    "listeners__enzyme_kinetics__target_fluxes_upper"  # kcat-derived upper bound
)
FLUX_LOWER = (
    "listeners__enzyme_kinetics__target_fluxes_lower"  # kcat-derived lower bound
)
COUNTS_TO_MOLAR = "listeners__enzyme_kinetics__counts_to_molar"  # mM per molecule (c2m)
REACTION_FLUXES = (
    "listeners__fba_results__reaction_fluxes"  # every FBA reaction's v_r (mM/s)
)
TARGET_CONC = (
    "listeners__fba_results__target_concentrations"  # homeostatic target conc (mM)
)
POOL_COUNTS = (
    "listeners__fba_results__homeostatic_metabolite_counts"  # pool size (molecules)
)
# Pool change FBA actually made this step (molecules; stochastically rounded):
EST_DMDT = "listeners__fba_results__estimated_homeostatic_dmdt"
# Pool change the homeostatic objective asked for (target - actual, molecules):
ASK_DMDT = "listeners__fba_results__target_homeostatic_dmdt"
# Exchange with the environment. FBA treats these as "special" fluxes, so they
# are NOT in reaction_fluxes; emitted per molecule in mmol/gDCW/h, pointing OUT
# of the cell (negative = uptake). x coefficient / 3600 -> mM per step.
EXCH_FLUXES = "listeners__fba_results__external_exchange_fluxes"
FBA_COEFF = "listeners__fba_results__coefficient"  # g DCW * s / L
EXCH_KEY = "external exchange (uptake)"

# The model stochastically rounds FBA's pool change to whole molecules, so the
# recomputed net may differ from the listener by up to ~1 molecule per step.
NET_CHECK_ABS_TOL = 1.5  # molecules
NET_CHECK_REL_TOL = 1e-4  # fraction of |listener value|

# Columns that identify a single cell (one generation of one lineage):
CELL_ID_COLS = ["experiment_id", "variant", "lineage_seed", "generation", "agent_id"]

# Kinetic plot colors:
COLOR_TARGET = "#1e1e1e"  # near-black dashed: kinetic target flux
COLOR_BOUND = "#2ca02c"  # green: kcat-derived bound band + kcat/Km annotation
COLOR_UP = "red"  # HO override drove flux ABOVE its kinetic target
COLOR_DOWN = "blue"  # HO override drove flux BELOW its kinetic target


def _shorten(name: str, width: int = 34) -> str:
    """Trim long EcoCyc reaction / metabolite ids so titles and legends fit."""
    return name if len(name) <= width else name[: width - 1] + "…"


def _km_ki_maps(metab, reactions):
    """
    Per-substrate Km/Ki (µM) for each reaction id in ``reactions``.

    Parsed from the saturation terms that actually build the kinetic flux
    (``metab._saturations``: flux = kcat x enzyme_conc x saturation(Km, Ki)), so
    these match the reaction even when its id is an expanded model id, and they
    are the literal µM values from ``metabolism_kinetics.tsv`` (Km/Ki are NOT
    temperature-adjusted, unlike kcat). A substrate can carry several values when
    multiple measurements were curated. Returns ``{rxn: (kms, kis)}`` where each
    map is ``{substrate_id: [µM, ...]}`` (sorted, de-duplicated); a reaction with
    no saturation term (kcat-only) maps to two empty dicts.
    """
    # Default: every requested reaction maps to empty Km/Ki dicts. Only kinetic
    # (constraint) reactions get overwritten below; the parse is best-effort so
    # an unexpected shape just leaves the annotation blank rather than crashing.
    out = {r: ({}, {}) for r in reactions}
    try:
        import re
        import sympy as sp

        # kcr / subs are the reaction and substrate index orders used to build
        # the saturation strings. metab._saturations is a source string that,
        # when evaluated with the substrate symbols s[j] in scope, yields a list of
        # per-reaction saturation factors (one list entry per kcr reaction).
        kcr = list(metab.kinetic_constraint_reactions)
        subs = list(metab.kinetic_constraint_substrates)
        s = [sp.symbols("s[{}]".format(j)) for j in range(len(subs))]  # noqa: F841
        sat_all = eval(metab._saturations)  # aligned with kcr
        # Each saturation factor is a product of (1 + Km/[S]) terms (and
        # (1 + [I]/Ki) for inhibitors). After sympify, Km shows up as the literal
        # "<value>/s[j]" and Ki as "s[j]/<value>", which these two patterns pull
        # back out together with the substrate index j.
        pat_km = re.compile(r"([0-9.eE+\-]+)/s\[(\d+)\]")  # Km/[S]
        pat_ki = re.compile(r"s\[(\d+)\]/([0-9.eE+\-]+)")  # [I]/Ki
        for r in reactions:
            if r not in kcr:
                continue  # kcat-only / non-constraint reaction has no saturation term
            kms: dict[str, list[float]] = {}
            kis: dict[str, list[float]] = {}
            for expr in sat_all[kcr.index(r)]:
                txt = str(sp.sympify(expr))
                for val, j in pat_km.findall(txt):
                    kms.setdefault(subs[int(j)], []).append(float(val))
                for j, val in pat_ki.findall(txt):
                    kis.setdefault(subs[int(j)], []).append(float(val))
            out[r] = (
                {k: sorted(set(v)) for k, v in kms.items()},
                {k: sorted(set(v)) for k, v in kis.items()},
            )
    except Exception as e:  # pragma: no cover - annotation is best-effort
        print(
            f"[metabolite_ho_reaction_breakdown] could not parse Km/Ki "
            f"({type(e).__name__}); skipping Km annotation."
        )
    return out


def _fmt_km_line(km_map, ki_map):
    """
    Creates compact 'Km metabolite1 92/110 · metabolite2 73/83' style lines
    for the graph keys.
    """

    def _join(d):
        return " · ".join(
            "{} {}".format(sub.split("[")[0], "/".join("{:g}".format(v) for v in vals))
            for sub, vals in d.items()
        )

    lines = []
    if km_map:
        lines.append("Km " + _join(km_map))
    if ki_map:
        lines.append("Ki " + _join(ki_map))
    return lines


def _add_generation_lines(axes, t, generation):
    """
    Dotted vertical line at each division boundary, on every panel.

    ``generation`` is the per-timestep generation index; a boundary is any step
    where it changes. ``k+1`` is the first timestep of the new generation, which
    is where the division line belongs.
    """
    if generation is None:
        return
    for k in np.where(np.diff(generation) != 0)[0]:
        for ax in axes:
            ax.axvline(t[k + 1], color="0.15", lw=0.8, ls=":", alpha=0.5)


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    """
    Config options (all optional)::

        {
            // A single metabolite id, OR a list of them (one figure each).
            // "metabolites" is an alias; either key accepts a str or a list.
            "metabolite": "GLN[c]",
            // "metabolites": ["GLN[c]", "ATP[c]", "ATP[p]"],
            // per-reaction blocks per grid row:
            "reactions_per_row": 4,
            // cap on number of reaction blocks:
            "max_reactions": 16,
            // 0 => show every reaction that is ever nonzero;
            "flux_threshold_frac": 0.0,
            // FBA-only reactions are aggregated by base reaction
            //   (net over all directions/substrate combos). true: break them out
            //   per FBA instantiation, like the kinetic reactions -- more detail
            //   but reintroduces large canceling forward/(reverse) pairs.
            "split_fba_by_instantiation": false,
            // Select DPI resolution:
            "dpi": 200,
            // Mark cell division lines:
            "mark_generations": true
        }
    """
    # Read in the user input params:
    raw_mets = params.get("metabolites", params.get("metabolite", "GLY[c]"))
    metabolites = [raw_mets] if isinstance(raw_mets, str) else list(raw_mets)
    rpr = int(params.get("reactions_per_row", 2))
    max_reactions = int(params.get("max_reactions", 16))
    thresh_frac = float(params.get("flux_threshold_frac", 0.0))
    split_fba = bool(params.get("split_fba_by_instantiation", False))
    dpi = int(params.get("dpi", 200))
    mark_generations = params.get("mark_generations", True)

    # Load in the sim data:
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)
    metab = sim_data.process.metabolism
    stoich = metab.reaction_stoich  # {rxn_id: {metabolite_id: S[met, r]}}

    # Maps each FBA instantiation id (forward/(reverse)/per-substrate split) back
    # to its underlying biological reaction; used to aggregate FBA-only pieces.
    rxn_to_base = metab.reaction_id_to_base_reaction_id

    # {rxn_id: [catalyst molecule ids]} -- for the per-block "enzyme" annotation.
    reaction_catalysts = getattr(metab, "reaction_catalysts", {})

    # kcat table (1/s), aligned with kinetic_constraint_reactions. _kcats is
    # (n_constraint_reactions, 3) holding [min, mean, max] turnover numbers, so
    # kcat_of[r] is a length-3 array. kcat is a per-reaction CONSTANT (annotated,
    # not plotted): the kinetic bound band is kcat * enzyme_conc * saturation.
    kcr = list(metab.kinetic_constraint_reactions)
    kcats_all = np.asarray(metab._kcats, dtype=float)
    kcat_of = {r: kcats_all[i] for i, r in enumerate(kcr)}

    # Resolve the emitted column orderings per experiment. reaction_fluxes and
    # target_fluxes are only populated by the classic metabolism.py process; on a
    # metabolism_redux* run they're missing, so we skip the whole analysis:
    try:
        fba_rxn_ids = field_metadata(conn, config_sql, REACTION_FLUXES)
        kinetic_ids = field_metadata(conn, config_sql, FLUX_TARGET)
    except Exception as e:
        print(
            f"[metabolite_ho_reaction_breakdown] required metabolism.py "
            f"listeners unavailable ({type(e).__name__}); skipping (redux run?)."
        )
        return

    # Name -> column-index maps for the two flux vectors, plus a fast membership
    # set for "is this reaction kinetically constrained?".
    fba_idx = {r: i for i, r in enumerate(fba_rxn_ids)}
    kin_idx = {r: i for i, r in enumerate(kinetic_ids)}
    kin_set = set(kinetic_ids)

    # Homeostatic pool + target for the top context panel:
    try:
        pool_ids = field_metadata(conn, config_sql, POOL_COUNTS)
        tconc_ids = field_metadata(conn, config_sql, TARGET_CONC)
    except Exception:
        pool_ids, tconc_ids = [], []
    try:
        dmdt_ids = field_metadata(conn, config_sql, ASK_DMDT)
    except Exception:
        dmdt_ids = []
    try:
        exch_ids = field_metadata(conn, config_sql, EXCH_FLUXES)
    except Exception:
        exch_ids = []

    for metabolite in metabolites:
        _run_metabolite(
            metabolite,
            conn,
            history_sql,
            outdir,
            metab,
            stoich,
            rxn_to_base,
            reaction_catalysts,
            kcat_of,
            fba_idx,
            kin_idx,
            kin_set,
            pool_ids,
            tconc_ids,
            dmdt_ids,
            exch_ids,
            rpr,
            max_reactions,
            thresh_frac,
            split_fba,
            dpi,
            mark_generations,
        )


def _run_metabolite(
    metabolite,
    conn,
    history_sql,
    outdir,
    metab,
    stoich,
    rxn_to_base,
    reaction_catalysts,
    kcat_of,
    fba_idx,
    kin_idx,
    kin_set,
    pool_ids,
    tconc_ids,
    dmdt_ids,
    exch_ids,
    rpr,
    max_reactions,
    thresh_frac,
    split_fba,
    dpi,
    mark_generations,
):
    """
    Read, decompose, and plot a single metabolite (one PNG per seed).
    """
    # Every reaction whose stoichiometry touches this metabolite AND that appears
    # in the emitted FBA flux vector. met_coeffs[r] is the signed coefficient
    # S[met, r] (>0 produced, <0 consumed) that turns a flux v_r into a
    # contribution to the pool's dm/dt.
    met_coeffs = {
        r: st[metabolite]
        for r, st in stoich.items()
        if metabolite in st and r in fba_idx
    }
    if not met_coeffs:
        print(
            f"[metabolite_ho_reaction_breakdown] {metabolite} not in FBA "
            "stoichiometry; skipping."
        )
        return
    met_rxns = list(met_coeffs.keys())

    # The kinetically-constrained subset: these get their own kcat band, target
    # trace, and override strip; the rest are FBA-only.
    met_kin = [r for r in met_rxns if r in kin_set]

    # Per-substrate Km/Ki (µM) for the kinetic reactions, for annotation.
    km_ki_of = _km_ki_maps(metab, met_kin)

    # Always pull every touching reaction's realized flux plus counts_to_molar
    # (c2m, mM/molecule) for the counts conversion. named_idx selects specific
    # positions out of an array-valued listener column and aliases them.
    columns = [
        named_idx(REACTION_FLUXES, met_rxns, [[fba_idx[r] for r in met_rxns]]),
        f"{COUNTS_TO_MOLAR} AS c2m",
    ]

    # For the kinetic reactions, also pull actual/target/upper/lower kinetic
    # fluxes (aliased kact::/ktar::/kup::/klo:: so we can look them up per rxn).
    if met_kin:
        k_i = [kin_idx[r] for r in met_kin]
        columns += [
            named_idx(FLUX_ACTUAL, [f"kact::{r}" for r in met_kin], [k_i]),
            named_idx(FLUX_TARGET, [f"ktar::{r}" for r in met_kin], [k_i]),
            named_idx(FLUX_UPPER, [f"kup::{r}" for r in met_kin], [k_i]),
            named_idx(FLUX_LOWER, [f"klo::{r}" for r in met_kin], [k_i]),
        ]

    # Pool size (molecules) and homeostatic target concentration for panel 3,
    # only if this metabolite is actually a homeostatic target.
    if metabolite in pool_ids:
        columns.append(
            named_idx(POOL_COUNTS, ["met_pool"], [[pool_ids.index(metabolite)]])
        )
    if metabolite in tconc_ids:
        columns.append(
            named_idx(TARGET_CONC, ["met_tconc"], [[tconc_ids.index(metabolite)]])
        )
    # Actual and targeted pool change per step, straight from the listeners,
    # for panel 2 and the recomputed-net check.
    if metabolite in dmdt_ids:
        d_i = [[dmdt_ids.index(metabolite)]]
        columns += [
            named_idx(EST_DMDT, ["met_est_dmdt"], d_i),
            named_idx(ASK_DMDT, ["met_ask_dmdt"], d_i),
        ]
    # Exchange with the environment (uptake/secretion), if this metabolite has
    # one; converted to a contribution below.
    if metabolite in exch_ids:
        columns += [
            named_idx(EXCH_FLUXES, ["met_exch"], [[exch_ids.index(metabolite)]]),
            f"{FBA_COEFF} AS fba_coeff",
        ]

    # remove_first=True drops each cell's first emitted row (a carried-over parent
    # value, not one this cell computed); order_results keeps the lineage in time
    # order. See ecoli.analysis.multigeneration.selected_fluxes for the rationale.
    df = read_stacked_columns(
        history_sql, columns, remove_first=True, order_results=True, conn=conn
    )
    if df.is_empty():
        print(f"[metabolite_ho_reaction_breakdown] no data for {metabolite}; skipping.")
        return

    # Also drop each cell's now-earliest row (its second timestep): the first FBA
    # solve after division overshoots because the pools were just halved, so this
    # per-cell min-time filter removes that unrepresentative spike.
    df = df.filter(pl.col("time") > pl.col("time").min().over(CELL_ID_COLS))

    # Report reactions that touch this metabolite but carry zero flux the entire
    # sim (so they are never plotted). Grouped the SAME way the plot groups them
    # (kinetic instantiation kept separate; other instantiations aggregated by
    # base reaction), so a base reaction that is active through any one
    # instantiation is NOT reported as dead.
    ever_nonzero = {r: bool(np.any(df[r].to_numpy() != 0)) for r in met_rxns}
    groups: dict[str, list[str]] = defaultdict(list)
    for r in met_rxns:
        key = r if (r in kin_set or split_fba) else rxn_to_base.get(r, r)
        groups[key].append(r)
    dead = sorted(
        g for g, members in groups.items() if not any(ever_nonzero[r] for r in members)
    )
    if dead:
        print(
            f"[metabolite_ho_reaction_breakdown] {metabolite}: {len(dead)} of "
            f"{len(groups)} reactions touching it carried ZERO flux all sim "
            f"(not plotted):"
        )
        for g in dead:
            extra = "" if groups[g] == [g] else f"  ({len(groups[g])} instantiations)"
            print(f"    {g}{extra}")
    else:
        print(
            f"[metabolite_ho_reaction_breakdown] {metabolite}: all "
            f"{len(groups)} reactions touching it carried some flux."
        )

    # Create one figure per seed:
    seeds = (
        sorted(df["lineage_seed"].unique().to_list())
        if "lineage_seed" in df.columns
        else [None]
    )
    for seed in seeds:
        sub = df if seed is None else df.filter(pl.col("lineage_seed") == seed)
        if sub.is_empty():
            continue
        sub = sub.sort("time")
        _plot_seed(
            sub,
            seed,
            metabolite,
            met_coeffs,
            met_rxns,
            met_kin,
            rxn_to_base,
            reaction_catalysts,
            kcat_of,
            km_ki_of,
            rpr,
            max_reactions,
            thresh_frac,
            split_fba,
            dpi,
            mark_generations,
            outdir,
            in_maintenance=metabolite in getattr(metab, "maintenance_reaction", {}),
        )


def _plot_seed(
    sub,
    seed,
    metabolite,
    met_coeffs,
    met_rxns,
    met_kin,
    rxn_to_base,
    reaction_catalysts,
    kcat_of,
    km_ki_of,
    rpr,
    max_reactions,
    thresh_frac,
    split_fba,
    dpi,
    mark_generations,
    outdir,
    in_maintenance=False,
):
    t = sub["time"].to_numpy().astype(float) / 60.0  # minutes for the x-axis
    c2m = sub["c2m"].to_numpy().astype(float)  # mM per molecule, per timestep
    generation = sub["generation"].to_numpy() if "generation" in sub.columns else None
    exp_id = str(sub["experiment_id"][0]) if "experiment_id" in sub.columns else "?"

    # Per-step duration (s): forward difference of emitted times WITHIN each
    # cell (a difference across a division would span the dropped rows); each
    # cell's last row takes the median step:
    dt_s = (
        sub.select(
            (pl.col("time").shift(-1) - pl.col("time")).over(CELL_ID_COLS).alias("dt")
        )["dt"]
        .to_numpy()
        .astype(float)
    )
    med_dt = float(np.nanmedian(dt_s)) if np.isfinite(dt_s).any() else 1.0
    dt_s = np.where(np.isfinite(dt_s) & (dt_s > 0), dt_s, med_dt)

    # mM/s -> molecules/step conversion factor: (mM/s) * dt_s[s] / c2m[mM/molecule]
    # = molecules. Guard against c2m==0 (NaN, so those steps drop out cleanly):
    to_counts = dt_s / np.where(c2m > 0, c2m, np.nan)

    # Plot kinetic reactions separate awlays, but aggregate FBA reactions by base.
    # (See the "Kinetic vs FBA-only split" section of the module docstring: a
    # kinetically-capped instantiation must not be averaged into its unconstrained
    # siblings or the kinetic signal would vanish.)

    # NOTE TO SELF: consider removing this later, but also the net/futile
    # cycling might be helpful to see too.
    series: dict[str, dict[str, Any]] = {}

    # Pre-fetch the four kinetic flux traces (mM/s) for each constrained reaction
    # so we can attach them to that reaction's series below. "raw_" values keep
    # the reaction-space flux; multiplying by coeff moves it into pool dm/dt space:
    kin_lookup = {
        r: {
            "act": sub[f"kact::{r}"].to_numpy().astype(float),
            "tar": sub[f"ktar::{r}"].to_numpy().astype(float),
            "up": sub[f"kup::{r}"].to_numpy().astype(float),
            "lo": sub[f"klo::{r}"].to_numpy().astype(float),
        }
        for r in met_kin
    }

    for r in met_rxns:
        coeff = met_coeffs[r]
        contrib = coeff * sub[r].to_numpy().astype(float)  # S[met,r]*v_r, mM/s
        if r in kin_lookup:
            # Kinetic reaction: its own series, keyed by the instantiation id.
            # Store both the raw (reaction-space) target/actual (needed for the
            # override strip) and the coeff-scaled target/bounds in pool dm/dt
            # space so panel (a) shares the contribution axis. kcat/Km/Ki/catalysts
            # ride along for the annotation.
            key = r
            entry = series.setdefault(
                key,
                {
                    "contrib": np.zeros_like(t),
                    "kinetic": True,
                    "coeff": coeff,
                    "raw_act": kin_lookup[r]["act"],
                    "raw_tar": kin_lookup[r]["tar"],
                    "tar": coeff * kin_lookup[r]["tar"],
                    "up": coeff * kin_lookup[r]["up"],
                    "lo": coeff * kin_lookup[r]["lo"],
                    "kcat": kcat_of.get(r),
                    "km": km_ki_of.get(r, ({}, {}))[0],
                    "ki": km_ki_of.get(r, ({}, {}))[1],
                    "catalysts": set(),
                },
            )
            entry["contrib"] = entry["contrib"] + contrib
        else:
            # FBA-only: aggregate by base reaction, or (split_fba) keep the
            # individual FBA instantiation as its own series. Aggregating collapses
            # the forward/(reverse)/per-substrate split into one signed net band:
            key = r if split_fba else rxn_to_base.get(r, r)
            entry = series.setdefault(
                key,
                {
                    "contrib": np.zeros_like(t),
                    "kinetic": False,
                    "coeff": coeff,
                    "catalysts": set(),
                },
            )
            entry["contrib"] = entry["contrib"] + contrib
        # Union the catalysts across every instantiation folded into this series:
        entry["catalysts"].update(reaction_catalysts.get(r, []))

    # Exchange with the environment as its own series: flip the outward sign
    # so uptake is a positive (producing) contribution, and convert
    # mmol/gDCW/h x (g DCW s/L) / 3600 -> mM per step -> mM/s.
    if "met_exch" in sub.columns:
        exch_mM_step = (
            -sub["met_exch"].to_numpy().astype(float)
            * sub["fba_coeff"].to_numpy().astype(float)
            / 3600.0
        )
        series[EXCH_KEY] = {
            "contrib": exch_mM_step / dt_s,
            "kinetic": False,
            "exchange": True,
            "coeff": 1.0,
            "catalysts": set(),
        }

    # Net dm/dt: sum of every series' signed contribution == the model's
    # homeostatic accumulation rate for this metabolite (mM/s):
    net = np.sum([s["contrib"] for s in series.values()], axis=0)

    # Average share of the metabolite's total production / consumption for each
    # series. Denominators use ALL series (not just the plotted ones) so the %
    # reflects the true totals. Time-summed shares (equivalent to time-averaged).
    # tot_prod / tot_cons sum every positive / negative contribution over all
    # reactions and all timesteps:
    all_c = np.stack([s["contrib"] for s in series.values()])
    tot_prod = float(np.clip(all_c, 0, None).sum())
    tot_cons = float(np.clip(-all_c, 0, None).sum())
    share_of: dict[str, tuple[float, str]] = {}
    for k, v in series.items():
        c = v["contrib"]
        # Classify a series as a net producer or consumer by its mean sign, then
        # express its throughput as a % of the matching gross total:
        if c.mean() >= 0:
            num, denom, role = float(np.clip(c, 0, None).sum()), tot_prod, "production"
        else:
            num, denom, role = (
                float(np.clip(-c, 0, None).sum()),
                tot_cons,
                "consumption",
            )
        share_of[k] = (100.0 * num / denom if denom > 0 else 0.0, role)

    # Rank series by mean |contribution| and keep those above a fraction of the
    # busiest one, capped at max_reactions. thresh_frac=0 keeps every reaction
    # that is ever nonzero:
    mean_abs = {k: float(np.abs(v["contrib"]).mean()) for k, v in series.items()}
    max_mean = max(mean_abs.values()) if mean_abs else 0.0
    thresh = thresh_frac * max_mean

    def keep(k):
        if mean_abs[k] > thresh:
            return True
        # Also keep a kinetic reaction whose realized flux is ~0 but whose kinetic
        # target is nonzero: it has capacity the objective chose not to use, which
        # is exactly the kind of tuning this plot is meant to expose:
        v = series[k]
        return v["kinetic"] and float(np.abs(v["tar"]).mean()) > thresh

    kept = sorted(
        (k for k in series if keep(k)), key=lambda k: mean_abs[k], reverse=True
    )[:max_reactions]
    if not kept:
        print(
            f"[metabolite_ho_reaction_breakdown] no active {metabolite} "
            f"reactions for seed {seed}."
        )
        return

    # One stable color per kept series (tab20 -> up to 20 distinct hues), reused
    # across every panel that references the series:
    cmap = plt.get_cmap("tab20")
    colors = {k: cmap(i % 20) for i, k in enumerate(kept)}

    # Create 3 full-width context panels (mM/s stack, counts stack, pool vs HO
    # target), then per-reaction blocks of 3 sub-panels each:
    n = len(kept)
    n_top = 3
    n_block_rows = int(np.ceil(n / rpr))
    n_rows = n_top + n_block_rows * 3
    height_ratios = [3.0, 3.0, 2.0] + [1.5, 1.5, 0.5] * n_block_rows
    fig_w = max(12.0, 5.2 * rpr)
    fig_h = 8.5 + n_block_rows * 3.8
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        n_rows, rpr, figure=fig, height_ratios=height_ratios, hspace=0.55, wspace=0.28
    )

    ctx_axes = []

    def stack(ax, values_of, net_line, net_label="net dm/dt"):
        """
        Sign-split stacked area of the kept series, with the net line on top.

        Each series' positive part fills upward from a running positive baseline
        and its negative part fills downward from a running negative baseline, so
        producers pile up above zero and consumers below. ``values_of(k)``
        picks which quantity to stack (mM/s contribution vs its counts/step version).
        """
        pos_base = np.zeros_like(t)
        neg_base = np.zeros_like(t)
        for k in kept:
            v = values_of(k)
            pos = np.clip(v, 0, None)
            neg = np.clip(v, None, 0)
            ax.fill_between(
                t,
                pos_base,
                pos_base + pos,
                color=colors[k],
                linewidth=0,
                label=series_lbl.get(k, _shorten(k)),
            )
            ax.fill_between(t, neg_base, neg_base + neg, color=colors[k], linewidth=0)
            pos_base = pos_base + pos
            neg_base = neg_base + neg
        ax.axhline(0, color="0.3", lw=0.8)
        ax.plot(t, net_line, color="black", lw=1.4, label=net_label)
        # Robust y-limits: clip to the 0.5/99.5 percentiles of the stacked totals
        # so a one-step post-division transient can't flatten the whole scale:
        hi = np.percentile(pos_base, 99.5)
        lo = np.percentile(neg_base, 0.5)
        span = max(hi, -lo, 1e-30)
        ax.set_ylim(min(lo, 0) - 0.08 * span, max(hi, 0) + 0.08 * span)

    # Net in counts/step, plus legend labels carrying the net's mean +/- std in
    # each unit:
    net_counts = net * to_counts

    # Listener values for panel 2: the actual pool change the model recorded
    # (est) and the targeted change it asked FBA for (ask), both molecules/step.
    est = (
        sub["met_est_dmdt"].to_numpy().astype(float)
        if "met_est_dmdt" in sub.columns
        else None
    )
    ask = (
        sub["met_ask_dmdt"].to_numpy().astype(float)
        if "met_ask_dmdt" in sub.columns
        else None
    )

    # Hard check: the net recomputed from every reaction touching this
    # metabolite must reproduce the model's own recorded pool change, up to the
    # model's stochastic rounding to whole molecules. Any larger gap means the
    # decomposition is missing or mis-scaling something, so the stacked bands
    # can't be trusted to add up to what the cell actually did.
    check_msg = None
    if est is not None:
        gap = np.abs(net_counts - est)
        tol = NET_CHECK_ABS_TOL + NET_CHECK_REL_TOL * np.abs(est)
        bad = np.isfinite(gap) & (gap > tol)
        if bad.any():
            k = int(np.nanargmax(np.where(bad, gap, np.nan)))
            mean_gap = float(np.nanmean(net_counts - est))
            check_msg = (
                f"recomputed net ≠ estimated_homeostatic_dmdt listener at "
                f"{int(bad.sum())}/{bad.size} steps (mean recomputed − listener "
                f"{mean_gap:+,.0f} molec/step; max |gap| {gap[k]:,.0f} at "
                f"t={t[k]:.1f} min)"
            )
            # FBA's maintenance fluxes (GAM, NGAM, translation energy) use the
            # maintenance-reaction stoichiometry but are "special" fluxes left
            # out of the emitted reaction_fluxes, so for these metabolites the
            # stacked bands omit that (usually dominant) term.
            if in_maintenance:
                check_msg += (
                    "\nlikely cause: maintenance reactions (growth-associated, "
                    "non-growth-associated, translation energy) use this "
                    "metabolite. They run inside FBA but are not emitted in "
                    "reaction_fluxes, so they are missing from the stack"
                )
            warnings.warn(
                f"[metabolite_ho_reaction_breakdown] {metabolite} seed {seed}: "
                f"{check_msg}. The per-reaction decomposition does not add up "
                f"to the model's recorded pool change."
            )
        else:
            print(
                f"[metabolite_ho_reaction_breakdown] {metabolite} seed {seed}: "
                f"recomputed net matches estimated_homeostatic_dmdt at all "
                f"{bad.size} steps (within rounding)."
            )
    else:
        print(
            f"[metabolite_ho_reaction_breakdown] {metabolite}: no "
            f"estimated_homeostatic_dmdt listener entry (not a homeostatic "
            f"target?); net check skipped."
        )
    net_lbl_mM = f"net dm/dt: {np.nanmean(net):.2e} ± {np.nanstd(net):.1e} mM/s"
    net_lbl_cnt = (
        f"net dm/dt: {np.nanmean(net_counts):,.0f} ± "
        f"{np.nanstd(net_counts):,.0f} molec/step"
    )

    # Mean signed contribution of each kept series over the plotted time window,
    # in both units (mM/s and molecules/step). This is the simple time-average of
    # the already-computed signed contribution array ``series[k]["contrib"]`` (and
    # its counts/step conversion), so the reader can read a reaction's average
    # realized flux straight off the legend. Sign convention matches the stack:
    # positive = net producer of the metabolite, negative = net consumer:
    def _mean_annot(k):
        c = series[k]["contrib"]
        m_mM = float(np.nanmean(c))
        m_cnt = float(np.nanmean(c * to_counts))
        return f"<{m_mM:+.3g} mM/s, {m_cnt:+,.0f} mol/step>"

    mean_annot = {k: _mean_annot(k) for k in kept}

    # Legend label for each series in the top mM/s stack: short id + mean annot.
    series_lbl = {k: f"{_shorten(k)}  {mean_annot[k]}" for k in kept}

    # Context panel 1: contributions in mM/s
    ax_top1 = fig.add_subplot(gs[0, :])
    stack(ax_top1, lambda k: series[k]["contrib"], net, net_label=net_lbl_mM)
    ax_top1.set_ylabel("contribution to\ndm/dt (mM/s)")
    ax_top1.set_title(
        f"{metabolite}: per-reaction contribution to homeostatic dm/dt  "
        f"(net dm/dt = black line)",
        fontsize=11,
    )
    ax_top1.legend(
        loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=7, frameon=False
    )
    ctx_axes.append(ax_top1)

    # Context panel 2: contributions in counts/step
    ax_top2 = fig.add_subplot(gs[1, :], sharex=ax_top1)
    stack(
        ax_top2,
        lambda k: series[k]["contrib"] * to_counts,
        net_counts,
        net_label=net_lbl_cnt,
    )
    ax_top2.set_ylabel("contribution to\ndm/dt (molecules/step)")

    # Targeted change (homeostatic objective's ask) on top of the actual net;
    # widen the y-range if the ask runs outside the stacked bands:
    net_handles = [ax_top2.get_legend_handles_labels()[0][-1]]
    net_labels = [net_lbl_cnt]
    if ask is not None:
        (h_ask,) = ax_top2.plot(t, ask, color="red", lw=1.4, ls=":", zorder=6)
        net_handles.append(h_ask)
        net_labels.append(
            f"targeted dm/dt: {np.nanmean(ask):,.0f} ± {np.nanstd(ask):,.0f} molec/step"
        )
        lo, hi = ax_top2.get_ylim()
        a_lo, a_hi = np.nanpercentile(ask, [0.5, 99.5])
        pad = 0.08 * max(hi - lo, a_hi - a_lo, 1e-30)
        ax_top2.set_ylim(min(lo, a_lo - pad), max(hi, a_hi + pad))
    ax_top2.set_title(
        "Actual pool change (black = net of reactions) vs targeted change (red dotted)",
        fontsize=10,
    )
    if check_msg is not None:
        ax_top2.text(
            0.01,
            0.97,
            f"WARNING: {check_msg}",
            transform=ax_top2.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="red",
            bbox=dict(boxstyle="round", fc="white", ec="red", alpha=0.9),
        )

    # Small legend with only the net and targeted lines:
    ax_top2.legend(
        net_handles,
        net_labels,
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        fontsize=7,
        frameon=False,
    )
    ctx_axes.append(ax_top2)

    # Context panel 3: homeostatic pool (pre- and post-FBA) vs target:
    ax_top3 = fig.add_subplot(gs[2, :], sharex=ax_top1)

    # Pre-FBA pool concentration = pool counts (as read at the start of the
    # metabolism step, before FBA) * counts_to_molar (mM). Post-FBA pool
    # concentration adds the change FBA actually made this step
    # (estimated_homeostatic_dmdt, molecules): post = (pool + est) * c2m, which
    # equals metabolite_counts_final * c2m. All three traces are optional (any
    # may be absent if this metabolite is not a homeostatic target or the
    # listeners weren't emitted).
    pre_conc = (
        sub["met_pool"].to_numpy().astype(float) * c2m
        if "met_pool" in sub.columns
        else None
    )
    post_conc = (
        (sub["met_pool"].to_numpy().astype(float) + est) * c2m
        if ("met_pool" in sub.columns and est is not None)
        else None
    )
    tconc = (
        sub["met_tconc"].to_numpy().astype(float)
        if "met_tconc" in sub.columns
        else None
    )

    # Average deviation is now measured POST-FBA vs target (+ = pool ends the
    # step below target), i.e. the residual homeostatic error after FBA has
    # acted -- the meaningful "how close did we end up?" number. (The pre-FBA
    # trace only shows the demand entering the solve; per-step FBA success is
    # already visible in the estimated-vs-target dm/dt panel above.)
    dev_txt = ""
    if post_conc is not None and tconc is not None:
        valid = tconc > 0
        if bool(np.any(valid)):
            dev = 100.0 * (tconc[valid] - post_conc[valid]) / tconc[valid]
            mean_dev, std_dev = float(np.nanmean(dev)), float(np.nanstd(dev))
            side = "below" if mean_dev >= 0 else "above"
            dev_txt = (
                f"  ·  post-FBA avg deviation {abs(mean_dev):.1f}% ± "
                f"{std_dev:.1f}% {side} target"
            )
    # Draw order matters here: pre-FBA (widest) at the bottom, then the slightly
    # thinner black post-FBA actual, then the red-dotted target ON TOP so its
    # dots stay visible where the black actual sits right on the target.
    if pre_conc is not None:
        ax_top3.plot(
            t,
            pre_conc,
            color="0.6",
            lw=2.4,
            ls="--",
            zorder=1,
            label=f"actual pre-FBA (start of step)  (avg {np.nanmean(pre_conc):.3g} "
            f"± {np.nanstd(pre_conc):.2g} mM)",
        )
    if post_conc is not None:
        ax_top3.plot(
            t,
            post_conc,
            color="black",
            lw=0.9,
            zorder=2,
            label=f"actual post-FBA (end of step)  (avg {np.nanmean(post_conc):.3g} "
            f"± {np.nanstd(post_conc):.2g} mM)",
        )
    if tconc is not None:
        ax_top3.plot(
            t,
            tconc,
            color="red",
            lw=1.6,
            ls=":",
            zorder=3,
            label=f"homeostatic target  (avg {np.nanmean(tconc):.3g} "
            f"± {np.nanstd(tconc):.2g} mM)",
        )
    ax_top3.set_ylabel(f"{metabolite}\nconc (mM)")
    ax_top3.set_title(
        f"Homeostatic objective  "
        f"(red dotted = target, black = post-FBA, grey dashed = pre-FBA)"
        f"{dev_txt}",
        fontsize=10,
    )
    ax_top3.legend(
        loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=7, frameon=False
    )
    ctx_axes.append(ax_top3)

    # REACTION BREAKDOWN PLOTS:
    # Each reaction to be plotted gets three subpanels within its plot:
    # contribution (mM/s), contribution (counts/step), and an HO override strip
    all_axes = list(ctx_axes)
    for j, k in enumerate(kept):
        v = series[k]
        br, col = divmod(j, rpr)
        r0 = n_top + br * 3
        ax_a = fig.add_subplot(gs[r0, col], sharex=ax_top1)
        ax_b = fig.add_subplot(gs[r0 + 1, col], sharex=ax_top1)
        ax_c = fig.add_subplot(gs[r0 + 2, col], sharex=ax_top1)
        all_axes += [ax_a, ax_b, ax_c]
        color = colors[k]
        contrib = v["contrib"]

        # Block title: reaction id, KINETIC vs FBA-only tag, up-to-3 catalyzing
        # enzymes (compartment tag stripped), and this reaction's % share of the
        # metabolite's gross (not net) production or consumption:
        tag = (
            "KINETIC"
            if v["kinetic"]
            else ("EXCHANGE" if v.get("exchange") else "FBA-only")
        )
        cats = sorted(c.split("[")[0] for c in v.get("catalysts", set()))
        if v.get("exchange"):
            cat_str = "medium exchange (not in reaction_fluxes)"
        elif cats:
            cat_str = ", ".join(cats[:3]) + ("…" if len(cats) > 3 else "")
        else:
            cat_str = "no annotated catalyst"
        share_pct, share_role = share_of[k]
        ax_a.set_title(
            f"{_shorten(k, 38)}\n"
            f"[{tag}]  ·  {_shorten(cat_str, 28)}  ·  "
            f"{share_pct:.0f}% of {share_role}",
            fontsize=8,
        )

        # Realized-flux stat boxed in the bottom-right corner of each of this
        # block's two contribution sub-panels: mean ± std of the actual signed
        # contribution over the plotted window:
        m_mM, s_mM = float(np.nanmean(contrib)), float(np.nanstd(contrib))
        cnt = contrib * to_counts
        m_cnt, s_cnt = float(np.nanmean(cnt)), float(np.nanstd(cnt))
        _corner = dict(
            ha="right",
            va="bottom",
            fontsize=6,
            color="0.15",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
        )

        # sub-panel a: contribution to dm/dt (mM/s). Solid = realized. For a
        # kinetic reaction, also draw the kcat-derived bound band and the kinetic
        # target (all in pool dm/dt space, i.e. already multiplied by coeff, so
        # bounds may appear flipped for a consumer -- hence the min/max below):
        ax_a.axhline(0, color="0.6", lw=0.6)
        ax_a.plot(t, contrib, color=color, lw=1.4, label="actual")
        if v["kinetic"]:
            # coeff<0 (consumer) flips lower/upper, so re-order into true low/high
            # before shading (Only shade when the two bounds actually differ):
            lo_b = np.minimum(v["lo"], v["up"])
            up_b = np.maximum(v["lo"], v["up"])
            if np.any(np.abs(up_b - lo_b) > 0):
                ax_a.fill_between(
                    t, lo_b, up_b, color=COLOR_BOUND, alpha=0.15, label="kcat band"
                )
            ax_a.plot(
                t, v["tar"], color=COLOR_TARGET, lw=1.0, ls="--", label="kinetic target"
            )
            # Text annotation for the constants (kcat 1/s min/mean/max; Km/Ki µM):
            kc = v["kcat"]
            anno = []
            if kc is not None:
                anno.append(f"kcat {kc[0]:.2g}/{kc[1]:.2g}/{kc[2]:.2g} 1/s")
            anno += _fmt_km_line(v.get("km", {}), v.get("ki", {}))
            if anno:
                ax_a.text(
                    0.98,
                    0.96,
                    "\n".join(anno),
                    transform=ax_a.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6,
                    color=COLOR_BOUND,
                    bbox=dict(boxstyle="round", fc="white", ec=COLOR_BOUND, alpha=0.8),
                )
        ax_a.set_ylabel("mM/s", fontsize=7)
        ax_a.tick_params(labelsize=6)
        ax_a.legend(fontsize=5.5, loc="upper left", frameon=False)
        ax_a.text(
            0.98, 0.03, f"{m_mM:+.3g} ± {s_mM:.2g}", transform=ax_a.transAxes, **_corner
        )

        # sub-panel b: the same contribution converted to molecules/step (the
        # units the homeostatic objective and division-mass accounting use):
        ax_b.axhline(0, color="0.6", lw=0.6)
        ax_b.plot(t, contrib * to_counts, color=color, lw=1.4, label="actual")
        if v["kinetic"]:
            ax_b.plot(
                t,
                v["tar"] * to_counts,
                color=COLOR_TARGET,
                lw=1.0,
                ls="--",
                label="kinetic target",
            )
        ax_b.set_ylabel("molec/step", fontsize=7)
        ax_b.tick_params(labelsize=6)
        ax_b.text(
            0.98,
            0.03,
            f"{m_cnt:+,.0f} ± {s_cnt:,.0f}",
            transform=ax_b.transAxes,
            **_corner,
        )

        # sub-panel c: override strip (kinetic) or note (FBA). Compares the raw
        # reaction-space actual vs kinetic-target flux (NOT the coeff-scaled
        # values). Red = the HO/FBA solution pushed the flux above its kinetic
        # target, blue = below:
        ax_c.set_yticks([])
        if v["kinetic"]:
            diff = v["raw_act"] - v["raw_tar"]
            scale = max(float(np.nanmax(np.abs(v["raw_tar"]))), 1e-30)
            tol = 1e-6 * scale
            up = diff > tol
            dn = diff < -tol
            # One vertical tick per overriding timestep (markers, not a line, so
            # isolated single-step overrides aren't dropped by NaN gaps):
            ax_c.scatter(
                t[up],
                np.zeros(up.sum()),
                color=COLOR_UP,
                marker="|",
                s=80,
                linewidths=0.6,
            )
            ax_c.scatter(
                t[dn],
                np.zeros(dn.sum()),
                color=COLOR_DOWN,
                marker="|",
                s=80,
                linewidths=0.6,
            )
            ax_c.set_ylim(-1, 1)
            ax_c.text(
                0.01,
                1.05,
                f"HO override: up {100 * up.mean():.0f}% (red) / "
                f"down {100 * dn.mean():.0f}% (blue)",
                transform=ax_c.transAxes,
                fontsize=6,
                va="bottom",
            )
        else:
            # FBA-only reactions have no kinetic target, so there is nothing to
            # override:
            ax_c.set_ylim(-1, 1)
            ax_c.text(
                0.5,
                0.5,
                "FBA reaction — no kinetic target / HO override",
                transform=ax_c.transAxes,
                ha="center",
                va="center",
                fontsize=6.5,
                color="0.4",
                style="italic",
            )

        # x-labels only on the bottom-most override panel of each column (the last
        # block row, or a block in the final partially-filled row):
        if br == n_block_rows - 1 or (j + rpr >= n):
            ax_c.set_xlabel("time (min)", fontsize=7)
        ax_c.tick_params(labelsize=6)

    # Division-boundary lines span every panel so features can be lined up with
    # generations across the whole figure:
    if mark_generations:
        _add_generation_lines(all_axes, t, generation)

    n_gens = sub["generation"].n_unique() if "generation" in sub.columns else "?"
    n_kin = sum(series[k]["kinetic"] for k in kept)
    n_exch = sum(bool(series[k].get("exchange")) for k in kept)
    exch_txt = " + medium exchange" if n_exch else ""
    fig.suptitle(
        f"{metabolite} homeostatic reaction breakdown — {exp_id} · seed {seed} · "
        f"{n_gens} gen(s)\n{len(kept) - n_exch} active reactions "
        f"({n_kin} kinetic, {len(kept) - n_kin - n_exch} FBA-only){exch_txt}",
        fontsize=12,
        y=0.995,
    )
    fig.subplots_adjust(top=0.94, bottom=0.04, left=0.07, right=0.86)

    # Make one PNG per metabolite per seed (and take out characters that
    # could cause error):
    safe_met = metabolite.replace("[", "_").replace("]", "")
    out = os.path.join(
        outdir, f"metabolite_ho_reaction_breakdown_{safe_met}_seed{seed}.png"
    )
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[metabolite_ho_reaction_breakdown] saved {out}")
