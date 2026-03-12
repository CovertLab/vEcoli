import marimo

__generated_with = "0.16.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Comparative sim_data: Reference vs "Expt" ParCa Outputs

        Load `sim_data` from two ParCa run directories and compare:
        - **Translation supply rates** (`minimal` and `minimal_plus_amino_acids`) per amino acid
        - **Condition doubling times**
        - **Config metadata** (e.g. Expt path)

        Use this to confirm how the new RNaseq basal changes CYS (and other AA) supply
        relative to the reference run.
        """
    )
    return (mo,)


@app.cell
def _():
    from pathlib import Path
    import pickle
    from wholecell.utils import constants
    import numpy as np
    import pandas as pd
    import altair as alt

    from wholecell.utils import units

    # Unit for translation supply rate: mmol/(g DCW * h), convention for supply rates
    SUPPLY_RATE_UNIT = units.mmol / units.g / units.h
    METABOLITE_CONCENTRATION_UNITS = units.mol / units.L
    K_CAT_UNITS = 1 / units.s

    # ParCa output directories (relative to repo root when run from project root)
    OUTDIR_REF = Path("out/test_rnaseq_ingestion_defaults")  # reference / no RNaseq
    OUTDIR_EXPT = Path("out/test_rnaseq_ingestion")  # new RNaseq config
    # When ParCa fails (e.g. at set_mechanistic_supply_constants), sim_data is still
    # saved after fit_condition if save_intermediates is true (see test_rnaseq_ingestion.json).
    EXPT_INTERMEDIATES_DIR = OUTDIR_EXPT / "intermediates"
    REF_INTERMEDIATES_DIR = OUTDIR_REF / "intermediates"

    cell_specs_ref_path = REF_INTERMEDIATES_DIR / "cell_specs_fit_condition.cPickle"
    cell_specs_expt_path = EXPT_INTERMEDIATES_DIR / "cell_specs_fit_condition.cPickle"

    SIM_DATA_FIT_CONDITION = "sim_data_fit_condition.cPickle"

    def sim_data_path(outdir: Path) -> Path:
        return outdir / constants.KB_DIR / constants.SERIALIZED_SIM_DATA_FILENAME

    path_ref = sim_data_path(OUTDIR_REF)
    path_expt = sim_data_path(OUTDIR_EXPT)
    path_expt_intermediate = EXPT_INTERMEDIATES_DIR / SIM_DATA_FIT_CONDITION
    return (
        K_CAT_UNITS,
        METABOLITE_CONCENTRATION_UNITS,
        SUPPLY_RATE_UNIT,
        alt,
        cell_specs_expt_path,
        cell_specs_ref_path,
        np,
        path_expt,
        path_expt_intermediate,
        path_ref,
        pd,
        pickle,
        units,
    )


@app.cell
def _(path_expt, path_expt_intermediate, path_ref, pickle):
    def load_sim_data(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    sim_data_ref = load_sim_data(path_ref) if path_ref.exists() else None
    # Prefer final kb/simData.cPickle; if ParCa failed before saving, use intermediate
    if path_expt.exists():
        sim_data_expt = load_sim_data(path_expt)
        expt_source = "kb"
    elif path_expt_intermediate.exists():
        sim_data_expt = load_sim_data(path_expt_intermediate)
        expt_source = "intermediate"
    else:
        sim_data_expt = None
        expt_source = None
    return expt_source, sim_data_expt, sim_data_ref


@app.cell
def _(
    expt_source,
    path_expt,
    path_expt_intermediate,
    path_ref,
    sim_data_expt,
    sim_data_ref,
):
    if sim_data_ref is None:
        print(f"**Warning:** Reference sim_data not found at `{path_ref}`.")
    elif sim_data_expt is None:
        print(
            f"**Warning:** expt sim_data not found at `{path_expt}` "
            f"or intermediate `{path_expt_intermediate}`. Run ParCa with "
            "`save_intermediates: true` so fit_condition output is saved when the run fails."
        )
    else:
        if expt_source == "intermediate":
            print(
                "**Reference:** sim_data from kb. **expt:** sim_data from intermediate "
                "`fit_condition` (ParCa did not complete; translation supply is still available)."
            )
        else:
            print("Both sim_data pickles loaded from final kb.")
    return


@app.cell
def _(cell_specs_expt_path, cell_specs_ref_path, pickle):
    with open(cell_specs_ref_path, "rb") as f:
        cell_specs_ref = pickle.load(f)

    with open(cell_specs_expt_path, "rb") as f:
        cell_specs_expt = pickle.load(f)
    return cell_specs_expt, cell_specs_ref


@app.cell
def _(mo):
    mo.md("""## Run metadata (expt config)""")
    return


@app.cell
def _(mo, sim_data_expt, sim_data_ref):
    def meta_row(label, sd):
        if sd is None:
            return (label, "—", "—")
        expt = getattr(sd, "rnaseq_manifest_path", None)
        dataset = getattr(sd, "rnaseq_basal_dataset_id", None)
        return (
            label,
            str(expt) if expt else "None (reference)",
            str(dataset) if dataset else "—",
        )

    rows_meta = [
        meta_row("Reference (defaults)", sim_data_ref),
        meta_row("New expt", sim_data_expt),
    ]
    mo.md(
        """
        | Run | `rnaseq_manifest_path` | `rnaseq_basal_dataset_id` |
        |-----|------------------------|---------------------------|
        | """
        + " | ".join(rows_meta[0])
        + " |\n        | "
        + " | ".join(rows_meta[1])
        + " |"
    )
    return


@app.cell
def _(mo):
    mo.md("""## Translation supply rate (mmol / g DCW / h)""")
    return


@app.cell
def _(extract_supply_rates, sim_data_expt, sim_data_ref):
    supply_ref = extract_supply_rates(sim_data_ref)
    supply_expt = extract_supply_rates(sim_data_expt)
    return supply_expt, supply_ref


@app.cell
def _(SUPPLY_RATE_UNIT, np):
    def extract_supply_rates(sim_data):
        if sim_data is None:
            return None
        tr = getattr(sim_data, "translation_supply_rate", None) or {}
        minimal = tr.get("minimal")
        minimal_plus_aa = tr.get("minimal_plus_amino_acids")
        if minimal is None or minimal_plus_aa is None:
            return None
        aa_ids = list(sim_data.molecule_groups.amino_acids)
        n = len(aa_ids)
        if len(minimal) != n or len(minimal_plus_aa) != n:
            return None

        def to_number(arr):
            if hasattr(arr, "asNumber"):
                return np.asarray(arr.asNumber(SUPPLY_RATE_UNIT))
            return np.asarray(arr)

        return {
            "aa_ids": aa_ids,
            "minimal": to_number(minimal),
            "minimal_plus_amino_acids": to_number(minimal_plus_aa),
        }

    return (extract_supply_rates,)


@app.cell
def _(pd):
    def supply_table(supply, run_label):
        if supply is None:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "run": run_label,
                "aa": supply["aa_ids"],
                "minimal": supply["minimal"],
                "minimal_plus_amino_acids": supply["minimal_plus_amino_acids"],
            }
        )

    # table_ref = supply_table(supply_ref, "reference")
    # table_expt = supply_table(supply_expt, "expt")
    return


@app.cell
def _(mo):
    mo.md("""### Side-by-side comparison (all AAs)""")
    return


@app.cell
def _(np, pd, supply_expt, supply_ref):
    if supply_ref is not None and supply_expt is not None:
        compare = pd.DataFrame(
            {
                "aa": supply_ref["aa_ids"],
                "minimal_ref": supply_ref["minimal"],
                "minimal_plus_aa_ref": supply_ref["minimal_plus_amino_acids"],
                "minimal_expt": supply_expt["minimal"],
                "minimal_plus_aa_expt": supply_expt["minimal_plus_amino_acids"],
            }
        )
        compare["minimal_ratio"] = compare["minimal_expt"] / np.maximum(
            compare["minimal_ref"], 1e-20
        )
        compare["minimal_plus_aa_ratio"] = compare["minimal_plus_aa_expt"] / np.maximum(
            compare["minimal_plus_aa_ref"], 1e-20
        )
    else:
        compare = pd.DataFrame()
    return (compare,)


@app.cell
def _(compare):
    compare
    return


@app.cell
def _(alt, compare, mo, pd):
    if compare.empty or "minimal_ref" not in compare.columns:
        plot_df = pd.DataFrame()
        chart_supply = mo.md("(Need both runs for plot.)")
    else:
        # --- 1. Absolute Supply Chart ---
        plot_df = compare.melt(
            id_vars=["aa"],
            value_vars=[
                "minimal_ref",
                "minimal_expt",
                "minimal_plus_aa_ref",
                "minimal_plus_aa_expt",
            ],
            var_name="condition_run",
            value_name="supply",
        )
        plot_df["condition"] = plot_df["condition_run"].str.replace(
            "_ref$|_expt$", "", regex=True
        )
        plot_df["run"] = plot_df["condition_run"].str.extract("_(ref|expt)$")[0]

    chart_supply = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("aa:N", sort="-y", title="Amino Acid"),
            xOffset="run:N",
            y="supply:Q",
            color="run:N",
            column=alt.Column("condition:N", header=alt.Header(titleOrient="bottom")),
        )
        .properties(width={"step": 8}, height=280)
    )

    mo.vstack(
        [
            mo.md("### Absolute Supply"),
            chart_supply,
        ]
    )
    return


@app.cell
def _(alt, compare, mo, np):
    # 1. Calculation
    compare_ratios = compare.copy()
    compare_ratios["ratio_ref"] = (
        compare_ratios["minimal_plus_aa_ref"] / compare_ratios["minimal_ref"]
    )
    compare_ratios["ratio_expt"] = (
        compare_ratios["minimal_plus_aa_expt"] / compare_ratios["minimal_expt"]
    )

    # 2. Create TWO distinct dataframes to avoid category "leakage"
    # This prevents Chart A from reserving space for Chart B's bars.
    df_left = (
        compare_ratios.melt(
            id_vars=["aa"],
            value_vars=["ratio_ref", "ratio_expt"],
            var_name="run",
            value_name="val",
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    df_left["run"] = df_left["run"].str.replace("ratio_", "")

    df_right = (
        compare_ratios.melt(
            id_vars=["aa"],
            value_vars=["minimal_ratio", "minimal_plus_aa_ratio"],
            var_name="cond",
            value_name="val",
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    df_right["cond"] = df_right["cond"].str.replace("_ratio", "")

    # 3. Define a shared base style to keep them identical
    def make_chart(df, encoding_col, title, y_title):
        return (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("aa:N", title="Amino Acid"),
                xOffset=f"{encoding_col}:N",
                y=alt.Y("val:Q", title=y_title),
                color=alt.Color(f"{encoding_col}:N", legend=alt.Legend(title=None)),
            )
            .properties(width={"step": 12}, height=280, title=title)
        )

    # 4. Generate and Concatenate
    c1 = make_chart(df_left, "run", "Plus AA / Minimal", "Ratio")
    c2 = make_chart(df_right, "cond", "Expt / Ref", "Ratio")

    # The magic fix: resolve x and color independently
    chart_ratios = alt.hconcat(c1, c2).resolve_scale(
        x="independent", color="independent"
    )

    mo.vstack([mo.md("### Comparative Ratios"), chart_ratios])
    return (compare_ratios,)


@app.cell
def _(compare_ratios):
    compare_ratios
    return


@app.cell
def _(mo):
    mo.md(r"""## Next let's look at transcript and protein levels""")
    return


@app.cell
def _(np, pd, sim_data_expt, sim_data_ref):
    # === Cell A: CYS pathway monomers and RNA mapping ===

    # 1. Get CYS enzymes (forward + reverse) from sim_data_ref
    met_ref = sim_data_ref.process.metabolism
    cys_key = "CYS[c]"
    cys_path = met_ref.aa_synthesis_pathways[cys_key]

    def strip_compartment(id_):
        # e.g. "CYSSYNMULTI-CPLX[c]" -> "CYSSYNMULTI-CPLX"
        return id_.split("[")[0]

    fwd_enzymes = [strip_compartment(e) for e in cys_path["enzymes"]]
    rev_enzymes = [strip_compartment(e) for e in cys_path["reverse enzymes"]]

    # 2. Expand each enzyme into monomer subunits via complexation in sim_data_ref
    cx = sim_data_ref.process.complexation
    cys_monomers = set()

    # monomer_with_compartment -> list of (enzyme_id, direction)
    monomer_to_enzyme = {}  # e.g. "G7622-MONOMER[c]" -> [("G7622-MONOMER", "reverse")]

    # Forward direction
    for eid in fwd_enzymes:
        eid_with_comp = f"{eid}[c]"
        info = cx.get_monomers(eid_with_comp)
        for sub_id in info["subunitIds"]:
            cys_monomers.add(sub_id)
            monomer_to_enzyme.setdefault(sub_id, []).append((eid, "forward"))

    # Reverse direction
    for eid in rev_enzymes:
        eid_with_comp = f"{eid}[c]"
        info = cx.get_monomers(eid_with_comp)
        for sub_id in info["subunitIds"]:
            cys_monomers.add(sub_id)
            monomer_to_enzyme.setdefault(sub_id, []).append((eid, "reverse"))

    # 3. Map CYS monomers -> cistrons -> RNAs (using sim_data_ref)
    rel = sim_data_ref.relation
    cistron_data = sim_data_ref.process.transcription.cistron_data
    rna_data = sim_data_ref.process.transcription.rna_data
    monomer_data = sim_data_ref.process.translation.monomer_data

    monomer_ids_all = monomer_data["id"]

    # monomer ID -> monomer index
    monomer_idxs = set()
    for mid in cys_monomers:
        idxs = np.where(monomer_ids_all == mid)[0]
        monomer_idxs.update(idxs.tolist())
    monomer_idxs = sorted(monomer_idxs)

    # monomer index -> cistron index
    rna_idxs = set()
    rna_to_monomers = {}  # RNA index -> set of monomer IDs (with compartment)

    for mon_idx in monomer_idxs:
        c_idx = rel.cistron_to_monomer_mapping[mon_idx]
        c_id = cistron_data["id"][c_idx]
        for r_idx in sim_data_ref.process.transcription.cistron_id_to_rna_indexes(c_id):
            rna_idxs.add(r_idx)
            rna_to_monomers.setdefault(r_idx, set()).add(monomer_ids_all[mon_idx])

    rna_idxs = sorted(rna_idxs)
    cys_rna_ids = rna_data["id"][rna_idxs]

    # === Cell B: CYS RNA expression table with enzyme + direction annotation ===

    expr_ref_basal = sim_data_ref.process.transcription.rna_expression["basal"]
    expr_ref_with_aa = sim_data_ref.process.transcription.rna_expression["with_aa"]
    expr_expt_basal = sim_data_expt.process.transcription.rna_expression["basal"]
    expr_expt_with_aa = sim_data_expt.process.transcription.rna_expression["with_aa"]

    rows = []
    for i, r_idx in enumerate(rna_idxs):
        rna_id = cys_rna_ids[i]

        # All monomers contributing to this RNA
        monomers_for_rna = rna_to_monomers.get(r_idx, set())

        # Collect (enzyme, direction) pairs for these monomers
        enzyme_dir_pairs = set()
        for m in monomers_for_rna:
            for eid, direction in monomer_to_enzyme.get(m, []):
                enzyme_dir_pairs.add((eid, direction))

        # Format for display: join enzyme IDs and directions
        if enzyme_dir_pairs:
            enzymes = sorted({eid for eid, _ in enzyme_dir_pairs})
            directions = sorted({direction for _, direction in enzyme_dir_pairs})
            enzyme_str = ", ".join(enzymes)
            direction_str = ", ".join(directions)
        else:
            enzyme_str = ""
            direction_str = ""

        rows.append(
            {
                "rna_id": rna_id,
                "enzymes": enzyme_str,
                "direction": direction_str,  # "forward", "reverse", or "forward, reverse"
                "expr_ref_basal": expr_ref_basal[r_idx],
                "expr_ref_with_aa": expr_ref_with_aa[r_idx],
                "expr_expt_basal": expr_expt_basal[r_idx],
                "expr_expt_with_aa": expr_expt_with_aa[r_idx],
            }
        )

    cys_rna_expr = pd.DataFrame(rows)
    eps = 1e-12
    cys_rna_expr["log2_fc_basal"] = np.log2(
        (cys_rna_expr["expr_expt_basal"] + eps) / (cys_rna_expr["expr_ref_basal"] + eps)
    )
    cys_rna_expr["log2_fc_with_aa"] = np.log2(
        (cys_rna_expr["expr_expt_with_aa"] + eps)
        / (cys_rna_expr["expr_ref_with_aa"] + eps)
    )
    return cys_rna_expr, fwd_enzymes, monomer_to_enzyme, rev_enzymes


@app.cell
def _(cys_rna_expr):
    cys_rna_expr.sort_values("direction")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""proteins""")
    return


@app.cell
def _(
    cell_specs_expt,
    cell_specs_ref,
    fwd_enzymes,
    monomer_to_enzyme,
    np,
    pd,
    rev_enzymes,
):
    from ecoli.library.schema import bulk_name_to_idx, counts

    # === Enzyme counts cell, reusing RNA Cell A outputs ===

    def enzyme_counts_for_run(cell_specs, label):
        """
        cell_specs: dict with 'basal' and 'with_aa' specs for one run (ref or rnaseq).
        label: 'ref' or 'rnaseq'.

        Uses fwd_enzymes, rev_enzymes, cys_monomers, monomer_to_enzyme
        defined in the earlier CYS RNA cell.
        """
        rows = []
        for cond in ["basal", "with_aa"]:
            bulk = cell_specs[cond].get(
                "bulkAverageContainer", cell_specs[cond]["bulkContainer"]
            )
            ids = bulk["id"]

            for direction, enzyme_ids in [
                ("forward", fwd_enzymes),
                ("reverse", rev_enzymes),
            ]:
                for eid in enzyme_ids:
                    eid_with_comp = f"{eid}[c]"

                    # --- Complex-only count (ParCa-style) ---
                    complex_idxs = []
                    if eid_with_comp in ids:
                        complex_idxs.append(bulk_name_to_idx([eid_with_comp], ids)[0])

                    if complex_idxs:
                        complex_count = counts(bulk, complex_idxs).sum()
                    else:
                        complex_count = 0

                    # --- Complex + monomer subunits (inclusive capacity view) ---
                    inclusive_idxs = list(complex_idxs)  # start from complexes
                    for mon_id, pairs in monomer_to_enzyme.items():
                        # mon_id is like "G7622-MONOMER[c]"
                        if any(
                            p_eid == eid and p_dir == direction
                            for p_eid, p_dir in pairs
                        ):
                            if mon_id in ids:
                                inclusive_idxs.append(
                                    bulk_name_to_idx([mon_id], ids)[0]
                                )

                    if inclusive_idxs:
                        inclusive_idxs = np.unique(inclusive_idxs)
                        total_with_monomers = counts(bulk, inclusive_idxs).sum()
                    else:
                        total_with_monomers = 0

                    rows.append(
                        {
                            "run": label,
                            "condition": cond,
                            "enzyme": eid,
                            "direction": direction,
                            "complex_count": complex_count,
                            "total_count_including_monomers": total_with_monomers,
                        }
                    )

        return pd.DataFrame(rows)

    # Build tables for reference and RNaseq runs
    cys_enzyme_counts_ref = enzyme_counts_for_run(cell_specs_ref, "ref")
    cys_enzyme_counts_expt = enzyme_counts_for_run(cell_specs_expt, "expt")

    cys_enzyme_counts = pd.concat(
        [cys_enzyme_counts_ref, cys_enzyme_counts_expt],
        ignore_index=True,
    )

    cys_enzyme_counts
    return bulk_name_to_idx, counts, cys_enzyme_counts


@app.cell
def _(cys_enzyme_counts):
    complex_summary = cys_enzyme_counts.pivot(
        index=["enzyme", "direction"],
        columns=["run", "condition"],
        values="complex_count",
    )
    complex_summary
    return (complex_summary,)


@app.cell
def _(complex_summary):
    complex_summary.groupby("direction").sum()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Solver-input view: what `set_mechanistic_supply_constants` sees

    For CYS and ARG we show supply (mmol/g/h and molecule/s), enzyme counts (fwd/rev in basal and with_aa),
    concentrations, and the implied kcat from the two-condition balance (with downstream=0).
    """
    )
    return


@app.cell
def _(
    K_CAT_UNITS,
    SUPPLY_RATE_UNIT,
    bulk_name_to_idx,
    cell_specs_expt,
    cell_specs_ref,
    counts,
    np,
    pd,
    sim_data_expt,
    sim_data_ref,
    supply_expt,
    supply_ref,
):
    def _aa_enzyme_totals(cell_specs, sim_data, aa_key):
        """Sum fwd/rev complex counts (basal, with_aa) for one AA and one run."""
        path = sim_data.process.metabolism.aa_synthesis_pathways[aa_key]
        fwd_ids = [e.split("[")[0] + "[c]" for e in path["enzymes"]]
        rev_ids = [e.split("[")[0] + "[c]" for e in path["reverse enzymes"]]
        out = {}
        for cond in ["basal", "with_aa"]:
            bulk = cell_specs[cond].get(
                "bulkAverageContainer", cell_specs[cond]["bulkContainer"]
            )
            ids = bulk["id"]
            for direction, mol_ids in [("forward", fwd_ids), ("reverse", rev_ids)]:
                idxs = []
                for mid in mol_ids:
                    if mid in ids:
                        idxs.append(bulk_name_to_idx([mid], ids)[0])
                out[(cond, direction)] = (
                    counts(bulk, np.array(idxs)).sum() if idxs else 0
                )
        return out

    aa_keys = ["CYS[c]", "ARG[c]"]
    supply_aa_ids = supply_ref["aa_ids"] if supply_ref else []
    aa_to_index = {aid: ix for ix, aid in enumerate(supply_aa_ids)}

    solver_input_rows = []
    for run_name, specs, sd, supply in [
        ("ref", cell_specs_ref, sim_data_ref, supply_ref),
        ("expt", cell_specs_expt, sim_data_expt, supply_expt),
    ]:
        if supply is None or sd is None:
            continue
        mass_basal = specs["basal"]["avgCellDryMassInit"]
        mass_with_aa = specs["with_aa"]["avgCellDryMassInit"]
        n_av = sd.constants.n_avogadro
        for aa_id in aa_keys:
            idx_aa = aa_to_index.get(aa_id)
            if idx_aa is None:
                continue
            supply_min = supply["minimal"][idx_aa]
            supply_aa = supply["minimal_plus_amino_acids"][idx_aa]
            supply_basal_1s = (
                SUPPLY_RATE_UNIT * supply_min * mass_basal * n_av
            ).asNumber(K_CAT_UNITS)
            supply_with_aa_1s = (
                SUPPLY_RATE_UNIT * supply_aa * mass_with_aa * n_av
            ).asNumber(K_CAT_UNITS)
            enz = _aa_enzyme_totals(specs, sd, aa_id)
            solver_input_rows.append(
                {
                    "run": run_name,
                    "aa": aa_id,
                    "supply_minimal_mmol_g_h": supply_min,
                    "supply_with_aa_mmol_g_h": supply_aa,
                    "supply_basal_1s": supply_basal_1s,
                    "supply_with_aa_1s": supply_with_aa_1s,
                    "fwd_basal": enz[("basal", "forward")],
                    "fwd_with_aa": enz[("with_aa", "forward")],
                    "rev_basal": enz[("basal", "reverse")],
                    "rev_with_aa": enz[("with_aa", "reverse")],
                }
            )

    solver_input_table = pd.DataFrame(solver_input_rows)
    solver_input_table.sort_values("aa")
    return (solver_input_table,)


@app.cell
def _(mo):
    mo.md("""### Concentrations (CYS pathway)""")
    return


@app.cell
def _(METABOLITE_CONCENTRATION_UNITS, np, pd, sim_data_expt, sim_data_ref):
    # Concentrations from metabolism: minimal vs minimal_plus_amino_acids
    def _conc_table(sim_data, run_label):
        conc_fn = sim_data.process.metabolism.concentration_updates.concentrations_based_on_nutrients
        minimal = conc_fn("minimal")
        with_aa = conc_fn("minimal_plus_amino_acids")
        cys_key = "CYS[c]"
        path = sim_data.process.metabolism.aa_synthesis_pathways[cys_key]
        # CYS + upstream AAs that appear in pathway
        upstream = list(path.get("upstream", {}).keys())
        aa_show = [cys_key] + [u for u in upstream if u != cys_key]
        rows = []
        for aa in aa_show:
            c_min = minimal.get(aa)
            c_aa = with_aa.get(aa)
            c_min_val = (
                c_min.asNumber(METABOLITE_CONCENTRATION_UNITS)
                if c_min is not None
                else np.nan
            )
            c_aa_val = (
                c_aa.asNumber(METABOLITE_CONCENTRATION_UNITS)
                if c_aa is not None
                else np.nan
            )
            rows.append(
                {
                    "run": run_label,
                    "aa": aa,
                    "conc_minimal_mol_L": c_min_val,
                    "conc_with_aa_mol_L": c_aa_val,
                }
            )
        return pd.DataFrame(rows)

    conc_ref = _conc_table(sim_data_ref, "ref")
    conc_expt = _conc_table(sim_data_expt, "expt")
    concentrations_table = pd.concat([conc_ref, conc_expt], ignore_index=True)
    concentrations_table
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Two-condition balance (downstream=0): implied kcat_fwd, kcat_rev

    Solve A @ [kcat_fwd, kcat_rev] = b with downstream terms set to zero.
    Ref should yield both ≥ 0; expt for CYS typically yields kcat_rev < 0 (solver rejects).

    **Columns:** `balance_basal` = CYS/ARG demand in **minimal** (molecules/s); `balance_with_aa` = demand in **with_aa** minus uptake (molecules/s). The implied **kcat_fwd_1s** and **kcat_rev_1s** are the rates that would satisfy both conditions; if **kcat_rev_1s** < 0 there is no feasible mechanistic solution.
    """
    )
    return


@app.cell
def _(
    K_CAT_UNITS,
    METABOLITE_CONCENTRATION_UNITS,
    SUPPLY_RATE_UNIT,
    bulk_name_to_idx,
    cell_specs_expt,
    cell_specs_ref,
    counts,
    np,
    pd,
    sim_data_expt,
    sim_data_ref,
    solver_input_table,
    units,
):
    def _implied_kcats(sim_data, cell_specs, supply_min, supply_aa, aa_key):
        """Compute fwd/rev fractions, A and b (downstream=0), solve for kcat_fwd, kcat_rev."""
        met = sim_data.process.metabolism
        path = met.aa_synthesis_pathways[aa_key]
        conc_fn = met.concentration_updates.concentrations_based_on_nutrients
        minimal_conc = conc_fn("minimal")
        with_aa_conc = conc_fn("minimal_plus_amino_acids")
        aa_conc_basal = minimal_conc[aa_key].asNumber(METABOLITE_CONCENTRATION_UNITS)
        aa_conc_with_aa = with_aa_conc[aa_key].asNumber(METABOLITE_CONCENTRATION_UNITS)

        ki_val = path["ki"]
        if ki_val is None:
            ki = np.inf
        elif hasattr(ki_val, "__len__") and len(ki_val) == 2:
            lower, upper = (
                ki_val[0].asNumber(METABOLITE_CONCENTRATION_UNITS),
                ki_val[1].asNumber(METABOLITE_CONCENTRATION_UNITS),
            )
            ki = np.clip(aa_conc_basal, lower, upper)
        else:
            ki = ki_val.asNumber(METABOLITE_CONCENTRATION_UNITS)
        # Match metabolism: finite km_reverse when pathway has reverse enzymes but no reverse stoich
        if path.get("reverse"):
            km_reverse_raw = path["km, reverse"]
            if units.isnan(km_reverse_raw):
                km_reverse = (minimal_conc[aa_key] * 10).asNumber(
                    METABOLITE_CONCENTRATION_UNITS
                )
            else:
                km_reverse = km_reverse_raw.asNumber(METABOLITE_CONCENTRATION_UNITS)
        elif (
            path.get("reverse enzymes")
            and not units.isfinite(path["km, degradation"])
            and aa_key != "L-SELENOCYSTEINE[c]"
        ):
            km_reverse = (minimal_conc[aa_key] * 10).asNumber(
                METABOLITE_CONCENTRATION_UNITS
            )
        else:
            km_reverse = np.inf

        km_degradation = path["km, degradation"]
        if not units.isfinite(km_degradation):
            km_degradation = np.inf
        else:
            km_degradation = km_degradation.asNumber(METABOLITE_CONCENTRATION_UNITS)
        upstream_aa = list(path.get("upstream", {}).keys())
        kms_upstream = path.get("km, upstream") or {}
        kms = np.array(
            [
                (
                    kms_upstream[aa].asNumber(METABOLITE_CONCENTRATION_UNITS)
                    if aa in kms_upstream
                    else minimal_conc[aa].asNumber(METABOLITE_CONCENTRATION_UNITS)
                )
                for aa in upstream_aa
            ]
        )
        km_conc_basal = (
            np.array(
                [
                    minimal_conc[aa].asNumber(METABOLITE_CONCENTRATION_UNITS)
                    for aa in upstream_aa
                ]
            )
            if upstream_aa
            else np.array([])
        )
        km_conc_with_aa = (
            np.array(
                [
                    with_aa_conc[aa].asNumber(METABOLITE_CONCENTRATION_UNITS)
                    for aa in upstream_aa
                ]
            )
            if upstream_aa
            else np.array([])
        )

        def frac_fwd(aa_conc, km_conc):
            f = 1.0 / (1.0 + aa_conc / ki)
            if len(kms) > 0 and len(km_conc) > 0:
                f *= np.prod(1.0 / (1.0 + kms / np.maximum(km_conc, 1e-20)))
            return f

        def frac_rev(aa_conc):
            rev = 1.0 / (
                1.0
                + km_reverse
                / np.maximum(aa_conc, 1e-20)
                * (1.0 + aa_conc / np.maximum(km_degradation, 1e-20))
            )
            deg = 1.0 / (
                1.0
                + km_degradation
                / np.maximum(aa_conc, 1e-20)
                * (1.0 + aa_conc / np.maximum(km_reverse, 1e-20))
            )
            return rev + deg

        fwd_frac_basal = frac_fwd(aa_conc_basal, km_conc_basal)
        fwd_frac_with_aa = frac_fwd(aa_conc_with_aa, km_conc_with_aa)
        loss_frac_basal = frac_rev(aa_conc_basal)
        loss_frac_with_aa = frac_rev(aa_conc_with_aa)

        bulk_basal = cell_specs["basal"].get(
            "bulkAverageContainer", cell_specs["basal"]["bulkContainer"]
        )
        bulk_with_aa = cell_specs["with_aa"].get(
            "bulkAverageContainer", cell_specs["with_aa"]["bulkContainer"]
        )
        fwd_ids = list(path["enzymes"])
        rev_ids = list(path["reverse enzymes"])
        ids_b = bulk_basal["id"]
        ids_w = bulk_with_aa["id"]
        fwd_basal = (
            counts(
                bulk_basal,
                bulk_name_to_idx([x for x in fwd_ids if x in ids_b], ids_b),
            ).sum()
            if any(x in ids_b for x in fwd_ids)
            else 0
        )
        fwd_with_aa = (
            counts(
                bulk_with_aa,
                bulk_name_to_idx([x for x in fwd_ids if x in ids_w], ids_w),
            ).sum()
            if any(x in ids_w for x in fwd_ids)
            else 0
        )
        rev_basal = (
            counts(
                bulk_basal,
                bulk_name_to_idx([x for x in rev_ids if x in ids_b], ids_b),
            ).sum()
            if any(x in ids_b for x in rev_ids)
            else 0
        )
        rev_with_aa = (
            counts(
                bulk_with_aa,
                bulk_name_to_idx([x for x in rev_ids if x in ids_w], ids_w),
            ).sum()
            if any(x in ids_w for x in rev_ids)
            else 0
        )

        fwd_cap_basal = fwd_basal * fwd_frac_basal
        rev_cap_basal = rev_basal * loss_frac_basal
        fwd_cap_with_aa = fwd_with_aa * fwd_frac_with_aa
        rev_cap_with_aa = rev_with_aa * loss_frac_with_aa

        uptake_rate = 0.0
        if aa_key[:-3] in met.amino_acid_uptake_rates:
            uptake_rate = met.amino_acid_uptake_rates[aa_key[:-3]]["uptake"].asNumber(
                units.mmol / units.g / units.h
            )
        mass_aa = cell_specs["with_aa"]["avgCellDryMassInit"]
        uptake_1s = (
            units.mmol
            / units.g
            / units.h
            * uptake_rate
            * mass_aa
            * sim_data.constants.n_avogadro
        ).asNumber(K_CAT_UNITS)

        supply_basal_1s = (
            SUPPLY_RATE_UNIT
            * supply_min
            * cell_specs["basal"]["avgCellDryMassInit"]
            * sim_data.constants.n_avogadro
        ).asNumber(K_CAT_UNITS)
        supply_with_aa_1s = (
            SUPPLY_RATE_UNIT * supply_aa * mass_aa * sim_data.constants.n_avogadro
        ).asNumber(K_CAT_UNITS)
        balance_basal = supply_basal_1s
        balance_with_aa = supply_with_aa_1s - uptake_1s

        A = np.array(
            [
                [fwd_cap_basal, -rev_cap_basal],
                [fwd_cap_with_aa, -rev_cap_with_aa],
            ],
            dtype=float,
        )
        b = np.array([balance_basal, balance_with_aa], dtype=float)
        try:
            x = np.linalg.solve(A, b)
            kcat_fwd, kcat_rev = float(x[0]), float(x[1])
        except np.linalg.LinAlgError:
            # Singular or ill-conditioned: use least-squares so we still get numbers
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            kcat_fwd, kcat_rev = float(x[0]), float(x[1])
        return kcat_fwd, kcat_rev, A, b

    balance_rows = []
    for run_label, cell_specs, sim_data in [
        ("ref", cell_specs_ref, sim_data_ref),
        ("expt", cell_specs_expt, sim_data_expt),
    ]:
        if sim_data is None:
            continue
        supply_df = solver_input_table[solver_input_table["run"] == run_label]
        if supply_df.empty:
            continue
        for aa in ["CYS[c]", "ARG[c]"]:
            row = supply_df[supply_df["aa"] == aa]
            if row.empty:
                continue
            row = row.iloc[0]
            kf, kr, A, b = _implied_kcats(
                sim_data,
                cell_specs,
                row["supply_minimal_mmol_g_h"],
                row["supply_with_aa_mmol_g_h"],
                aa,
            )
            balance_rows.append(
                {
                    "run": run_label,
                    "aa": aa,
                    "kcat_fwd_1s": kf,
                    "kcat_rev_1s": kr,
                    "balance_basal": b[0],
                    "balance_with_aa": b[1],
                }
            )

    balance_result_table = pd.DataFrame(balance_rows)
    balance_result_table
    return (balance_result_table,)


@app.cell
def _(alt, balance_result_table, mo):
    # CYS-focused: implied kcat_fwd and kcat_rev for ref vs expt
    cys_balance = balance_result_table[balance_result_table["aa"] == "CYS[c]"]
    if cys_balance.empty:
        chart_cys_kcats = mo.md("(No CYS balance data.)")
    else:
        plot_cys = cys_balance.melt(
            id_vars=["run"],
            value_vars=["kcat_fwd_1s", "kcat_rev_1s"],
            var_name="kcat",
            value_name="value_1s",
        )
        chart_cys_kcats = (
            alt.Chart(plot_cys)
            .mark_bar()
            .encode(
                x=alt.X("kcat:N", title=""),
                y=alt.Y("value_1s:Q", title="Implied kcat (1/s)"),
                xOffset="run:N",
                color="run:N",
            )
            .properties(
                title="CYS: implied kcat_fwd and kcat_rev (two-condition balance, downstream=0)"
            )
        )
    mo.vstack(
        [mo.md("### CYS: implied kcats from two-condition balance"), chart_cys_kcats]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
