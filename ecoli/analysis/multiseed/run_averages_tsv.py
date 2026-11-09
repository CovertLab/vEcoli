"""Average molecule counts and synthesis probabilities over the completed cells
of one vEcoli experiment variant, for several numbers of skipped generations
at once.

Each cell is averaged over its time steps (skipping the first, where some
listeners are still zero), then the cell averages are averaged with equal
weight; sd columns are the spread of the cell averages. A cell counts as
completed when its seed has a next generation, or when it is the last
generation of a successful lineage. vEcoli generations start at 1, so skipping
N generations keeps generation N + 1 onward (the same convention as
skip_n_gens in ecoli.library.parquet_emitter).

Writes one folder per skip value, <analysis outdir>/skip_<N>/, with:
    protein_counts.tsv          total (listeners__monomer_counts: free plus
                                every copy in a complex, including active
                                ribosomes, RNAP, replisomes and bound TFs) and
                                free (bulk count) of each monomer;
                                mean_in_complexes = mean_total - mean_free;
                                plus both as concentrations (mM)
    free_complex_counts.tsv     bulk count of each complexation, equilibrium
                                and two-component system complex (a complex
                                inside a larger complex is not counted), and
                                its concentration (mM)
    metabolite_counts.tsv       count and concentration (mM) of each small
                                molecule with a nonzero average count. source
                                fba_target: FBA homeostatic targets, from
                                listeners__enzyme_kinetics__metabolite_counts_
                                final (counts after the metabolism update).
                                source bulk_not_fba: other bulk molecules that
                                are not proteins, complexes or RNAs, from bulk
                                counts (e.g. ppGpp when ppGpp regulation is on,
                                since FBA then has no ppGpp target; the
                                polymerized_* entries are nucleotides and amino
                                acids built into polymers, not free molecules)
    mrna_tu_counts.tsv          total, full, partial and decaying mRNA per TU
    mrna_cistron_counts.tsv     total, full, partial and available mRNA per
                                cistron
    partial_rrna_tu_counts.tsv  partial transcripts of each rRNA operon TU
    rrna_counts.tsv             per rRNA cistron (16S, 23S, 5S): free bulk
                                count and partial transcripts, with gene ID
    trna_counts.tsv             per tRNA cistron: charged and uncharged counts
                                and their sum (tRNAs are in no complexes, so
                                these are all copies), with gene ID. Both forms
                                come from the same transcription, so
                                mean_total pairs with the cistron's synthesis
                                in synth_prob_cistron.tsv. Some charged forms
                                have their own IDs (RNA0-...), in charged_id
    rrna_totals.tsv             16S, 23S and 5S rRNA: free, in inactive
                                ribosomes (free 30S or 50S subunits), in
                                active ribosomes, and total
    synth_prob_tu.tsv           actual and target RNA synthesis probability
                                and promoter copy number per TU; for the
                                attenuated TUs, the amino acid(s) whose charged
                                tRNA controls attenuation, the attenuation stop
                                probability, and transcripts stopped per second
                                (nan and blank for other TUs)
    synth_prob_cistron.tsv      actual and target synthesis probability per
                                cistron (normalized over cistrons, so about
                                half the summed TU probability), and expected
                                transcription initiations per second covering
                                each cistron (not normalized). All three are
                                computed from the per-TU columns as
                                RnaSynthProb does. Expected initiations use
                                listeners__rna_synth_prob__total_rna_init: the
                                listener's own expected_rna_init_per_cistron
                                column implies about 5x fewer initiations than
                                that column records, so it is not used. Every
                                cistron of a TU is credited with each
                                initiation, which overcounts for attenuated
                                TUs: their stopped transcripts are dropped
                                before RNAP makes any of them
                                (attenuation_location is 1 nt)
    tf_binding.tsv              promoter-bound count of each TF
                                (n_actual_bound), promoters it can bind
                                (n_available_promoters), and occupancy (bound /
                                available, averaged per time step)
    unique_molecule_counts.tsv  count of each unique molecule type (active
                                ribosomes, active RNAP, RNAs, promoters,
                                genes, oriC, ...). active_replisome (one per
                                replication fork) is written as
                                replication_fork when the run's config has
                                mechanistic_replisome off
    reaction_fluxes.tsv         flux of every full FBA reaction ID (forward and
                                reverse are separate IDs; not base reaction
                                IDs), from listeners__fba_results__
                                reaction_fluxes, in mmol/L/s (the FBA solution
                                divided by the time step)
    cells.tsv                   the cells used

Concentrations are count times listeners__enzyme_kinetics__counts_to_molar
(1 / (Avogadro * cell volume), in mmol/L) at each time step, then averaged
like the counts, so they are not mean count / mean volume. They are per
whole-cell volume.

Columns a run did not record (e.g. available or decaying mRNA counts in older
runs) are written as nan. Cells are read by params["workers"] processes in
parallel (default: the Slurm job's CPUs).

# todo (mia): figure out what is up with the expected_rna_init_per_cistron listener it looks wrong so I used a workaround for now

"""

import multiprocessing
import os
import pickle
import re
import time
from typing import Any

from duckdb import DuckDBPyConnection
import numpy as np

from ecoli.library.parquet_emitter import (
    config_value,
    create_duckdb_conn,
    field_metadata,
    list_columns,
    ndlist_to_ndarray,
    open_arbitrary_sim_data,
)

RNA_COUNTS = "listeners__rna_counts__"
ENZYME_KINETICS = "listeners__enzyme_kinetics__"
SYNTH = "listeners__rna_synth_prob__"
# Count columns that also get a concentration column (key + "_conc")
CONC_KEYS = (
    "protein_total",
    "protein_free",
    "bulk",
    "metabolite_counts",
    "small_molecules",
)
ELONGATION = "listeners__transcript_elongation_listener__"
UNIQUE = "listeners__unique_molecule_counts__"

# Output key (same as the wcEcoli script) -> history column
RNA_COLUMNS = {
    name: RNA_COUNTS + name
    for name in (
        "mRNA_counts",
        "full_mRNA_counts",
        "partial_mRNA_counts",
        "decaying_mRNA_counts",
        "mRNA_cistron_counts",
        "full_mRNA_cistron_counts",
        "partial_mRNA_cistron_counts",
        "available_mRNA_cistron_counts",
        "partial_rRNA_counts",
        "partial_rRNA_cistron_counts",
    )
}


class RunningStats:
    """Running mean and sd of cell averages (Welford), one vector per key."""

    def __init__(self):
        self.n = 0
        self.mean = {}
        self.m2 = {}

    def add(self, values):
        self.n += 1
        for key, x in values.items():
            if key not in self.mean:
                self.mean[key] = np.zeros_like(x)
                self.m2[key] = np.zeros_like(x)
            delta = x - self.mean[key]
            self.mean[key] += delta / self.n
            self.m2[key] += delta * (x - self.mean[key])

    def sd(self, key):
        if self.n < 2:
            return np.full_like(self.mean[key], np.nan)
        return np.sqrt(self.m2[key] / (self.n - 1))


def completed_cells(conn, config_sql, success_sql, variant, first_gen):
    """(lineage_seed, generation, agent_id) of completed cells from first_gen on."""
    cells = conn.sql(f"""
        SELECT DISTINCT lineage_seed, generation, agent_id FROM ({config_sql})
        WHERE variant = {variant}""").fetchall()
    present = {(seed, gen) for seed, gen, _ in cells}
    successful = {
        tuple(row)
        for row in conn.sql(f"""
        SELECT DISTINCT lineage_seed, generation FROM ({success_sql})
        WHERE variant = {variant}""").fetchall()
    }
    return sorted(
        (seed, gen, agent)
        for seed, gen, agent in cells
        if gen >= first_gen
        and ((seed, gen + 1) in present or (seed, gen) in successful)
    )


def complex_ids(sim_data):
    """(complex ID, source) for every complex, each ID once."""
    sources = [
        ("complexation", sim_data.process.complexation.ids_complexes),
        ("equilibrium", sim_data.process.equilibrium.ids_complexes),
        (
            "two_component_system",
            sim_data.process.two_component_system.complex_to_monomer.keys(),
        ),
    ]
    seen, result = set(), []
    for source, ids in sources:
        for complex_id in ids:
            if complex_id not in seen:
                seen.add(complex_id)
                result.append((complex_id, source))
    return result


def free_rna_ids(sim_data):
    """(bulk RNA ID, type, cistron ID, gene ID) for free rRNAs and tRNAs."""
    groups = sim_data.molecule_groups
    transcription = sim_data.process.transcription
    cistron_to_gene = dict(
        zip(transcription.cistron_data["id"], transcription.cistron_data["gene_id"])
    )
    result = []
    for label, ids in (
        ("16S rRNA", groups.s30_16s_rRNA),
        ("23S rRNA", groups.s50_23s_rRNA),
        ("5S rRNA", groups.s50_5s_rRNA),
        ("uncharged tRNA", transcription.uncharged_trna_names),
    ):
        for rna_id in ids:
            result.append(
                (rna_id, label, rna_id[:-3], cistron_to_gene.get(rna_id[:-3], ""))
            )
    # Charged tRNA names line up with the uncharged names; some charged forms
    # have their own IDs (RNA0-...), so the cistron comes from the pair
    for charged, uncharged in zip(
        transcription.charged_trna_names, transcription.uncharged_trna_names
    ):
        result.append(
            (
                charged,
                "charged tRNA",
                uncharged[:-3],
                cistron_to_gene.get(uncharged[:-3], ""),
            )
        )
    return result


def attenuated_tu_ids(sim_data):
    """(TU ID, amino acids whose charged tRNA controls attenuation) for each
    attenuated TU, in listener order."""
    transcription = sim_data.process.transcription
    k = transcription.attenuation_k
    k = np.asarray(k.asNumber() if hasattr(k, "asNumber") else k)
    aas = sim_data.molecule_groups.amino_acids
    return [
        (
            transcription.rna_data["id"][idx],
            [aas[i][:-3] for i in np.nonzero(k[:, j])[0]],
        )
        for j, idx in enumerate(transcription.attenuated_rna_indices)
    ]


# Set once per worker process by init_worker
worker = {}


def init_worker(settings):
    worker.update(settings)
    worker["conn"] = create_duckdb_conn()


def read_cell(cell):
    """Worker entry point: (cell, per-cell averages keyed like the wcEcoli
    script)."""
    seed, gen, agent = cell
    path = os.path.join(
        worker["history_dir"],
        f"lineage_seed={seed}",
        f"generation={gen}",
        f"agent_id={agent}",
        "*.pq",
    )
    present = worker["present_columns"]
    selects = {
        "time": "time",
        "protein_total": "listeners__monomer_counts",
        "protein_free": f"list_select(bulk, {worker['monomer_bulk_idx']})",
        "bulk": f"list_select(bulk, {worker['other_bulk_idx']})",
        "actual_rna_synth_prob": SYNTH + "actual_rna_synth_prob",
        "target_rna_synth_prob": SYNTH + "target_rna_synth_prob",
        "nActualBound": SYNTH + "n_actual_bound",
        "n_available_promoters": SYNTH + "n_available_promoters",
        "promoter_copy_number": SYNTH + "promoter_copy_number",
        "total_rna_init": SYNTH + "total_rna_init",
        "attenuation_probability": ELONGATION + "attenuation_probability",
        "counts_attenuated": ELONGATION + "counts_attenuated",
        "reactionFluxes": "listeners__fba_results__reaction_fluxes",
        "metabolite_counts": ENZYME_KINETICS + "metabolite_counts_final",
        "counts_to_molar": ENZYME_KINETICS + "counts_to_molar",
    }
    if worker["small_molecule_bulk_idx"]:
        selects["small_molecules"] = (
            f"list_select(bulk, {worker['small_molecule_bulk_idx']})"
        )
    selects.update(RNA_COLUMNS)
    for name in worker["unique_names"]:
        selects[f"unique__{name}"] = UNIQUE + name
    selects = {
        key: expr
        for key, expr in selects.items()
        if expr.startswith("list_select") or expr in present
    }
    sql = ", ".join(f'{expr} AS "{key}"' for key, expr in selects.items())
    data = (
        worker["conn"]
        .sql(f"SELECT {sql} FROM read_parquet('{path}') ORDER BY time")
        .pl()
    )

    def column(key):
        values = data[key]
        if values.dtype.is_nested():
            return ndlist_to_ndarray(values).astype(np.float64)
        return values.to_numpy().astype(np.float64)

    time_steps = np.diff(column("time"))
    # mmol/L per count at each step (1 / (Avogadro * cell volume)), so each
    # concentration is averaged per step rather than mean count / mean volume
    counts_to_molar = (
        column("counts_to_molar")[1:] if "counts_to_molar" in selects else None
    )
    values = {}
    for key in selects:
        if key in ("time", "counts_to_molar") or key.startswith("unique__"):
            continue
        array = column(key)[1:]
        if key == "counts_attenuated":
            array = array / time_steps[:, None]
        values[key] = array.mean(axis=0) if key != "total_rna_init" else array
        if key in CONC_KEYS and counts_to_molar is not None:
            values[f"{key}_conc"] = (array * counts_to_molar[:, None]).mean(axis=0)

    # Per-cistron values from the per-TU columns, as RnaSynthProb computes them
    mapping = worker["cistron_tu_mapping"]
    total_rna_init = values.pop("total_rna_init")
    for label in ("actual", "target"):
        per_tu = column(f"{label}_rna_synth_prob")[1:]
        per_cistron = np.asarray(mapping.dot(per_tu.T)).T
        if label == "actual":
            values["expected_rna_init_per_cistron"] = (
                per_cistron * total_rna_init[:, None] / time_steps[:, None]
            ).mean(axis=0)
        sums = per_cistron.sum(axis=1, keepdims=True)
        per_cistron = np.divide(per_cistron, sums, out=per_cistron, where=sums != 0)
        values[f"{label}_rna_synth_prob_per_cistron"] = per_cistron.mean(axis=0)

    # Occupancy averaged per step; steps with no available promoters skipped
    bound = column("nActualBound")[1:]
    available = column("n_available_promoters")[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        occupancy = np.where(available > 0, bound / available, np.nan)
    values["tf_occupancy"] = np.nanmean(occupancy, axis=0)

    values["unique"] = np.array(
        [column(f"unique__{name}")[1:].mean() for name in worker["unique_names"]]
    )
    return cell, values


def write_table(path, header, row_ids, columns, n_cells):
    """One row per ID. columns is a list of (name, values); float values are
    written with 10 significant digits, and a column whose values are None
    (not recorded in this run) as nan."""
    with open(path, "w") as f:
        f.write(
            "\t".join([header] + [name for name, _ in columns] + ["n_cells"]) + "\n"
        )
        for i, row_id in enumerate(row_ids):
            fields = [str(row_id)] + [
                "nan"
                if values is None
                else f"{values[i]:.10g}"
                if isinstance(values[i], (float, np.floating))
                else str(values[i])
                for _, values in columns
            ]
            f.write("\t".join(fields + [str(n_cells)]) + "\n")


def write_outputs(
    folder,
    stats,
    ids,
    complexes,
    rnas,
    cells,
    mechanistic_replisome,
    subunit_ids,
    attenuated_tus,
):
    os.makedirs(folder, exist_ok=True)
    n = stats.n

    def mean_sd(key, label):
        if key not in stats.mean:
            return [(f"mean_{label}", None), (f"sd_{label}", None)]
        return [(f"mean_{label}", stats.mean[key]), (f"sd_{label}", stats.sd(key))]

    def table(name, header, row_ids, columns):
        write_table(os.path.join(folder, name), header, row_ids, columns, n)

    table(
        "protein_counts.tsv",
        "monomer_id",
        ids["monomer"],
        mean_sd("protein_total", "total")
        + mean_sd("protein_free", "free")
        + [
            (
                "mean_in_complexes",
                stats.mean["protein_total"] - stats.mean["protein_free"],
            )
        ]
        + mean_sd("protein_total_conc", "total_conc_mM")
        + mean_sd("protein_free_conc", "free_conc_mM"),
    )

    n_complexes = len(complexes)
    bulk_mean, bulk_sd = stats.mean["bulk"], stats.sd("bulk")
    conc_columns = [
        (name, None if values is None else values[:n_complexes])
        for name, values in mean_sd("bulk_conc", "free_conc_mM")
    ]
    table(
        "free_complex_counts.tsv",
        "complex_id",
        [c for c, _ in complexes],
        [
            ("source", [s for _, s in complexes]),
            ("mean_free", bulk_mean[:n_complexes]),
            ("sd_free", bulk_sd[:n_complexes]),
        ]
        + conc_columns,
    )

    # Metabolites with a nonzero average count: FBA targets first, then other
    # small bulk molecules (e.g. ppGpp when ppGpp regulation is on)
    metabolite_ids, sources, metabolite_columns = [], [], {}
    for key, id_key, source in (
        ("metabolite_counts", "metabolite", "fba_target"),
        ("small_molecules", "small_molecule", "bulk_not_fba"),
    ):
        if key not in stats.mean:
            continue
        nonzero = np.nonzero(stats.mean[key] > 0)[0]
        metabolite_ids += [ids[id_key][i] for i in nonzero]
        sources += [source] * len(nonzero)
        for name, values in mean_sd(key, "count") + mean_sd(f"{key}_conc", "conc_mM"):
            part = np.full(len(nonzero), np.nan) if values is None else values[nonzero]
            metabolite_columns.setdefault(name, []).append(part)
    if metabolite_ids:
        table(
            "metabolite_counts.tsv",
            "metabolite_id",
            metabolite_ids,
            [("source", sources)]
            + [
                (name, np.concatenate(parts))
                for name, parts in metabolite_columns.items()
            ],
        )

    rna_mean, rna_sd = bulk_mean[n_complexes:], bulk_sd[n_complexes:]
    rrna_rows = [i for i, r in enumerate(rnas) if "rRNA" in r[1]]
    partial_index = {c: i for i, c in enumerate(ids["rRNA_cistron"])}

    def partial_column(values):
        if values is None:
            return None
        return np.array(
            [
                values[partial_index[rnas[i][2]]]
                if rnas[i][2] in partial_index
                else np.nan
                for i in rrna_rows
            ]
        )

    partial = mean_sd("partial_rRNA_cistron_counts", "partial")
    table(
        "rrna_counts.tsv",
        "rna_id",
        [rnas[i][0] for i in rrna_rows],
        [
            ("type", [rnas[i][1] for i in rrna_rows]),
            ("cistron_id", [rnas[i][2] for i in rrna_rows]),
            ("gene_id", [rnas[i][3] for i in rrna_rows]),
            ("mean_free", rna_mean[rrna_rows]),
            ("sd_free", rna_sd[rrna_rows]),
        ]
        + [(name, partial_column(values)) for name, values in partial],
    )

    row_of = {(r[1], r[2]): i for i, r in enumerate(rnas)}
    trna_cistrons = [r[2] for r in rnas if r[1] == "uncharged tRNA"]
    charged = [row_of[("charged tRNA", c)] for c in trna_cistrons]
    uncharged = [row_of[("uncharged tRNA", c)] for c in trna_cistrons]
    table(
        "trna_counts.tsv",
        "cistron_id",
        trna_cistrons,
        [
            ("gene_id", [rnas[i][3] for i in uncharged]),
            ("charged_id", [rnas[i][0] for i in charged]),
            ("mean_charged", rna_mean[charged]),
            ("sd_charged", rna_sd[charged]),
            ("mean_uncharged", rna_mean[uncharged]),
            ("sd_uncharged", rna_sd[uncharged]),
            ("mean_total", rna_mean[charged] + rna_mean[uncharged]),
        ],
    )

    # rRNA totals: free rRNA (all operon copies) plus one copy per free 30S or
    # 50S subunit and per active ribosome. Averages add, so these are exact;
    # no sd is given for the sums.
    complex_index = {c: i for i, (c, _) in enumerate(complexes)}
    free_30s = bulk_mean[complex_index[subunit_ids[0]]]
    free_50s = bulk_mean[complex_index[subunit_ids[1]]]
    active = stats.mean["unique"][ids["unique"].index("active_ribosome")]
    free, in_subunits = [], []
    for label, subunit in (
        ("16S rRNA", free_30s),
        ("23S rRNA", free_50s),
        ("5S rRNA", free_50s),
    ):
        free.append(rna_mean[[r[1] == label for r in rnas]].sum())
        in_subunits.append(subunit)
    free, in_subunits = np.array(free), np.array(in_subunits)
    table(
        "rrna_totals.tsv",
        "rrna_type",
        ["16S rRNA", "23S rRNA", "5S rRNA"],
        [
            ("mean_free", free),
            ("mean_in_inactive_ribosomes", in_subunits),
            ("mean_in_active_ribosomes", np.full(3, active)),
            ("mean_total", free + in_subunits + active),
        ],
    )

    table(
        "mrna_tu_counts.tsv",
        "tu_id",
        ids["mRNA"],
        mean_sd("mRNA_counts", "total")
        + mean_sd("full_mRNA_counts", "full")
        + mean_sd("partial_mRNA_counts", "partial")
        + mean_sd("decaying_mRNA_counts", "decaying"),
    )
    table(
        "mrna_cistron_counts.tsv",
        "cistron_id",
        ids["mRNA_cistron"],
        mean_sd("mRNA_cistron_counts", "total")
        + mean_sd("full_mRNA_cistron_counts", "full")
        + mean_sd("partial_mRNA_cistron_counts", "partial")
        + mean_sd("available_mRNA_cistron_counts", "available"),
    )
    table(
        "partial_rrna_tu_counts.tsv",
        "tu_id",
        ids["rRNA"],
        mean_sd("partial_rRNA_counts", "partial"),
    )

    tu_index = {tu: i for i, tu in enumerate(ids["tu"])}
    attenuating_aa = [""] * len(ids["tu"])
    for tu, aas in attenuated_tus:
        attenuating_aa[tu_index[tu]] = ",".join(aas)

    def per_tu(values):
        if values is None:
            return None
        full = np.full(len(ids["tu"]), np.nan)
        full[[tu_index[tu] for tu, _ in attenuated_tus]] = values
        return full

    attenuation_columns = [("attenuating_aa", attenuating_aa)] + [
        (name, per_tu(values))
        for key, label in (
            ("attenuation_probability", "attenuation_prob"),
            ("counts_attenuated", "attenuated_per_s"),
        )
        for name, values in mean_sd(key, label)
    ]
    table(
        "synth_prob_tu.tsv",
        "tu_id",
        ids["tu"],
        mean_sd("actual_rna_synth_prob", "actual")
        + mean_sd("target_rna_synth_prob", "target")
        + mean_sd("promoter_copy_number", "promoter_copies")
        + attenuation_columns,
    )
    table(
        "synth_prob_cistron.tsv",
        "cistron_id",
        ids["cistron"],
        mean_sd("actual_rna_synth_prob_per_cistron", "actual")
        + mean_sd("target_rna_synth_prob_per_cistron", "target")
        + mean_sd("expected_rna_init_per_cistron", "expected_inits_per_s"),
    )
    table(
        "tf_binding.tsv",
        "tf_id",
        ids["tf"],
        mean_sd("nActualBound", "bound")
        + mean_sd("n_available_promoters", "available_promoters")
        + mean_sd("tf_occupancy", "occupancy"),
    )

    # Each active_replisome is one replication fork. Without the mechanistic
    # replisome they hold no subunits, so they are labeled as forks.
    unique_ids = [
        "replication_fork"
        if name == "active_replisome" and not mechanistic_replisome
        else name
        for name in ids["unique"]
    ]
    table(
        "unique_molecule_counts.tsv",
        "unique_molecule",
        unique_ids,
        mean_sd("unique", "count"),
    )
    table(
        "reaction_fluxes.tsv",
        "reaction_id",
        ids["reaction"],
        mean_sd("reactionFluxes", "flux"),
    )

    with open(os.path.join(folder, "cells.tsv"), "w") as f:
        f.write("lineage_seed\tgeneration\tagent_id\n")
        for seed, gen, agent in cells:
            f.write(f"{seed}\t{gen}\t{agent}\n")


def history_dir_from_sql(history_sql: str, experiment_id: str, variant: int) -> str:
    """Folder holding one variant's per-cell parquet files, taken from the
    parquet glob in history_sql (see ecoli.library.parquet_emitter.dataset_sql),
    so each worker can read one cell's files directly."""
    match = re.search(r"'([^']*)/history/\*/\*/\*/\*/\*/\*\.pq'", history_sql)
    if match is None:
        raise ValueError("could not find the history parquet path in history_sql")
    return os.path.join(
        match.group(1),
        "history",
        f"experiment_id={experiment_id}",
        f"variant={variant}",
    )


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_paths: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    """Write the averaged tables for one experiment and variant into
    <outdir>/skip_<N>/ for each N in params["skip_gens"].

    Params::

        {
            # Numbers of initial generations to skip; one output folder each
            "skip_gens": list[int] (default [0, 4, 8]),
            # Processes reading cells in parallel (default: SLURM_CPUS_PER_TASK,
            # else 1)
            "workers": int,
        }
    """
    skip_gens = list(params.get("skip_gens", [0, 4, 8]))
    experiment_ids = [
        row[0]
        for row in conn.sql(
            f"SELECT DISTINCT experiment_id FROM ({config_sql})"
        ).fetchall()
    ]
    variants = [
        row[0]
        for row in conn.sql(f"SELECT DISTINCT variant FROM ({config_sql})").fetchall()
    ]
    if len(experiment_ids) != 1 or len(variants) != 1:
        raise ValueError(
            f"expected one experiment and variant, got {experiment_ids} {variants}"
        )
    experiment_id, variant = experiment_ids[0], variants[0]

    with open_arbitrary_sim_data(sim_data_paths) as f:
        sim_data = pickle.load(f)
    complexes = complex_ids(sim_data)
    rnas = free_rna_ids(sim_data)
    subunit_ids = (
        sim_data.molecule_ids.s30_full_complex,
        sim_data.molecule_ids.s50_full_complex,
    )
    attenuated_tus = attenuated_tu_ids(sim_data)

    present = set(list_columns(conn, history_sql))
    try:
        mechanistic_replisome = bool(
            config_value(conn, config_sql, "mechanistic_replisome")
        )
    except Exception:
        # Missing from the config means the LoadSimData default (False)
        mechanistic_replisome = False
    print(f"mechanistic_replisome: {mechanistic_replisome}")

    bulk_index = {
        name: i + 1 for i, name in enumerate(field_metadata(conn, config_sql, "bulk"))
    }
    ids = {
        "monomer": list(field_metadata(conn, config_sql, "listeners__monomer_counts")),
        "mRNA": list(field_metadata(conn, config_sql, RNA_COUNTS + "mRNA_counts")),
        "mRNA_cistron": list(
            field_metadata(conn, config_sql, RNA_COUNTS + "mRNA_cistron_counts")
        ),
        "rRNA": list(
            field_metadata(conn, config_sql, RNA_COUNTS + "partial_rRNA_counts")
        ),
        "rRNA_cistron": list(
            field_metadata(conn, config_sql, RNA_COUNTS + "partial_rRNA_cistron_counts")
        ),
        "tu": list(field_metadata(conn, config_sql, SYNTH + "actual_rna_synth_prob")),
        "cistron": [str(c) for c in sim_data.process.transcription.cistron_data["id"]],
        "tf": list(field_metadata(conn, config_sql, SYNTH + "n_actual_bound")),
        "reaction": list(
            field_metadata(conn, config_sql, "listeners__fba_results__reaction_fluxes")
        ),
        "unique": sorted(c[len(UNIQUE) :] for c in present if c.startswith(UNIQUE)),
        "metabolite": list(
            field_metadata(
                conn, config_sql, ENZYME_KINETICS + "metabolite_counts_final"
            )
        ),
    }
    # Bulk small molecules that are not FBA targets: everything in bulk except
    # monomers, complexes, RNAs and the FBA targets themselves
    transcription = sim_data.process.transcription
    groups = sim_data.molecule_groups
    not_small = (
        set(ids["monomer"])
        | {c for c, _ in complexes}
        | set(transcription.rna_data["id"])
        | set(transcription.charged_trna_names)
        | set(transcription.uncharged_trna_names)
        | set(groups.s30_16s_rRNA)
        | set(groups.s50_23s_rRNA)
        | set(groups.s50_5s_rRNA)
        | set(ids["metabolite"])
    )
    ids["small_molecule"] = [name for name in bulk_index if name not in not_small]
    settings = {
        "history_dir": history_dir_from_sql(history_sql, experiment_id, variant),
        "present_columns": present,
        "monomer_bulk_idx": [bulk_index[m] for m in ids["monomer"]],
        "other_bulk_idx": [bulk_index[c] for c, _ in complexes]
        + [bulk_index[r[0]] for r in rnas],
        "small_molecule_bulk_idx": [bulk_index[m] for m in ids["small_molecule"]],
        "unique_names": ids["unique"],
        "cistron_tu_mapping": sim_data.process.transcription.cistron_tu_mapping_matrix,
    }

    vecoli_first_gen = min(skip_gens) + 1
    cells = completed_cells(conn, config_sql, success_sql, variant, vecoli_first_gen)
    if not cells:
        print("Skipping analysis -- no completed generations.")
        return
    stats = {skip: RunningStats() for skip in skip_gens}
    used = {skip: [] for skip in skip_gens}
    workers = params.get("workers") or int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"{len(cells)} cells, {workers} worker processes", flush=True)
    start = time.time()
    with multiprocessing.Pool(
        workers, initializer=init_worker, initargs=(settings,)
    ) as pool:
        # imap keeps the original cell order, so results do not depend on
        # which worker finishes first
        for k, (cell, values) in enumerate(pool.imap(read_cell, cells), 1):
            for skip in skip_gens:
                if cell[1] > skip:
                    stats[skip].add(values)
                    used[skip].append(cell)
            if k % 25 == 0 or k == len(cells):
                print(
                    f"{k} of {len(cells)} cells read, {(time.time() - start) / k:.2f} s per cell",
                    flush=True,
                )

    for skip in skip_gens:
        folder = os.path.join(outdir, f"skip_{skip}")
        write_outputs(
            folder,
            stats[skip],
            ids,
            complexes,
            rnas,
            used[skip],
            mechanistic_replisome,
            subunit_ids,
            attenuated_tus,
        )
        print(f"wrote {folder}: {stats[skip].n} cells")
