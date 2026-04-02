"""
Experiment summary report for multi-parca vEcoli workflow runs.

Produces:
  - parca_summary.csv  — one row per parca run (status, sim counts, cell stats)
  - cell_distributions.png — violin + strip plots faceted by metric, hued by dataset_id

Usage:
    uv run wholecell/io/multiparca_analysis.py --out_dir out/my_experiment -o out/reports/
"""

import argparse
import copy
import os
import glob
import importlib.util
import json
import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ecoli.library.parquet_emitter import dataset_sql, read_stacked_columns

REPO_ROOT = Path(__file__).parents[2]
RUNSCRIPTS_DIR = REPO_ROOT / "runscripts"


# ---------------------------------------------------------------------------
# Parca map
# ---------------------------------------------------------------------------


def _count_pickles_per_parca(config: dict) -> int:
    """
    Return the number of variant sim_data pickles each ParCa run produces.
    Mirrors the same function in runscripts/workflow.py.
    """
    variant_config = config.get("variants", {})
    skip_baseline = config.get("skip_baseline", False)
    if not variant_config:
        return 0 if skip_baseline else 1
    spec = importlib.util.spec_from_file_location(
        "create_variants",
        RUNSCRIPTS_DIR / "create_variants.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    variant_name = list(variant_config.keys())[0]
    params_copy = copy.deepcopy(variant_config[variant_name])
    n_variants = len(mod.parse_variants(params_copy))
    return n_variants + (0 if skip_baseline else 1)


def build_parca_map(out_dir: Path) -> dict:
    """
    Read workflow_config.json and per-parca config files to build:
      {parca_id: {dataset_id, variant_start, variant_end}}
    """
    nextflow_dir = out_dir / "nextflow"
    with open(nextflow_dir / "workflow_config.json") as f:
        wf_config = json.load(f)

    pickles_per_parca = _count_pickles_per_parca(wf_config)
    parca_variants_list = wf_config.get("parca_variants") or [{}]
    n_parca = len(parca_variants_list)

    parca_map = {}
    for i in range(n_parca):
        parca_config_path = nextflow_dir / f"parca_config_{i}.json"
        if parca_config_path.exists():
            with open(parca_config_path) as f:
                parca_config = json.load(f)
            dataset_id = (
                parca_config.get("parca_options", {}).get("rnaseq_basal_dataset_id")
                or "legacy"
            )
        else:
            dataset_id = f"parca_{i}"

        parca_map[i] = {
            "dataset_id": dataset_id,
            "variant_start": i * pickles_per_parca,
            "variant_end": (i + 1) * pickles_per_parca,
        }

    return parca_map


# ---------------------------------------------------------------------------
# Nextflow trace
# ---------------------------------------------------------------------------


def load_trace(experiment_id: str) -> dict:
    """
    Glob trace--{experiment_id}--*.csv in the repo root and parse runParca rows.
    Returns {parca_id: {status, duration_min, workdir}}.
    If no trace is found, returns an empty dict (status will appear as "unknown").
    """
    pattern = str(REPO_ROOT / f"trace--{experiment_id}--*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return {}

    # Use the most recent trace file
    df = pd.read_csv(files[-1], sep=",")

    result = {}
    for _, row in df.iterrows():
        name = str(row.get("name", ""))
        m = re.match(r"runParca \((\d+)\)", name)
        if not m:
            continue
        k = int(m.group(1))
        parca_id = k - 1  # Nextflow 1-indexes submissions

        # Keep the row with the highest attempt number for each parca_id
        attempt = int(row.get("attempt", 1))
        if parca_id in result and result[parca_id].get("_attempt", 0) >= attempt:
            continue

        try:
            duration_min = float(row.get("duration", 0)) / 60000.0
        except (TypeError, ValueError):
            duration_min = None

        result[parca_id] = {
            "status": str(row.get("status", "unknown")),
            "duration_min": duration_min,
            "workdir": str(row.get("workdir", "")),
            "_attempt": attempt,
        }

    # Strip internal tracking key
    for v in result.values():
        v.pop("_attempt", None)

    return result


def load_sim_errors(experiment_id: str, parca_map: dict) -> dict:
    """
    Parse trace CSV for FAILED/IGNORED sim tasks.
    Returns {parca_id: error_string} summarising unique errors and which gen/seed
    they came from.  Empty string if no failures for that parca.
    """
    pattern = str(REPO_ROOT / f"trace--{experiment_id}--*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return {pid: "" for pid in parca_map}

    df = pd.read_csv(files[-1])

    variant_to_parca = {
        v: parca_id
        for parca_id, info in parca_map.items()
        for v in range(info["variant_start"], info["variant_end"])
    }

    errors_by_parca: dict[int, list[str]] = {pid: [] for pid in parca_map}

    for _, row in df.iterrows():
        status = str(row.get("status", ""))
        if status not in ("FAILED", "IGNORED"):
            continue
        name = str(row.get("name", ""))
        m = re.match(
            r"sim_gen_\d+ \(variant=(\d+)/seed=(\d+)/generation=(\d+)/agent_id=\S+\)",
            name,
        )
        if not m:
            continue
        variant, seed, generation = int(m.group(1)), int(m.group(2)), int(m.group(3))
        parca_id = variant_to_parca.get(variant)
        if parca_id is None:
            continue
        error = _get_parca_error(str(row.get("workdir", "")))
        errors_by_parca[parca_id].append(
            f"gen{generation}/seed{seed}: {error or '(no message)'}"
        )

    # Deduplicate on error text; keep first occurrence's gen/seed as example
    result = {}
    for parca_id, errs in errors_by_parca.items():
        seen_msgs: set[str] = set()
        unique: list[str] = []
        for entry in errs:
            msg = entry.split(": ", 1)[1] if ": " in entry else entry
            if msg not in seen_msgs:
                seen_msgs.add(msg)
                unique.append(entry)
        result[parca_id] = "; ".join(unique)
    return result


def _get_parca_error(workdir: str) -> str:
    """
    Extract a short error summary from .command.err in a Nextflow work directory.
    Returns the last XxxError/Exception line, or the last non-empty line.
    """
    if not workdir:
        return ""
    err_path = Path(workdir) / ".command.err"
    if not err_path.exists():
        return ""
    try:
        with open(err_path) as f:
            lines = f.readlines()
        non_empty = [ln.rstrip() for ln in lines if ln.strip()]
        for ln in reversed(non_empty):
            if re.search(r"\w+Error:", ln) or re.search(r"\w+Exception:", ln):
                return ln
        return non_empty[-1] if non_empty else ""
    except OSError:
        return ""


def _get_genes_filled_from_ref(workdir: str) -> int | None:
    """
    Parse the UserWarning logged when missing genes are filled from the reference
    dataset.  Returns the count, 0 if the warning was not emitted (no missing
    genes), or None if .command.err is not available.
    """
    if not workdir:
        return None
    err_path = Path(workdir) / ".command.err"
    if not err_path.exists():
        return None
    try:
        with open(err_path) as f:
            text = f.read()
        m = re.search(r"(\d+) genes were missing from experimental RNA-seq", text)
        return int(m.group(1)) if m else 0
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Higher-order properties (from cd1_higher_order_properties multiseed analysis)
# ---------------------------------------------------------------------------


def load_higher_order_properties(out_dir: Path, parca_map: dict) -> pd.DataFrame:
    """
    Load higher_order_properties.tsv files written by the cd1_higher_order_properties
    multiseed analysis.  Path pattern:
        {out_dir}/analyses/variant={v}/lineage_seed=*/plots/higher_order_properties.tsv

    Returns a long-form DataFrame with columns [dataset_id, parca_id, metric, value]
    where each row is one cell × one metric.  Returns an empty DataFrame if no files
    are found (i.e. the analysis hasn't run yet).
    """
    all_frames = []
    for parca_id, info in parca_map.items():
        for v in range(info["variant_start"], info["variant_end"]):
            pattern = str(
                out_dir / f"analyses/variant={v}/plots/higher_order_properties.tsv"
            )
            for fpath in glob.glob(pattern):
                try:
                    df = pd.read_csv(fpath, sep="\t")
                except Exception as exc:
                    print(f"Warning: could not read {fpath}: {exc}")
                    continue
                cell_cols = [
                    c for c in df.columns if c not in ("Properties", "mean", "std")
                ]
                if not cell_cols:
                    continue
                long = df.melt(
                    id_vars=["Properties"],
                    value_vars=cell_cols,
                    var_name="cell_id",
                    value_name="value",
                ).rename(columns={"Properties": "metric"})
                # cell_id format: "Cell: {lineage_seed}_{agent_id}"
                # agent_id length encodes generation depth (0→gen1, 00→gen2, …)
                seed_agent = long["cell_id"].str[6:]  # strip "Cell: "
                long["lineage_seed"] = (
                    seed_agent.str.rsplit("_", n=1).str[0].astype(int)
                )
                long["generation"] = seed_agent.str.rsplit("_", n=1).str[1].str.len()
                long["dataset_id"] = info["dataset_id"]
                long["parca_id"] = parca_id
                all_frames.append(
                    long[
                        [
                            "dataset_id",
                            "parca_id",
                            "lineage_seed",
                            "generation",
                            "metric",
                            "value",
                        ]
                    ]
                )

    if not all_frames:
        return pd.DataFrame(columns=["dataset_id", "parca_id", "metric", "value"])
    return pd.concat(all_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Sim data loaders (patterns from compare_sims.py)
# ---------------------------------------------------------------------------


def load_doubling_times(
    history_sql: str, success_sql: str, experiment_id: str
) -> pd.DataFrame:
    """
    Per-cell doubling times.
    Columns: [Doubling Time (hr), experiment_id, variant, lineage_seed, generation, agent_id]
    """
    time_subquery = read_stacked_columns(
        history_sql,
        ["time"],
        remove_first=False,
        success_sql=success_sql,
    )
    conn = duckdb.connect()
    df = conn.sql(
        f"""
        SELECT
            (MAX(time) - MIN(time)) / 3600.0 AS "Doubling Time (hr)",
            experiment_id,
            variant,
            lineage_seed,
            generation,
            agent_id
        FROM ({time_subquery})
        WHERE experiment_id = '{experiment_id}'
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        ORDER BY experiment_id, variant, lineage_seed, generation, agent_id
        """
    ).to_df()
    conn.close()
    return df


def load_protein_mass_per_cell(
    history_sql: str, success_sql: str, experiment_id: str
) -> pd.DataFrame:
    """
    Time-averaged protein mass per cell.
    Columns: [protein_mass_fg, experiment_id, variant, lineage_seed, generation, agent_id]
    """
    subquery = read_stacked_columns(
        history_sql,
        ["listeners__mass__protein_mass"],
        remove_first=False,
        success_sql=success_sql,
    )
    conn = duckdb.connect()
    df = conn.sql(
        f"""
        SELECT
            AVG(listeners__mass__protein_mass) AS protein_mass_fg,
            experiment_id,
            variant,
            lineage_seed,
            generation,
            agent_id
        FROM ({subquery})
        WHERE experiment_id = '{experiment_id}'
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        ORDER BY experiment_id, variant, lineage_seed, generation, agent_id
        """
    ).to_df()
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


_ORDERED_METRICS = [
    "Doubling Time (hr)",
    "Protein mass (fg)",
    "Cell mass (mg/10^9 cells)",
    "Cell volume (um^3)",
]


def _plot_cell_distributions(
    dt_all: pd.DataFrame,
    prot_all: pd.DataFrame,
    higher_order_long: pd.DataFrame,
    parca_map: dict,
    output_dir: Path,
    min_generation: int | None = None,
):
    """2x2 violin + strip plots, one panel per metric, hued by dataset_id."""
    variant_to_dataset = {
        v: info["dataset_id"]
        for info in parca_map.values()
        for v in range(info["variant_start"], info["variant_end"])
    }

    long_rows = []
    if not dt_all.empty:
        for _, row in dt_all.iterrows():
            long_rows.append(
                {
                    "dataset_id": variant_to_dataset.get(
                        int(row["variant"]), "unknown"
                    ),
                    "generation": int(row["generation"]),
                    "metric": "Doubling Time (hr)",
                    "value": row["Doubling Time (hr)"],
                }
            )
    if not prot_all.empty:
        for _, row in prot_all.iterrows():
            long_rows.append(
                {
                    "dataset_id": variant_to_dataset.get(
                        int(row["variant"]), "unknown"
                    ),
                    "generation": int(row["generation"]),
                    "metric": "Protein mass (fg)",
                    "value": row["protein_mass_fg"],
                }
            )

    _VIOLIN_METRICS = {"Cell mass (mg/10^9 cells)", "Cell volume (um^3)"}

    frames = [pd.DataFrame(long_rows)] if long_rows else []
    if not higher_order_long.empty:
        ho_filtered = higher_order_long[
            higher_order_long["metric"].isin(_VIOLIN_METRICS)
        ]
        frames.append(ho_filtered[["dataset_id", "generation", "metric", "value"]])

    if not frames:
        print("No cell data to plot.")
        return

    metrics_df = pd.concat(frames, ignore_index=True)

    if min_generation is not None:
        metrics_df = metrics_df[metrics_df["generation"] >= min_generation]
        if metrics_df.empty:
            print(
                f"No data remaining after filtering to generation >= {min_generation}."
            )
            return

    metrics = [m for m in _ORDERED_METRICS if m in metrics_df["metric"].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    dataset_ids = sorted(metrics_df["dataset_id"].unique())
    palette = sns.color_palette("tab10", n_colors=len(dataset_ids))
    color_map = dict(zip(dataset_ids, palette))

    for i, ax in enumerate(axes_flat):
        if i >= len(metrics):
            ax.set_visible(False)
            continue
        metric = metrics[i]
        data = metrics_df[metrics_df["metric"] == metric]
        sns.violinplot(
            data=data,
            x="dataset_id",
            y="value",
            hue="dataset_id",
            palette=color_map,
            inner=None,
            alpha=0.4,
            ax=ax,
            legend=False,
        )
        sns.stripplot(
            data=data,
            x="dataset_id",
            y="value",
            color="black",
            alpha=0.5,
            ax=ax,
        )
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=50)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    title = "Cell distributions by dataset"
    if min_generation is not None:
        title += f"  (generation \u2265 {min_generation})"
    fig.suptitle(title)
    plt.tight_layout()

    suffix = f"_gen{min_generation}plus" if min_generation is not None else ""
    path = output_dir / f"cell_distributions{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def _plot_timecourse(
    dt_all: pd.DataFrame,
    prot_all: pd.DataFrame,
    higher_order_long: pd.DataFrame,
    parca_map: dict,
    output_dir: Path,
):
    """
    One PNG per dataset showing per-cell metrics vs generation.
    Each line is one lineage seed; all metrics in separate subplots.
    Uses only already-loaded per-cell aggregate data — no new Parquet reads.
    """
    for parca_id, info in parca_map.items():
        dataset_id = info["dataset_id"]
        v_start, v_end = info["variant_start"], info["variant_end"]

        frames = []

        if not dt_all.empty:
            sub = dt_all[dt_all["variant"].between(v_start, v_end - 1)][
                ["generation", "lineage_seed", "Doubling Time (hr)"]
            ].copy()
            sub = sub.rename(columns={"Doubling Time (hr)": "value"})
            sub["metric"] = "Doubling Time (hr)"
            frames.append(sub[["generation", "lineage_seed", "metric", "value"]])

        if not prot_all.empty:
            sub = prot_all[prot_all["variant"].between(v_start, v_end - 1)][
                ["generation", "lineage_seed", "protein_mass_fg"]
            ].copy()
            sub = sub.rename(columns={"protein_mass_fg": "value"})
            sub["metric"] = "Protein mass (fg)"
            frames.append(sub[["generation", "lineage_seed", "metric", "value"]])

        if not higher_order_long.empty:
            sub = higher_order_long[higher_order_long["parca_id"] == parca_id][
                ["generation", "lineage_seed", "metric", "value"]
            ].copy()
            frames.append(sub)

        if not frames:
            continue

        data = pd.concat(frames, ignore_index=True)
        if data.empty:
            continue
        metrics = list(data["metric"].unique())
        n = len(metrics)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False
        )
        axes_flat = axes.flatten()

        for ax, metric in zip(axes_flat, metrics):
            metric_data = data[data["metric"] == metric]
            sns.lineplot(
                data=metric_data,
                x="generation",
                y="value",
                hue="lineage_seed",
                marker="o",
                estimator=None,
                ax=ax,
                legend=False,
            )
            ax.set_title(metric, fontsize=9)
            ax.set_xlabel("Generation")
            ax.set_ylabel("")
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Hide any unused subplot panels
        for ax in axes_flat[n:]:
            ax.set_visible(False)

        fig.suptitle(dataset_id, y=1.01)
        plt.tight_layout()
        path = output_dir / f"{dataset_id}_timecourse.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(out_dir: str, output_dir: str):
    out_dir = Path(out_dir).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = out_dir.name
    out_base = str(out_dir.parent)

    print(f"Experiment: {experiment_id}")
    print(f"Output dir: {out_dir}")

    # 1. Build variant → parca map; also read generations from workflow config
    parca_map = build_parca_map(out_dir)
    with open(out_dir / "nextflow" / "workflow_config.json") as f:
        wf_config = json.load(f)
    generations = wf_config.get("generations", 1)

    print(f"Found {len(parca_map)} parca run(s), {generations} generation(s):")
    for pid, info in parca_map.items():
        print(
            f"  parca_{pid}: dataset={info['dataset_id']!r}, "
            f"variants [{info['variant_start']}, {info['variant_end']})"
        )

    # 2. Load Nextflow trace for parca status
    trace = load_trace(experiment_id)
    if not trace:
        print(
            f"Warning: No trace CSV found for experiment '{experiment_id}' "
            f"in {REPO_ROOT} — parca status will be 'unknown'."
        )

    # 3. Load per-cell sim stats
    history_sql, _, success_sql = dataset_sql(out_base, [experiment_id])

    dt_all = pd.DataFrame()
    prot_all = pd.DataFrame()
    try:
        dt_all = load_doubling_times(history_sql, success_sql, experiment_id)
        print(f"Loaded doubling times for {len(dt_all)} cells.")
    except Exception as exc:
        print(f"Warning: Could not load doubling times: {exc}")
    try:
        prot_all = load_protein_mass_per_cell(history_sql, success_sql, experiment_id)
        print(f"Loaded protein mass for {len(prot_all)} cells.")
    except Exception as exc:
        print(f"Warning: Could not load protein mass: {exc}")

    # Per-generation success counts (grouped by variant + generation)
    if not dt_all.empty:
        gen_counts = (
            dt_all.groupby(["variant", "generation"]).size().reset_index(name="count")
        )
    else:
        gen_counts = pd.DataFrame(columns=["variant", "generation", "count"])

    # Sim error messages from trace
    sim_errors = load_sim_errors(experiment_id, parca_map)

    # 3b. Load higher-order properties if available
    higher_order_long = load_higher_order_properties(out_dir, parca_map)
    if not higher_order_long.empty:
        metrics_found = higher_order_long["metric"].unique().tolist()
        print(f"Loaded higher-order properties: {metrics_found}")
    else:
        print(
            "No higher_order_properties.tsv files found (run cd1_higher_order_properties analysis to include cell mass/volume)."
        )

    # 4. Assemble summary rows
    rows = []
    for parca_id, info in parca_map.items():
        trace_row = trace.get(parca_id, {})
        status = trace_row.get("status", "unknown")
        duration_min = trace_row.get("duration_min")
        workdir = trace_row.get("workdir", "")

        v_start, v_end = info["variant_start"], info["variant_end"]

        # Per-generation counts for this parca's variant range
        per_gen = {}
        for gen in range(1, generations + 1):
            if not gen_counts.empty:
                mask = gen_counts["variant"].between(v_start, v_end - 1) & (
                    gen_counts["generation"] == gen
                )
                per_gen[f"n_gen{gen}"] = int(gen_counts.loc[mask, "count"].sum())
            else:
                per_gen[f"n_gen{gen}"] = 0

        parca_error = ""
        if status in ("FAILED", "IGNORED"):
            parca_error = _get_parca_error(workdir)

        rows.append(
            {
                "parca_id": parca_id,
                "dataset_id": info["dataset_id"],
                "parca_status": status,
                "parca_error": parca_error,
                "parca_duration_min": duration_min,
                "n_genes_filled_from_ref": _get_genes_filled_from_ref(workdir),
                **per_gen,
                "sim_errors": sim_errors.get(parca_id, ""),
            }
        )

    summary_df = pd.DataFrame(rows)

    # 5. Write CSV
    csv_path = output_dir / "parca_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    print(summary_df.to_string(index=False))

    # 6. Plots
    if not dt_all.empty or not prot_all.empty or not higher_order_long.empty:
        _plot_cell_distributions(
            dt_all, prot_all, higher_order_long, parca_map, output_dir
        )
        _plot_cell_distributions(
            dt_all, prot_all, higher_order_long, parca_map, output_dir, min_generation=3
        )
        _plot_timecourse(dt_all, prot_all, higher_order_long, parca_map, output_dir)
    else:
        print("No simulation data found; skipping cell distribution plots.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a summary report for a vEcoli multi-parca workflow run."
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Path to the experiment output directory (e.g. out/my_experiment_20240101)",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        help="Directory to write report files (CSV + plots). "
        "Defaults to {out_dir}/summary_report.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or os.path.join(args.out_dir, "summary_report")
    run(args.out_dir, output_dir)


if __name__ == "__main__":
    main()
