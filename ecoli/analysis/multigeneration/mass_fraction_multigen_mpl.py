"""
mass_fraction_multigen_mpl
==========================
Multigeneration mass breakdown for a single experiment. This plot creates one
PNG per lineage seed; with each seed's generations are concatenated into one continuous
time trace (minutes from the lineage's own t=0).

Layout:
  1. FOLD CHANGE overview -- every submass (+ total dry mass) divided by its own
     value at the lineage's first timepoint, exactly like the single-cell
     ``mass_fraction_summary`` normalization, but carried across generations. Because
     mass roughly doubles then halves at each division, these curves are sawtoothed;
     the generation lines mark the divisions. Each legend label carries that type's
     mean fraction of dry mass (the number ``mass_fraction_summary`` shows in
     parentheses).
  2. MASS FRACTION overview -- every submass divided by dry mass AT EACH TIMESTEP
     (true composition, ~0-0.55, summing to ~1). This is the composition view the
     fold-change panel can't show.
  3..N. One panel PER submass, DUAL AXIS:
        * left y  -- that type's ACTUAL mass in fg (solid line),
        * right y -- that type's mass FRACTION of dry mass (dash-dot line, same hue).
     So each type's absolute growth and its share of the cell are read together.

Generation (division) boundaries are drawn as vertical dotted lines on every panel.

Assumes ``single_daughters: true`` (the default): each lineage is one cell per
generation, so sorting a seed's rows by time gives a single continuous trace. With
both daughters kept, sibling cells would be interleaved into one line.

Submasses come straight from the MassListener emitted columns
(``ecoli/processes/listeners/mass_listener.py``): protein / rRna / tRna / mRna / dna
/ smallMolecule / dry, all in fg.
"""

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import read_stacked_columns

# Emitted MassListener columns (all fg). Order sets legend / panel order.
MASS_COLUMNS = {
    "Protein": "listeners__mass__protein_mass",
    "rRNA": "listeners__mass__rRna_mass",
    "tRNA": "listeners__mass__tRna_mass",
    "mRNA": "listeners__mass__mRna_mass",
    "DNA": "listeners__mass__dna_mass",
    "Small Mol": "listeners__mass__smallMolecule_mass",
}
DRY_COLUMN = "listeners__mass__dry_mass"

# One stable color per submass (colorbrewer2 qualitative 8-class Set1, skipping the
# hard-to-read yellow). Dry mass is black.
TYPE_COLORS = {
    "Protein": "#e41a1c",
    "rRNA": "#377eb8",
    "tRNA": "#4daf4a",
    "mRNA": "#984ea3",
    "DNA": "#ff7f00",
    "Small Mol": "#a65628",
}
DRY_COLOR = "#000000"


def _draw_generation_lines(ax, t, gen):
    """Vertical dotted line at each division (where the generation index changes)."""
    for k in np.where(np.diff(gen) != 0)[0]:
        ax.axvline(t[k + 1], color="0.6", lw=1, ls=":", alpha=0.7, zorder=0)


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
    all_cols = list(MASS_COLUMNS.values()) + [DRY_COLUMN]
    df = pl.DataFrame(read_stacked_columns(history_sql, all_cols, conn=conn))

    seeds = (
        sorted(df["lineage_seed"].unique().to_list())
        if "lineage_seed" in df.columns
        else [None]
    )

    for seed in seeds:
        sub = df if seed is None else df.filter(pl.col("lineage_seed") == seed)
        if sub.height == 0:
            continue
        sub = sub.sort("time")

        t = ((sub["time"] - sub["time"].min()) / 60).to_numpy()
        dry = sub[DRY_COLUMN].to_numpy().astype(float)
        gen = (
            sub["generation"].to_numpy()
            if "generation" in sub.columns
            else np.zeros_like(t)
        )
        exp_id = sub["experiment_id"][0] if "experiment_id" in sub.columns else "?"
        n_gens = int(np.unique(gen).size)

        # Per-type absolute mass (fg), fraction of dry mass, and their mean/std.
        absolute = {k: sub[v].to_numpy().astype(float) for k, v in MASS_COLUMNS.items()}
        fraction = {
            k: np.divide(m, dry, out=np.full_like(m, np.nan), where=dry != 0)
            for k, m in absolute.items()
        }
        mean_frac = {k: float(np.nanmean(f)) for k, f in fraction.items()}
        std_frac = {k: float(np.nanstd(f)) for k, f in fraction.items()}
        mean_mass = {k: float(np.nanmean(m)) for k, m in absolute.items()}
        std_mass = {k: float(np.nanstd(m)) for k, m in absolute.items()}

        n_types = len(MASS_COLUMNS)
        # 2 overview panels + one dual-axis panel per submass + a total dry-mass panel.
        fig, axes = plt.subplots(
            3 + n_types,
            1,
            figsize=(12, 4 + 2.4 * (n_types + 1)),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 2.2] + [1.4] * n_types + [1.4]},
        )
        ax_fold, ax_frac = axes[0], axes[1]
        type_axes = axes[2 : 2 + n_types]
        ax_dry = axes[-1]

        for ax in axes:
            _draw_generation_lines(ax, t, gen)

        # ---- panel 1: fold change vs lineage t=0 (incl. dry mass) ---------- #
        for k, m in absolute.items():
            m0 = m[0] if m.size and m[0] != 0 else np.nan
            ax_fold.plot(t, m / m0, color=TYPE_COLORS[k], lw=1.5, label=k)
        dry0 = dry[0] if dry.size and dry[0] != 0 else np.nan
        ax_fold.plot(
            t, dry / dry0, color=DRY_COLOR, lw=1.8, ls=":", label="Dry (total)"
        )
        ax_fold.set_ylabel("Mass / lineage t=0")
        ax_fold.set_title("Fold change vs lineage start", loc="left")
        ax_fold.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

        # ---- panel 2: literal mass fraction of dry mass -------------------- #
        for k in MASS_COLUMNS:
            ax_frac.plot(
                t,
                fraction[k],
                color=TYPE_COLORS[k],
                lw=1.5,
                label=f"{k} ({mean_frac[k]:.3f})",
            )
        ax_frac.set_ylabel("Fraction of dry mass")
        ax_frac.set_ylim(bottom=0)
        ax_frac.set_title(
            "Mass fraction (submass / dry mass, per timestep)", loc="left"
        )
        ax_frac.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

        # ---- panels 3..N: one dual-axis panel per submass ------------------ #
        for ax, k in zip(type_axes, MASS_COLUMNS):
            color = TYPE_COLORS[k]
            (l_abs,) = ax.plot(
                t, absolute[k], color=color, lw=1.6, label=f"{k} mass (fg)"
            )
            ax.set_ylabel(f"{k} mass (fg)", color=color, fontsize=9)
            ax.tick_params(axis="y", labelcolor=color, labelsize=8)
            # Autoscale (not pinned to 0) so the sawtooth sits centered in the panel.
            ax.margins(y=0.08)

            ax_r = ax.twinx()
            (l_frac,) = ax_r.plot(
                t,
                fraction[k],
                color=color,
                lw=1.2,
                ls="-.",
                alpha=0.7,
                label=f"{k} fraction",
            )
            ax_r.set_ylabel("fraction", color=color, fontsize=8)
            ax_r.tick_params(axis="y", labelcolor=color, labelsize=8)
            ax_r.margins(y=0.08)

            ax.set_title(
                f"{k}  (mass {mean_mass[k]:.1f}±{std_mass[k]:.1f} fg; "
                f"fraction {mean_frac[k]:.3f}±{std_frac[k]:.3f})",
                loc="left",
                fontsize=9,
            )
            ax.legend(
                handles=[l_abs, l_frac],
                loc="center left",
                bbox_to_anchor=(1.06, 0.5),
                fontsize=7,
            )

        # ---- bottom panel: total dry mass (absolute only, no fraction twin) --- #
        mean_dry, std_dry = float(np.nanmean(dry)), float(np.nanstd(dry))
        ax_dry.plot(t, dry, color=DRY_COLOR, lw=1.7, label="Dry (total) mass (fg)")
        ax_dry.set_ylabel("Dry mass (fg)", fontsize=9)
        ax_dry.margins(y=0.08)
        ax_dry.set_title(
            f"Dry (total)  (mass {mean_dry:.1f}±{std_dry:.1f} fg)",
            loc="left",
            fontsize=9,
        )
        ax_dry.legend(loc="center left", bbox_to_anchor=(1.06, 0.5), fontsize=7)
        ax_dry.set_xlabel("Time (min)")

        fig.suptitle(
            f"Mass fractions & absolute submasses\n{exp_id} · seed {seed} · "
            f"{n_gens} generation(s)",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        out = os.path.join(outdir, f"mass_fraction_multigen_seed{seed}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[mass_fraction_multigen_mpl] saved {out}")
