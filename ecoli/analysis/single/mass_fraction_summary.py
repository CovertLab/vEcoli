import os
from typing import Any

from duckdb import DuckDBPyConnection
import hvplot.polars

from ecoli.analysis.template import num_cells

hvplot.extension("matplotlib")

COLORS_256 = [  # From colorbrewer2.org, qualitative 8-class set 1
    [228, 26, 28],
    [55, 126, 184],
    [77, 175, 74],
    [152, 78, 163],
    [255, 127, 0],
    [255, 255, 51],
    [166, 86, 40],
    [247, 129, 191],
]

COLORS = ["#%02x%02x%02x" % (color[0], color[1], color[2]) for color in COLORS_256]


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    sim_data_paths: list[str],
    validation_data_paths: list[str],
    outdir: str,
):
    assert num_cells(conn, "configuration"
        ) == 1, "Mass fraction summary plot requires single-cell data."

    conn.register("all_mass_data", conn.sql("""
        SELECT
            (time - MIN(time) OVER ()) / 60 AS "Time (min)",
            listeners__mass__protein_mass AS Protein,
            listeners__mass__tRna_mass AS tRNA,
            listeners__mass__rRna_mass AS rRNA,
            listeners__mass__mRna_mass AS mRNA,
            listeners__mass__dna_mass AS DNA,
            listeners__mass__smallMolecule_mass AS smallMol,
            listeners__mass__dry_mass AS Dry
        FROM history
        ORDER BY "Time (min)"
        """))

    fractions = conn.sql("""
        SELECT
            avg(Protein / Dry) AS Protein,
            avg(tRNA / Dry) AS tRNA,
            avg(rRNA / Dry) AS rRNA,
            avg(mRNA / Dry) AS mRNA,
            avg(DNA / Dry) AS DNA,
            avg(smallMol / Dry) AS "Small Mol.s",
        FROM all_mass_data
        """).fetchnumpy()
    mass_data = conn.sql(f"""
        SELECT
            "Time (min)",
            Protein / first(Protein) OVER all_times AS "Protein ({fractions["Protein"][0]:.3f})",
            tRNA / first(tRNA) OVER all_times AS "tRNA ({fractions["tRNA"][0]:.3f})",
            rRNA / first(rRNA) OVER all_times AS "rRNA ({fractions["rRNA"][0]:.3f})",
            mRNA / first(mRNA) OVER all_times AS "mRNA ({fractions["mRNA"][0]:.3f})",
            DNA / first(DNA) OVER all_times AS "DNA ({fractions["DNA"][0]:.3f})",
            smallMol / first(smallMol) OVER all_times AS "Small Mol.s ({fractions["Small Mol.s"][0]:.3f})",
        FROM all_mass_data
        WINDOW all_times AS (ORDER BY "Time (min)")
        ORDER BY "Time (min)"
        """).pl()
    plot_namespace = mass_data.plot
    # hvplot.output(backend='matplotlib')
    plotted_data = plot_namespace.line(
        x="Time (min)",
        ylabel="Mass (normalized by t = 0 min)",
        title="Biomass components (average fraction of total dry mass in parentheses)",
        color=COLORS,
    )
    hvplot.save(plotted_data, os.path.join(outdir, "mass_fraction_summary.html"))
    # hvplot.save(plotted_data, 'mass_fraction_summary.png', dpi=300)
