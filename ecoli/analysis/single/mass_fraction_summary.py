import os
from typing import Any

import duckdb
import hvplot.polars

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
    configuration: duckdb.DuckDBPyRelation,
    history: duckdb.DuckDBPyRelation,
    sim_data_paths: list[str],
    validation_data_paths: list[str],
    outdir: str,
):
    assert (
        duckdb.sql("""
            SELECT count(time) as num_cells
            FROM configuration
            GROUP BY experiment_id, variant, generation,
                lineage_seed, agent_id
        """).fetchnumpy()['num_cells'][0]
        == 1
    ), "Mass fraction summary plot requires single-cell data."

    duckdb.sql("""
        CREATE TABLE all_mass_data AS SELECT
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
        """)

    fractions = duckdb.sql("""
        SELECT
            avg(Protein / Dry) AS Protein,
            avg(tRNA / Dry) AS tRNA,
            avg(rRNA / Dry) AS rRNA,
            avg(mRNA / Dry) AS mRNA,
            avg(DNA / Dry) AS DNA,
            avg(smallMol / Dry) AS "Small Mol.s",
        FROM all_mass_data
        """).fetchnumpy()
    mass_data = duckdb.sql(f"""
        WITH firsts AS (SELECT first(COLUMNS(*) ORDER BY "Time (min)") FROM all_mass_data)
        SELECT
            all_mass_data."Time (min)",
            all_mass_data.Protein / firsts.Protein AS "Protein ({fractions["Protein"][0]})",
            all_mass_data.tRNA / firsts.tRNA AS "tRNA ({fractions["tRNA"][0]})",
            all_mass_data.rRNA / firsts.rRNA AS "rRNA ({fractions["rRNA"][0]})",
            all_mass_data.mRNA / firsts.mRNA AS "mRNA ({fractions["mRNA"][0]})",
            all_mass_data.DNA / firsts.DNA AS "DNA ({fractions["DNA"][0]})",
            all_mass_data.smallMol / firsts.smallMol AS "Small Mol.s ({fractions["Small Mol.s"][0]})",
        FROM all_mass_data, firsts
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
