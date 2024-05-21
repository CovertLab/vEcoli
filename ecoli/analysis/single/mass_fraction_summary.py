from typing import Any

import polars as pl
import hvplot.polars

hvplot.extension('matplotlib')

COLORS_256 = [  # From colorbrewer2.org, qualitative 8-class set 1
    [228, 26, 28],
    [55, 126, 184],
    [77, 175, 74],
    [152, 78, 163],
    [255, 127, 0],
    [255, 255, 51],
    [166, 86, 40],
    [247, 129, 191]
]

COLORS = [
    '#%02x%02x%02x' % (color[0], color[1], color[2])
    for color in COLORS_256
]

def plot(
    params: dict[str, Any],
    config_lf: pl.LazyFrame,
    history_lf: pl.LazyFrame,
    sim_data_paths: list[str],
    validation_data_paths: list[str]
):
    assert config_lf.collect(streaming=True).n_unique(subset=[
        'experiment_id', 'variant', 'generation', 'lineage_seed', 'agent_id']
        ) == 1, "Mass fraction summary plot requires single-cell data."
    
    mass_columns = {
        'Protein': 'listeners__mass__protein_mass',
        'tRNA': 'listeners__mass__tRna_mass',
        'rRNA': 'listeners__mass__rRna_mass',
        'mRNA': 'listeners__mass__mRna_mass',
        'DNA': 'listeners__mass__dna_mass',
        'Small Mol.s': 'listeners__mass__smallMolecule_mass',
    }
    other_columns = {
        'Time (min)': (pl.col('time') - pl.col('time').min()) / 60,
        'Total dry mass': 'listeners__mass__dry_mass'
    }
    
    mass_data = history_lf.select(**other_columns, **mass_columns
        ).sort('Time (min)').collect(streaming=True)
    fractions = (mass_data[list(mass_columns.keys())] /
                 mass_data['Total dry mass']).mean()
    mass_data = mass_data.select('Time (min)',
        mass_data['Total dry mass'] / mass_data[0, 'Total dry mass'],
        *(mass_data[col] / mass_data[0, col] for col in mass_columns)
    )

    mass_data = mass_data.rename({label: '{} ({:.3f})'.format(
        label, fractions[0, label]) for label in mass_columns})
    plot_namespace = mass_data.plot
    # hvplot.output(backend='matplotlib')
    plotted_data = plot_namespace.line(
        x='Time (min)',
        ylabel='Mass (normalized by t = 0 min)',
        title='Biomass components (average fraction of total dry mass in parentheses)',
        color=COLORS)
    hvplot.save(plotted_data, 'mass_fraction_summary.html')
    # hvplot.save(plotted_data, 'mass_fraction_summary.png', dpi=300)
