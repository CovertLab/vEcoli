"""
Labkey Data Transformation
"""


from typing import Any

from duckdb import DuckDBPyConnection
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from ecoli.library.parquet_emitter import read_stacked_columns, field_metadata
from ecoli.library.sim_data import LoadSimData

from wholecell.utils import units
from wholecell.analysis.analysis_tools import exportFigure
from wholecell.utils.voronoi_plot_main import VoronoiMaster


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
    voronoi_columns = [
        "listeners__mass__rna_mass",
        "listeners__mass__protein_mass",
        "listeners__mass__tRna_mass",
        "listeners__mass__rRna_mass",
        "listeners__mass__mRna_mass",
        "listeners__mass__dna_mass",
        "listeners__mass__smallMolecule_mass",
        "bulk",
    ]

    voronoi_data = pl.DataFrame(
        read_stacked_columns(history_sql, voronoi_columns, conn=conn)
    )

    bulk_molecule_counts = np.stack(voronoi_data["bulk"])

    bulk_molecule_ids = field_metadata(conn, config_sql, "bulk")

    bulk_molecule_idx = {name: idx for idx, name in enumerate(bulk_molecule_ids)}

    exp_id = list(sim_data_paths.keys())[0]

    sim_data_path = list(sim_data_paths[exp_id].values())[0]

    sim_data = LoadSimData(sim_data_path).sim_data

    nAvogadro = sim_data.constants.n_avogadro

    def find_mass_molecule_group(group_id):
        temp_ids = getattr(sim_data.molecule_groups, str(group_id))
        temp_indexes = np.array([bulk_molecule_idx[temp] for temp in temp_ids])
        temp_counts = bulk_molecule_counts[:, temp_indexes]
        temp_mw = sim_data.getter.get_masses(temp_ids)
        return (units.dot(temp_counts, temp_mw) / nAvogadro).asNumber(units.fg)

    def find_mass_single_molecule(molecule_id):
        temp_id = getattr(sim_data.molecule_ids, str(molecule_id))
        temp_index = bulk_molecule_idx[temp_id]
        temp_counts = bulk_molecule_counts[:, temp_index]
        temp_mw = sim_data.getter.get_mass(temp_id)
        return (units.multiply(temp_counts, temp_mw) / nAvogadro).asNumber(units.fg)

    lipid = find_mass_molecule_group("lipids")
    polyamines = find_mass_molecule_group("polyamines")
    lps = find_mass_single_molecule("LPS")
    murein = find_mass_single_molecule("murein")
    glycogen = find_mass_single_molecule("glycogen")

    protein = voronoi_data["listeners__mass__protein_mass"]
    rna = voronoi_data["listeners__mass__rna_mass"]
    tRna = voronoi_data["listeners__mass__tRna_mass"]
    rRna = voronoi_data["listeners__mass__rRna_mass"]
    mRna = voronoi_data["listeners__mass__mRna_mass"]
    miscRna = rna - (tRna + rRna + mRna)
    dna = voronoi_data["listeners__mass__dna_mass"]
    smallMolecules = voronoi_data["listeners__mass__smallMolecule_mass"]
    metabolites = smallMolecules - (lipid + lps + murein + polyamines + glycogen)

    dic_initial = {
        "nucleic_acid": {
            "DNA": dna[0],
            "mRNA": mRna[0],
            "miscRNA": miscRna[0],
            "rRNA": rRna[0],
            "tRNA": tRna[0],
        },
        "metabolites": {
            "LPS": lps[0],
            "glycogen": glycogen[0],
            "lipid": lipid[0],
            "metabolites": metabolites[0],
            "peptidoglycan": murein[0],
            "polyamines": polyamines[0],
        },
        "protein": protein[0],
    }
    dic_final = {
        "nucleic_acid": {
            "DNA": dna[-1],
            "mRNA": mRna[-1],
            "miscRNA": miscRna[-1],
            "rRNA": rRna[-1],
            "tRNA": tRna[-1],
        },
        "metabolites": {
            "LPS": lps[-1],
            "glycogen": glycogen[-1],
            "lipid": lipid[-1],
            "metabolites": metabolites[-1],
            "peptidoglycan": murein[-1],
            "polyamines": polyamines[-1],
        },
        "protein": protein[-1],
    }
    vm = VoronoiMaster()
    vm.plot(
        [[dic_initial, dic_final]],
        title=[["Initial biomass components", "Final biomass components"]],
        ax_shape=(1, 2),
        chained=True,
    )

    plotOutFileName = "mass_fractions_voronoi"

    exportFigure(plt, outdir, plotOutFileName, extension=".png")
