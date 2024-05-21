"""
Generates a table of genes that are subgenerationally expressed, with their
expression frequencies and average/maximum mRNA/protein counts.
"""

import pickle
import os
from typing import Any

# noinspection PyUnresolvedReferences
import numpy as np
import polars as pl

from ecoli.analysis.template import get_field_metadata, named_idx


IGNORE_FIRST_N_GENS = 8


def plot(
    params: dict[str, Any],
    config_lf: pl.LazyFrame,
    history_lf: pl.LazyFrame,
    sim_data_path: list[str],
    validation_data_path: list[str],
    outdir: str
):
    with open(sim_data_path[0], 'rb') as f:
        sim_data = pickle.load(f)

    # Ignore first N generations
    history_lf = history_lf.filter(pl.col('generation') >= IGNORE_FIRST_N_GENS)
    config_lf = config_lf.filter(pl.col('generation') >= IGNORE_FIRST_N_GENS)

    if config_lf.select('time').count().collect(streaming=True)['time'][0] == 0:
        print('Skipping analysis - not enough generations run.')
        return

    # Get list of cistron IDs from sim_data
    cistron_data = sim_data.process.transcription.cistron_data
    cistron_ids = cistron_data['id']

    # Filter list for cistron IDs with associated protein ids
    cistron_id_to_protein_id = {
        protein['cistron_id']: protein['id']
        for protein in sim_data.process.translation.monomer_data
        }
    mRNA_cistron_ids = [
        cistron_id for cistron_id in cistron_ids
        if cistron_id in cistron_id_to_protein_id]

    # Get IDs of associated monomers and genes
    monomer_ids = [
        cistron_id_to_protein_id.get(cistron_id, None)
        for cistron_id in mRNA_cistron_ids]
    cistron_id_to_gene_id = {
        cistron['id']: cistron['gene_id'] for cistron in cistron_data
        }
    gene_ids = [
        cistron_id_to_gene_id[cistron_id]
        for cistron_id in mRNA_cistron_ids]

    # Get subcolumn for mRNA cistron IDs in RNA counts table
    mRNA_cistron_ids_rna_counts_table = get_field_metadata(config_lf,
        'listeners__rna_counts__mRNA_cistron_counts')

    # Get indexes of mRNA cistrons in this subcolumn
    mRNA_cistron_id_to_index = {
        cistron_id: i for (i, cistron_id)
        in enumerate(mRNA_cistron_ids_rna_counts_table)
        }
    mRNA_cistron_indexes = np.array([
        mRNA_cistron_id_to_index[cistron_id] for cistron_id
        in mRNA_cistron_ids
        ])

    # Get boolean matrix for whether each gene's mRNA exists in each
    # generation or not
    mRNA_exists_in_gen = history_lf.select(**{
        'lineage_seed': 'lineage_seed',
        'generation': 'generation',
        'agent_id': 'agent_id',
        **named_idx('listeners__rna_counts__mRNA_cistron_counts',
                    mRNA_cistron_ids, mRNA_cistron_indexes)
    }).groupby(['lineage_seed', 'generation', 'agent_id']
    ).agg(pl.all().sum() > 0).drop(['lineage_seed', 'generation', 'agent_id']
    ).collect(streaming=True)

    # Divide by total number of cells to get probability
    p_mRNA_exists_in_gen = (
        mRNA_exists_in_gen.sum() / mRNA_exists_in_gen.shape[0])

    # Get maximum counts of mRNAs for each gene across all timepoints
    max_mRNA_counts = history_lf.select(**named_idx(
        'listeners__rna_counts__mRNA_cistron_counts',
        mRNA_cistron_ids, mRNA_cistron_indexes)).max().collect(streaming=True)

    # Get subcolumn for monomer IDs in monomer counts table
    monomer_ids_monomer_counts_table = get_field_metadata(
        config_lf, 'listeners__monomer_counts')

    # Get indexes of monomers in this subcolumn
    monomer_id_to_index = {
        monomer_id: i for (i, monomer_id)
        in enumerate(monomer_ids_monomer_counts_table)
        }
    monomer_indexes = np.array([
        monomer_id_to_index[monomer_id] for monomer_id in monomer_ids
        ])

    # Get maximum counts of monomers for each gene across all timepoints
    max_monomer_counts = history_lf.select(**named_idx(
        'listeners__monomer_counts',
        monomer_ids, monomer_indexes)).max().collect(streaming=True)

    # Write data to table
    out_df = pl.DataFrame({
        'gene_name': gene_ids,
        'cistron_name': mRNA_cistron_ids,
        'protein_name': [i[:-3] for i in monomer_ids],
        'p_expressed': p_mRNA_exists_in_gen.transpose(),
        'max_mRNA_count': max_mRNA_counts.transpose(),
        'max_monomer_counts': max_monomer_counts.transpose(),
    }).filter((0 < pl.col('p_expressed')) & (pl.col('p_expressed') < 1))
    out_df.write_csv(os.path.join(outdir, 'subgen.tsv'), separator='\t')
