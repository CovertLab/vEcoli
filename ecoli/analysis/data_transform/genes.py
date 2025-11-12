"""
Gene Data Relabeling/Aggregation Transformation (Eco/BioCyc)
"""


from pathlib import Path
from typing import Any

from duckdb import DuckDBPyConnection
import polars as pl

from ecoli.library.transforms import REPO_ROOT
from ecoli.library.transforms.data_transformer_biocyc import DataTransformerGenes
from ecoli.library.transforms.models import SimulationConfigData


def partition_log(experiment_id: str, variant: int, seed: int, generation: int, agent_id: str) -> None:
    print(f'# ======================================================================================= #\n>> Experiment_id: {experiment_id}, Variant: {variant}, Seed: {seed}, Generation: {generation}, AgentID: {agent_id}')


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
) -> None:
    # get config params
    config_df = SimulationConfigData(query=config_sql)
    experiment_id, variant, seed, generation, agent_id, sim_outdir = list(map(
        lambda column: config_df.get(column),
        ["experiment_id", "variant", "lineage_seed", "generation", "agent_id", "emitter_arg__out_dir"]
    ))

    # log (sanity check)
    partition_log(experiment_id, variant, seed, generation, agent_id)

    # set up transformer
    simdata_path = Path(sim_data_paths[experiment_id][variant])
    transformer = DataTransformerGenes(sim_data_path=simdata_path)

    # generate genes_df
    genes_df: pl.LazyFrame = transformer.transform(
        experiment_id=experiment_id,
        simulation_outdir=sim_outdir,
        observable_ids=params['observable_ids'],
        variant=variant,
        seed=seed,
        generation=generation,
        agent_id=agent_id,
        history_sql=history_sql,
        lazy=True
    )
    print(f'GENES DF:\n{genes_df.collect_schema()}')

    def record_iter():
        fp = REPO_ROOT / "iter.txt"
        with open(fp, 'r') as f:
            i = int(f.read())

        i += 1
        with open(fp, 'w') as f:
            f.write(str(i))

    record_iter()

    # export lazyframe to parquet
    filename = f"genes_{experiment_id}"
    transformer.export()
    transformer.export_parquet(genes_df, outdir, filename)

