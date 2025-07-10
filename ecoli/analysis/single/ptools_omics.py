import os
from typing import Any

from duckdb import DuckDBPyConnection
import numpy as np
import pandas as pd

from ecoli.library.sim_data import LoadSimData


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
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_paths: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    with open(os.path.join(outdir, "history_sql.txt"), "w") as f:
        f.write(f"\nwd={os.getcwd()}\n")
        f.write(f"\nwd_top={os.getcwd().split('/out/')[0]}\n")
        f.write(f"\nsim_data_paths={sim_data_paths}\n")
        f.write(f"\nvariant_metadata={variant_metadata}\n")
        f.write(f"\nvariant_names={variant_names}\n")
        f.write(f"\nparams={params}\n")
        f.write(history_sql)

    exp_id = list(sim_data_paths.keys())[0]

    wd_top = os.getcwd().split("/out/")[0]

    wd_raw = os.path.join(wd_top, "reconstruction", "ecoli", "flat")

    sim_data_path = list(sim_data_paths[exp_id].values())[0]

    sim_data = LoadSimData(sim_data_path).sim_data

    rna_data = sim_data.process.transcription.rna_data

    mrna_ids = rna_data["id"][rna_data["is_mRNA"]]

    mrna_ids = [id[:-3] for id in mrna_ids]

    query_sql = f"""
        SELECT listeners__rna_counts__full_mRNA_counts, time FROM ({history_sql})
        ORDER BY time
    """

    output = conn.sql(query_sql).df()
    mrna_mtx = np.stack(
        output["listeners__rna_counts__full_mRNA_counts"].values
    ).astype(int)
    np.savetxt(os.path.join(outdir, "mrna_mtx.txt"), mrna_mtx, delimiter="\t", fmt="%d")

    tps = np.linspace(0, np.shape(mrna_mtx)[0], 6, dtype=int)

    # Sum over each block
    block_sums = [
        mrna_mtx[tps[i] : tps[i + 1]].sum(axis=0) for i in range(len(tps) - 1)
    ]

    # Stack into final result
    mrna_summed = np.stack(block_sums, axis=0)
    mrna_summed = np.insert(mrna_summed, 0, mrna_mtx[0], axis=0)

    tp_columns = ["t" + str(i) for i in range(len(tps))]

    tu_id_mapping = pd.read_csv(
        os.path.join(wd_raw, "transcription_units.tsv"), sep="\t", header=5, index_col=0
    )
    tu_id_mapping = tu_id_mapping["common_name"]

    mrna_names = []

    for i in range(len(mrna_ids)):
        try:
            mrna_name = tu_id_mapping[mrna_ids[i]]
            if isinstance(mrna_name, float):
                mrna_name = mrna_ids[i]
        except KeyError:
            mrna_name = mrna_ids[i]
        mrna_names.append(mrna_name)

    ptools_rna = pd.DataFrame(
        data=mrna_summed.transpose(), columns=tp_columns, index=mrna_names
    )
    ptools_rna.to_csv(
        os.path.join(outdir, "ptools_rna.txt"), sep="\t", index=True, header=True
    )
