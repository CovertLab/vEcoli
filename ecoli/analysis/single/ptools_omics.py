import os
from typing import Any

from duckdb import DuckDBPyConnection
import numpy as np

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
    # with open(os.path.join(outdir, "history_sql.txt"), "w") as f:
    #     f.write(history_sql)

    query_sql = f"""
        SELECT listeners__rna_counts__full_mRNA_counts, time FROM ({history_sql})
        ORDER BY time
    """

    output = conn.sql(query_sql).df()
    mrna_mtx = np.stack(
        output["listeners__rna_counts__full_mRNA_counts"].values
    ).astype(int)
    np.savetxt(os.path.join(outdir, "mrna_mtx.txt"), mrna_mtx, delimiter="\t", fmt="%d")
