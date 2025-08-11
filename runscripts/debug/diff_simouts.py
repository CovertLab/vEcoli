import argparse
import duckdb
import numpy as np

from ecoli.library.parquet_emitter import dataset_sql, ndlist_to_ndarray

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare simulation output of two experiment IDs."
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory containing data for two experiment IDs.",
    )
    parser.add_argument(
        "exp_ids", metavar="EXP_ID", nargs=2, help="The two experiment IDs to compare."
    )

    args = parser.parse_args()
    exp_id_1, exp_id_2 = args.exp_ids

    history_sql, _, _ = dataset_sql(args.output, list(args.exp_ids))
    id_cols = "experiment_id, variant, lineage_seed, generation, agent_id, time"
    ordered_sql = f"SELECT * FROM ({{sql_query}}) WHERE experiment_id = '{{exp_id}}' ORDER BY {id_cols}"
    data_1 = duckdb.sql(
        ordered_sql.format(sql_query=history_sql, exp_id=exp_id_1)
    ).arrow()
    data_2 = duckdb.sql(
        ordered_sql.format(sql_query=history_sql, exp_id=exp_id_2)
    ).arrow()
    assert data_1.column_names == data_2.column_names, "Different columns."
    for i, (col_1, col_2) in enumerate(zip(data_1, data_2)):
        if col_1 != col_2 and data_1.column_names[i] not in [
            "experiment_id",
            "agent_id",
        ]:
            np.testing.assert_allclose(
                ndlist_to_ndarray(col_1),
                ndlist_to_ndarray(col_2),
                atol=1e-12,
                err_msg=f"{data_1.column_names[i]} not equal",
            )
