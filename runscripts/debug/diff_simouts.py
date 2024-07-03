import argparse
import duckdb

from ecoli.library.parquet_emitter import get_dataset_sql

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare simulation output in two ouput directories")
    parser.add_argument('path', metavar='PATH', nargs=2,
        help="The two output directories to compare.")

    args = parser.parse_args()
    path1, path2 = args.path

    h_1, c_1 = get_dataset_sql(path1)
    h_2, c_2 = get_dataset_sql(path2)
    id_cols = "experiment_id, variant, lineage_seed, generation, agent_id, time"
    ordered_sql = f"SELECT * FROM ({{sql_query}}) ORDER BY {id_cols}"
    for sql_1, sql_2 in [(h_1, h_2), (c_1, c_2)]:
        data_1 = duckdb.sql(ordered_sql.format(sql_query=sql_1)).arrow()
        data_2 = duckdb.sql(ordered_sql.format(sql_query=sql_2)).arrow()
        assert data_1.column_names == data_2.column_names, "Different columns."
        for i, (col_1, col_2) in enumerate(zip(data_1, data_2)):
            assert col_1 == col_2, f"{data_1.column_names[i]} is different."
