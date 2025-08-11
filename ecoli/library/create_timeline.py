import numpy as np
import pandas as pd
from vivarium.core.emitter import data_from_database, get_experiment_database


def get_csv_from_database(
    experiment_id,
    query_to_colname_mapping,
    outfile=None,
    port=27017,
    database_name="simulations",
):
    db = get_experiment_database(port, database_name)
    data, _ = data_from_database(
        experiment_id, db, list(query_to_colname_mapping.keys())
    )

    colname_to_query_mapping = {v: k for k, v in query_to_colname_mapping.items()}

    def query_dict(dictionary, query):
        for elem in query:
            dictionary = dictionary[elem]
        return dictionary

    df = pd.DataFrame(columns=["Time", *query_to_colname_mapping.values()])
    for time, values in data.items():
        df = df.append(
            {
                "Time": time,
                **{
                    col: query_dict(values, colname_to_query_mapping[col])
                    for col in df.columns
                    if col != "Time"
                },
            },
            ignore_index=True,
        )

    return df.to_csv(outfile, index=False)


def create_timeline_from_csv(filepath, column_path_mapping, time_column="Time"):
    """ """
    df = pd.read_csv(filepath, skipinitialspace=True)
    return create_timeline_from_df(df, column_path_mapping, time_column="Time")


def create_timeline_from_df(df, column_path_mapping, time_column="Time"):
    """ """
    time = df[time_column].to_numpy()
    output_cols = []
    output_paths = []
    for column, path in column_path_mapping.items():
        output_cols.append(df[column].to_numpy())
        output_paths.append(path)
    output_cols = np.array(output_cols)

    result = {
        "timeline": [
            (t, {path: output_cols[p, i] for p, path in enumerate(output_paths)})
            for i, t in enumerate(time)
        ]
    }
    return result


def create_bulk_timeline_from_df(df, column_path_mapping, time_column="Time"):
    """ """
    time = df[time_column].to_numpy()
    output_cols = []
    output_paths = []
    for column, path in column_path_mapping.items():
        output_cols.append(df[column].to_numpy())
        output_paths.append(path)
    output_cols = np.array(output_cols)

    result = {
        "timeline": {
            t: {path: output_cols[p, i] for p, path in enumerate(output_paths)}
            for i, t in enumerate(time)
        }
    }
    return result


def add_computed_value(timeline, func):
    """func: (time, values) => dict of calculated values"""
    return {
        "timeline": [
            (time, {**values, **func(time, values)})
            for time, values in timeline["timeline"]
        ]
    }


def add_computed_value_bulk(timeline, func):
    """func: (time, values) => dict of calculated values"""
    return {
        "timeline": {
            time: {**values, **func(time, values)}
            for time, values in timeline["timeline"].items()
        }
    }


def test_add_timeline():
    TEST_FILE = "data/cell_wall/cell_wall_test_rig_17_09_2022_00_41_51.csv"

    timeline = create_timeline_from_csv(
        TEST_FILE,
        {
            "CPD-12261[p]": ("bulk", "CPD-12261[p]"),
            "CPLX0-7717[i]": ("bulk", "CPLX0-7717[i]"),
            "CPLX0-3951[i]": ("bulk", "CPLX0-3951[i]"),
        },
    )

    timeline = add_computed_value(
        timeline, lambda t, val: {"murein*4": val[("bulk", "CPD-12261[p]")] * 4}
    )

    print(timeline)


def main():
    test_add_timeline()


if __name__ == "__main__":
    main()
