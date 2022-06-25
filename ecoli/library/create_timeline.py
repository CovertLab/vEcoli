import numpy as np
import pandas as pd


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


def test_add_timeline():
    TEST_FILE = "data/cell_wall/test_murein_21_06_2022_17_42_11.csv"
    timeline = create_timeline_from_csv(
        TEST_FILE,
        {
            "CPD-12261[p]": ("bulk", "CPD-12261[p]"),
            "CPLX0-7717[m]": ("bulk", "CPLX0-7717[m]"),
            "CPLX0-3951[i]": ("bulk", "CPLX0-3951[i]"),
        },
    )

    print(timeline)


def main():
    test_add_timeline()


if __name__ == "__main__":
    main()
