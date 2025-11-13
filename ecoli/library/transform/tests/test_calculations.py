# -- tests -- #

import itertools
import math
import subprocess
from pathlib import Path

import pandas as pd
import polars as pl
import numpy as np

from ecoli.library.transform.data_transformer import load_outputs, downsample


def test_run() -> None:
    subprocess.run("rm -rf /Users/alexanderpatrie/sms/vecoli_repo/out/multiseed_multigen_analysis && uv run --env-file .env runscripts/analysis.py --config configs/data_transformation.json".split(" "))


def test_downsample(sim_data_path: Path) -> None:
    def downsample_eager(df_long: pl.DataFrame) -> pl.DataFrame:
        tp_all = np.unique(df_long["time"].to_numpy()).astype(int)
        ds_ratio = int(math.ceil(len(df_long) / 20000))
        tp_ds = list(itertools.islice(tp_all, 0, tp_all.max(), ds_ratio))
        return df_long.filter(pl.col("time").is_in(tp_ds))

    def downsample_pd(df_long: pd.DataFrame) -> pd.DataFrame:
        tp_all = np.unique(df_long["time"]).astype(int)
        ds_ratio = int(np.ceil(np.shape(df_long)[0] / 20000))
        tp_ds = list(itertools.islice(tp_all, 0, max(tp_all), ds_ratio))
        df_ds = df_long[np.isin(df_long["time"], tp_ds)]
        return df_ds

    pddf = pd.DataFrame({
        "time": np.arange(0, 100),
        "bulk": np.random.rand(100),
    })

    pldf = pl.from_pandas(pddf)
    lf = pldf.lazy()

    eager_out = downsample_eager(pldf)
    lazy_out = downsample(lf).collect()
    pd_out = downsample_pd(pddf)

    assert eager_out.equals(lazy_out)
    assert pd_out.equals(lazy_out.to_pandas())