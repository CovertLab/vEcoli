# -- tests -- #

import polars as pl
import numpy as np


def lazyframe_fixture(upper: int = 1111) -> pl.LazyFrame:
    def fake_data(upper: int = 1111) -> dict[str, list[float]]:
        return {
            "time": np.arange(0, upper).tolist(),
            "x": list(map(lambda i: (i ** 0.3) / (2.2 ** 11.11), list(range(upper)))),
            "y": list(map(lambda i: (-i ** 0.3 ** 0.2) / (2.2 ** 11.11), list(range(upper)))),
            "z": list(map(lambda i: (i ** 0.3) / (2.2 ** 11.11 ** 2 / 3), list(range(upper))))
        }
    return pl.LazyFrame(fake_data(upper))

