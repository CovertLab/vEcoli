import polars as pl

from ecoli.library.transform.models import DataTransformation
from ecoli.library.transform.tests.fixtures import lazyframe_fixture


def test_implement_transformer() -> None:
    def mean(x: pl.DataFrame):
        new_data = {}
        for i, col in enumerate(x.iter_columns()):
            col_avg = col.sum() / len(col)
            new_data[x.columns[i]] = col_avg
        return pl.DataFrame(new_data)

    x = lazyframe_fixture(100_000).collect()
    y = mean(x)
    transform = DataTransformation(x=x, y=y, function=mean)
    for i, row in enumerate(y.iter_rows()):
        assert row == list(transform.y.iter_rows())[i]
    assert transform.cardinality == (1e-05, 1.0)
    assert transform.f == {'id': 'mean', 'source': 'ecoli.library.transform.tests'}