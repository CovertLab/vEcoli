"""
CSV reader and writer that default to TAB delimiters.
"""

import csv
import _csv

from io import TextIOWrapper
from typing import (
    Any,
    cast,
    Collection,
    IO,
    Iterable,
    Optional,
    Sequence,
    Text,
    Type,
    Union,
)


DIALECT = Union[str, _csv.Dialect, Type[_csv.Dialect], csv.Dialect, Type[csv.Dialect]]


class reader(object):
    def __init__(
        self,
        csvfile: IO[bytes],
        dialect: DIALECT = "excel",
        delimiter: str = "\t",
        **fmtparams: Any,
    ):
        """Open a csv reader() defaulting to TAB delimiters.

        REQUIRES: `csvfile` must be a buffered byte reader, e.g. from
        io.open(filename, 'rb') or io.BytesIO(buffer).

        This does Unicode by constructing the csv.reader with a TextIO.
        """
        self.input_file = TextIOWrapper(csvfile, encoding="utf-8", newline="")
        self.reader = csv.reader(
            self.input_file, dialect=dialect, delimiter=delimiter, **fmtparams
        )

    def __iter__(self):
        return self

    def __next__(self) -> list[Text]:
        row = next(self.reader)
        return row

    next = __next__

    @property
    def dialect(self) -> DIALECT:
        return self.reader.dialect

    @property
    def line_num(self) -> int:
        return self.reader.line_num


class writer(object):
    def __init__(
        self,
        csvfile: IO[bytes],
        dialect: DIALECT = "excel",
        delimiter: str = "\t",
        **fmtparams: Any,
    ):
        """Open a csv writer() defaulting to TAB delimiters.

        REQUIRES: `csvfile` must be a buffered byte writer, e.g. from
        io.open(filename, 'wb') or io.BytesIO(buffer).

        This does Unicode by constructing the csv.writer with a TextIO.
        """
        self.output_file = TextIOWrapper(
            csvfile, encoding="utf-8", newline="", line_buffering=True
        )
        self.writer = csv.writer(
            self.output_file, dialect=dialect, delimiter=delimiter, **fmtparams
        )

    def writerow(self, row: Sequence[Any]):
        self.writer.writerow(row)

    def writerows(self, rows: Iterable[Sequence[Any]]):
        for row in rows:
            self.writerow(row)

    @property
    def dialect(self) -> DIALECT:
        return self.writer.dialect


class dict_reader(object):
    def __init__(
        self, f: IO[bytes], fieldnames: Optional[list[str]] = None, **kwargs: Any
    ):
        """
        Open a csv DictReader() defaulting to TAB delimiters. Fields whose
        names start with an underscore are removed from self._fieldnames, and
        discarded from each row during iteration.

        REQUIRES: `f` must be a buffered byte reader, e.g. from
        io.open(filename, 'rb') or io.BytesIO(buffer).
        """
        tsv_reader = reader(f, **kwargs)
        self.tsv_dict_reader = csv.DictReader(
            tsv_reader.input_file, fieldnames=fieldnames, **kwargs
        )
        self.tsv_dict_reader.reader = cast(Any, tsv_reader)

        # Discard private field names that begin with underscore
        self._fieldnames = [
            fieldname
            for fieldname in (self.tsv_dict_reader.fieldnames or [])
            if not fieldname.startswith("_")
        ]

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, str]:
        row = self.tsv_dict_reader.__next__()

        # Discard entries with private field names
        new_row = {k: row[k] for k in self._fieldnames}

        return new_row

    @property
    def fieldnames(self) -> list[str]:
        return self._fieldnames

    @fieldnames.setter
    def fieldnames(self, values: list[str]):
        self.tsv_dict_reader.fieldnames = values

        self._fieldnames = [
            fieldname
            for fieldname in self.tsv_dict_reader.fieldnames
            if not fieldname.startswith("_")
        ]

    @property
    def dialect(self) -> DIALECT:
        return self.tsv_dict_reader.dialect

    @property
    def line_num(self) -> int:
        return self.tsv_dict_reader.line_num


def dict_writer(
    f: IO[bytes], fieldnames: Collection[str], dialect: DIALECT = "excel", **kwargs: Any
) -> csv.DictWriter:
    """Open a csv DictWriter() defaulting to TAB delimiters.

    REQUIRES: `csvfile` must be a buffered byte writer, e.g. from
    io.open(filename, 'wb') or io.BytesIO(buffer).
    """
    tsv_writer = writer(f, dialect=dialect, **kwargs)
    tsv_dict_writer = csv.DictWriter(
        tsv_writer.output_file, fieldnames, dialect=dialect, **kwargs
    )
    tsv_dict_writer.writer = cast(Any, tsv_writer)
    # TODO(jerry): Call `tsv_dict_writer.writeheader()` for convenience?

    return tsv_dict_writer
