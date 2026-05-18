
"""
Utilities for patching execution environments, configurations and functions.
"""


from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import reduce
from inspect import ismethod
from unittest.mock import Mock, DEFAULT, _patch, patch
from typing import Any

import pytest

from ecoli.library.xarray_emitter.utils import WarningFilter


# ==============================================================================
# warnings
# ==============================================================================


def filter_warnings(filters: list[WarningFilter]) -> Callable[[Callable], Callable]:
    """
    Analogue of :py:func:`ecoli.library.xarray_emitter.utils.filter_warnings`,
    but with the effect of applying :py:func:`pytest.mark.filterwarnings`
    decorators, instead of :py:func:`warnings.filterwarnings` context modifiers.
    """
    return (lambda func: reduce(
        lambda fun, wf: pytest.mark.filterwarnings(str(wf))(fun),
        filters, func))



# ==============================================================================
# config patching
# ==============================================================================


class PatchConfig(ABC):
    """
    Test parameter for modifying an already loaded baseline JSON configuration.
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Materialise changes to the JSON configuration.
        """
        ...


# ==============================================================================
# code patching
# ==============================================================================


def patch_func(func: str, *, cb: Callable | None = None) -> _patch:
    """
    Create a context manager which patches a module-level function, in order to
    trace its calls, and to optionally pre-apply a callback.

    .. note::
      ``func`` is passed as the argument ``target`` to
      :py:func:`unittest.mock.patch`.
    """
    mocked = None
    def side_effect(*args, **kwargs) -> Any:
        nonlocal cb, mocked
        if cb is not None:
            cb(*args, **kwargs)
        return mocked.temp_original(*args, **kwargs)  # type: ignore[attr-defined]
    mocked = patch(func, side_effect=side_effect)
    return mocked


# ------------------------------------------------------------------------------


def patch_meth(
    obj: object, meth: str, *,
    cb: Callable[..., None] | None = None,
    modargs: Callable[..., tuple[tuple, dict]] | None = None
) -> None:
    """
    Patch an object instance method, in order to trace its calls, and to
    optionally pre-apply a callback or argument modification.
    """
    assert ismethod(getattr(obj, meth))
    assert cb is None or modargs is None
    def side_effect(*args, **kwargs):
        nonlocal obj, meth, cb, modargs
        if modargs is not None:
            _args, _kwargs = modargs(obj, *args, **kwargs)
            return getattr(obj, meth)._mock_wraps(*_args, **_kwargs)
        elif cb is not None:
            cb(obj, *args, **kwargs)
        return DEFAULT
    setattr(obj, meth, Mock(wraps=getattr(obj, meth), side_effect=side_effect))
