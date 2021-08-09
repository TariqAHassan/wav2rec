"""

    Test Printing

    References:
        * https://github.com/TariqAHassan/alsek/blob/master/tests/_utils/test_printing.py

"""
from datetime import datetime
from typing import Any, Dict, Optional

import pytest

from wav2rec._utils.printing import _format_params, _format_value, auto_repr

START_DATETIME = datetime.now()


class ReprClass:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.mark.parametrize(
    "value,expected",
    [
        ("a", "'a'"),
        (1, 1),
        ({}, {}),
        (START_DATETIME, f"'{START_DATETIME}'"),
    ],
)
def test_format_value(value: Any, expected: Any) -> None:
    assert _format_value(value) == expected


@pytest.mark.parametrize(
    "params,join_on,expected",
    [
        ({"a": 1}, ", ", "a=1"),
        ({"a": 1, "b": "c"}, ", ", "a=1, b='c'"),
        ({"a": 1, "b": "c"}, "-!-", "a=1-!-b='c'"),
    ],
)
def test_format_params(
    params: Dict[str, Any],
    join_on: str,
    expected: str,
) -> None:
    actual = _format_params(params, join_on=join_on)
    assert actual == expected


@pytest.mark.parametrize(
    "obj,new_line_threshold,params,expected",
    [
        (ReprClass(), 3, dict(a=1, b=2), "ReprClass(a=1, b=2)"),
        (ReprClass(), 1, dict(a=1, b=2), "ReprClass(\n    a=1,\n    b=2\n)"),
        (
            ReprClass(),
            3,
            dict(a=1, b=2, c=3, d=4, e=5),
            "ReprClass(\n    a=1,\n    b=2,\n    c=3,\n    d=4,\n    e=5\n)",
        ),
    ],
)
def test_auto_repr(
    obj: Any,
    new_line_threshold: Optional[int],
    params: Dict[str, Any],
    expected: str,
) -> None:
    actual = auto_repr(obj, new_line_threshold=new_line_threshold, **params)
    assert actual == expected
