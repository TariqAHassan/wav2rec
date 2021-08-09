"""

    Printing Utils

    References:
        * https://github.com/TariqAHassan/alsek/blob/master/tests/_utils/test_printing.py

"""
from datetime import datetime
from typing import Any, Dict, Optional


def _format_value(value: Any) -> Any:
    if isinstance(value, (str, datetime)):
        return f"'{value}'"
    else:
        return value


def _format_params(params: Dict[str, Any], join_on: str) -> str:
    return join_on.join((f"{k}={_format_value(v)}" for k, v in params.items()))


def auto_repr(obj: object, new_line_threshold: Optional[int] = 5, **params: Any) -> str:
    """Autogenerate a class repr string.

    Args:
        obj (object): an object to generate a repr for
        new_line_threshold (int, optional): number of ``params``
            required to split the parameters over multiple lines.
        **params (Keyword Args): parameters to include in the
            repr string

    Returns:
        repr (str): repr string

    """
    class_name = obj.__class__.__name__
    if new_line_threshold is None or len(params) <= new_line_threshold:
        start, join_on, end = "", ", ", ""
    else:
        start, join_on, end = "\n    ", ",\n    ", "\n"
    return f"{class_name}({start}{_format_params(params, join_on=join_on)}{end})"
