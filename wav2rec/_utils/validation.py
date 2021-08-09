"""

    Validation

"""
from typing import Any, Callable


def check_is_fitted(method: Callable[..., Any]) -> Callable[..., Any]:
    """Validate a class is fitted prior to ``method`` running

    Args:
        method (callable): a callable class method

    Returns:
        wrapper (callable): a wrapper function which will validate
            that the class ``method`` belongs to has been fit prior
            to it being executed

    """

    def wrapper(self: object, *args: Any, **kwargs: Any) -> Any:
        if getattr(self, "fitted") is True:
            return method(self, *args, **kwargs)
        else:
            raise AttributeError(f"{self.__class__.__name__} not fit.")

    return wrapper
