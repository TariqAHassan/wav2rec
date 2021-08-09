"""

    Test Validation

"""
from __future__ import annotations

import pytest

from wav2rec._utils.validation import check_is_fitted


@pytest.mark.parametrize("do_fit", [True, False])
def test_check_is_fitted(do_fit: bool) -> None:
    class Model:
        def __init__(self) -> None:
            self.fitted: bool = False

        def fit(self) -> Model:
            self.fitted = True

        @check_is_fitted
        def predict(self) -> bool:
            return True

    model = Model()
    if do_fit:
        model.fit()
        assert model.predict() is True
    else:
        with pytest.raises(AttributeError):
            model.predict()
