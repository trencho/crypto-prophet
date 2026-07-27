"""Tests for modeling.train_model pure helpers.

Only the helpers that do not require a trained pipeline are exercised here:
``split_dataframe`` (with ``selected_features`` supplied so the heavy
``backward_elimination`` OLS path is skipped) and ``check_best_regression_model``
(with ``os.path.getmtime`` monkeypatched to drive the fresh/stale/error paths).
"""

import time

from pandas import DataFrame

import modeling.train_model as train_model
from modeling.train_model import split_dataframe

_MONTH_IN_SECONDS = 2629800


def test_split_dataframe_alignment():
    frame = DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feat_a": [10.0, 12.0, 15.0, 19.0, 24.0, 30.0],
            "feat_b": [5.0, 4.0, 6.0, 3.0, 7.0, 2.0],
        }
    )

    # Supplying selected_features avoids the OLS backward-elimination path.
    x, y = split_dataframe(frame, "value", selected_features=["feat_a"])

    assert len(x) == len(y)
    # previous_value_overwrite + tail-drop shorten by exactly one row.
    assert len(x) == len(frame) - 1
    assert list(x.columns) == ["feat_a"]


def test_check_best_regression_model_freshness(monkeypatch):
    coin_symbol = "btc"

    # Fresh: modified just now -> within the one-month window -> True.
    monkeypatch.setattr(train_model.path, "getmtime", lambda _p: time.time())
    assert train_model.check_best_regression_model(coin_symbol) is True

    # Stale: modified two months ago -> older than the window -> False.
    monkeypatch.setattr(
        train_model.path,
        "getmtime",
        lambda _p: time.time() - 2 * _MONTH_IN_SECONDS,
    )
    assert train_model.check_best_regression_model(coin_symbol) is False

    # Missing file: getmtime raises OSError -> False.
    def _raise(_p):
        raise OSError("no such file")

    monkeypatch.setattr(train_model.path, "getmtime", _raise)
    assert train_model.check_best_regression_model(coin_symbol) is False
