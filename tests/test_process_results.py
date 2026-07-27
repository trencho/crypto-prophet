"""Tests for modeling.process_results error-metric helpers."""

from pathlib import Path

import pytest
from numpy import inf, nan
from pandas import Series, read_csv

import definitions
from modeling.process_results import (
    filter_invalid_values,
    mean_absolute_percentage_error,
    save_errors,
)


def test_save_errors_returns_mae_and_writes_csv(results_errors_dir):
    coin_symbol, model_name = "btc", "LinearRegressionModel"
    results_errors_dir(coin_symbol, model_name)

    y_true = Series([1.0, 2.0, 3.0, 4.0])
    y_predicted = Series([1.0, 2.0, 3.0, 5.0])

    mae = save_errors(coin_symbol, model_name, y_true, y_predicted)

    # errors of 0, 0, 0, 1 -> MAE == 0.25
    assert mae == pytest.approx(0.25)

    csv_path = (
        Path(definitions.RESULTS_ERRORS_PATH)
        / "data"
        / coin_symbol
        / model_name
        / "error.csv"
    )
    assert csv_path.exists()

    written = read_csv(csv_path)
    assert written["Mean Absolute Error"].iloc[0] == pytest.approx(0.25)


def test_mape_none_when_infinite():
    # A zero in y_true forces a division by zero -> inf -> None.
    y_true = Series([0.0, 1.0, 2.0])
    y_predicted = Series([1.0, 1.0, 2.0])

    assert mean_absolute_percentage_error(y_true, y_predicted) is None


def test_mape_finite_value():
    # Sanity check the non-infinite branch returns a real percentage.
    y_true = Series([100.0, 200.0])
    y_predicted = Series([110.0, 180.0])

    result = mean_absolute_percentage_error(y_true, y_predicted)
    assert result == pytest.approx(10.0)


def test_filter_invalid_values_drops_inf_nan():
    y_true = Series([1.0, inf, 3.0, nan])
    y_predicted = Series([1.0, 2.0, 3.0, 4.0])

    filtered_true, filtered_predicted = filter_invalid_values(y_true, y_predicted)

    assert list(filtered_true) == [1.0, 3.0]
    assert list(filtered_predicted) == [1.0, 3.0]
