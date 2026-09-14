"""The train/test split in ``generate_regression_model`` is chronological.

Nothing executed this function, so swapping the two ``iloc`` slices - training the model on
the LATER three quarters and scoring it on the earlier one - changed nothing observable.
That is look-ahead: the split is the only thing keeping the future out of the training set.

The stubs here are deliberate. ``split_dataframe`` is replaced by a recorder so the frames
it is handed can be asserted on, and ``regression_models`` is emptied so the loop body (real
training, real hyper-parameter search) never runs. The slicing itself is real.
"""

import asyncio

import pytest
from pandas import DataFrame, date_range

import modeling.train_model as train_model


@pytest.fixture
def recorded_splits(monkeypatch) -> list:
    """Run ``generate_regression_model`` far enough to record every ``split_dataframe`` call."""
    calls = []

    def recorder(dataframe, target, selected_features=None, scaler=None):
        calls.append(dataframe)
        x = dataframe.drop(columns=[target])
        return x, dataframe[target], object()

    monkeypatch.setattr(train_model, "split_dataframe", recorder)
    monkeypatch.setattr(train_model, "regression_models", {})
    monkeypatch.setattr(
        train_model,
        "generate_features",
        lambda series: DataFrame({"feature": range(len(series))}, index=series.index),
    )
    monkeypatch.setattr(train_model, "encode_categorical_data", lambda frame: None)
    return calls


def _frame(rows: int) -> DataFrame:
    return DataFrame(
        {"value": [float(i) for i in range(rows)]},
        index=date_range("2026-01-01", periods=rows, freq="D"),
    )


def test_training_data_comes_entirely_before_the_test_data(recorded_splits):
    asyncio.run(train_model.generate_regression_model(_frame(100), "btc"))

    train_frame, test_frame = recorded_splits[0], recorded_splits[1]
    assert len(train_frame.index) and len(test_frame.index)
    # The property, not the arithmetic: no training row may sit at or after the first test row.
    assert train_frame.index.max() < test_frame.index.min()


def test_the_split_is_three_quarters_and_the_two_halves_are_disjoint(recorded_splits):
    rows = 100
    asyncio.run(train_model.generate_regression_model(_frame(rows), "btc"))

    train_frame, test_frame = recorded_splits[0], recorded_splits[1]
    total = len(train_frame.index) + len(test_frame.index)
    assert len(train_frame.index) == total * 3 // 4
    # Every row used exactly once: an off-by-one that drops or repeats a row fails here.
    assert not set(train_frame.index) & set(test_frame.index)
    assert len(set(train_frame.index) | set(test_frame.index)) == total


def test_the_test_split_is_scored_on_the_features_the_training_split_chose(
    recorded_splits, monkeypatch
):
    """The second call must reuse the first call's columns, not re-select from the test frame."""
    seen = []

    def recorder(dataframe, target, selected_features=None, scaler=None):
        seen.append(selected_features)
        x = dataframe.drop(columns=[target])
        return x, dataframe[target], object()

    monkeypatch.setattr(train_model, "split_dataframe", recorder)
    asyncio.run(train_model.generate_regression_model(_frame(100), "btc"))

    assert seen[0] is None, "the training split selects the features"
    assert seen[1] == [
        "feature"
    ], "the test split is handed the training split's columns"
