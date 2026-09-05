"""Tests for `recursive_forecast` (`processing.forecast_data`).

The named remaining gap in the project's test-coverage report. The function is recursive -- each
step feeds the previous prediction back in -- so the properties worth asserting are structural
(horizon length, index, the feed-back actually happening) and the failure path, none of which a
happy-path smoke test distinguishes.

The model is a stub implementing only `predict`, which is the whole scikit-learn surface this
function uses. A real estimator here would test scikit-learn, not this code.
"""

from numpy import isnan
from pandas import DataFrame, Timedelta, date_range, to_datetime

from processing import forecast_data as fd


class RecordingModel:
    """Returns a fixed value and records every feature frame it was asked to predict on."""

    def __init__(self, value=42.0):
        self.value = value
        self.seen = []

    def predict(self, features):
        self.seen.append(features)
        return [self.value]


class ExplodingModel:
    def predict(self, features):
        raise ValueError("feature matrix is not usable")


def _write_series(tmp_path, rows=40):
    external = tmp_path / "external"
    (external / "btc").mkdir(parents=True)
    # Millisecond epochs one day apart, which is what the loader divides by 10**3.
    start = 1_700_000_000_000
    DataFrame(
        {
            "time": [start + i * 86_400_000 for i in range(rows)],
            "value": [100.0 + i for i in range(rows)],
        }
    ).to_csv(external / "btc" / "data.csv", index=False)
    return external


def _run(monkeypatch, tmp_path, model, n_steps=5):
    external = _write_series(tmp_path)
    monkeypatch.setattr(fd, "DATA_EXTERNAL_PATH", str(external))
    monkeypatch.setattr(fd, "value_scaling", lambda f: f)
    monkeypatch.setattr(fd, "encode_categorical_data", lambda f: None)
    monkeypatch.setattr(
        fd,
        "generate_features",
        lambda target, lags: DataFrame({"lag_1": target.values}, index=target.index),
    )
    return fd.recursive_forecast("btc", model, ["lag_1"], lags=3, n_steps=n_steps)


def test_the_forecast_has_one_value_per_requested_step(monkeypatch, tmp_path):
    """R4: the horizon length is the contract callers index into."""
    result = _run(monkeypatch, tmp_path, RecordingModel(), n_steps=5)

    assert len(result) == 5


def test_the_forecast_is_indexed_by_dates_after_the_last_observation(
    monkeypatch, tmp_path
):
    """R4: an off-by-one here silently forecasts a day already in the data.

    The expected range is anchored to the LAST OBSERVED date, not to the result's own first
    element. Building it from `result.index[0]` made the assertion tautological: it could only
    check that three consecutive daily stamps existed, never where they started, so a mutation
    starting the forecast five days BEFORE the last observation passed. That is the exact defect
    the docstring names.
    """
    result = _run(monkeypatch, tmp_path, RecordingModel(), n_steps=3)

    # _write_series writes 40 daily rows from a fixed epoch, so the last observation and the first
    # forecast date are both computable without consulting the result.
    last_observed = to_datetime(
        1_700_000_000_000 + 39 * 86_400_000, unit="ms"
    ).normalize()
    expected_first = last_observed + Timedelta(days=1)

    assert result.index[0].normalize() == expected_first
    assert list(result.index) == list(
        date_range(result.index[0], periods=3, freq=fd.FORECAST_PERIOD)
    )
    assert result.index.is_monotonic_increasing


def test_each_step_predicts_on_a_series_grown_by_the_previous_prediction(
    monkeypatch, tmp_path
):
    """R5: the recursion itself.

    If the feed-back were dropped, every step would predict on the same frame and the function
    would be a constant-length loop rather than a recursive forecast -- with identical output for
    a constant model, so only the input frames tell the two apart.
    """
    model = RecordingModel()

    _run(monkeypatch, tmp_path, model, n_steps=4)

    assert len(model.seen) == 4
    lengths = [len(frame) for frame in model.seen]
    assert (
        lengths == sorted(lengths) and lengths[0] < lengths[-1]
    ), f"each step should see one more row than the last, got {lengths}"

    # Lengths alone do not prove the feed-back: the frames still grow if the appended value is a
    # constant, or zero, and a mutation replacing the fed-back prediction with 0.0 survived this
    # test on exactly that. Assert the VALUE that came back.
    #
    # `_run` stubs generate_features to `lag_1 = target.values`, so the last row of each frame is
    # the most recently appended point. From step 2 onward that point is the previous prediction.
    fed_back = [frame["lag_1"].iloc[-1] for frame in model.seen[1:]]
    assert fed_back == [model.value] * 3, (
        f"each step after the first must see the previous prediction ({model.value}) as its most "
        f"recent point, got {fed_back}"
    )
    # And the first step must NOT see a fabricated point: it predicts from real data only.
    assert (
        model.seen[0]["lag_1"].iloc[-1] == 139.0
    ), "the first step should predict from the last OBSERVED value, not from a placeholder"


def test_a_model_that_rejects_the_features_yields_nan_rather_than_raising(
    monkeypatch, tmp_path
):
    """R5: the `except ValueError` path.

    The job forecasts every coin in a loop, so one unusable feature matrix must not abort the
    others. The horizon still has to be the full length or the caller's index breaks.
    """
    result = _run(monkeypatch, tmp_path, ExplodingModel(), n_steps=3)

    assert len(result) == 3
    assert all(isnan(v) for v in result.values)
