"""Tests for modeling.train_model pure helpers.

Only the helpers that do not require a trained pipeline are exercised here:
``split_dataframe`` (with ``selected_features`` supplied so the heavy
``backward_elimination`` OLS path is skipped) and ``check_best_regression_model``
(with ``os.path.getmtime`` monkeypatched to drive the fresh/stale/error paths).
"""

import time

from numpy import corrcoef, cumsum, random
from pandas import DataFrame, Series

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
    x, y, scaler = split_dataframe(frame, "value", selected_features=["feat_a"])

    assert len(x) == len(y)
    # The split drops exactly one row: the last, which has no next value to predict.
    assert len(x) == len(frame) - 1
    assert list(x.columns) == ["feat_a"]

    # x(t) must predict y(t+1). This is the assertion the old version of this test could not
    # make: it checked only lengths and column names, all of which a tail-drop satisfies on its
    # own, so it passed while x was shifted to t+1 and y left at t.
    assert list(y) == [2.0, 3.0, 4.0, 5.0, 6.0]
    # feat_a is unscaled here only because the scaler is a no-op on a single column with this
    # fixture; what matters is the ROW it came from, which must be t, not t+1.
    assert x.index.tolist() == [0, 1, 2, 3, 4]


def test_no_feature_column_equals_the_target():
    """The target must not be an input.

    `previous_value_overwrite` shifted x to t+1 while y stayed at t, so the lag_1 feature (the
    previous value at t+1, which IS the value at t) was bit-identical to y. Every error metric
    was then near-zero by construction, model selection was arbitrary, and the served forecast
    degenerated to persistence. Measured on a random walk: pearson r = 1.000000, against 0.9969
    for the same column unshifted.

    Asserted on CORRELATION, not identity. `split_dataframe` scales x through a RobustScaler, so
    a leaked column is no longer numerically equal to y even though it carries exactly the same
    information. Correlation is invariant under that affine transform and identity is not, so an
    `allclose` check here passes on the broken code: it was tried, and it did.

    A random walk is used deliberately. An ordinary lag is ALREADY correlated at ~0.997 with the
    next value, so the threshold has to sit above that to mean anything.
    """
    rng = random.default_rng(0)
    values = 100 + cumsum(rng.normal(0, 1, 200))
    frame = DataFrame({"value": values, "lag_1": Series(values).shift(1).bfill()})

    x, y, _ = split_dataframe(frame, "value", selected_features=["lag_1"])

    for column in x.columns:
        r = corrcoef(x[column].to_numpy(), y.to_numpy())[0, 1]
        assert abs(r) < 0.9999, (
            f"feature {column!r} correlates with the target at r={r:.6f}: the label is in the "
            f"feature matrix (an honest lag on this fixture sits near 0.997)"
        )


def test_check_best_regression_model_freshness(monkeypatch, tmp_path):
    """Skipping training needs a model that is both recent AND usable.

    The manifest half matters as much as the mtime: a model saved before the scaler was persisted
    carries no pipeline.json and the forecast loader refuses it. Calling that "fresh" here would
    strand the coin - refused at serving time, skipped at training time - until the mtime window
    expired a month later.
    """
    coin_symbol = "btc"
    monkeypatch.setattr(train_model, "MODELS_PATH", tmp_path)
    coin_path = tmp_path / coin_symbol
    coin_path.mkdir()
    manifest = coin_path / "pipeline.json"
    current = '{"schema": %d, "features": []}' % train_model.PIPELINE_SCHEMA
    manifest.write_text(current, encoding="utf-8")

    # Fresh, with a current manifest -> within the one-month window -> True.
    monkeypatch.setattr(train_model.path, "getmtime", lambda _p: time.time())
    assert train_model.check_best_regression_model(coin_symbol) is True

    # Stale: modified two months ago -> older than the window -> False.
    monkeypatch.setattr(
        train_model.path,
        "getmtime",
        lambda _p: time.time() - 2 * _MONTH_IN_SECONDS,
    )
    assert train_model.check_best_regression_model(coin_symbol) is False

    # Recent, but the manifest is from a superseded pipeline -> retrain anyway.
    monkeypatch.setattr(train_model.path, "getmtime", lambda _p: time.time())
    manifest.write_text('{"schema": 0, "features": []}', encoding="utf-8")
    assert train_model.check_best_regression_model(coin_symbol) is False

    # Recent, but no manifest at all -> every artefact predating this change -> retrain.
    manifest.unlink()
    assert train_model.check_best_regression_model(coin_symbol) is False

    # Missing model file: getmtime raises OSError -> False.
    manifest.write_text(current, encoding="utf-8")

    def _raise(_p):
        raise OSError("no such file")

    monkeypatch.setattr(train_model.path, "getmtime", _raise)
    assert train_model.check_best_regression_model(coin_symbol) is False
