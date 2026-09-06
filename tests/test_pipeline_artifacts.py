"""The model, its feature list and its scaler must describe one training run, or be refused."""

from json import loads
from pickle import dump, HIGHEST_PROTOCOL

from numpy import allclose, array
from pandas import DataFrame
from pytest import raises

from definitions import PIPELINE_SCHEMA
from modeling.train_model import save_pipeline, split_dataframe
from processing import forecast_data as fd
from processing.feature_scaling import apply_scaler, fit_scaler


def _frame(rows: int = 12) -> DataFrame:
    return DataFrame(
        {
            "value": [float(i) for i in range(rows)],
            "feat_a": [float(i * 2) for i in range(rows)],
        }
    )


def test_fit_scaler_is_the_only_thing_that_fits():
    """apply_scaler must not fit. A scaler with no `fit` proves it by construction."""

    class TransformOnly:
        def transform(self, frame):
            return frame * 0 + 1

    result = apply_scaler(_frame(3)[["feat_a"]], TransformOnly())
    assert allclose(result.to_numpy(), 1.0)


def test_the_test_split_is_scaled_with_the_training_scaler():
    """Train and test used to be scaled by separate scalers fitted on their own rows.

    Passing the training scaler through is the whole point: the same input column must map to the
    same number regardless of which split it arrived in.
    """
    frame = _frame(12)
    train, test = frame.iloc[:8], frame.iloc[8:]

    _, _, scaler = split_dataframe(train, "value", selected_features=["feat_a"])
    x_test, _, returned = split_dataframe(
        test, "value", selected_features=["feat_a"], scaler=scaler
    )

    assert returned is scaler
    expected = apply_scaler(test[["feat_a"]], scaler).iloc[:-1]
    assert allclose(x_test["feat_a"].to_numpy(), expected["feat_a"].to_numpy())


def test_a_directory_without_a_manifest_is_refused(tmp_path, monkeypatch):
    """Every artefact trained before the scaler was persisted is retired by this check."""
    monkeypatch.setattr(fd, "MODELS_PATH", tmp_path)
    coin = tmp_path / "btc"
    coin.mkdir()
    for name in ("best_regression_model.pkl", "selected_features.pkl", "scaler.pkl"):
        with open(coin / name, "wb") as out_file:
            dump(["feat_a"], out_file, HIGHEST_PROTOCOL)

    assert fd.load_regression_model("btc") is None


def test_an_incomplete_directory_is_skipped_not_raised(tmp_path, monkeypatch):
    """A half-written directory used to 500 the whole endpoint, not just drop one coin."""
    monkeypatch.setattr(fd, "MODELS_PATH", tmp_path)
    coin = tmp_path / "btc"
    coin.mkdir()
    (coin / "pipeline.json").write_text(
        '{"schema": %d, "features": ["feat_a"]}' % PIPELINE_SCHEMA, encoding="utf-8"
    )
    with open(coin / "best_regression_model.pkl", "wb") as out_file:
        dump(object(), out_file, HIGHEST_PROTOCOL)

    assert fd.load_regression_model("btc") is None


def test_disagreeing_artefacts_are_refused():
    class Model:
        n_features_in_ = 3

    class Scaler:
        feature_names_in_ = array(["feat_a", "feat_b"])

    manifest = {"schema": PIPELINE_SCHEMA, "features": ["feat_a", "feat_b"]}

    with raises(fd.ModelArtifactMismatch, match="takes 3 features"):
        fd.verify_pipeline("btc", manifest, Model(), ["feat_a", "feat_b"], Scaler())

    with raises(fd.ModelArtifactMismatch, match="feature list"):
        fd.verify_pipeline("btc", manifest, Model(), ["feat_a"], Scaler())


def test_save_pipeline_writes_a_manifest_that_matches_what_it_saved(
    tmp_path, monkeypatch
):
    import modeling.train_model as tm

    monkeypatch.setattr(tm, "MODELS_PATH", tmp_path)
    _, scaler = fit_scaler(_frame(6)[["feat_a"]])
    save_pipeline("btc", ["feat_a"], scaler)

    manifest = loads((tmp_path / "btc" / "pipeline.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == PIPELINE_SCHEMA
    assert manifest["features"] == ["feat_a"]
    assert manifest["scaler"]["n_features_in"] == 1
    assert (tmp_path / "btc" / "scaler.pkl").exists()
