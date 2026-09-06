from functools import lru_cache
from json import JSONDecodeError, loads
from logging import getLogger
from math import isnan
from math import nan
from pathlib import Path
from pickle import load
from typing import Optional

from pandas import (
    concat,
    DataFrame,
    date_range,
    read_csv,
    Series,
    Timedelta,
    to_datetime,
)

from definitions import coins, DATA_EXTERNAL_PATH, MODELS_PATH, PIPELINE_SCHEMA
from models.base_regression_model import BaseRegressionModel
from .feature_generation import (
    encode_categorical_data,
    generate_features,
)
from .feature_scaling import apply_scaler

logger = getLogger(__name__)

# A forecast is recomputed from artefacts the scheduler rewrites at most every 15 minutes, and
# computing one costs ~0.58s per coin (a disk unpickle plus a 30-step recursive prediction that
# regenerates features at every step). Serving the same numbers twice in that window is pure waste,
# and with -w 4 a handful of concurrent requests is enough to occupy every worker.
#
# Keyed on the model file's mtime as well as the coin, so a retrain invalidates the entry by
# construction rather than by waiting out a TTL.
FORECAST_CACHE_SIZE = 32

FORECAST_PERIOD = "1D"
FORECAST_STEPS = 30


def fetch_forecast_result(coin_id: str = None) -> dict:
    """Forecast every configured coin, or just one when `coin_id` is given.

    The single-coin path exists because the route had no way to ask for one: a caller wanting
    bitcoin paid for the whole configured set, and each coin unpickles a model and runs a 30-step
    recursive forecast.
    """
    forecast_result = {}
    coin_list = read_csv(Path(DATA_EXTERNAL_PATH) / "coin_list.csv").to_dict("records")
    for coin in coin_list:
        if coin["id"] in coins and (coin_id is None or coin["id"] == coin_id):
            if (predictions := forecast_coin(coin["symbol"])) is None:
                continue

            for index, value in predictions.items():
                timestamp_dict = forecast_result.get(int(index.timestamp()), {})
                timestamp_dict.update(
                    {
                        "time": int(index.timestamp()),
                        coin["id"]: None if isnan(value) else value,
                    }
                )
                forecast_result.update({int(index.timestamp()): timestamp_dict})

    return forecast_result


class ModelArtifactMismatch(ValueError):
    """A coin's model, feature list and scaler do not describe the same training run."""


def _model_fingerprint(coin_symbol: str) -> Optional[float]:
    try:
        return (
            (Path(MODELS_PATH) / coin_symbol / "best_regression_model.pkl")
            .stat()
            .st_mtime
        )
    except OSError:
        return None


def forecast_coin(coin_symbol: str) -> Optional[Series]:
    fingerprint = _model_fingerprint(coin_symbol)
    if fingerprint is None:
        return None
    return _forecast_coin_cached(coin_symbol, fingerprint)


@lru_cache(maxsize=FORECAST_CACHE_SIZE)
def _forecast_coin_cached(coin_symbol: str, _fingerprint: float) -> Optional[Series]:
    """`_fingerprint` is unused in the body and load-bearing in the key.

    It is the model file's mtime, so a retrain produces a different key and the previous entry
    stops being reachable. Without it this cache would serve the old model's numbers until the
    process restarted.
    """
    if (load_model := load_regression_model(coin_symbol)) is None:
        return None

    model, model_features, scaler = load_model

    return recursive_forecast(coin_symbol, model, model_features, scaler)


def load_regression_model(coin_symbol: str):
    """Load a coin's model, features and scaler, or return None if they do not agree.

    Returning None means "this coin has nothing servable", which the caller already handles by
    skipping the coin. It used to check only that the model file existed and then open the
    features file unguarded, so a half-written directory raised FileNotFoundError out of the HTTP
    handler and 500'd the whole endpoint - every coin, because of one.

    The schema check is also what retires every artefact trained before the scaler was persisted:
    those directories have no pipeline.json, so they are refused here and the scheduler retrains
    them. That matters more than it looks, because it means correctness does not depend on someone
    remembering to delete models/.
    """
    coin_path = Path(MODELS_PATH) / coin_symbol
    manifest_path = coin_path / "pipeline.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = loads(manifest_path.read_text(encoding="utf-8"))
    except JSONDecodeError:
        logger.warning("%s: pipeline.json is not readable JSON; skipping", coin_symbol)
        return None

    if manifest.get("schema") != PIPELINE_SCHEMA:
        logger.info(
            "%s: pipeline schema %s, expected %s; skipping until it is retrained",
            coin_symbol,
            manifest.get("schema"),
            PIPELINE_SCHEMA,
        )
        return None

    required = (
        coin_path / "best_regression_model.pkl",
        coin_path / "selected_features.pkl",
        coin_path / "scaler.pkl",
    )
    if not all(artefact.exists() for artefact in required):
        logger.warning("%s: model directory is incomplete; skipping", coin_symbol)
        return None

    with open(coin_path / "best_regression_model.pkl", "rb") as in_file:
        model = load(in_file)
    with open(coin_path / "selected_features.pkl", "rb") as in_file:
        model_features = load(in_file)
    with open(coin_path / "scaler.pkl", "rb") as in_file:
        scaler = load(in_file)

    try:
        verify_pipeline(coin_symbol, manifest, model, model_features, scaler)
    except ModelArtifactMismatch as mismatch:
        logger.error("%s: %s", coin_symbol, mismatch)
        return None

    return model, model_features, scaler


def verify_pipeline(coin_symbol, manifest, model, model_features, scaler) -> None:
    """Refuse a set whose parts disagree, rather than serving numbers from mismatched columns."""
    if list(manifest.get("features", [])) != list(model_features):
        raise ModelArtifactMismatch(
            "the manifest's feature list and selected_features.pkl differ"
        )

    scaler_features = getattr(scaler, "feature_names_in_", None)
    if scaler_features is not None and len(scaler_features) != len(model_features):
        raise ModelArtifactMismatch(
            f"the scaler was fitted on {len(scaler_features)} features, "
            f"the model expects {len(model_features)}"
        )

    model_inputs = getattr(model, "n_features_in_", None)
    if model_inputs is not None and model_inputs != len(model_features):
        raise ModelArtifactMismatch(
            f"the model takes {model_inputs} features, "
            f"the feature list has {len(model_features)}"
        )


def recursive_forecast(
    coin_symbol: str,
    model: BaseRegressionModel,
    model_features: list,
    scaler,
    lags: int = FORECAST_STEPS,
    n_steps: int = FORECAST_STEPS,
    # A pandas frequency string (FORECAST_PERIOD is "1D"), not a count of periods.
    # This was annotated `int`, which is the annotation being wrong rather than the value.
    step: str = FORECAST_PERIOD,
) -> Series:
    """Multistep recursive forecasting using the input time series data and a pre-trained machine learning model

    Parameters
    ----------
    coin_symbol: The symbol of the coin that is used as a forecasting target
    model: An already trained machine learning model implementing the scikit-learn interface
    model_features: Selected model features for forecasting
    lags: List of lags used for training the model
    n_steps: Number of time periods in the forecasting horizon
    step: The period of forecasting

    Returns
    -------
    forecast_values: pd.Series with forecasted values indexed by forecast horizon dates
    """

    dataframe = read_csv(Path(DATA_EXTERNAL_PATH) / coin_symbol / "data.csv")
    dataframe = dataframe.set_index("time")
    dataframe.index = to_datetime(dataframe.index / 10**3, unit="s")

    # Get the dates to forecast
    last_date = dataframe.index[-1] + Timedelta(days=1)
    forecast_range = date_range(last_date, periods=n_steps, freq=step)

    forecasted_values = []
    target = dataframe["value"].copy()

    for date in forecast_range:
        # Predict from the features at the LAST OBSERVED row, which is what the model was trained
        # on: x(t) -> y(t+1). The previous loop appended a placeholder row for `date` FIRST and
        # then predicted from features computed AT that row, which is one step out of step with the
        # training pairing. It also seeded that placeholder with 0.0 on the first iteration, a
        # value the series never takes, so the first forecast was made from a fabricated lag.
        features = generate_features(target, lags)
        features = concat(
            [
                features,
                DataFrame(columns=list(set(model_features) - set(features.columns))),
            ]
        )
        encode_categorical_data(features)
        features = features[model_features]
        try:
            # Transform with the TRAINING scaler. This used to refit a fresh scaler on every
            # step's frame, so each of the 30 steps was scaled by different statistics and
            # none of them matched the ones the model was trained under.
            features = apply_scaler(features, scaler)
            prediction = model.predict(features)[-1]
        except ValueError:
            prediction = nan
        forecasted_values.append(prediction)
        # Feed the prediction back in so the next step's lags see it.
        target = concat([target, Series(prediction, [date])])

    return Series(forecasted_values, forecast_range)
