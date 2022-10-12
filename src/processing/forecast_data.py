from math import isnan
from math import nan
from os import path
from pickle import load
from typing import Optional

from pandas import concat, DataFrame, date_range, read_csv, Series, Timedelta, to_datetime

from definitions import coins, DATA_EXTERNAL_PATH, MODELS_PATH
from models.base_regression_model import BaseRegressionModel
from .feature_generation import encode_categorical_data, generate_features, generate_lag_features, \
    generate_time_features
from .feature_scaling import value_scaling

FORECAST_PERIOD = '1D'
FORECAST_STEPS = 30


async def fetch_forecast_result() -> dict:
    forecast_result = {}
    coin_list = read_csv(path.join(DATA_EXTERNAL_PATH, 'coin_list.csv')).to_dict('records')
    for coin in coin_list:
        if coin['id'] in coins:
            if (predictions := await forecast_coin(coin['symbol'])) is None:
                continue

            for index, value in predictions.items():
                timestamp_dict = forecast_result.get(int(index.timestamp()), {})
                timestamp_dict.update({'time': int(index.timestamp()), coin['id']: None if isnan(value) else value})
                forecast_result.update({int(index.timestamp()): timestamp_dict})

    return forecast_result


async def forecast_coin(coin_symbol: str) -> Optional[Series]:
    if (load_model := await load_regression_model(coin_symbol)) is None:
        return None

    model, model_features = load_model

    return await recursive_forecast(coin_symbol, model, model_features)


async def load_regression_model(coin_symbol: str):
    if not path.exists(path.join(MODELS_PATH, coin_symbol, 'best_regression_model.pkl')):
        return None

    with open(path.join(MODELS_PATH, coin_symbol, 'best_regression_model.pkl'), 'rb') as in_file:
        model = load(in_file)

    with open(path.join(MODELS_PATH, coin_symbol, 'selected_features.pkl'), 'rb') as in_file:
        model_features = load(in_file)

    return model, model_features


async def direct_forecast(y, model, lags=FORECAST_STEPS, n_steps=FORECAST_STEPS, step=FORECAST_PERIOD) -> Series:
    """Multistep direct forecasting using a machine learning model to forecast each time period ahead

    Parameters
    ----------
    y: pd.Series holding the input time-series to forecast
    model: A model for iterative training
    lags: List of lags used for training the model
    n_steps: Number of time periods in the forecasting horizon
    step: The period of forecasting

    Returns
    -------
    forecast_values: pd.Series with forecasted values indexed by forecast horizon dates
    """

    async def one_step_features(date, step):
        # Features must be obtained using data lagged by the desired number of steps (the for loop index)
        tmp = y[y.index <= date]
        lags_features = await generate_lag_features(tmp, lags)
        time_features = await generate_time_features(tmp)
        features = lags_features.join(time_features, how='inner').dropna()

        # Build target to be ahead of the features built by the desired number of steps (the for loop index)
        target = y[y.index >= features.index[0] + Timedelta(days=step)]
        assert len(features.index) == len(target.index)

        return features, target

    forecast_values = []
    forecast_range = date_range(y.index[-1] + Timedelta(days=1), periods=n_steps, freq=step)
    forecast_features, _ = await one_step_features(y.index[-1], 0)

    for s in range(1, n_steps + 1):
        last_date = y.index[-1] - Timedelta(days=s)
        features, target = await one_step_features(last_date, s)

        model.train(features, target)

        # Use the model to predict s steps ahead
        predictions = model.predict(forecast_features)
        forecast_values.append(predictions[-1])

    return Series(forecast_values, forecast_range)


async def recursive_forecast(coin_symbol: str, model: BaseRegressionModel, model_features: list,
                             lags: int = FORECAST_STEPS, n_steps: int = FORECAST_STEPS,
                             step: int = FORECAST_PERIOD) -> Series:
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

    dataframe = read_csv(path.join(DATA_EXTERNAL_PATH, coin_symbol, 'data.csv'))
    dataframe.set_index('time', inplace=True)
    dataframe.index = to_datetime(dataframe.index / 10 ** 3, unit='s')

    # Get the dates to forecast
    last_date = dataframe.index[-1] + Timedelta(days=1)
    forecast_range = date_range(last_date, periods=n_steps, freq=step)

    forecasted_values = []
    target = dataframe['value'].copy()

    for date in forecast_range:
        # Build target time series using previously forecast value
        new_point = forecasted_values[-1] if len(forecasted_values) > 0 else 0.0
        target = concat([target, Series(new_point, [date])])

        features = await generate_features(target, lags)
        features = concat([features, DataFrame(columns=list(set(model_features) - set(list(features.columns))))])
        await encode_categorical_data(features)
        features = features[model_features]
        try:
            features = await value_scaling(features)
            predictions = model.predict(features)
            forecasted_values.append(predictions[-1])
        except ValueError:
            forecasted_values.append(nan)
        target.update(Series(forecasted_values[-1], [target.index[-1]]))

    return Series(forecasted_values, forecast_range)
