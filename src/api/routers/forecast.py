from datetime import datetime
from math import ceil
from os import path
from pickle import load as pickle_load

from fastapi import APIRouter, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse
from pandas import read_csv, to_datetime

from definitions import DATA_EXTERNAL_PATH, MODELS_PATH, coins
from modeling import train_coin_models
from preparation import check_coin
from processing import closest_hour, current_hour, next_hour, recursive_forecast

forecast_router = APIRouter(tags=['forecast'])


def append_forecast_data(coin, forecast_value, forecast_results):
    forecast_results.append({
        'coin': coin,
        'value': forecast_value if forecast_value is None else float(forecast_value)
    })


@forecast_router.get('/forecast/')
async def forecast(coin_id: str = None, timestamp: int = None):
    if coin_id is not None and coin_id not in coins:
        message = 'Value cannot be predicted because the coin is not found or is invalid.'
        return ORJSONResponse(jsonable_encoder({'error_message': message}), status_code=status.HTTP_400_BAD_REQUEST)

    timestamp = retrieve_forecast_timestamp(timestamp)
    if isinstance(timestamp, ORJSONResponse):
        return timestamp

    forecast_results = []
    if coin_id is None:
        for coin_id in coins:
            coin = check_coin(coin_id)
            forecast_value = forecast_coin(coin, timestamp)
            append_forecast_data(coin_id, forecast_value, forecast_results)

        return ORJSONResponse(jsonable_encoder(forecast_results), status_code=status.HTTP_200_OK)

    coin = check_coin(coin_id)
    forecast_value = forecast_coin(coin, timestamp)
    append_forecast_data(coin_id, forecast_value, forecast_results)
    return ORJSONResponse(jsonable_encoder(forecast_results), status_code=status.HTTP_200_OK)


def load_regression_model(coin):
    if not path.exists(path.join(MODELS_PATH, coin['symbol'], 'best_regression_model.pkl')):
        train_coin_models(coin)
        return None

    with open(path.join(MODELS_PATH, coin['symbol'], 'best_regression_model.pkl'), 'rb') as in_file:
        model = pickle_load(in_file)

    with open(path.join(MODELS_PATH, coin['symbol'], 'selected_features.pkl'), 'rb') as in_file:
        model_features = pickle_load(in_file)

    return model, model_features


def forecast_coin(coin, timestamp):
    load_model = load_regression_model(coin)
    if load_model is None:
        return load_model

    model, model_features = load_model

    dataframe = read_csv(path.join(DATA_EXTERNAL_PATH, coin['symbol'], 'data.csv'))
    dataframe.set_index('time', inplace=True)
    dataframe.index = to_datetime(dataframe.index / 10 ** 3, unit='s')

    current_datetime = current_hour(datetime.now())
    date_time = datetime.fromtimestamp(timestamp)
    n_steps = ceil((date_time - current_datetime).total_seconds() / 86400)
    return recursive_forecast(dataframe['value'], model, model_features, n_steps).iloc[-1]


def retrieve_forecast_timestamp(timestamp):
    next_hour_time = next_hour(datetime.now())
    next_hour_timestamp = int(datetime.timestamp(next_hour_time))
    timestamp = timestamp or next_hour_timestamp
    if timestamp < next_hour_timestamp:
        message = ('Cannot forecast coin because the timestamp is in the past. Send a GET request to the history '
                   'endpoint for past values.')
        return ORJSONResponse(jsonable_encoder({'error_message': message}), status_code=status.HTTP_400_BAD_REQUEST)

    return closest_hour(datetime.fromtimestamp(timestamp)).timestamp()
