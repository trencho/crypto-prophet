from datetime import datetime
from os import path
from pickle import load as pickle_load

from fastapi import APIRouter, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse
from pandas import read_csv, to_datetime

from definitions import DATA_EXTERNAL_PATH, MODELS_PATH
from modeling import train_coin_models
from processing import closest_hour, current_hour, next_hour, recursive_forecast

forecast_router = APIRouter(tags=['forecast'])


@forecast_router.get('/forecast/{coin_id}')
async def forecast(coin_id: str = None, timestamp: int = None):
    pass


def load_regression_model(coin):
    if not path.exists(
            path.join(MODELS_PATH, coin['symbol'], 'best_regression_model.pkl')):
        train_coin_models(coin)
        return None

    with open(path.join(MODELS_PATH, coin['symbol'], 'best_regression_model.pkl'),
              'rb') as in_file:
        model = pickle_load(in_file)

    with open(path.join(MODELS_PATH, coin['symbol'], 'selected_features.pkl'),
              'rb') as in_file:
        model_features = pickle_load(in_file)

    return model, model_features


def forecast_city_sensor(coin, timestamp):
    load_model = load_regression_model(coin)
    if load_model is None:
        return load_model

    model, model_features = load_model

    dataframe = read_csv(path.join(DATA_EXTERNAL_PATH, coin['symbol'], 'data.csv'))
    dataframe.set_index(to_datetime(dataframe['time'] / 10 ** 3, unit='s'), inplace=True)

    current_datetime = current_hour(datetime.now())
    date_time = datetime.fromtimestamp(timestamp)
    n_steps = (date_time - current_datetime).total_seconds() // 3600
    return recursive_forecast(dataframe['value'], model, model_features, n_steps).iloc[-1]


def retrieve_forecast_timestamp(timestamp):
    next_hour_time = next_hour(datetime.now())
    next_hour_timestamp = int(datetime.timestamp(next_hour_time))
    timestamp = timestamp or next_hour_timestamp
    if timestamp < next_hour_timestamp:
        message = ('Cannot forecast pollutant because the timestamp is in the past. Send a GET request to the history '
                   'endpoint for past values.')
        return ORJSONResponse(jsonable_encoder(f'error_message={message}'), status_code=status.HTTP_400_BAD_REQUEST)

    return closest_hour(datetime.fromtimestamp(timestamp)).timestamp()
