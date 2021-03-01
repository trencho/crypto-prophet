from os import path
from pickle import load as pickle_load

from fastapi import APIRouter

from definitions import MODELS_PATH
from modeling import train_coin_models

forecast_router = APIRouter(tags=['forecast'])


@forecast_router.get('/forecast')
async def forecast():
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
