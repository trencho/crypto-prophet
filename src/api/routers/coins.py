from os import path

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse
from pandas import read_csv

from definitions import DATA_EXTERNAL_PATH

coins_router = APIRouter(tags=['coins'])


@coins_router.get('/coins/')
async def fetch_coins():
    return ORJSONResponse(jsonable_encoder(read_csv(path.join(DATA_EXTERNAL_PATH, 'coin_list.csv')).to_dict('records')))


@coins_router.get('/coins/{coin_id}')
async def fetch_coin(coin_id: str = None):
    coin_list = read_csv(path.join(DATA_EXTERNAL_PATH, 'coin_list.csv')).to_dict('records')
    for coin in coin_list:
        if coin['id'] == coin_id:
            return ORJSONResponse(jsonable_encoder(coin))
