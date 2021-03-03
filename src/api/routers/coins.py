from os.path import join as path_join

from fastapi import APIRouter, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse
from pandas import read_csv

from definitions import DATA_EXTERNAL_PATH
from preparation import check_coin

coins_router = APIRouter(tags=['coins'])


@coins_router.get('/coins/')
async def fetch_coins():
    return ORJSONResponse(jsonable_encoder(read_csv(path_join(DATA_EXTERNAL_PATH, 'coin_list.csv')).to_dict('records')),
                          status_code=status.HTTP_200_OK)


@coins_router.get('/coins/{coin_id}/')
async def fetch_coin(coin_id: str = None):
    coin = check_coin(coin_id)
    if coin is None:
        message = 'Cannot return data because the coin is not found or is invalid.'
        return ORJSONResponse(jsonable_encoder({'error_message': message}), status_code=status.HTTP_404_NOT_FOUND)

    return ORJSONResponse(jsonable_encoder(coin), status_code=status.HTTP_200_OK)
