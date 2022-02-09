from fastapi import APIRouter, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse

from processing import fetch_forecast_result

forecast_router = APIRouter(tags=['forecast'])


def append_forecast_data(coin, forecast_value, forecast_results):
    forecast_results.append({
        'coin': coin,
        'value': forecast_value if forecast_value is None else float(forecast_value)
    })


@forecast_router.get('/forecast/')
async def forecast():
    return ORJSONResponse(jsonable_encoder(list(fetch_forecast_result().values())), status_code=status.HTTP_200_OK)
