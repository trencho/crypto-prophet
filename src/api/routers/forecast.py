from fastapi import APIRouter, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse

from processing import fetch_forecast_result

forecast_router = APIRouter(tags=['forecast'])


@forecast_router.get('/forecast/')
async def forecast():
    return ORJSONResponse(jsonable_encoder(list(fetch_forecast_result().values())), status_code=status.HTTP_200_OK)
