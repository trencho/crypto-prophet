from fastapi import APIRouter

forecast_router = APIRouter(tags=['forecast'])


@forecast_router.get('/forecast')
async def forecast():
    pass
