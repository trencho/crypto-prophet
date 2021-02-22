from fastapi import APIRouter

forecast_router = APIRouter()


@forecast_router.get('/forecast')
async def forecast():
    pass
