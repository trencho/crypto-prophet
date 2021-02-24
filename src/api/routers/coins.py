from fastapi import APIRouter

coins_router = APIRouter(tags=['coins'])


@coins_router.get('/coins/')
async def fetch_coins():
    return None


@coins_router.get('/coins/{coin_id}')
async def fetch_coin(coin_id: str = None):
    return coin_id
