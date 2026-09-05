from fastapi import APIRouter, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse

from processing import fetch_forecast_result

forecast_router = APIRouter(tags=["forecast"])


@forecast_router.get("/forecast/")
def forecast():
    """Deliberately `def`, not `async def`.

    `fetch_forecast_result` is synchronous and expensive: it unpickles a model per coin and runs a
    30-step recursive forecast, regenerating features and rescaling at every step. Measured at
    ~0.58s per coin on 365 rows, so ~1.7s for the three configured coins.

    As an `async def` that work ran ON the event loop, so it blocked every other request on the
    worker for its full duration. Measured: a single forecast request kept `/health` from even
    ENTERING its coroutine for 1004ms. With `-w 4`, a handful of concurrent anonymous requests
    stalls all four workers and the liveness probe cannot answer to say why.

    A plain `def` makes FastAPI run this in a threadpool instead, which is what a synchronous
    handler needs. Note the health route is already mounted at import time so it answers during
    slow STARTUP (see app.py); this preserves the same property at REQUEST time.
    """
    return ORJSONResponse(
        jsonable_encoder(list((fetch_forecast_result()).values())),
        status_code=status.HTTP_200_OK,
    )
