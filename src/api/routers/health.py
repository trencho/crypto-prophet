from fastapi import APIRouter, status
from fastapi.responses import ORJSONResponse

health_router = APIRouter(tags=["health"])


@health_router.get("/health")
async def health() -> ORJSONResponse:
    """Liveness probe: is the process up and serving?

    Deliberately does no I/O -- no CoinGecko call, no model load, no disk read. A health
    check that depends on a downstream service reports that service's outage as this
    process being unhealthy, which is how a restart loop starts.
    """
    return ORJSONResponse({"status": "ok"}, status_code=status.HTTP_200_OK)


@health_router.get("/")
async def root() -> ORJSONResponse:
    """Service descriptor, so hitting the bare host answers something other than 404."""
    return ORJSONResponse(
        {"service": "crypto-prophet", "docs": "/docs", "api": "/api/v1"},
        status_code=status.HTTP_200_OK,
    )
