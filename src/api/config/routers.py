from api.routers.coins import coins_router
from api.routers.forecast import forecast_router

__all__ = ["coins_router", "forecast_router"]


def register_routers(app) -> None:
    for router in __all__:
        app.include_router(globals()[router], prefix="/api/v1")
