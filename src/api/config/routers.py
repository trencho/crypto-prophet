from api.routers.forecast import forecast_router

__all__ = [
    'forecast_router',
]


def register_routers(app):
    for router in __all__:
        app.include_router(globals()[router], prefix='/api/v1')
