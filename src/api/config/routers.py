from api.routers.forecast import forecast_router


def register_routers(app):
    app.include_router(forecast_router, prefix='/api/v1')
