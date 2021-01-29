__all__ = []


def register_routers(app):
    for router in __all__:
        app.include_router(globals()[router], prefix='/api/v1')
