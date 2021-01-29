from fastapi import FastAPI

from .routers import register_routers


def create_app():
    app = FastAPI()

    register_routers(app)

    # Comment this line to skip training regression models for all available locations during application startup
    # model_training()

    return app
