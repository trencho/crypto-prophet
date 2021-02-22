from fastapi import FastAPI

from .routers import register_routers
from .schedule import model_training, schedule_jobs


def create_app():
    # Comment this line to skip the scheduling of api operations when running application in debug mode
    schedule_jobs()

    app = FastAPI()

    register_routers(app)

    # Comment this line to skip training regression models for all available locations during application startup
    model_training()

    return app
