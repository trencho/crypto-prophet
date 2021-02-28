from fastapi import FastAPI

from .environment import check_environment_variables
from .routers import register_routers
from .schedule import fetch_coin_info, model_training, schedule_jobs


def create_app():
    # Comment these 2 lines to skip the environment variable check and scheduling of api operations when running
    # application in debug mode
    check_environment_variables()
    schedule_jobs()

    app = FastAPI()

    register_routers(app)

    fetch_coin_info()

    # Comment this line to skip training regression models for all available locations during application startup
    model_training()

    return app
