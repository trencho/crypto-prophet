from os import environ

from fastapi import FastAPI

from definitions import app_dev, app_env, app_prod
from .environment import check_environment_variables, fetch_data
from .routers import register_routers
from .schedule import model_training, schedule_jobs


def create_app():
    if environ.get(app_env, app_dev) == app_prod:
        check_environment_variables()
        schedule_jobs()

    app = FastAPI()

    register_routers(app)

    fetch_data()

    return app
