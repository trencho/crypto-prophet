from os import environ

from fastapi import FastAPI

from definitions import app_dev, app_env, app_prod
from .environment import check_environment_variables, fetch_data, init_system_paths
from .logger import configure_logger
from .routers import register_routers
from .schedule import model_training, schedule_jobs

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    await init_system_paths()

    if environ.get(app_env, app_dev) == app_prod:
        await check_environment_variables()
        await schedule_jobs()

    await configure_logger()
    await register_routers(app)

    await fetch_data()
