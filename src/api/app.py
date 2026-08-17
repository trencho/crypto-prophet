from contextlib import asynccontextmanager
from os import environ

from fastapi import FastAPI
from uvicorn import run

from api.config import (
    check_environment_variables,
    configure_gc,
    configure_logger,
    fetch_data,
    init_system_paths,
    register_routers,
    schedule_jobs,
)
from api.routers.health import health_router
from definitions import app_dev, app_env, app_prod


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_gc()
    init_system_paths()

    if environ.get(app_env, app_dev) == app_prod:
        check_environment_variables()
        schedule_jobs()

    configure_logger()
    register_routers(app)

    await fetch_data()

    yield


app = FastAPI(lifespan=lifespan)

# Mounted at import time, NOT inside lifespan with the others, and deliberately unprefixed.
# register_routers() runs during startup and is followed by `await fetch_data()`, which blocks
# until a network pull finishes -- so a health route registered there would be unavailable for
# exactly as long as startup is slow, which is precisely when a probe is asked the question.
app.include_router(health_router)
if __name__ == "__main__":
    run("app:app", host="0.0.0.0", reload=True, reload_dirs="..")
