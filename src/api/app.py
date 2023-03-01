from os import environ

from fastapi import FastAPI
from uvicorn import run

from api.config import check_environment_variables, configure_logger, fetch_data, init_system_paths, register_routers, \
    schedule_jobs
from definitions import app_dev, app_env, app_prod

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


if __name__ == '__main__':
    run('app:app', host='0.0.0.0', reload=True)
