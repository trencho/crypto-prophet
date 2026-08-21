from asyncio import CancelledError, create_task
from contextlib import asynccontextmanager
from logging import getLogger
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

    # Started, not awaited. `fetch_data` walks the whole coin list against CoinGecko with a
    # one-second courtesy sleep per coin, so awaiting it here gated readiness on a network pull
    # that takes minutes -- and a probe asking "are you up?" during exactly that window got no
    # answer. The task is kept on a local name because asyncio only holds a weak reference to a
    # running task, and a garbage-collected one cancels itself mid-fetch.
    warmup = create_task(_warm_up_data())

    try:
        yield
    finally:
        # A shutdown mid-fetch is normal, not a failure: cancel and let it go.
        warmup.cancel()


async def _warm_up_data() -> None:
    """Populate the external data cache in the background.

    Swallows nothing silently. A background task that raises has no caller to notice, so the
    failure is logged here or it is invisible -- which is worse than the blocking startup this
    replaced, because the app would then serve requests over an empty cache and look healthy.
    """
    try:
        await fetch_data()
    except CancelledError:
        raise
    except Exception:
        getLogger(__name__).exception(
            "startup data warm-up failed; serving without a warm cache"
        )


app = FastAPI(lifespan=lifespan)

# Mounted at import time, NOT inside lifespan with the others, and deliberately unprefixed.
# register_routers() runs during startup, so a health route registered there is unavailable until
# lifespan completes -- and it is asked precisely when startup is slow. The warm-up no longer
# blocks startup, but the ordering guarantee is still the health route's, not a side effect of it.
app.include_router(health_router)
if __name__ == "__main__":
    run("app:app", host="0.0.0.0", reload=True, reload_dirs="..")
