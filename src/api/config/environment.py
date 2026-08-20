from asyncio import sleep
from datetime import datetime
from os import environ, makedirs, path

from pandas import DataFrame, json_normalize
from pycoingecko import CoinGeckoAPI

from definitions import (
    DATA_EXTERNAL_PATH,
    DATA_PROCESSED_PATH,
    DATA_RAW_PATH,
    coins,
    environment_variables,
    LOG_PATH,
    MODELS_PATH,
    RESULTS_ERRORS_PATH,
    RESULTS_PREDICTIONS_PATH,
)
from preparation import trim_dataframe
from processing.normalize_data import current_hour

system_paths = [
    DATA_EXTERNAL_PATH,
    DATA_PROCESSED_PATH,
    DATA_RAW_PATH,
    LOG_PATH,
    MODELS_PATH,
    RESULTS_ERRORS_PATH,
    RESULTS_PREDICTIONS_PATH,
]


def check_environment_variables() -> None:
    """Raise when a required environment variable is unset.

    Raises rather than calling ``exit``: this runs inside the ASGI lifespan, where a
    ``SystemExit`` unwinds through the server's startup machinery instead of being reported as
    the configuration error it is. ``exit`` is also a ``site`` builtin and simply does not
    exist under ``python -S``. Reporting every missing variable at once beats failing on the
    first, so a misconfigured deployment needs one restart rather than one per variable.
    """
    missing = [name for name in environment_variables if environ.get(name) is None]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )


async def fetch_data() -> None:
    coin_gecko = CoinGeckoAPI()
    coin_list = coin_gecko.get_coins_list()
    await sleep(1)
    json_normalize(coin_list).to_csv(
        path.join(DATA_EXTERNAL_PATH, "coin_list.csv"), index=False
    )
    for coin in coin_list:
        if coin["id"] not in coins:
            continue

        current_datetime = datetime.now()
        coin_data = coin_gecko.get_coin_market_chart_range_by_id(
            coin["id"],
            "usd",
            current_hour().replace(year=current_datetime.year - 1).timestamp(),
            current_hour().timestamp(),
        )
        dataframe = DataFrame(coin_data["prices"], columns=["time", "value"])
        dataframe = trim_dataframe(dataframe, "time")
        makedirs(path.join(DATA_EXTERNAL_PATH, coin["symbol"]), exist_ok=True)
        dataframe.to_csv(
            path.join(DATA_EXTERNAL_PATH, coin["symbol"], "data.csv"), index=False
        )

        await sleep(1)


def init_system_paths() -> None:
    for system_path in system_paths:
        makedirs(system_path, exist_ok=True)
