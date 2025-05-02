from datetime import datetime
from os import environ, makedirs, path
from time import sleep

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


async def check_environment_variables() -> None:
    for environment_variable in environment_variables:
        if environ.get(environment_variable) is None:
            print(f'The environment variable "{environment_variable}" is missing')
            exit(-1)


async def fetch_data() -> None:
    coin_gecko = CoinGeckoAPI()
    coin_list = coin_gecko.get_coins_list()
    sleep(1)
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
        await trim_dataframe(dataframe, "time")
        makedirs(path.join(DATA_EXTERNAL_PATH, coin["symbol"]), exist_ok=True)
        dataframe.to_csv(
            path.join(DATA_EXTERNAL_PATH, coin["symbol"], "data.csv"), index=False
        )

        sleep(1)


async def init_system_paths() -> None:
    for system_path in system_paths:
        makedirs(system_path, exist_ok=True)
