from datetime import datetime
from os import environ, makedirs, path
from time import sleep

from pandas import DataFrame, json_normalize
from pycoingecko import CoinGeckoAPI

from definitions import DATA_EXTERNAL_PATH, coins, environment_variables
from preparation import trim_dataframe
from processing.normalize_data import current_hour

collections = ['summary', 'pollution', 'weather']


def check_environment_variables():
    for environment_variable in environment_variables:
        if environ.get(environment_variable) is None:
            print(f'The environment variable "{environment_variable}" is missing')
            exit(-1)


def fetch_data():
    coin_gecko = CoinGeckoAPI()
    coin_list = coin_gecko.get_coins_list()
    sleep(1)
    json_normalize(coin_list).to_csv(path.join(DATA_EXTERNAL_PATH, 'coin_list.csv'), index=False)
    for coin in coin_list:
        if coin['id'] not in coins:
            continue

        current_datetime = datetime.now()
        coin_data = coin_gecko.get_coin_market_chart_range_by_id(
            coin['id'], 'usd', current_hour(current_datetime).replace(year=current_datetime.year - 5).timestamp(),
            current_hour(current_datetime).timestamp())
        coin_dataframe = DataFrame(coin_data['prices'], columns=['time', 'value'])
        trim_dataframe(coin_dataframe, 'time')
        makedirs(path.join(DATA_EXTERNAL_PATH, coin['symbol']), exist_ok=True)
        coin_dataframe.to_csv(path.join(DATA_EXTERNAL_PATH, coin['symbol'], 'data.csv'), index=False)

        sleep(1)
