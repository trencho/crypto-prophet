from os import path

from pandas import read_csv

from definitions import DATA_EXTERNAL_PATH


def check_coin(coin_id):
    coin_list = read_csv(path.join(DATA_EXTERNAL_PATH, 'coin_list.csv')).to_dict('records')
    for coin in coin_list:
        if coin['id'] == coin_id:
            return coin

    return None
