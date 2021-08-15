from base64 import b64encode
from datetime import datetime
from os import environ, walk
from os.path import join as path_join
from time import sleep

from apscheduler.schedulers.background import BackgroundScheduler
from pandas import DataFrame, json_normalize, read_csv
from pycoingecko import CoinGeckoAPI

from definitions import coins, DATA_EXTERNAL_PATH, repo_name, ROOT_PATH
from modeling import train_regression_model
from preparation import trim_dataframe
from .git import append_commit_files, merge_csv_files, update_git_files

scheduler = BackgroundScheduler()


@scheduler.scheduled_job(trigger='cron', day=1)
def data_dump():
    repository_name = environ[repo_name]

    file_list = []
    file_names = []
    for root, directories, files in walk(ROOT_PATH):
        for file in files:
            file_path = path_join(root, file)
            if file.endswith('.csv'):
                data = read_csv(file_path).to_csv(index=False)
                data = merge_csv_files(repository_name, file_path, data)
                append_commit_files(file_list, file_names, root, data, file)
            elif file.endswith('.png'):
                with open(file_path, 'rb') as in_file:
                    data = b64encode(in_file.read())
                append_commit_files(file_list, file_names, root, data, file)

    if file_list:
        branch = 'master'
        commit_message = f'Scheduled data dump - {datetime.now().strftime("%H:%M:%S %d-%m-%Y")}'
        update_git_files(file_names, file_list, repository_name, branch, commit_message)


@scheduler.scheduled_job(trigger='cron', day=2)
def model_training():
    coin_list = read_csv(path_join(DATA_EXTERNAL_PATH, 'coin_list.csv')).to_dict('records')
    for coin in coin_list:
        if coin['id'] in coins:
            train_regression_model(coin)


@scheduler.scheduled_job(trigger='cron', hour=0)
def update_coin_info():
    coin_gecko = CoinGeckoAPI()
    coin_list = coin_gecko.get_coins_list()
    sleep(1)
    json_normalize(coin_list).to_csv(path_join(DATA_EXTERNAL_PATH, 'coin_list.csv'), index=False)
    for coin in coin_list:
        if coin['id'] not in coins:
            continue

        updated_coin_data = coin_gecko.get_coin_market_chart_by_id(coin['id'], 'usd', 1)
        updated_coin_dataframe = DataFrame(updated_coin_data['prices'], columns=['time', 'value'])
        trim_dataframe(updated_coin_dataframe, 'time')
        coin_dataframe = read_csv(path_join(DATA_EXTERNAL_PATH, coin['symbol'], 'data.csv'))
        last_timestamp = coin_dataframe['time'].iloc[-1]
        updated_coin_dataframe.drop(
            index=updated_coin_dataframe.loc[updated_coin_dataframe['time'] < last_timestamp].index, inplace=True,
            errors='ignore')
        coin_dataframe.append(updated_coin_dataframe, ignore_index=True).to_csv(
            path_join(DATA_EXTERNAL_PATH, coin['symbol'], 'data.csv'), index=False)

        sleep(1)


def schedule_jobs():
    scheduler.start()
