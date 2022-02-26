from base64 import b64encode
from datetime import datetime
from os import environ, path, remove, walk
from time import sleep

from apscheduler.schedulers.background import BackgroundScheduler
from pandas import DataFrame, json_normalize, read_csv
from pycoingecko import CoinGeckoAPI

from definitions import coins, DATA_EXTERNAL_PATH, DATA_PATH, MODELS_PATH, repo_name
from modeling import train_regression_model
from preparation import trim_dataframe
from .git import append_commit_files, create_archive, update_git_files

scheduler = BackgroundScheduler()


@scheduler.scheduled_job(trigger='cron', day='*/15')
def dump_data() -> None:
    print('Started dumping data...')

    file_list, file_names = [], []
    for root, directories, files in walk(DATA_PATH):
        if not directories and files:
            file_path = f'{root}.zip'
            create_archive(source=root, destination=file_path)
            with open(file_path, 'rb') as in_file:
                data = b64encode(in_file.read())
            append_commit_files(file_list, data, path.dirname(path.abspath(root)), path.basename(file_path), file_names)
            remove(file_path)

    if file_list:
        branch = 'master'
        commit_message = f'Scheduled data dump - {datetime.now().strftime("%H:%M:%S %d-%m-%Y")}'
        update_git_files(file_list, file_names, environ[repo_name], branch, commit_message)

    print('Finished dumping data!')


@scheduler.scheduled_job(trigger='cron', minute='*/15')
def model_training() -> None:
    print('Started training regression models...')

    for file in [path.join(root, file) for root, directories, files in walk(MODELS_PATH) for file in files if
                 file.endswith('.lock')]:
        remove(path.join(MODELS_PATH, file))

    coin_list = read_csv(path.join(DATA_EXTERNAL_PATH, 'coin_list.csv')).to_dict('records')
    for coin in coin_list:
        if coin['id'] in coins:
            train_regression_model(coin)

    print('Finished training regression models!')


@scheduler.scheduled_job(trigger='cron', hour=0)
def update_coin_info() -> None:
    print('Started fetching coins...')

    coin_gecko = CoinGeckoAPI()
    coin_list = coin_gecko.get_coins_list()
    sleep(1)
    json_normalize(coin_list).to_csv(path.join(DATA_EXTERNAL_PATH, 'coin_list.csv'), index=False)
    for coin in coin_list:
        if coin['id'] not in coins:
            continue

        updated_coin_data = coin_gecko.get_coin_market_chart_by_id(coin['id'], 'usd', 1)
        updated_coin_dataframe = DataFrame(updated_coin_data['prices'], columns=['time', 'value'])
        trim_dataframe(updated_coin_dataframe, 'time')
        coin_dataframe = read_csv(path.join(DATA_EXTERNAL_PATH, coin['symbol'], 'data.csv'))
        last_timestamp = coin_dataframe['time'].iloc[-1]
        updated_coin_dataframe.drop(
            index=updated_coin_dataframe.loc[updated_coin_dataframe['time'] < last_timestamp].index, inplace=True,
            errors='ignore')
        coin_dataframe.append(updated_coin_dataframe, ignore_index=True).to_csv(
            path.join(DATA_EXTERNAL_PATH, coin['symbol'], 'data.csv'), index=False)

        sleep(1)

    print('Finished fetching coins!')


def schedule_jobs() -> None:
    scheduler.start()
