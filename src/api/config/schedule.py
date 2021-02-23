from base64 import b64encode
from datetime import datetime
from os import environ, path, walk

from apscheduler.schedulers.background import BackgroundScheduler
from pandas import DataFrame
from pandas import read_csv
from pycoingecko import CoinGeckoAPI

from definitions import DATA_EXTERNAL_PATH, ROOT_DIR, app_name_env
from .git import append_commit_files, merge_csv_files, update_git_files

scheduler = BackgroundScheduler()


@scheduler.scheduled_job(trigger='cron', day=1)
def data_dump():
    repo_name = environ[app_name_env]

    file_list = []
    file_names = []
    for root, directories, files in walk(ROOT_DIR):
        for file in files:
            file_path = path.join(root, file)
            if file.endswith('.csv'):
                data = read_csv(file_path).to_csv(index=False)
                data = merge_csv_files(repo_name, file_path, data)
                append_commit_files(file_list, file_names, root, data, file)
            elif file.endswith('.png'):
                with open(file_path, 'rb') as input_file:
                    data = b64encode(input_file.read())
                append_commit_files(file_list, file_names, root, data, file)

    if file_list:
        branch = 'master'
        commit_message = f'Scheduled data dump - {datetime.now().strftime("%H:%M:%S %d-%m-%Y")}'
        update_git_files(file_names, file_list, repo_name, branch, commit_message)


@scheduler.scheduled_job(trigger='cron', day=1)
def model_training():
    pass


@scheduler.scheduled_job(trigger='cron', hour=0)
def update_coin_info():
    coin_gecko = CoinGeckoAPI()
    DataFrame(coin_gecko.get_coins_list()).to_csv(path.join(DATA_EXTERNAL_PATH, 'coin_list.csv'), index=False)


def schedule_jobs():
    scheduler.start()
