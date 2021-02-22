from os import path

from apscheduler.schedulers.background import BackgroundScheduler
from pandas import DataFrame
from pycoingecko import CoinGeckoAPI

from definitions import DATA_EXTERNAL_PATH

scheduler = BackgroundScheduler()


@scheduler.scheduled_job(trigger='cron', day=1)
def model_training():
    pass


@scheduler.scheduled_job(trigger='cron', hour=0)
def update_coin_info():
    coin_gecko = CoinGeckoAPI()
    DataFrame(coin_gecko.get_coins_list()).to_csv(path.join(DATA_EXTERNAL_PATH, 'coin_list.csv'), index=False)


def schedule_jobs():
    scheduler.start()
