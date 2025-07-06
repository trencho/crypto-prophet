from asyncio import sleep
from base64 import b64encode
from datetime import datetime
from os import environ, path, remove, walk
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from pandas import DataFrame, json_normalize, read_csv
from pycoingecko import CoinGeckoAPI

from definitions import coins, DATA_EXTERNAL_PATH, DATA_PATH, MODELS_PATH, repo_name
from modeling import train_regression_model
from preparation import trim_dataframe
from .git import append_commit_files, create_archive, update_git_files

scheduler = BackgroundScheduler()


@scheduler.scheduled_job(trigger="cron", day="*/15")
async def dump_data() -> None:
    file_list, file_names = [], []
    for root, directories, files in walk(DATA_PATH):
        if not directories and files:
            file_path = f"{root}.zip"
            create_archive(source=root, destination=file_path)
            with open(file_path, "rb") as in_file:
                data = b64encode(in_file.read())
            append_commit_files(
                file_list,
                data,
                Path(root).resolve().parent,
                path.basename(file_path),
                file_names,
            )
            remove(file_path)

    if file_list:
        branch = "master"
        commit_message = (
            f"Scheduled data dump - {datetime.now().strftime('%H:%M:%S %d-%m-%Y')}"
        )
        await update_git_files(
            file_list, file_names, environ[repo_name], branch, commit_message
        )


@scheduler.scheduled_job(trigger="cron", minute="*/15")
async def model_training() -> None:
    for file in [
        Path(root) / file
        for root, directories, files in walk(MODELS_PATH)
        for file in files
        if file.endswith(".lock")
    ]:
        remove(Path(MODELS_PATH) / file)

    coin_list = read_csv(Path(DATA_EXTERNAL_PATH) / "coin_list.csv").to_dict("records")
    for coin in coin_list:
        if coin["id"] in coins:
            await train_regression_model(coin)


@scheduler.scheduled_job(trigger="cron", hour=0)
async def update_coin_info() -> None:
    coin_gecko = CoinGeckoAPI()
    coin_list = coin_gecko.get_coins_list()
    await sleep(1)
    json_normalize(coin_list).to_csv(
        Path(DATA_EXTERNAL_PATH) / "coin_list.csv", index=False
    )
    for coin in coin_list:
        if coin["id"] not in coins:
            continue

        updated_coin_data = coin_gecko.get_coin_market_chart_by_id(coin["id"], "usd", 1)
        updated_coin_dataframe = DataFrame(
            updated_coin_data["prices"], columns=["time", "value"]
        )
        updated_coin_dataframe = trim_dataframe(updated_coin_dataframe, "time")
        coin_dataframe = read_csv(
            Path(DATA_EXTERNAL_PATH) / coin["symbol"] / "data.csv"
        )
        last_timestamp = coin_dataframe["time"].iloc[-1]
        updated_coin_dataframe = updated_coin_dataframe.drop(
            index=updated_coin_dataframe.loc[
                updated_coin_dataframe["time"] < last_timestamp
            ].index,
            errors="ignore",
        )
        coin_dataframe.append(updated_coin_dataframe, ignore_index=True).to_csv(
            Path(DATA_EXTERNAL_PATH) / coin["symbol"] / "data.csv", index=False
        )

        await sleep(1)


def schedule_jobs() -> None:
    scheduler.start()
