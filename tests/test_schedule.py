"""Tests for the scheduled jobs (`api.config.schedule`).

These jobs run unattended, so a defect in one is invisible until data goes missing. Nothing here
starts a scheduler or touches the network: the job coroutines are awaited directly and their
collaborators are replaced, which is what makes the guards observable at all.
"""

import asyncio
import os
import time
from pathlib import Path

from pandas import DataFrame, read_csv

from api.config import schedule as schedule_module


def test_dump_data_does_not_commit_when_there_is_nothing_to_dump(monkeypatch, tmp_path):
    """The `if file_list:` guard.

    Without it every 15-day run would open an empty commit against the data repo. Asserted on the
    collaborator, because "committed nothing" and "did not commit" look identical from outside.
    """
    calls = []

    async def record(*args, **kwargs):
        calls.append(args)

    monkeypatch.setattr(schedule_module, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(schedule_module, "update_git_files", record)

    asyncio.run(schedule_module.dump_data())

    assert calls == [], "an empty data tree must not produce a commit"


def test_dump_data_commits_the_archives_it_built(monkeypatch, tmp_path):
    """The positive case, so the guard above cannot pass by the job doing nothing at all."""
    leaf = tmp_path / "bitcoin"
    leaf.mkdir()
    (leaf / "data.csv").write_text("time,value\n1,2\n", encoding="utf-8")

    calls = []

    async def record(file_list, file_names, repo, branch, message):
        calls.append((len(file_list), message))

    monkeypatch.setattr(schedule_module, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(schedule_module, "update_git_files", record)
    monkeypatch.setenv("REPO_NAME", "some-repo")
    monkeypatch.setattr(schedule_module, "repo_name", "REPO_NAME")

    asyncio.run(schedule_module.dump_data())

    assert len(calls) == 1
    assert calls[0][0] == 1, "the one leaf directory should have produced one archive"
    assert calls[0][1].startswith("Scheduled data dump")


def test_model_training_clears_stale_locks_and_trains_only_configured_coins(
    monkeypatch, tmp_path
):
    """Three guarantees in one job, and all three are silent when broken.

    A STALE `.lock` left behind blocks the next training run forever. A FRESH one must survive:
    it belongs to a run that is still going, and this job used to delete every lock it found,
    which destroyed the only cross-process interlock the design has. With `gunicorn -w 4` there
    are four schedulers, so one worker's tick was removing another worker's in-flight lock.
    Third: training a coin outside `coins` burns a slot on data the app never serves.
    """
    models = tmp_path / "models"
    models.mkdir()
    stale = models / "bitcoin.lock"
    stale.write_text("", encoding="utf-8")
    # Backdate it past the reap window. Age is the only thing that distinguishes a crashed run
    # from a live one, so it is the only thing the job may act on.
    old = time.time() - schedule_module._STALE_LOCK_SECONDS - 60
    os.utime(stale, (old, old))

    fresh = models / "ethereum.lock"
    fresh.write_text("", encoding="utf-8")

    (models / "keep.txt").write_text("", encoding="utf-8")

    external = tmp_path / "external"
    external.mkdir()
    DataFrame(
        [{"id": "bitcoin", "symbol": "btc"}, {"id": "not-configured", "symbol": "xxx"}]
    ).to_csv(external / "coin_list.csv", index=False)

    trained = []

    async def record(coin):
        trained.append(coin["id"])

    monkeypatch.setattr(schedule_module, "MODELS_PATH", str(models))
    monkeypatch.setattr(schedule_module, "DATA_EXTERNAL_PATH", str(external))
    monkeypatch.setattr(schedule_module, "coins", ["bitcoin"])
    monkeypatch.setattr(schedule_module, "train_regression_model", record)

    asyncio.run(schedule_module.model_training())

    assert not stale.exists(), "the stale lock was not cleared"
    assert fresh.exists(), (
        "a FRESH lock was deleted: that lock belongs to a run still in progress, and removing it "
        "lets a second worker train the same coin against the same files"
    )
    assert (models / "keep.txt").exists(), "a non-lock file must be left alone"
    assert trained == ["bitcoin"], f"trained the wrong set: {trained}"


def test_update_coin_info_appends_only_rows_newer_than_the_last_stored(
    monkeypatch, tmp_path
):
    """The dedup that keeps the series from growing a duplicated tail every midnight.

    CoinGecko returns a rolling window that overlaps what is already on disk, so without the
    `time < last_timestamp` drop each run re-appends hours it already has.
    """
    external = tmp_path / "external"
    (external / "btc").mkdir(parents=True)
    DataFrame({"time": [100, 200], "value": [1.0, 2.0]}).to_csv(
        external / "btc" / "data.csv", index=False
    )

    class FakeCoinGecko:
        def get_coins_list(self):
            return [{"id": "bitcoin", "symbol": "btc"}]

        def get_coin_market_chart_by_id(self, _id, _cur, _days):
            # 100 and 200 are already stored; only 300 is new.
            return {"prices": [[100, 1.0], [200, 2.0], [300, 3.0]]}

    monkeypatch.setattr(schedule_module, "DATA_EXTERNAL_PATH", str(external))
    monkeypatch.setattr(schedule_module, "coins", ["bitcoin"])
    monkeypatch.setattr(schedule_module, "CoinGeckoAPI", FakeCoinGecko)
    monkeypatch.setattr(schedule_module, "trim_dataframe", lambda df, _col: df)
    monkeypatch.setattr(schedule_module, "sleep", _no_sleep)

    asyncio.run(schedule_module.update_coin_info())

    result = read_csv(Path(external) / "btc" / "data.csv")
    assert list(result["time"]) == [
        100,
        200,
        300,
    ], f"expected the stored rows plus only the new one, got {list(result['time'])}"


async def _no_sleep(_seconds):
    return None
