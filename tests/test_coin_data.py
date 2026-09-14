"""Tests for ``check_coin`` itself.

``test_routers.py`` monkeypatches ``check_coin`` away on both router tests, which is right
for those tests and leaves the real function executed by nothing: replacing its body with
``return None`` kept the whole suite green. These drive the real lookup against a real CSV.
"""

import asyncio
from pathlib import Path

import pytest
from pandas import DataFrame

import preparation.coin_data as coin_data


@pytest.fixture
def coin_list(tmp_path, monkeypatch) -> Path:
    """Point ``check_coin`` at a throwaway ``coin_list.csv`` and return its directory."""

    def _write(records: list) -> Path:
        DataFrame(records).to_csv(tmp_path / "coin_list.csv", index=False)
        monkeypatch.setattr(coin_data, "DATA_EXTERNAL_PATH", str(tmp_path))
        return tmp_path

    return _write


def test_a_known_id_returns_that_coins_record(coin_list):
    coin_list(
        [
            {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
            {"id": "ethereum", "symbol": "eth", "name": "Ethereum"},
        ]
    )

    coin = asyncio.run(coin_data.check_coin("ethereum"))

    # The whole record, not just a truthy hit: the router encodes this straight into the body.
    assert coin["id"] == "ethereum"
    assert coin["symbol"] == "eth"
    assert coin["name"] == "Ethereum"


def test_an_unknown_id_returns_none(coin_list):
    coin_list([{"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"}])

    assert asyncio.run(coin_data.check_coin("dogecoin")) is None


def test_the_match_is_on_id_and_not_on_symbol_or_name(coin_list):
    """`btc` is a symbol, never an id. Matching on the wrong column would 200 here."""
    coin_list([{"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"}])

    assert asyncio.run(coin_data.check_coin("btc")) is None
    assert asyncio.run(coin_data.check_coin("Bitcoin")) is None


def test_an_empty_list_returns_none_rather_than_raising(coin_list):
    coin_list([{"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"}])
    DataFrame(columns=["id", "symbol", "name"]).to_csv(
        Path(coin_data.DATA_EXTERNAL_PATH) / "coin_list.csv", index=False
    )

    assert asyncio.run(coin_data.check_coin("bitcoin")) is None


def test_the_first_match_wins_when_an_id_repeats(coin_list):
    """Documents the contract the router depends on: one record back, not a list."""
    coin_list(
        [
            {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
            {"id": "bitcoin", "symbol": "xbt", "name": "Bitcoin (duplicate)"},
        ]
    )

    assert asyncio.run(coin_data.check_coin("bitcoin"))["symbol"] == "btc"
