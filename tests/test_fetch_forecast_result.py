"""Tests for ``fetch_forecast_result``: which coins reach the response, and what happens
to the ones that do not.

Nothing covered this function. The gap that mattered is the ``continue``: a coin whose model
cannot be loaded is dropped, and the response looks exactly like one for a fleet that never
had that coin configured. The tests below pin both the omission and the log line that now
explains it.
"""

from logging import WARNING
from pathlib import Path

import pytest
from pandas import DataFrame, date_range, Series

import processing.forecast_data as forecast_data


@pytest.fixture
def coin_list(tmp_path, monkeypatch):
    """Write a coin_list.csv and point the module at it."""

    def _write(records: list) -> Path:
        DataFrame(records).to_csv(tmp_path / "coin_list.csv", index=False)
        monkeypatch.setattr(forecast_data, "DATA_EXTERNAL_PATH", str(tmp_path))
        return tmp_path

    return _write


def _predictions(values: list) -> Series:
    return Series(values, index=date_range("2026-01-01", periods=len(values), freq="D"))


def test_every_configured_coin_with_a_model_appears_at_every_timestamp(
    coin_list, monkeypatch
):
    coin_list(
        [
            {"id": "bitcoin", "symbol": "btc"},
            {"id": "ethereum", "symbol": "eth"},
        ]
    )
    monkeypatch.setattr(forecast_data, "coins", ["bitcoin", "ethereum"])
    monkeypatch.setattr(
        forecast_data, "forecast_coin", lambda symbol: _predictions([1.0, 2.0])
    )

    result = forecast_data.fetch_forecast_result()

    assert len(result) == 2
    for row in result.values():
        assert set(row) == {"time", "bitcoin", "ethereum"}


def test_a_coin_without_a_usable_model_is_omitted_and_the_rest_survive(
    coin_list, monkeypatch
):
    coin_list(
        [
            {"id": "bitcoin", "symbol": "btc"},
            {"id": "ethereum", "symbol": "eth"},
        ]
    )
    monkeypatch.setattr(forecast_data, "coins", ["bitcoin", "ethereum"])
    monkeypatch.setattr(
        forecast_data,
        "forecast_coin",
        lambda symbol: None if symbol == "eth" else _predictions([1.0, 2.0]),
    )

    result = forecast_data.fetch_forecast_result()

    # The contract: one bad coin does not empty the response, and it does not appear as null.
    assert len(result) == 2
    for row in result.values():
        assert "bitcoin" in row
        assert "ethereum" not in row


def test_the_omission_is_logged_rather_than_silent(coin_list, monkeypatch, caplog):
    coin_list([{"id": "bitcoin", "symbol": "btc"}])
    monkeypatch.setattr(forecast_data, "coins", ["bitcoin"])
    monkeypatch.setattr(forecast_data, "forecast_coin", lambda symbol: None)

    with caplog.at_level(WARNING, logger=forecast_data.__name__):
        result = forecast_data.fetch_forecast_result()

    assert result == {}
    assert "bitcoin" in caplog.text
    assert "no usable model" in caplog.text


def test_nothing_is_logged_when_every_coin_forecasts(coin_list, monkeypatch, caplog):
    """The warning must mean something. A line on every healthy call would not."""
    coin_list([{"id": "bitcoin", "symbol": "btc"}])
    monkeypatch.setattr(forecast_data, "coins", ["bitcoin"])
    monkeypatch.setattr(
        forecast_data, "forecast_coin", lambda symbol: _predictions([1.0])
    )

    with caplog.at_level(WARNING, logger=forecast_data.__name__):
        forecast_data.fetch_forecast_result()

    assert "no usable model" not in caplog.text


def test_a_coin_id_narrows_the_work_to_that_coin(coin_list, monkeypatch):
    """The single-coin path must not forecast the others, which is why it exists."""
    coin_list(
        [
            {"id": "bitcoin", "symbol": "btc"},
            {"id": "ethereum", "symbol": "eth"},
        ]
    )
    monkeypatch.setattr(forecast_data, "coins", ["bitcoin", "ethereum"])
    asked = []

    def _forecast(symbol):
        asked.append(symbol)
        return _predictions([1.0])

    monkeypatch.setattr(forecast_data, "forecast_coin", _forecast)

    result = forecast_data.fetch_forecast_result("bitcoin")

    assert asked == ["btc"]
    for row in result.values():
        assert "ethereum" not in row


def test_a_coin_in_the_csv_but_not_configured_is_never_forecast(coin_list, monkeypatch):
    coin_list(
        [
            {"id": "bitcoin", "symbol": "btc"},
            {"id": "dogecoin", "symbol": "doge"},
        ]
    )
    monkeypatch.setattr(forecast_data, "coins", ["bitcoin"])
    asked = []

    def _forecast(symbol):
        asked.append(symbol)
        return _predictions([1.0])

    monkeypatch.setattr(forecast_data, "forecast_coin", _forecast)

    forecast_data.fetch_forecast_result()

    # Not merely absent from the response: never computed. A 30-step recursive forecast is
    # the expensive thing this branch exists to avoid.
    assert asked == ["btc"]
