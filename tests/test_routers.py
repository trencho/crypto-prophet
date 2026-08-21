"""Tests for the FastAPI routers, driven through TestClient.

Each router is mounted on a fresh FastAPI app; the data-access dependency the
router pulled in at import time (``check_coin`` / ``fetch_forecast_result``) is
monkeypatched so nothing touches the network or the filesystem.
"""

import asyncio
from asyncio import Event
from logging import ERROR

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.app as app_module
import api.routers.coins as coins_module
import api.routers.forecast as forecast_module
import api.routers.health as health_module


@pytest.fixture
def coins_client():
    app = FastAPI()
    app.include_router(coins_module.coins_router)
    return TestClient(app)


@pytest.fixture
def forecast_client():
    app = FastAPI()
    app.include_router(forecast_module.forecast_router)
    return TestClient(app)


def test_fetch_coin_404_when_missing(monkeypatch, coins_client):
    async def fake_check_coin(coin_id):
        return None

    monkeypatch.setattr(coins_module, "check_coin", fake_check_coin)

    response = coins_client.get("/coins/unknown-coin/")
    assert response.status_code == 404
    assert "error_message" in response.json()


def test_fetch_coin_200_when_found(monkeypatch, coins_client):
    coin = {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"}

    async def fake_check_coin(coin_id):
        return coin

    monkeypatch.setattr(coins_module, "check_coin", fake_check_coin)

    response = coins_client.get("/coins/bitcoin/")
    assert response.status_code == 200
    assert response.json() == coin


def test_forecast_endpoint_returns_list(monkeypatch, forecast_client):
    forecast_result = {
        1700000000: {"time": 1700000000, "bitcoin": 42000.0},
        1700086400: {"time": 1700086400, "bitcoin": 42500.0},
    }

    def fake_fetch_forecast_result():
        return forecast_result

    monkeypatch.setattr(
        forecast_module, "fetch_forecast_result", fake_fetch_forecast_result
    )

    response = forecast_client.get("/forecast/")
    assert response.status_code == 200
    assert response.json() == list(forecast_result.values())


def test_health_reports_ok():
    """The liveness probe answers without any dependency being available."""
    app = FastAPI()
    app.include_router(health_module.health_router)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_answers_without_lifespan_having_run():
    """Regression guard for the reason this route is mounted where it is.

    The app registers its other routers inside `lifespan`, which then blocks on
    `await fetch_data()`. A health route mounted there is unreachable for the whole of a
    slow startup -- the one moment a probe is actually asking. So it must be attached to
    the app object at import time.

    Asserted by BEHAVIOUR, not by introspecting `app.routes`: FastAPI 0.140 stopped
    flattening included routers into that list (it holds a single `_IncludedRouter`), so a
    structural check silently stops meaning anything. TestClient only runs lifespan when
    used as a context manager, so a plain get() here proves exactly the property claimed --
    the endpoint answers with no startup having happened.
    """
    response = TestClient(app_module.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_describes_the_service():
    app = FastAPI()
    app.include_router(health_module.health_router)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "crypto-prophet"


def test_startup_does_not_wait_for_the_data_warm_up(monkeypatch):
    """Startup must complete while the CoinGecko pull is still running.

    `fetch_data` walks the whole coin list with a one-second sleep per coin. Awaiting it in
    `lifespan` gated readiness on that pull, so the app was unreachable for minutes at exactly
    the moment a probe asks. Asserted by BEHAVIOUR: a warm-up that never finishes must not stop
    lifespan from completing, which is the property that regressed before.

    A wall-clock assertion would only prove "fast", and fast is what a network call is on a good
    day. Blocking on an event that is never set is the whole failure and it cannot pass by luck.
    """
    started = Event()
    release = Event()

    async def never_finishes():
        started.set()
        await release.wait()

    monkeypatch.setattr(app_module, "fetch_data", never_finishes)
    monkeypatch.setattr(app_module, "register_routers", lambda app: None)
    monkeypatch.setattr(app_module, "configure_gc", lambda: None)
    monkeypatch.setattr(app_module, "init_system_paths", lambda: None)
    monkeypatch.setattr(app_module, "configure_logger", lambda: None)

    with TestClient(app_module.app) as client:
        # Reached only because lifespan returned without awaiting the warm-up.
        assert client.get("/health").status_code == 200
        assert started.is_set(), "the warm-up should have been started, not skipped"

    release.set()


def test_a_failing_warm_up_is_logged_and_does_not_break_startup(monkeypatch, caplog):
    """A background task has no caller to notice it raised, so it logs or it is invisible.

    Without this the app would serve requests over an empty cache and report itself healthy --
    worse than the blocking startup it replaced, because nothing anywhere would say why.
    """

    async def explodes():
        raise RuntimeError("coingecko is down")

    monkeypatch.setattr(app_module, "fetch_data", explodes)

    with caplog.at_level(ERROR):
        asyncio.run(app_module._warm_up_data())

    assert "warm-up failed" in caplog.text
    assert "coingecko is down" in caplog.text
