# crypto-prophet

A FastAPI service that forecasts cryptocurrency prices. It ingests price history from CoinGecko,
trains a bench of regression models per coin on a schedule, keeps the best model (lowest MAE), and
serves a 30-day price forecast over a small REST API.

## How it works

- **Ingest** — `pycoingecko` pulls the coin list and hourly price history for the configured coins
  (`bitcoin`, `ethereum`, `ravencoin` — see `definitions.py`) into `data/`. Runs on startup and on a
  daily cron.
- **Train** (cron-driven, not on request) — for each coin: feature generation → train/test split →
  scaling + backward-elimination feature selection → `RandomizedSearchCV` over each model → the
  lowest-MAE model is retrained on the full data and pickled under `models/`. A model younger than one
  month is reused rather than retrained.
- **Models** — Decision Tree, LightGBM, Linear, MLP, Support Vector, XGBoost (Random Forest is present
  in the registry but currently disabled).
- **Forecast** — `GET /api/v1/forecast/` runs a 30-step recursive prediction (each day's prediction
  feeds the next) and returns the merged series. Returns empty until a training run has produced models.
- **Scheduling** — APScheduler runs the training, daily coin refresh, and an optional periodic dump of
  `data/` to a GitHub repo (for off-box persistence).

## API

All routes are under `/api/v1`:

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/coins/` | list all known coins |
| GET | `/api/v1/coins/{coin_id}/` | one coin (404 if unknown) |
| GET | `/api/v1/forecast/` | 30-day forecast points across the configured coins |

## Requirements

- Python **3.14**
- `pip install -r requirements.txt`

## Configuration (environment variables)

| Variable | Required | Purpose |
|---|---|---|
| `APP_ENV` | no (default `development`) | `production` enables the scheduler + enforces the vars below; `development` skips both |
| `GITHUB_TOKEN` | prod only | PyGithub auth for the scheduled data dump |
| `REPO_NAME` | prod only | target repo for the data dump |
| `VOLUME_PATH` | no | overrides the root path for all `data/`, `models/`, `results/`, `logs/` |

Env vars are read from the process environment (there is no `.env` auto-loading).

## Run

**Local (development):**

```sh
pip install -r requirements.txt
uvicorn main:app --reload      # APP_ENV defaults to "development"
```

The first startup pulls price history from CoinGecko (needs network). Forecasts are empty until a
training run has produced models.

**Docker:**

```sh
docker compose -f docker/docker-compose.dev.yml up --build
```

Serves via gunicorn + uvicorn workers behind traefik.

## Development

- **Tests:** `pytest -q` (suite under `tests/`; data paths are redirected to a temp dir via `VOLUME_PATH`).
- **Format:** `black .` (checked in CI).
- **CI** (`.github/workflows/ci.yml`): black + compile-check + pytest on Python 3.14.
- **Security:** a weekly OSV scan (`security-scan.yml`) reads `requirements.lock`; regenerate the lock
  (`uv pip compile requirements.txt -o requirements.lock`) whenever `requirements.txt` changes.
- **Dependencies:** Dependabot opens weekly update PRs; patch/minor land automatically once CI is green.

## Roadmap

- Add a `GET /health` (and root) endpoint — there is none today.
- Decide Random Forest: re-enable it in the registry or remove the dead entry.
- Make the startup CoinGecko fetch non-blocking / lazy so readiness isn't gated on a network pull.
