from os import environ
from pathlib import Path

app_env = "APP_ENV"

github_token = "GITHUB_TOKEN"

repo_name = "REPO_NAME"

volume_path = "VOLUME_PATH"

environment_variables = [app_env, github_token, repo_name]

VOLUME_PATH = environ.get(volume_path, "")
ROOT_PATH = VOLUME_PATH or Path(__file__).resolve().parent

DATA_PATH = Path(ROOT_PATH) / "data"
DATA_EXTERNAL_PATH = Path(DATA_PATH) / "external"
DATA_PROCESSED_PATH = Path(DATA_PATH) / "processed"
DATA_RAW_PATH = Path(DATA_PATH) / "raw"

LOG_PATH = Path(ROOT_PATH) / "logs"

MODELS_PATH = Path(ROOT_PATH) / "models"

RESULTS_PATH = Path(ROOT_PATH) / "results"
RESULTS_ERRORS_PATH = Path(RESULTS_PATH) / "errors"
RESULTS_PREDICTIONS_PATH = Path(RESULTS_PATH) / "predictions"

app_dev = "development"
app_prod = "production"

chunk_size = 15000

coins = ["bitcoin", "ethereum", "ravencoin"]

regression_models = {
    "DecisionTreeRegressionModel": "Decision Tree",
    "LightGBMRegressionModel": "LightGBM",
    "LinearRegressionModel": "Linear",
    "MLPRegressionModel": "Multilayer Perceptron",
    # 'RandomForestRegressionModel': 'Random Forest',
    "SupportVectorRegressionModel": "Support Vector",
    "XGBoostRegressionModel": "XGBoost",
}
