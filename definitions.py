from os import environ, path

app_env = "APP_ENV"

github_token = "GITHUB_TOKEN"

repo_name = "REPO_NAME"

volume_path = "VOLUME_PATH"

environment_variables = [app_env, github_token, repo_name]

VOLUME_PATH = environ.get(volume_path, "")
ROOT_PATH = VOLUME_PATH or path.dirname(path.abspath(__file__))

DATA_PATH = path.join(ROOT_PATH, "data")
DATA_EXTERNAL_PATH = path.join(DATA_PATH, "external")
DATA_PROCESSED_PATH = path.join(DATA_PATH, "processed")
DATA_RAW_PATH = path.join(DATA_PATH, "raw")

LOG_PATH = path.join(ROOT_PATH, "logs")

MODELS_PATH = path.join(ROOT_PATH, "models")

RESULTS_PATH = path.join(ROOT_PATH, "results")
RESULTS_ERRORS_PATH = path.join(RESULTS_PATH, "errors")
RESULTS_PREDICTIONS_PATH = path.join(RESULTS_PATH, "predictions")

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
