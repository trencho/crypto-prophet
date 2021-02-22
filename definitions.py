from os import path

ROOT_DIR = path.dirname(path.abspath(__file__))

DATA_PATH = path.join(ROOT_DIR, 'data')
DATA_EXTERNAL_PATH = path.join(DATA_PATH, 'external')
DATA_INTERIM_PATH = path.join(DATA_PATH, 'interim')
DATA_PROCESSED_PATH = path.join(DATA_PATH, 'processed')
DATA_RAW_PATH = path.join(DATA_PATH, 'raw')

MODELS_PATH = path.join(ROOT_DIR, 'models')

RESULTS_PATH = path.join(ROOT_DIR, 'results')
RESULTS_ERRORS_PATH = path.join(RESULTS_PATH, 'errors')
RESULTS_PREDICTIONS_PATH = path.join(RESULTS_PATH, 'predictions')

github_token_env = 'GITHUB_TOKEN'

environment_variables = [
    github_token_env
]

regression_models = {
    'DecisionTreeRegressionModel': 'Decision Tree',
    'DummyRegressionModel': 'Dummy',
    'LightGBMRegressionModel': 'LightGBM',
    'LinearRegressionModel': 'Linear',
    'MLPRegressionModel': 'Multilayer Perceptron',
    'RandomForestRegressionModel': 'Random Forest',
    'SupportVectorRegressionModel': 'Support Vector',
    # 'TPOTRegressionModel': 'TPOT',
    'XGBoostRegressionModel': 'XGBoost'
}
