from datetime import datetime
from math import inf
from os import environ, makedirs, path, remove
from pickle import dump, HIGHEST_PROTOCOL
from threading import Thread

from pandas import DataFrame, read_csv, to_datetime
from sklearn.model_selection import RandomizedSearchCV

from api.config.logger import logger
from definitions import app_dev, app_env, DATA_EXTERNAL_PATH, MODELS_PATH, regression_models, RESULTS_ERRORS_PATH, \
    RESULTS_PREDICTIONS_PATH
from models import make_model
from models.base_regression_model import BaseRegressionModel
from processing import backward_elimination, generate_features, value_scaling, encode_categorical_data
from processing.normalize_data import current_hour
from visualization import draw_errors, draw_predictions
from .process_results import save_errors, save_results

lock_file = '.lock'


async def previous_value_overwrite(dataframe: DataFrame) -> DataFrame:
    dataframe = dataframe.shift(periods=-1, axis=0)
    dataframe.drop(dataframe.tail(1).index, inplace=True)

    return dataframe


async def split_dataframe(dataframe: DataFrame, target: str, selected_features: list = None) -> tuple:
    x = dataframe.drop(columns=target, errors='ignore')
    x = await value_scaling(x)
    y = dataframe['value']

    x = await previous_value_overwrite(x)
    y.drop(y.tail(1).index, inplace=True)

    selected_features = await backward_elimination(x, y) if selected_features is None else selected_features
    x = x[selected_features]

    return x, y


async def save_selected_features(coin_symbol: str, selected_features: list) -> None:
    makedirs(path.join(MODELS_PATH, coin_symbol), exist_ok=True)
    with open(path.join(MODELS_PATH, coin_symbol, 'selected_features.pkl'), 'wb') as out_file:
        dump(selected_features, out_file, HIGHEST_PROTOCOL)


async def read_model(coin_symbol: str, algorithm: str, error_type: str) -> tuple:
    dataframe_errors = read_csv(path.join(RESULTS_ERRORS_PATH, 'data', coin_symbol, algorithm, 'error.csv'))
    model = await make_model(algorithm)
    model.load(path.join(MODELS_PATH, coin_symbol))
    return model, dataframe_errors.iloc[0][error_type]


async def create_models_path(coin_symbol: str, model_name: str) -> None:
    makedirs(path.join(MODELS_PATH, coin_symbol, model_name), exist_ok=True)


async def create_results_path(results_path: str, coin_symbol: str, model_name: str) -> None:
    makedirs(path.join(results_path, 'data', coin_symbol, model_name), exist_ok=True)


async def create_paths(coin_symbol: str, model_name: str) -> None:
    await create_models_path(coin_symbol, model_name)
    await create_results_path(RESULTS_ERRORS_PATH, coin_symbol, model_name)
    await create_results_path(RESULTS_PREDICTIONS_PATH, coin_symbol, model_name)


async def check_coin_lock(coin_symbol: str) -> bool:
    return path.exists(path.join(MODELS_PATH, coin_symbol, lock_file))


async def create_coin_lock(coin_symbol: str) -> None:
    makedirs(path.join(MODELS_PATH, coin_symbol), exist_ok=True)
    with open(path.join(MODELS_PATH, coin_symbol, lock_file), 'w'):
        pass


async def hyper_parameter_tuning(model: BaseRegressionModel, x_train, y_train, coin_symbol):
    model_cv = RandomizedSearchCV(model.reg, model.param_grid, cv=5)
    model_cv.fit(x_train, y_train)

    if environ.get(app_env, app_dev) == app_dev:
        with open(path.join(MODELS_PATH, coin_symbol, type(model).__name__, 'HyperparameterOptimization.pkl'),
                  'wb') as out_file:
            dump(model_cv.best_params_, out_file, HIGHEST_PROTOCOL)

    return model_cv.best_params_


async def remove_coin_lock(coin_symbol: str) -> None:
    try:
        remove(path.join(MODELS_PATH, coin_symbol, lock_file))
    except OSError:
        pass


async def check_best_regression_model(coin_symbol: str) -> bool:
    try:
        last_modified = int(
            path.getmtime(path.join(MODELS_PATH, coin_symbol, 'best_regression_model.pkl')))
        month_in_seconds = 2629800
        if last_modified < int(datetime.timestamp(current_hour())) - month_in_seconds:
            return False

        return True
    except OSError:
        return False


async def save_best_regression_model(coin_symbol: str, best_model: BaseRegressionModel) -> None:
    with open(path.join(MODELS_PATH, coin_symbol, 'best_regression_model.pkl'), 'wb') as out_file:
        dump(best_model, out_file, HIGHEST_PROTOCOL)


async def generate_regression_model(dataframe: DataFrame, coin_symbol: str) -> None:
    dataframe = dataframe.join(await generate_features(dataframe['value']), how='inner')
    await encode_categorical_data(dataframe)
    validation_split = len(dataframe.index) * 3 // 4

    train_dataframe = dataframe.iloc[:validation_split]
    x_train, y_train = await split_dataframe(train_dataframe, 'value')
    selected_features = x_train.columns.values.tolist()

    test_dataframe = dataframe.iloc[validation_split:]
    x_test, y_test = await split_dataframe(test_dataframe, 'value', selected_features)

    best_model_error = inf
    best_model = None
    for model_name in regression_models:
        if (env_var := environ.get(app_env, app_dev)) == app_dev and path.exists(
                path.join(MODELS_PATH, coin_symbol, model_name)):
            model, model_error = await read_model(coin_symbol, model_name, 'Mean Absolute Error')
            if model_error < best_model_error:
                best_model = model
                best_model_error = model_error
            continue

        await create_paths(coin_symbol, model_name)

        model = await make_model(model_name)
        params = await hyper_parameter_tuning(model, x_train, y_train, coin_symbol)
        model.set_params(**params)
        model.train(x_train, y_train)

        if env_var == app_dev:
            model.save(path.join(MODELS_PATH, coin_symbol))

        y_predicted = model.predict(x_test)

        results = DataFrame({'Actual': y_test, 'Predicted': y_predicted}, x_test.index)
        await save_results(coin_symbol, model_name, results)

        if (model_error := await save_errors(coin_symbol, model_name, y_test, y_predicted)) < best_model_error:
            best_model = model
            best_model_error = model_error

    if best_model is not None:
        await save_selected_features(coin_symbol, selected_features)
        x_train, y_train = await split_dataframe(dataframe, 'value', selected_features)
        best_model.train(x_train, y_train)
        await save_best_regression_model(coin_symbol, best_model.reg)


async def train_regression_model(coin: dict) -> None:
    if await check_best_regression_model(coin['symbol']) or await check_coin_lock(coin['symbol']):
        return
    try:
        dataframe = read_csv(path.join(DATA_EXTERNAL_PATH, coin['symbol'], 'data.csv'), index_col='time')
        dataframe.index = to_datetime(dataframe.index / 10 ** 3, unit='s')
        await create_coin_lock(coin['symbol'])
        await generate_regression_model(dataframe, coin['symbol'])
        await draw_errors(coin)
        await draw_predictions(coin)
    except Exception:
        logger.error(f'Error occurred while training regression model for {coin["name"]}', exc_info=1)
    finally:
        await remove_coin_lock(coin['symbol'])


async def train_coin_models(coin: dict) -> None:
    Thread(target=train_regression_model, args=(coin,), daemon=True).start()
