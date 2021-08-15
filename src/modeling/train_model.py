from datetime import datetime
from math import inf
from os import cpu_count, makedirs, path, remove as os_remove
from pickle import dump as pickle_dump, HIGHEST_PROTOCOL
from threading import Thread
from traceback import print_exc

from pandas import DataFrame, read_csv, to_datetime
from sklearn.model_selection import RandomizedSearchCV

from definitions import DATA_EXTERNAL_PATH, MODELS_PATH, RESULTS_ERRORS_PATH, RESULTS_PREDICTIONS_PATH, \
    regression_models, week_in_seconds
from models import make_model
from processing import backward_elimination, current_hour, generate_features, value_scaling, encode_categorical_data
from visualization import draw_errors, draw_predictions
from .process_results import save_errors, save_results

lock_file = '.lock'


def previous_value_overwrite(dataframe):
    dataframe = dataframe.shift(periods=-1, axis=0)
    dataframe.drop(dataframe.tail(1).index, inplace=True)

    return dataframe


def split_dataframe(dataframe, target, selected_features=None):
    x = dataframe.drop(columns=target, errors='ignore')
    x = value_scaling(x)
    y = dataframe['value']

    x = previous_value_overwrite(x)
    y.drop(y.tail(1).index, inplace=True)

    selected_features = backward_elimination(x, y) if selected_features is None else selected_features
    x = x[selected_features]

    return x, y


def save_selected_features(coin_symbol, selected_features):
    makedirs(path.join(MODELS_PATH, coin_symbol), exist_ok=True)
    with open(path.join(MODELS_PATH, coin_symbol, 'selected_features.pkl'), 'wb') as out_file:
        pickle_dump(selected_features, out_file, HIGHEST_PROTOCOL)


def read_model(coin_symbol, algorithm, error_type):
    dataframe_errors = read_csv(path.join(RESULTS_ERRORS_PATH, 'data', coin_symbol, algorithm, 'error.csv'))
    model = make_model(algorithm)
    model.load(path.join(MODELS_PATH, coin_symbol))
    return model, dataframe_errors.iloc[0][error_type]


def create_models_path(coin_symbol, model_name):
    makedirs(path.join(MODELS_PATH, coin_symbol, model_name), exist_ok=True)


def create_results_path(results_path, coin_symbol, model_name):
    makedirs(path.join(results_path, 'data', coin_symbol, model_name), exist_ok=True)


def create_paths(coin_symbol, model_name):
    create_models_path(coin_symbol, model_name)
    create_results_path(RESULTS_ERRORS_PATH, coin_symbol, model_name)
    create_results_path(RESULTS_PREDICTIONS_PATH, coin_symbol, model_name)


def check_coin_lock(coin_symbol):
    return path.exists(path.join(MODELS_PATH, coin_symbol, lock_file))


def create_coin_lock(coin_symbol):
    makedirs(path.join(MODELS_PATH, coin_symbol), exist_ok=True)
    with open(path.join(MODELS_PATH, coin_symbol, lock_file), 'w'):
        pass


def hyper_parameter_tuning(model, x_train, y_train, coin_symbol):
    model_cv = RandomizedSearchCV(model.reg, model.param_grid, n_jobs=cpu_count() // 2, cv=5)
    model_cv.fit(x_train, y_train)

    with open(path.join(MODELS_PATH, coin_symbol, type(model).__name__, 'HyperparameterOptimization.pkl'),
              'wb') as out_file:
        pickle_dump(model_cv.best_params_, out_file, HIGHEST_PROTOCOL)

    return model_cv.best_params_


def remove_coin_lock(coin_symbol):
    try:
        os_remove(path.join(MODELS_PATH, coin_symbol, lock_file))
    except OSError:
        pass


def check_best_regression_model(coin_symbol):
    try:
        last_modified = int(
            path.getmtime(path.join(MODELS_PATH, coin_symbol, 'best_regression_model.pkl')))
        if last_modified < int(datetime.timestamp(current_hour())) - week_in_seconds:
            return True
    except OSError:
        return False


def save_best_regression_model(coin_symbol, best_model):
    with open(path.join(MODELS_PATH, coin_symbol, 'best_regression_model.pkl'), 'wb') as out_file:
        pickle_dump(best_model, out_file, HIGHEST_PROTOCOL)


def generate_regression_model(dataframe, coin_symbol):
    dataframe = dataframe.join(generate_features(dataframe['value']), how='inner')
    encode_categorical_data(dataframe)
    validation_split = len(dataframe) * 3 // 4

    train_dataframe = dataframe.iloc[:validation_split]
    x_train, y_train = split_dataframe(train_dataframe, 'value')
    selected_features = list(x_train.columns)

    test_dataframe = dataframe.iloc[validation_split:]
    x_test, y_test = split_dataframe(test_dataframe, 'value', selected_features)

    save_selected_features(coin_symbol, selected_features)

    best_model_error = inf
    best_model = None
    for model_name in regression_models:
        if path.exists(path.join(MODELS_PATH, coin_symbol, model_name)):
            model, model_error = read_model(coin_symbol, model_name, 'Mean Absolute Error')
            if model_error < best_model_error:
                best_model = model
                best_model_error = model_error
            continue

        create_paths(coin_symbol, model_name)

        model = make_model(model_name)
        params = hyper_parameter_tuning(model, x_train, y_train, coin_symbol)
        model.set_params(**params)
        model.train(x_train, y_train)
        model.save(path.join(MODELS_PATH, coin_symbol))

        y_predicted = model.predict(x_test)

        results = DataFrame({'Actual': y_test, 'Predicted': y_predicted}, x_test.index)
        save_results(coin_symbol, model_name, results)

        model_error = save_errors(coin_symbol, model_name, y_test, y_predicted)
        if model_error < best_model_error:
            best_model = model
            best_model_error = model_error

    if best_model is not None:
        x_train, y_train = split_dataframe(dataframe, 'value', selected_features)
        best_model.train(x_train, y_train)
        save_best_regression_model(coin_symbol, best_model.reg)


def train_regression_model(coin):
    if check_best_regression_model(coin['symbol']):
        return
    try:
        dataframe = read_csv(path.join(DATA_EXTERNAL_PATH, coin['symbol'], 'data.csv'), index_col='time')
        dataframe.index = to_datetime(dataframe.index / 10 ** 3, unit='s')
        if not check_coin_lock(coin['symbol']):
            create_coin_lock(coin['symbol'])
        generate_regression_model(dataframe, coin['symbol'])
        draw_errors(coin)
        draw_predictions(coin)
    except Exception:
        print_exc()
    finally:
        remove_coin_lock(coin['symbol'])


def train_coin_models(coin):
    Thread(target=train_regression_model, args=(coin,)).start()
