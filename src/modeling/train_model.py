from datetime import datetime
from logging import getLogger
from math import inf
from os import (
    O_CREAT,
    O_EXCL,
    O_WRONLY,
    close,
    environ,
    makedirs,
    open as os_open,
    path,
    remove,
)
from pathlib import Path
from pickle import dump, HIGHEST_PROTOCOL

from pandas import DataFrame, read_csv, to_datetime
from sklearn.model_selection import RandomizedSearchCV

from definitions import (
    app_dev,
    app_env,
    DATA_EXTERNAL_PATH,
    MODELS_PATH,
    regression_models,
    RESULTS_ERRORS_PATH,
    RESULTS_PREDICTIONS_PATH,
)
from models import make_model
from models.base_regression_model import BaseRegressionModel
from processing import (
    backward_elimination,
    generate_features,
    value_scaling,
    encode_categorical_data,
)
from processing.normalize_data import current_hour
from visualization import draw_errors, draw_predictions
from .process_results import save_errors, save_results

logger = getLogger(__name__)

lock_file = ".lock"


def split_dataframe(
    dataframe: DataFrame, target: str, selected_features: list = None
) -> tuple:
    """Pair the features at t with the value at t+1.

    This used to shift X FORWARD by one row and leave y where it was, via a helper called
    `previous_value_overwrite`. That pairs features from t+1 with a target at t, so the model was
    asked to predict the past from the future - and because `lag_1` at t+1 is by definition the
    value at t, the target ended up as a column of X. Measured on a random walk: the leaked column
    correlated with y at r = 1.000000, against 0.996881 for the same column honestly aligned.

    Nothing was visibly wrong. Every error in results/errors/ was near-zero by construction, the
    best model was chosen on that error, and the served forecast collapsed to persistence.
    """
    x = dataframe.drop(columns=target, errors="ignore")
    x = value_scaling(x)
    y = dataframe["value"]

    # Take the target from the NEXT row, and drop the final row of both: it has no next value.
    y = y.shift(-1)
    x = x.drop(x.tail(1).index)
    y = y.drop(y.tail(1).index)

    selected_features = (
        backward_elimination(x, y) if selected_features is None else selected_features
    )
    x = x[selected_features]

    return x, y


def save_selected_features(coin_symbol: str, selected_features: list) -> None:
    makedirs(Path(MODELS_PATH) / coin_symbol, exist_ok=True)
    with open(
        Path(MODELS_PATH) / coin_symbol / "selected_features.pkl", "wb"
    ) as out_file:
        dump(selected_features, out_file, HIGHEST_PROTOCOL)


async def read_model(coin_symbol: str, algorithm: str, error_type: str) -> tuple:
    dataframe_errors = read_csv(
        path.join(RESULTS_ERRORS_PATH, "data", coin_symbol, algorithm, "error.csv")
    )
    model = await make_model(algorithm)
    model.load(path.join(MODELS_PATH, coin_symbol))
    return model, dataframe_errors.iloc[0][error_type]


def create_models_path(coin_symbol: str, model_name: str) -> None:
    makedirs(path.join(MODELS_PATH, coin_symbol, model_name), exist_ok=True)


def create_results_path(results_path: str, coin_symbol: str, model_name: str) -> None:
    makedirs(path.join(results_path, "data", coin_symbol, model_name), exist_ok=True)


def create_paths(coin_symbol: str, model_name: str) -> None:
    create_models_path(coin_symbol, model_name)
    create_results_path(RESULTS_ERRORS_PATH, coin_symbol, model_name)
    create_results_path(RESULTS_PREDICTIONS_PATH, coin_symbol, model_name)


def acquire_coin_lock(coin_symbol: str) -> bool:
    """Take the training lock for a coin. True if THIS process now owns it.

    O_CREAT|O_EXCL is one syscall, so two processes cannot both succeed. The previous form was a
    separate `check_coin_lock` several statements before `create_coin_lock`, with a read_csv in
    between - a window wide enough for four gunicorn workers, each running its own scheduler, to
    pass the check together and then all train the same coin against the same files.
    """
    makedirs(path.join(MODELS_PATH, coin_symbol), exist_ok=True)
    try:
        descriptor = os_open(
            path.join(MODELS_PATH, coin_symbol, lock_file), O_CREAT | O_EXCL | O_WRONLY
        )
    except FileExistsError:
        return False
    close(descriptor)
    return True


def hyper_parameter_tuning(model: BaseRegressionModel, x_train, y_train, coin_symbol):
    model_cv = RandomizedSearchCV(model.reg, model.param_grid, cv=5)
    model_cv.fit(x_train, y_train)

    if environ.get(app_env, app_dev) == app_dev:
        with open(
            path.join(
                MODELS_PATH,
                coin_symbol,
                type(model).__name__,
                "HyperparameterOptimization.pkl",
            ),
            "wb",
        ) as out_file:
            dump(model_cv.best_params_, out_file, HIGHEST_PROTOCOL)

    return model_cv.best_params_


def remove_coin_lock(coin_symbol: str) -> None:
    try:
        remove(path.join(MODELS_PATH, coin_symbol, lock_file))
    except OSError:
        pass


def check_best_regression_model(coin_symbol: str) -> bool:
    try:
        last_modified = int(
            path.getmtime(
                path.join(MODELS_PATH, coin_symbol, "best_regression_model.pkl")
            )
        )
        month_in_seconds = 2629800
        if last_modified < int(datetime.timestamp(current_hour())) - month_in_seconds:
            return False

        return True
    except OSError:
        return False


def save_best_regression_model(
    coin_symbol: str, best_model: BaseRegressionModel
) -> None:
    with open(
        path.join(MODELS_PATH, coin_symbol, "best_regression_model.pkl"), "wb"
    ) as out_file:
        dump(best_model, out_file, HIGHEST_PROTOCOL)


async def generate_regression_model(dataframe: DataFrame, coin_symbol: str) -> None:
    dataframe = dataframe.join(generate_features(dataframe["value"]), how="inner")
    encode_categorical_data(dataframe)
    validation_split = len(dataframe.index) * 3 // 4

    train_dataframe = dataframe.iloc[:validation_split]
    x_train, y_train = split_dataframe(train_dataframe, "value")
    selected_features = x_train.columns.values.tolist()

    test_dataframe = dataframe.iloc[validation_split:]
    x_test, y_test = split_dataframe(test_dataframe, "value", selected_features)

    best_model_error = inf
    best_model = None
    for model_name in regression_models:
        if (env_var := environ.get(app_env, app_dev)) == app_dev and path.exists(
            path.join(MODELS_PATH, coin_symbol, model_name)
        ):
            model, model_error = await read_model(
                coin_symbol, model_name, "Mean Absolute Error"
            )
            if model_error < best_model_error:
                best_model = model
                best_model_error = model_error
            continue

        create_paths(coin_symbol, model_name)

        model = await make_model(model_name)
        params = hyper_parameter_tuning(model, x_train, y_train, coin_symbol)
        model.set_params(**params)
        model.train(x_train, y_train)

        if env_var == app_dev:
            model.save(Path(MODELS_PATH) / coin_symbol)

        y_predicted = model.predict(x_test)

        results = DataFrame({"Actual": y_test, "Predicted": y_predicted}, x_test.index)
        save_results(coin_symbol, model_name, results)

        if (
            model_error := save_errors(coin_symbol, model_name, y_test, y_predicted)
        ) < best_model_error:
            best_model = model
            best_model_error = model_error

    if best_model is not None:
        save_selected_features(coin_symbol, selected_features)
        x_train, y_train = split_dataframe(dataframe, "value", selected_features)
        best_model.train(x_train, y_train)
        save_best_regression_model(coin_symbol, best_model.reg)


async def train_regression_model(coin: dict) -> None:
    if check_best_regression_model(coin["symbol"]):
        return
    # Acquire BEFORE any other work, and bail if someone else holds it. Ordering matters: the lock
    # used to be taken after the read_csv below, so the check and the create were separated by a
    # file read.
    if not acquire_coin_lock(coin["symbol"]):
        return
    try:
        dataframe = read_csv(
            Path(DATA_EXTERNAL_PATH, coin["symbol"], "data.csv"), index_col="time"
        )
        dataframe.index = to_datetime(dataframe.index / 10**3, unit="s")
        await generate_regression_model(dataframe, coin["symbol"])
        draw_errors(coin)
        draw_predictions(coin)
    except Exception:
        logger.error(
            f'Error occurred while training regression model for {coin["name"]}',
            exc_info=True,
        )
    finally:
        remove_coin_lock(coin["symbol"])
