from math import isinf
from os import path
from typing import Optional

from numpy import abs, array, inf, mean, nan, ndarray, sqrt
from pandas import DataFrame, Series
from sklearn.metrics import mean_absolute_error, mean_squared_error

from definitions import RESULTS_ERRORS_PATH, RESULTS_PREDICTIONS_PATH


async def filter_invalid_values(y_true: [ndarray, Series], y_predicted: [ndarray, Series]) -> tuple:
    dataframe = DataFrame({'y_true': y_true, 'y_predicted': y_predicted}).replace([-inf, inf], nan).dropna()
    return dataframe['y_true'], dataframe['y_predicted']


async def mean_absolute_percentage_error(y_true: Series, y_predicted: Series) -> Optional[float]:
    y_true, y_predicted = array(y_true), array(y_predicted)
    mape = mean(abs((y_true - y_predicted) / y_true)) * 100
    return None if isinf(mape) else mape


async def save_errors(coin_symbol: str, model_name: str, y_true: Series, y_predicted: Series) -> float:
    y_true, y_predicted = await filter_invalid_values(y_true, y_predicted)
    try:
        mae = mean_absolute_error(y_true, y_predicted)
        mse = mean_squared_error(y_true, y_predicted)
        dataframe = DataFrame({
            'Mean Absolute Error': None if isinf(mae) else mae,
            'Mean Absolute Percentage Error': await mean_absolute_percentage_error(y_true, y_predicted),
            'Mean Squared Error': mse,
            'Root Mean Squared Error': sqrt(mse)
        }, index=[0], columns=['Mean Absolute Error', 'Mean Absolute Percentage Error', 'Mean Squared Error',
                               'Root Mean Squared Error'])
        dataframe.to_csv(path.join(RESULTS_ERRORS_PATH, 'data', coin_symbol, model_name, 'error.csv'), index=False)

        return mae
    except Exception:
        return inf


async def save_results(coin_symbol: str, model_name: str, dataframe: DataFrame) -> None:
    dataframe.to_csv(path.join(RESULTS_PREDICTIONS_PATH, 'data', coin_symbol, model_name, 'prediction.csv'))
