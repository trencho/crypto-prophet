from os import path
from warnings import filterwarnings

from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv, to_datetime

from definitions import RESULTS_ERRORS_PATH, RESULTS_PREDICTIONS_PATH, regression_models
from .handle_plot import save_plot

filterwarnings(action='once')


def draw_predictions(cryptocurrency):
    dataframe_algorithms = DataFrame(columns=['algorithm', cryptocurrency['symbol']])
    for algorithm in regression_models:
        dataframe_errors = read_csv(
            path.join(RESULTS_ERRORS_PATH, 'data', cryptocurrency['symbol'], algorithm,
                      'error.csv'))
        dataframe_algorithms = dataframe_algorithms.append(
            [{
                'algorithm': algorithm,
                cryptocurrency['symbol']: dataframe_errors.iloc[0]['Mean Absolute Error']
            }], ignore_index=True)

    algorithm_index = dataframe_algorithms[cryptocurrency['symbol']].idxmin()
    dataframe_predictions = read_csv(
        path.join(RESULTS_PREDICTIONS_PATH, 'data', cryptocurrency['symbol'],
                  dataframe_algorithms.iloc[algorithm_index]['algorithm'], 'prediction.csv'), index_col='time')

    x = to_datetime(dataframe_predictions.index).normalize()
    y1 = dataframe_predictions['Actual']
    y2 = dataframe_predictions['Predicted']

    # Plot Line (left Y axis)
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=80)
    ax.plot(x, y1, color='tab:red', label='Actual')
    ax.plot(x, y2, color='tab:green',
            label=f'Predicted: {regression_models[dataframe_algorithms.iloc[algorithm_index]["algorithm"]]}')
    # Decorations
    # ax (left Y axis)
    ax.set_xlabel('Dates', fontsize=22)
    ax.tick_params(axis='x', rotation=0, labelsize=18)
    ax.set_ylabel(f'{cryptocurrency["name"]} values', fontsize=22)
    ax.tick_params(axis='y', rotation=0)
    ax.set_title(f'{cryptocurrency["name"]} predictions', fontsize=22)
    ax.grid(alpha=.4)
    ax.legend(fontsize=16)

    plt.gcf().autofmt_xdate()

    file_path = path.join(RESULTS_PREDICTIONS_PATH, 'plots', cryptocurrency['symbol'])
    save_plot(fig, plt, file_path, 'prediction')
