from os import path

import seaborn
from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv

from definitions import RESULTS_ERRORS_PATH, regression_models
from .handle_plot import save_plot


async def draw_errors(coin: dict) -> None:
    error_types = [
        "Mean Absolute Error",
        "Mean Absolute Percentage Error",
        "Mean Squared Error",
        "Root Mean Squared Error",
    ]

    large, med = 22, 16
    params = {
        "legend.fontsize": med,
        "figure.figsize": (16, 10),
        "axes.labelsize": med,
        "axes.titlesize": med,
        "xtick.labelsize": med,
        "ytick.labelsize": med,
        "figure.titlesize": large,
        "xtick.major.pad": 8,
    }
    plt.rcParams.update(params)
    plt.style.use("seaborn-v0_8-whitegrid")
    seaborn.set_style("white")

    for error_type in error_types:
        data = []
        for model_name in regression_models:
            dataframe_errors = read_csv(
                path.join(
                    RESULTS_ERRORS_PATH, "data", coin["symbol"], model_name, "error.csv"
                )
            )
            data.append(
                [regression_models[model_name], dataframe_errors.iloc[0][error_type]]
            )

        dataframe_algorithms = DataFrame(
            data, columns=["algorithm", coin["symbol"]]
        ).dropna()
        if len(dataframe_algorithms.index) == 0:
            continue

        dataframe_algorithms.sort_values(
            by=coin["symbol"], ascending=False, inplace=True
        )
        dataframe_algorithms.reset_index(drop=True, inplace=True)

        fig, ax = plt.subplots(figsize=(16, 10), facecolor="white", dpi=80)
        ax.vlines(
            x=dataframe_algorithms.index,
            ymin=0,
            ymax=dataframe_algorithms[coin["symbol"]],
            color="firebrick",
            alpha=0.7,
            linewidth=40,
            label=coin["name"],
        )
        ax.legend()

        for i, value in enumerate(dataframe_algorithms[coin["symbol"]]):
            ax.text(
                i,
                value + 0.5,
                round(value, 1),
                fontsize=22,
                horizontalalignment="center",
            )

        ax.set_title(coin["symbol"], fontdict={"size": 22})
        ax.set_ylabel(error_type, fontsize=22)
        y_lim = (
            dataframe_algorithms[coin["symbol"]].max()
            + 10
            - (dataframe_algorithms[coin["symbol"]].max() % 10)
        )
        ax.set_ylim(ymin=0, ymax=y_lim)
        plt.xticks(
            dataframe_algorithms.index,
            dataframe_algorithms["algorithm"],
            horizontalalignment="center",
            fontsize=22,
            rotation=30,
        )

        await save_plot(
            fig,
            plt,
            path.join(RESULTS_ERRORS_PATH, "plots", coin["symbol"]),
            error_type,
        )
