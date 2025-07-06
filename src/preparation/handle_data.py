from numpy import nan
from pandas import DataFrame


def trim_dataframe(dataframe: DataFrame, column: str) -> DataFrame:
    dataframe = dataframe.replace(to_replace=0, value=nan)
    dataframe = dataframe.dropna(axis="columns", how="all")
    dataframe = dataframe.drop_duplicates(subset=column, keep="last")
    dataframe = dataframe.reset_index(drop=True)
    return dataframe.sort_values(by=column)
