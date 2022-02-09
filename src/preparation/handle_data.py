from numpy import nan
from pandas import DataFrame


def trim_dataframe(dataframe: DataFrame, column: str) -> None:
    dataframe.replace(to_replace=0, value=nan, inplace=True)
    dataframe.drop(columns=dataframe.columns[dataframe.isna().all()].tolist(), inplace=True, errors='ignore')
    dataframe.drop_duplicates(subset=column, keep='last', inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    dataframe.sort_values(by=column, inplace=True)
