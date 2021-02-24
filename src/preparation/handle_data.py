def trim_dataframe(dataframe, column):
    dataframe.drop_duplicates(subset=column, inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    dataframe.sort_values(by=column, inplace=True)
