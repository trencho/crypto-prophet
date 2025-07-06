from datetime import date, datetime

from numpy import abs
from pandas import DataFrame, Index, Series
from statsmodels.tsa.stattools import pacf


def get_season(time: datetime) -> str:
    # Dummy leap year to allow input 29-02-X (leap day)
    dummy_leap_year = 2000

    dt = time.date()
    dt = dt.replace(year=dummy_leap_year)

    seasons = [
        ("winter", (date(dummy_leap_year, 1, 1), date(dummy_leap_year, 3, 20))),
        ("spring", (date(dummy_leap_year, 3, 21), date(dummy_leap_year, 6, 20))),
        ("summer", (date(dummy_leap_year, 6, 21), date(dummy_leap_year, 9, 22))),
        ("autumn", (date(dummy_leap_year, 9, 23), date(dummy_leap_year, 12, 20))),
        ("winter", (date(dummy_leap_year, 12, 21), date(dummy_leap_year, 12, 31))),
    ]
    return next(season for season, (start, end) in seasons if start <= dt <= end)


def encode_categorical_data(dataframe: DataFrame) -> None:
    obj_columns = dataframe.select_dtypes("object").columns
    dataframe[obj_columns] = dataframe[obj_columns].astype("category")
    cat_columns = dataframe.select_dtypes("category").columns
    dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.cat.codes)


def generate_lag_features(target: Series, lags: int) -> DataFrame:
    partial = Series(
        data=pacf(
            target, nlags=lags if lags < target.size // 2 else target.size // 2 - 1
        )
    )
    lags = list(partial[abs(partial) >= 0.2].index)

    if 0 in lags:
        # Do not consider itself as a lag feature
        lags.remove(0)

    features = DataFrame()
    for lag in lags:
        features[f"lag_{lag}"] = target.shift(lag)

    return features


def generate_time_features(target) -> DataFrame:
    features = DataFrame()
    features["month"] = target.index.month
    features["day"] = target.index.day
    features["hour"] = target.index.hour
    features["weekOfYear"] = Index(target.index.isocalendar().week, dtype="int64")
    features["dayOfWeek"] = target.index.dayofweek
    features["dayOfYear"] = target.index.dayofyear
    features["weekdayName"] = target.index.day_name()
    features["quarter"] = target.index.quarter
    features["daysInMonth"] = target.index.days_in_month
    features["isMonthStart"] = target.index.is_month_start
    features["isMonthEnd"] = target.index.is_month_end
    features["isQuarterStart"] = target.index.is_quarter_start
    features["isQuarterEnd"] = target.index.is_quarter_end
    features["isYearStart"] = target.index.is_year_start
    features["isYearEnd"] = target.index.is_year_end
    features["isLeapYear"] = target.index.is_leap_year
    features["isWeekend"] = (
        target.index.to_series()
        .apply(lambda x: 0 if x.dayofweek in (5, 6) else 1)
        .values
    )
    features["season"] = target.index.to_series().apply(get_season).values

    features = features.set_index(target.index)

    return features


def generate_features(target: Series, lags: int = 24) -> DataFrame:
    lag_features = generate_lag_features(target, lags)
    time_features = generate_time_features(target)
    features = (
        time_features
        if len(lag_features.index) == 0
        else lag_features.join(time_features, how="inner")
    )

    return features.dropna(axis="columns", how="all").dropna(axis="index", how="any")
