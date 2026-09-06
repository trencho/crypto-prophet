from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SCALERS = {
    "min_max": MinMaxScaler,
    "standard": StandardScaler,
    "robust": RobustScaler,
}


def _as_frame(values, template: DataFrame) -> DataFrame:
    return DataFrame(values, template.index, template.columns)


def fit_scaler(
    dataframe: DataFrame, scale: str = "robust"
) -> tuple[DataFrame, TransformerMixin]:
    """Fit a scaler on this frame and return the scaled frame alongside it.

    The ONLY place `fit_transform` appears in this codebase. Everything that scales at inference
    time goes through `apply_scaler`, so "no fit_transform outside fit_scaler" is a one-line check
    rather than a convention.

    This used to be a single `value_scaling` that always fitted, called from training AND from the
    forecast loop. So the scaler that served a request was fitted on the request's own frame and
    had nothing to do with the one the model was trained under, and train and test were each
    scaled with their own statistics.
    """
    scaler = SCALERS.get(scale, RobustScaler)()
    return _as_frame(scaler.fit_transform(dataframe), dataframe), scaler


def apply_scaler(dataframe: DataFrame, scaler: TransformerMixin) -> DataFrame:
    """Scale with an already-fitted scaler. No `fit` in this function, deliberately."""
    return _as_frame(scaler.transform(dataframe), dataframe)
