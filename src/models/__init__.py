from .decision_tree_regression import DecisionTreeRegressionModel
from .light_gbm_regression import LightGBMRegressionModel
from .linear_regression import LinearRegressionModel
from .mlp_regression import MLPRegressionModel
from .random_forest_regression import RandomForestRegressionModel
from .support_vector_regression import SupportVectorRegressionModel
from .xgboost_regression import XGBoostRegressionModel

__all__ = [
    "DecisionTreeRegressionModel",
    "LightGBMRegressionModel",
    "LinearRegressionModel",
    "MLPRegressionModel",
    "RandomForestRegressionModel",
    "SupportVectorRegressionModel",
    "XGBoostRegressionModel",
]


class UnknownModelError(ValueError):
    """Raised when a model name is not in the registry.

    A real type rather than a bare ``Exception``: the lookup-miss path is the only failure these
    factories are meant to produce, and a test asserting ``pytest.raises(Exception)`` could not tell
    it apart from a TypeError or an ImportError raised by a factory broken some other way.

    Subclasses ``ValueError`` because an unknown name is a bad argument, and because the one caller
    (``modeling.train_model``) catches broadly and logs -- so narrowing the type changes nothing at
    runtime while making the failure nameable in a test.
    """


async def get_model_class(model):
    if model not in __all__:
        raise UnknownModelError(f"The agent name {model} does not exist")
    return globals()[model]


async def make_model(model):
    # Delegates so the registry lookup and its error message exist once. The two used to be
    # byte-identical apart from the trailing `()`, which is how they came to share a defect.
    return (await get_model_class(model))()
