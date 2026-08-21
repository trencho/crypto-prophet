"""Tests for the models package factory and BaseRegressionModel behavior."""

import asyncio

import pytest
from numpy import allclose, arange, column_stack

from models import __all__, get_model_class, make_model, UnknownModelError
from models.linear_regression import LinearRegressionModel


def _linear_dataset():
    # y = 2*x0 + 3*x1 + 1, a relationship LinearRegression fits exactly.
    x0 = arange(0, 20, dtype=float)
    x1 = arange(20, 40, dtype=float)
    x = column_stack([x0, x1])
    y = 2.0 * x0 + 3.0 * x1 + 1.0
    return x, y


def test_make_model_known_and_unknown():
    model = asyncio.run(make_model("LinearRegressionModel"))
    assert isinstance(model, LinearRegressionModel)

    # Both the TYPE and the message. `pytest.raises(Exception)` alone accepted any failure at
    # all -- a TypeError or ImportError from a factory broken some other way satisfied it just as
    # well as the intended rejection.
    with pytest.raises(
        UnknownModelError, match="The agent name NoSuchModel does not exist"
    ):
        asyncio.run(make_model("NoSuchModel"))


def test_get_model_class_returns_the_class_without_instantiating():
    # Had no test at all. It is the half of the registry that callers use when they want to
    # inspect or subclass a model rather than build one, and it shares make_model's lookup.
    cls = asyncio.run(get_model_class("LinearRegressionModel"))

    assert cls is LinearRegressionModel
    assert isinstance(cls(), LinearRegressionModel)


def test_get_model_class_rejects_an_unknown_name():
    with pytest.raises(
        UnknownModelError, match="The agent name NoSuchModel does not exist"
    ):
        asyncio.run(get_model_class("NoSuchModel"))


def test_unknown_model_error_is_a_value_error():
    # The base class is part of the contract: `modeling.train_model` catches broadly, and any
    # caller narrowing to ValueError must keep catching this.
    assert issubclass(UnknownModelError, ValueError)


def test_model_train_predict_shape():
    x, y = _linear_dataset()

    model = LinearRegressionModel()
    model.train(x, y)
    predictions = model.predict(x)

    assert predictions.shape == (x.shape[0],)
    # An exact linear relationship should be recovered closely.
    assert allclose(predictions, y, atol=1e-6)


def test_model_save_load_roundtrip(models_dir):
    base = models_dir("LinearRegressionModel")
    x, y = _linear_dataset()

    trained = LinearRegressionModel()
    trained.train(x, y)
    expected = trained.predict(x)
    trained.save(str(base))

    restored = LinearRegressionModel()
    restored.load(str(base))
    actual = restored.predict(x)

    assert allclose(actual, expected)


def test_every_registered_model_accepts_its_own_param_grid():
    """Each model's param_grid must actually be valid for the estimator it configures.

    ``RandomizedSearchCV`` validates a parameter only when it SAMPLES it, so an invalid value sits
    dormant until a search happens to pick it -- and then fails inside a training run, not at
    construction. ``RandomForestRegressionModel`` shipped ``max_features="auto"``, removed in
    scikit-learn 1.3 against a 1.9 pin, and nothing caught it.

    ``set_params`` performs the same validation without fitting, so this walks every value in every
    registry entry's grid and is still fast.
    """

    from sklearn.base import clone

    for name in __all__:
        model = asyncio.run(make_model(name))
        for parameter, values in model.param_grid.items():
            for value in values:
                estimator = clone(model.reg)
                try:
                    estimator.set_params(**{parameter: value})
                except (
                    Exception
                ) as error:  # pragma: no cover - the failure message is the point
                    raise AssertionError(
                        f"{name}.param_grid[{parameter!r}] contains {value!r}, "
                        f"which {type(model.reg).__name__} rejects: {error}"
                    ) from error

    # Guard the guard: a registry that walked nothing would pass silently.
    assert len(__all__) >= 6
