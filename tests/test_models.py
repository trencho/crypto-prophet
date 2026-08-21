"""Tests for the models package factory and BaseRegressionModel behavior."""

import asyncio

import pytest
from numpy import allclose, arange, column_stack

from models import make_model
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

    # `pytest.raises(Exception)` alone accepted any failure at all -- a TypeError or
    # ImportError from a factory broken on the lookup-miss path satisfied it just as well as
    # the intended rejection. `match` pins the message, which only the intended path produces.
    # (make_model raises a bare Exception; narrowing that type is a separate change.)
    with pytest.raises(Exception, match="The agent name NoSuchModel does not exist"):
        asyncio.run(make_model("NoSuchModel"))


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
