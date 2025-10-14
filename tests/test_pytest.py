import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def test_addition():
    assert 2 + 2 == 4


def test_string():
    s = "mlops"
    assert "ml" in s


@pytest.fixture
def sample_data():
    return [1, 2, 3]


def test_sum(sample_data):
    assert sum(sample_data) == 6


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (2, 3, 5),
        (0, 0, 0),
    ],
)
def test_add(a, b, expected):
    assert a + b == expected


@pytest.mark.slow
def test_big_comp():
    assert sum(range(10000000)) > 0


def divide(a, b):
    return a / b


def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)


@pytest.mark.skip(reason="функция не реализована")
def test_future_feature():
    pass


@pytest.mark.xfail(sys.platform == "darwin", reason="баг на Винде")
def test_platform_spec():
    assert 1 / 0


@pytest.fixture
def input_data():
    return pd.DataFrame(
        {"age": [20, 30, 40], "gender": ["M", "F", "M"], "income": [100, 200, 300]}
    )


def test_no_missing_vals(input_data):
    assert not input_data.isnull().any().any(), "должно быть без пропусков"


def test_age_range(input_data):
    assert input_data["age"].between(0, 120).all(), "возраст вне диапазона"


BASELINE_ACC = 0.7


def test_model_regression():
    X = np.random.rand(200, 3)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression().fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    assert acc >= BASELINE_ACC, f"Модель деградировала! {acc} < {BASELINE_ACC}"


def test_model_prediction_consistency():
    X = np.random.rand(100, 3)
    y = (X[:, 0] > 0.5).astype(int)

    model = LogisticRegression().fit(X, y)
    preds1 = model.predict(X[:5])
    preds2 = model.predict(X[:5])

    assert (preds1 == preds2).all(), "Модель должна быть детерминированной"
