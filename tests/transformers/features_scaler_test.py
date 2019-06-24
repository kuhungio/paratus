import numpy as np
from paratus.transformers import FeatureScaler

data = np.array([[1], [2], [3]])
data2 = np.array([[1, 0], [2, 4], [3, 2]])
data3 = np.array([[1, 0], [1, 1]])
data4 = np.array([[1], [2], [3], [2]])


def test_one_feature_scaler():
    """
    Tests that an array with as single column can be scaled
    """
    scaler = FeatureScaler()
    scaler.fit(data)
    transformed = scaler.transform(data)
    assert (transformed == np.array([[0], [0.5], [1]])).all()
    assert (scaler.inverse_transform(transformed) == data).all()


def test_two_feature_scaler():
    """
    Tests that an array with as single column can be scaled
    """
    scaler = FeatureScaler()
    transformed = scaler.fit_transform(data2)
    assert (transformed == np.array([[0, 0], [0.5, 1], [1, 0.5]])).all()
    assert (scaler.inverse_transform(transformed) == data2).all()
    assert (scaler.transform(data3) == np.array([[0, 0], [0, 0.25]])).all()


def test_feature_scale_different_row_sizer():
    """
    Tests that an array with as single column can be scaled
    """
    scaler = FeatureScaler()
    scaler.fit(data)
    transformed = scaler.transform(data4)
    assert (transformed == np.array([[0], [0.5], [1], [0.5]])).all()
    assert (scaler.inverse_transform(transformed) == data4).all()
