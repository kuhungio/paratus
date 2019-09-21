import numpy as np
import pandas as pd
from pytest import approx

from paratus.feature_extraction import CyclicalEncoder

data = pd.DataFrame(
    [[1, 2, 3], [2, 5, 6], [3, 8, 9], [4, 5, 9]],
    columns=['a', 'b', 'c'])

data2 = pd.DataFrame(
    [[1, 2, 3], [2, 5, 6], [np.nan, 8, 9], [4, 5, 9]],
    columns=['a', 'b', 'c'])


def test_encode_one_column():
    model = CyclicalEncoder(['a'])
    transformed = model.fit_transform(data)
    assert (transformed.columns.values == [
        'a', 'b', 'c', 'cyc_a_x', 'cyc_a_y'
    ]).all()
    assert (model.get_new_column_names() == ['cyc_a_x', 'cyc_a_y'])
    assert (transformed.values == approx(np.array([
        [1, 2, 3, 1, 0],
        [2, 5, 6, 0, 1],
        [3, 8, 9, -1, 0],
        [4, 5, 9, 0, -1]
    ])))


def test_encode_missing_number():
    model = CyclicalEncoder(['a'])
    transformed = model.fit_transform(data2)
    assert (transformed.columns.values == [
        'a', 'b', 'c', 'cyc_a_x', 'cyc_a_y'
    ]).all()
    print(transformed.values)
    np.testing.assert_array_almost_equal(
        transformed.values,
        np.array([
            [1., 2., 3., 1., 0.],
            [2., 5., 6., 0., 1.],
            [np.nan, 8., 9., np.nan, np.nan],
            [4., 5., 9., 0., -1.]
        ])
    )
