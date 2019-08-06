import numpy as np
import pandas as pd
import timeit

from paratus.feature_extraction import FrequencyEncoding

data = pd.DataFrame(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 5, 9]],
    columns=['a', 'b', 'c'])

data2 = pd.DataFrame(
    [["1", np.nan, 3], ["4", 5, 6], ["7", 8, 9], ["1", 5, 9]],
    columns=['a', 'b', 'c'])


def test_encode_one_column():
    model = FrequencyEncoding(['a'])
    transformed = model.fit_transform(data)
    print(transformed)
    assert (transformed.columns.values == [
        'a', 'b', 'c', 'freq_a'
    ]).all()
    assert (model.get_new_column_names() == ['freq_a'])
    assert (transformed.values == np.array([
        [1, 2, 3, 0.5],
        [4, 5, 6, 0.25],
        [7, 8, 9, 0.25],
        [1, 5, 9, 0.5]
    ])).all()


def test_encode_two_columns():
    model = FrequencyEncoding(['c', 'a'])
    transformed = model.fit_transform(data)
    print(transformed)
    assert (transformed.columns.values == [
        'a', 'b', 'c', 'freq_c', 'freq_a'
    ]).all()
    assert (model.get_new_column_names() == ['freq_c', 'freq_a'])
    assert (transformed.values == np.array([
        [1, 2, 3, 0.25, 0.5],
        [4, 5, 6, 0.25, 0.25],
        [7, 8, 9, 0.5, 0.25],
        [1, 5, 9, 0.5, 0.5]
    ])).all()


def test_encode_string():
    model = FrequencyEncoding(['a'], 'new')
    transformed = model.fit_transform(data2)
    assert (transformed.columns.values == [
        'a', 'b', 'c', 'new_a'
    ]).all()
    assert transformed.equals(
        pd.DataFrame(
            [
                ["1", np.nan, 3, 0.5],
                ["4", 5, 6, 0.25],
                ["7", 8, 9, 0.25],
                ["1", 5, 9, 0.5]
            ],
            columns=['a', 'b', 'c', 'new_a']).astype(transformed.dtypes)
    )


def test_encode_missing_number():
    model = FrequencyEncoding(['b'], 'new')
    transformed = model.fit_transform(data2)
    assert (transformed.columns.values == [
        'a', 'b', 'c', 'new_b'
    ]).all()
    assert transformed.equals(
        pd.DataFrame(
            [
                ["1", np.nan, 3, 0.25],
                ["4", 5, 6, 0.5],
                ["7", 8, 9, 0.25],
                ["1", 5, 9, 0.5]
            ],
            columns=['a', 'b', 'c', 'new_b']).astype(transformed.dtypes)
    )


def test_handle_unseen_values():
    model = FrequencyEncoding(['b'], 'new')
    model.fit(data2)
    transformed = model.transform(data)
    assert (transformed.columns.values == [
        'a', 'b', 'c', 'new_b'
    ]).all()
    print(transformed)
    assert transformed.equals(
        pd.DataFrame(
            [
                [1, 2, 3, 0],
                [4, 5, 6, 0.5],
                [7, 8, 9, 0.25],
                [1, 5, 9, 0.5]
            ],
            columns=['a', 'b', 'c', 'new_b']).astype(transformed.dtypes)
    )
