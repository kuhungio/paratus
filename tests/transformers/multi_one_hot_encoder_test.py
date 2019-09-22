import numpy as np
import pandas as pd

from paratus.transformers import MultiOneHotEncoder

data = pd.DataFrame(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 5, 9]],
    columns=['a', 'b', 'c'])

data2 = pd.DataFrame(
    [["1", np.nan, 3], ["4", 5, 6], ["7", 8, 9], ["1", 5, 9]],
    columns=['a', 'b', 'c'])


def test_encode_one_column():
    model = MultiOneHotEncoder(['b'], drop=None)
    transformed = model.fit_transform(data)
    assert (transformed.columns.values == [
        'a', 'c', 'b_0', 'b_1', 'b_2'
    ]).all()
    assert (transformed.values == np.array([
        [1, 3, 1, 0, 0],
        [4, 6, 0, 1, 0],
        [7, 9, 0, 0, 1],
        [1, 9, 0, 1, 0]
    ])).all()


def test_encode_one_column_drop_first():
    model = MultiOneHotEncoder(['b'])
    transformed = model.fit_transform(data)
    assert (transformed.columns.values == [
        'a', 'c', 'b_0', 'b_1'
    ]).all()
    assert (transformed.values == np.array([
        [1, 3, 0, 0],
        [4, 6, 1, 0],
        [7, 9, 0, 1],
        [1, 9, 1, 0]
    ])).all()


def test_encode_two_columns():
    model = MultiOneHotEncoder(['b', 'a'], drop=None)
    transformed = model.fit_transform(data)
    assert (transformed.columns.values == [
        'c', 'b_0', 'b_1', 'b_2', 'a_0', 'a_1', 'a_2'
    ]).all()
    assert (transformed.values == np.array([
        [3, 1, 0, 0, 1, 0, 0],
        [6, 0, 1, 0, 0, 1, 0],
        [9, 0, 0, 1, 0, 0, 1],
        [9, 0, 1, 0, 1, 0, 0]
    ])).all()


def test_encode_two_columns_drop_first():
    model = MultiOneHotEncoder(['b', 'a'])
    transformed = model.fit_transform(data)
    assert (transformed.columns.values == [
        'c', 'b_0', 'b_1', 'a_0', 'a_1'
    ]).all()
    assert (transformed.values == np.array([
        [3, 0, 0, 0, 0],
        [6, 1, 0, 1, 0],
        [9, 0, 1, 0, 1],
        [9, 1, 0, 0, 0]
    ])).all()


def test_encode_string():
    model = MultiOneHotEncoder(['a'], drop=None)

    transformed = model.fit_transform(data2)

    assert (transformed.columns.values == [
        'b', 'c', 'a_0', 'a_1', 'a_2'
    ]).all()

    assert transformed.equals(
        pd.DataFrame(
            [
                [np.nan, 3, 1, 0, 0],
                [5, 6, 0, 1, 0],
                [8, 9, 0, 0, 1],
                [5, 9, 1, 0, 0]
            ],
            columns=['b', 'c', 'a_0', 'a_1', 'a_2']).astype({
                'a_0': np.uint8, 'a_1': np.uint8, 'a_2': np.uint8
            })
    )


def test_encode_missing_number():
    model = MultiOneHotEncoder(['b'], drop=None)

    transformed = model.fit_transform(data2)

    assert (transformed.columns.values == [
        'a', 'c', 'b_0', 'b_1', 'b_2'
    ]).all()

    assert (transformed.equals(pd.DataFrame([
        ["1", 3, 1, 0, 0],
        ["4", 6, 0, 1, 0],
        ["7", 9, 0, 0, 1],
        ["1", 9, 0, 1, 0]
    ], columns=['a', 'c', 'b_0', 'b_1', 'b_2']).astype({
        'b_0': np.uint8, 'b_1': np.uint8, 'b_2': np.uint8
    })))
