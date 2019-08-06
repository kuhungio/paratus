import numpy as np
import pandas as pd
import timeit

from paratus.feature_extraction import CategoricalCombinations

data = pd.DataFrame(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 5, 9]],
    columns=['a', 'b', 'c'])

data2 = pd.DataFrame(
    [["1", np.nan, 3], ["4", 5, 6], ["7", 8, 9], ["1", 5, 9]],
    columns=['a', 'b', 'c'])


def test_combination_of_two_columns():
    model = CategoricalCombinations(['b', 'c'], 2, 2, 'new')
    transformed = model.fit_transform(data)
    assert (transformed.columns.values == [
        'a', 'b', 'c', 'new_b_c'
    ]).all()
    assert (model.get_new_column_names() == ['new_b_c'])
    assert (transformed.values == np.array([
        [1, 2, 3, 0],
        [4, 5, 6, 1],
        [7, 8, 9, 3],
        [1, 5, 9, 2]
    ])).all()


def test_encode_string():
    model = CategoricalCombinations(['a', 'c'], 2, 2, 'new')
    transformed = model.fit_transform(data2)
    assert (transformed.columns.values == [
        'a', 'b', 'c', 'new_a_c'
    ]).all()
    assert transformed.equals(
        pd.DataFrame(
            [
                ["1", np.nan, 3, 0],
                ["4", 5, 6, 2],
                ["7", 8, 9, 3],
                ["1", 5, 9, 1]
            ],
            columns=['a', 'b', 'c', 'new_a_c']).astype(transformed.dtypes)
    )


def test_encode_missing_number():
    model = CategoricalCombinations(['c', 'b'], 2, 2, 'new')
    transformed = model.fit_transform(data2)
    assert (transformed.columns.values == [
        'a', 'b', 'c', 'new_c_b'
    ]).all()

    assert (transformed.equals(pd.DataFrame([
        ["1", np.nan, 3, 0],
        ["4", 5, 6, 1],
        ["7", 8, 9, 3],
        ["1", 5, 9, 2]
    ], columns=['a', 'b', 'c', 'new_c_b']).astype(transformed.dtypes)))


def test_speed():
    dataset = np.random.randint(0, 100, (100000, 2))
    df = pd.DataFrame(dataset, columns=['a', 'b'])
    model = CategoricalCombinations(['a', 'b'], 2, 2, 'new')
    res = timeit.timeit(lambda: model.fit_transform(df), number=10)
    assert(res < 10)
