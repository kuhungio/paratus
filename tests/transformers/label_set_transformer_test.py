import numpy as np

from paratus.transformers import LabelSetTransformer


def test_lst_one_element():
    data = np.array([
        [1],
        [3],
        [2]
    ])
    transformer = LabelSetTransformer()
    assert (transformer.fit_transform(data) == np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])).all()
    assert transformer.names() == [1, 2, 3]

    assert (transformer.transform(np.array([
        [4], [2]
    ])) == np.array([
        [0, 0, 0],
        [0, 1, 0]
    ])).all()

    assert (transformer.inverse_transform(np.array([
        [0, 0, 0],
        [0, 1, 0]
    ])) == np.array([
        [],
        [2]
    ])).all()


def test_lst_multi_element():
    transformer = LabelSetTransformer()
    assert (transformer.fit_transform(np.array([
        [1, 2, 3],
        [2, 4],
        [3]
    ])) == np.array([
        [1, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ])).all()

    assert transformer.names() == [1, 2, 3, 4]

    assert (transformer.inverse_transform(np.array([
        [0, 1, 0, 1],
        [1, 0, 0, 1]
    ])) == np.array([[2, 4], [1, 4]])).all()
