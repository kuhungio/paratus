import numpy as np

from paratus.transformers import SequenceTransformer

data = np.array([
    [1, 2, 2, 1],
    [0, 1, 1, 0],
    [0, 1, 2, 1, 0]
])


def test_transform_and_inverse_cut():
    seq = SequenceTransformer(4)

    transformed = seq.fit_transform(data)
    assert (transformed == np.array([
        [[0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]],
        [[0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
    ])).all()

    assert (seq.inverse_transform(transformed) == np.array([
        [1, 2, 2, 1],
        [0, 1, 1, 0],
        [1, 2, 1, 0]
    ])).all()


def test_transform_and_inverse_add_padding():
    seq = SequenceTransformer(5)

    transformed = seq.fit_transform(data)
    assert (transformed == np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
    ])).all()

    assert (seq.inverse_transform(transformed) == np.array([
        [1, 2, 2, 1],
        [0, 1, 1, 0],
        [0, 1, 2, 1, 0]
    ])).all()
