import numpy as np
from random import random
from tensorflow.keras import layers
from tensorflow import set_random_seed

from paratus.cat2vec import Cat2Vec
np.random.seed(123)
set_random_seed(123)


def test_small_reconstruction_error():
    data = np.random.randint(0, 2, (1024, 3))

    model = Cat2Vec(2, output_activation='sigmoid', batch_size=128, epochs=512,
                    extra_layers=lambda: [
                        layers.Dense(8, activation='tanh',
                                     name="tanh"+str(random()))
                    ])
    model.fit(data)
    reconstruction_error = (
        np.sqrt(np.mean((data - model.inverse_transform(model.transform(data)))**2)))
    assert reconstruction_error < 0.05  # Small reconstruction error
