import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, callbacks, models

from paratus.autoencoder import Autoencoder


class Cat2Vec(Autoencoder):
    def _input_layers(self, X):
        n_features = X.shape[1]
        print("#features: {}".format(n_features))
        inputs = []
        outputs = []
        for i in range(n_features):
            size = len(np.unique(X[:, i]))
            model = models.Sequential()
            model.add(layers.Embedding(size, int(np.ceil(np.log2(size))),
                                       input_length=1,
                                       name="Embedding_{}".format(i)))
            model.add(layers.Flatten())
            inputs.append(model.input)
            outputs.append(model.output)
        return inputs, outputs

    def _format_input(self, X):
        n_features = X.shape[1]
        return [X[:, i] for i in range(n_features)]
