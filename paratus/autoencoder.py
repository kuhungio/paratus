import tempfile
import tensorflow as tf
from tensorflow.keras import layers, losses, callbacks, models
import matplotlib.pyplot as plt

from paratus.baseModel import BaseModel


class Autoencoder(BaseModel):
    def __init__(self, embedding_size, batch_size=256, epochs=50, patience=50, output_activation=None, extra_layers=lambda: []):
        self.trainer = None
        self.transformer = None
        self.embedding_size = embedding_size
        self._output_activation = output_activation
        self._batch_size = batch_size
        self._epochs = epochs
        self._extra_layers = extra_layers
        self.__post = None
        self._history = None
        #self._model_tmp_path = tempfile.mktemp(prefix="paratus_")
        self._callbacks = [
            callbacks.EarlyStopping(
                patience=patience, restore_best_weights=True)
            # callbacks.ModelCheckpoint(filepath=self._model_tmp_path,
            #                           verbose=1, save_best_only=True)
        ]

    def fit(self, X, y=None):
        if y is None:
            y = X
        shape = X.shape
        n_features = shape[1]
        output_n_features = y.shape[1]
        (input_models, input_layers) = self._input_layers(X)
        if isinstance(input_layers, list) and len(input_layers) > 0:
            inputs = layers.Concatenate()(input_layers)
        else:
            inputs = input_layers
        pre = self._pre_embedding(inputs)
        embedding = self._embedding_layer(pre)
        post = self._post_embedding
        output = layers.Dense(output_n_features, name='output',
                              activation=self._output_activation)
        output_layer = output(post(embedding))
        self.trainer = models.Model(
            input_models, output_layer, name='autoencoder')
        self.trainer.summary()

        self.transformer = models.Model(
            input_models, embedding, name='transformer')
        self.transformer.summary()

        decoder_input = layers.Input(shape=(self.embedding_size,))
        self.inv_transformer = models.Model(
            decoder_input, output(post(decoder_input)), name='inv_transformer')
        self.inv_transformer.summary()

        self.trainer.compile(optimizer='rmsprop', loss=self._loss())
        self._history = self.trainer.fit(self._format_input(X), y,
                                         epochs=self._epochs,
                                         batch_size=self._batch_size,
                                         shuffle=True,
                                         validation_split=0.3,
                                         callbacks=self._callbacks,
                                         verbose=2)

    def transform(self, X):
        return self.transformer.predict(self._format_input(X))

    def inverse_transform(self, X):
        return self.inv_transformer.predict(X)

    def predict(self, X):
        return self.trainer.predict(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X if y is None else y)

    def _input_layers(self, X):
        n_features = X.shape[1]
        inputs = layers.Input(shape=(n_features,), name='autoencoder_input')
        return inputs, inputs

    def _format_input(self, X):
        return X

    def _embedding_layer(self, input_layer):
        return layers.Dense(self.embedding_size, activation='relu', name='embedding')(input_layer)

    def _pre_embedding(self, input_layer):
        res = input_layer
        for layer in self._extra_layers():
            res = layer(res)
        return res

    def _post_embedding(self, input_layer):
        if self.__post is None:
            self.__post = list(reversed(self._extra_layers()))
        res = input_layer
        for layer in self.__post:
            res = layer(res)
        return res

    def _loss(self):
        return 'mse'
        return losses.binary_crossentropy

    def plot_training_loss(self):
        if self._history is None:
            raise Exception(
                "Model should be trained before calling this method.")
        plt.plot(self._history.history['loss'])
        plt.plot(self._history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
