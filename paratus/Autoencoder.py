import tempfile
import tensorflow as tf
from tensorflow.keras import layers, losses, callbacks, models
import matplotlib.pyplot as plt

from paratus.BaseModel import BaseModel


class Autoencoder(BaseModel):
    def __init__(self, embedding_size, batch_size=256, epochs=50, output_activation=None, extra_layers=lambda: []):
        self.trainer = None
        self.transformer = None
        self.embedding_size = embedding_size
        self._output_activation = None
        self._batch_size = batch_size
        self._epochs = epochs
        self._extra_layers = extra_layers
        self.__post = None
        self._history = None
        #self._model_tmp_path = tempfile.mktemp(prefix="paratus_")
        self._callbacks = [
            callbacks.EarlyStopping(patience=50, restore_best_weights=True)
            # callbacks.ModelCheckpoint(filepath=self._model_tmp_path,
            #                           verbose=1, save_best_only=True)
        ]

    def fit(self, X):
        shape = X.shape
        n_features = shape[1]
        self._batch_size
        inputs = layers.Input(
            shape=(n_features,), name='autoencoder_input')
        pre = self._pre_embedding(inputs)
        embedding = self._embedding_layer(pre)
        post = self._post_embedding
        output = layers.Dense(n_features, name='output',
                              activation=self._output_activation)
        output_layer = output(post(embedding))
        self.trainer = models.Model(
            inputs, output_layer, name='autoencoder')
        self.trainer.summary()

        self.transformer = models.Model(
            inputs, embedding, name='transformer')
        self.transformer.summary()

        decoder_input = layers.Input(shape=(self.embedding_size,))
        self.inv_transformer = models.Model(
            decoder_input, output(post(decoder_input)), name='inv_transformer')
        self.inv_transformer.summary()

        #self.trainer.add_loss(self._loss(inputs, output_layer))
        self.trainer.compile(optimizer='rmsprop', loss=self._loss())
        print(X.shape)
        self._history = self.trainer.fit(X, X,
                                         epochs=self._epochs,
                                         batch_size=self._batch_size,
                                         shuffle=True,
                                         validation_split=0.2,
                                         callbacks=self._callbacks)
        # self.trainer.load_weights(self._model_tmp_path)

    def transform(self, X):
        return self.transformer.predict(X)

    def inverse_transform(self, X):
        return self.inv_transformer.predict(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X if y is None else y)

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
            print("res -->", res)
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
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
