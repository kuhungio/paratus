from tensorflow.keras import layers, losses
from tensorflow.keras import backend as K

from paratus.autoencoder import Autoencoder


class VariationalAutoencoder(Autoencoder):
    def __init__(self, embedding_size,  *args, mse_loss=False, **kwargs):
        self.z_mean = None
        self.z_log_var = None
        self._mse_loss = mse_loss
        return super().__init__(embedding_size, *args, **kwargs)

    def _embedding_layer(self, input_layer):
        def _sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(self.embedding_size,))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        self.z_mean = layers.Dense(self.embedding_size)(input_layer)
        self.z_log_var = layers.Dense(self.embedding_size)(input_layer)
        return layers.Lambda(lambda x: _sampling(x), output_shape=(self.embedding_size,))([self.z_mean, self.z_log_var])

    def _loss(self):
        def loss(y_true, y_pred):
            if self._mse_loss:
                reconstruction_loss = losses.mse(y_true, y_pred)
            else:
                reconstruction_loss = losses.binary_crossentropy(
                    y_true, y_pred)

            reconstruction_loss *= self.embedding_size
            kl_loss = 1 + self.z_log_var - \
                K.square(self.z_mean) - K.exp(self.z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            return K.mean(reconstruction_loss + kl_loss)
        return loss
