import train_model
import tensorflow as tf
import numpy as np
import keras.api._v2.keras as K


class ResidualLayer(K.layers.Layer):
    def __init__(self, ffnn, **kwargs):
        super().__init__(**kwargs)
        self.ffnn = K.models.Sequential(ffnn)

    def call(self, inputs, *args, **kwargs):
        return tf.concat((self.ffnn(inputs), inputs), axis=-1)


def get_dense_net():
    inputs = K.layers.Input(shape=(128, 128, 1))
    x = K.layers.Conv2D(filters=256, kernel_size=1, padding="SAME", activation="linear")(inputs)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    concatenated_inputs = x
    for f in [128] * 7:
        x = K.layers.Conv2D(filters=64, kernel_size=1, padding="SAME", activation="linear")(x)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.Activation(tf.nn.leaky_relu)(x)
        x = K.layers.Conv2D(filters=f, kernel_size=4, padding="SAME", activation="linear")(x)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.Activation(tf.nn.leaky_relu)(x)
        concatenated_inputs = K.layers.Concatenate()([concatenated_inputs, x])

    x = K.layers.Conv2D(filters=64, kernel_size=1, padding="SAME", activation="linear")(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(3, 3, activation="sigmoid", padding="SAME")(x)
    dense_net = K.models.Model(inputs=[inputs], outputs=[x])
    return dense_net


if __name__ == "__main__":
    print("0%")
    with open('cache/X_train.npy', 'rb') as f:
        X_train = np.load(f)
    print("33%")
    with open('cache/X_val.npy', 'rb') as f:
        X_val = np.load(f)
    print("66%")
    with open('cache/X_test.npy', 'rb') as f:
        X_test = np.load(f)
    print("100%")

    dense_net = get_dense_net()

    dense_net.compile(
        tf.optimizers.legacy.Adam(1e-3),
        loss=tf.losses.MeanSquaredError()
    )

    train_model.train_dnn(X_train, X_val, dense_net, "dense_net", batch_size=16, save_only_weights=True)
