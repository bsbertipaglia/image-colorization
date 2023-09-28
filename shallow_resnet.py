import train_model
import tensorflow as tf
import numpy as np
import keras.api._v2.keras as K


class ResidualLayer(K.layers.Layer):
    def __init__(self, ffnn, **kwargs):
        super().__init__(**kwargs)
        self.ffnn = K.models.Sequential(ffnn)

    def call(self, inputs, *args, **kwargs):
        return self.ffnn(inputs) + inputs


def get_shallow_resnet():
    inputs = K.layers.Input(shape=(128, 128, 1))
    x = K.layers.BatchNormalization(axis=-1)(inputs)
    x = K.layers.Conv2D(256, 1, activation="linear", padding="SAME")(x)

    for i in range(5):
        x = ResidualLayer([
            K.layers.BatchNormalization(axis=-1),
            K.layers.Activation(tf.nn.leaky_relu),
            K.layers.Conv2D(128, 1, activation="linear", padding="SAME", use_bias=False),
            K.layers.BatchNormalization(axis=-1),
            K.layers.Activation(tf.nn.leaky_relu),
            K.layers.Conv2D(128, 3, activation="linear", padding="SAME", use_bias=False),
            K.layers.BatchNormalization(axis=-1),
            K.layers.Activation(tf.nn.leaky_relu),
            K.layers.Conv2D(256, 1, activation="linear", padding="SAME", use_bias=False),
        ])(x)

    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(3, 3, activation="sigmoid", padding="SAME")(x)

    shallow_resnet = K.models.Model(inputs=inputs, outputs=x)
    return shallow_resnet


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

    shallow_resnet = get_shallow_resnet()

    shallow_resnet.compile(
        tf.optimizers.legacy.Adam(1e-3),
        loss=tf.losses.MeanSquaredError()
    )

    train_model.train_dnn(X_train, X_val, shallow_resnet, "shallow_resnet", batch_size=16, save_only_weights=True)
