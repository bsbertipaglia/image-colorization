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


def get_unet():
    inputs = K.layers.Input(shape=(128, 128, 1))
    x = K.layers.BatchNormalization(axis=-1)(inputs)
    x = K.layers.Conv2D(16, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(16, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.MaxPool2D(2)(x)
    x = ResidualLayer([
        K.layers.Conv2D(32, 3, activation="linear", padding="SAME", use_bias=False),
        K.layers.BatchNormalization(axis=-1),
        K.layers.Activation(tf.nn.leaky_relu),
        K.layers.Conv2D(32, 3, activation="linear", padding="SAME", use_bias=False),
        K.layers.BatchNormalization(axis=-1),
        K.layers.Activation(tf.nn.leaky_relu),
        K.layers.MaxPool2D(2),
        ResidualLayer([
            K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False),
            K.layers.BatchNormalization(axis=-1),
            K.layers.Activation(tf.nn.leaky_relu),
            K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False),
            K.layers.BatchNormalization(axis=-1),
            K.layers.Activation(tf.nn.leaky_relu),
            K.layers.MaxPool2D(2),
            ResidualLayer([
                K.layers.Conv2D(128, 3, activation="linear", padding="SAME", use_bias=False),
                K.layers.BatchNormalization(axis=-1),
                K.layers.Activation(tf.nn.leaky_relu),
                K.layers.Conv2D(128, 3, activation="linear", padding="SAME", use_bias=False),
                K.layers.BatchNormalization(axis=-1),
                K.layers.Activation(tf.nn.leaky_relu),
                K.layers.MaxPool2D(2),
                ResidualLayer([
                    K.layers.Conv2D(256, 1, activation="linear", padding="SAME", use_bias=False),
                    K.layers.BatchNormalization(axis=-1),
                    K.layers.Activation(tf.nn.leaky_relu),
                    K.layers.Conv2D(256, 3, activation="linear", padding="SAME", use_bias=False),
                    K.layers.BatchNormalization(axis=-1),
                    K.layers.Activation(tf.nn.leaky_relu),
                ]),
                K.layers.UpSampling2D(2),
                K.layers.Conv2D(128, 3, activation="linear", padding="SAME", use_bias=False),
                K.layers.BatchNormalization(axis=-1),
                K.layers.Activation(tf.nn.leaky_relu),
                K.layers.Conv2D(128, 3, activation="linear", padding="SAME", use_bias=False),
                K.layers.BatchNormalization(axis=-1),
                K.layers.Activation(tf.nn.leaky_relu),
            ]),
            K.layers.UpSampling2D(2),
            K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False),
            K.layers.BatchNormalization(axis=-1),
            K.layers.Activation(tf.nn.leaky_relu),
            K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False),
            K.layers.BatchNormalization(axis=-1),
            K.layers.Activation(tf.nn.leaky_relu),
        ]),
        K.layers.UpSampling2D(2),
        K.layers.Conv2D(32, 3, activation="linear", padding="SAME", use_bias=False),
        K.layers.BatchNormalization(axis=-1),
        K.layers.Activation(tf.nn.leaky_relu),
        K.layers.Conv2D(32, 3, activation="linear", padding="SAME", use_bias=False),
        K.layers.BatchNormalization(axis=-1),
        K.layers.Activation(tf.nn.leaky_relu),
    ])(x)
    x = K.layers.UpSampling2D(2)(x)
    x = K.layers.Conv2D(16, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(16, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(3, 3, activation="sigmoid", padding="SAME")(x)

    unet = K.models.Model(inputs=inputs, outputs=x)
    return unet


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

    unet = get_unet()

    unet.compile(
        tf.optimizers.legacy.Adam(1e-3),
        loss=tf.losses.MeanSquaredError()
    )
    train_model.train_dnn(X_train, X_val, unet, "unet", batch_size=16, save_only_weights=True)
