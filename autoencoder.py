import tensorflow as tf
import numpy as np
import keras.api._v2.keras as K
import train_model


def get_autoencoder():
    inputs = K.layers.Input(shape=(128, 128, 1))
    x = K.layers.BatchNormalization()(inputs)
    x = K.layers.Conv2D(32, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(32, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(32, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)

    x = K.layers.MaxPool2D(2)(x)

    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)

    x = K.layers.MaxPool2D(2)(x)

    x = K.layers.Conv2D(128, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(128, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(128, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)

    x = K.layers.UpSampling2D(2)(x)

    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)

    x = K.layers.UpSampling2D(2)(x)

    x = K.layers.Conv2D(32, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(32, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(3, 1, activation="sigmoid", padding="SAME", use_bias=True)(x)

    autoencoder = K.models.Model(inputs=inputs, outputs=x)
    return autoencoder


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

    autoencoder = get_autoencoder()

    autoencoder.compile(
        tf.optimizers.legacy.Adam(1e-3),
        loss=tf.losses.MeanSquaredError()
    )
    train_model.train_dnn(X_train, X_val, autoencoder, "autoencoder", batch_size=128)
