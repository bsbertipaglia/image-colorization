import train_model
import tensorflow as tf
import numpy as np
import keras.api._v2.keras as K


def get_dnet():
    inputs = K.layers.Input(shape=(128, 128, 1))
    x = K.layers.BatchNormalization(axis=-1)(inputs)
    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)

    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)

    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)

    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)

    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(3, 3, activation="sigmoid", padding="SAME", use_bias=True)(x)

    dnet = K.models.Model(inputs=inputs, outputs=x)
    return dnet


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

    dnet = get_dnet()

    dnet.compile(
        tf.optimizers.legacy.Adam(1e-3),
        loss=tf.losses.MeanSquaredError()
    )
    train_model.train_dnn(X_train, X_val, dnet, "dnet", batch_size=64)
