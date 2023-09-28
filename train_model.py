import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras.api._v2.keras as K
import tensorflow_datasets as tfd
import time
from tqdm import tqdm
import pickle


def train_dnn(X_train, X_val, model, name, batch_size=128, epochs=30, save_only_weights=False):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss_value = model.loss(y, preds)
        grads = tape.gradient(loss_value, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train)) \
        .map(lambda x, y: (tf.reduce_mean(x, axis=-1, keepdims=True), tf.cast(y, tf.float32) / 255.)) \
        .cache() \
        .shuffle(buffer_size=1024) \
        .batch(batch_size)

    # .map(lambda x,y: (tf.image.random_flip_left_right(x),tf.image.random_flip_left_right(y))) \

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, X_val)) \
        .map(lambda x, y: (tf.reduce_mean(x, axis=-1, keepdims=True), tf.cast(y, tf.float32) / 255.)) \
        .cache() \
        .shuffle(buffer_size=1024) \
        .batch(batch_size)

    losses_history = []
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        train_losses = []
        for step, (x_batch_train, y_batch_train) in enumerate(tqdm(train_dataset)):
            train_losses.append(
                train_step(x_batch_train, y_batch_train)
            )

        # Run a validation loop at the end of each epoch.
        val_losses = []
        for x_batch_val, y_batch_val in val_dataset:
            val_losses.append(model.evaluate(x_batch_val, y_batch_val, verbose=0))

        print("Train loss: %.4f" % (float(np.mean(train_losses)),))
        print("Validation loss: %.4f" % (float(np.mean(val_losses)),))
        print("Time taken: %.2fs" % (time.time() - start_time))

        losses_history.append([float(np.mean(train_losses)), float(np.mean(val_losses))])

        if save_only_weights:
            model.save_weights(f"models/{name}/{epoch}")
        else:
            model.save(f"models/{name}/{epoch}")
        with open(f"models/{name}/loss_history", "wb+") as f:
            pickle.dump(losses_history, f)

    return losses_history


def train_pix2pix(X_train, generator, discriminator, name, alpha=1.0, batch_size=32, epochs=30, save_only_weights=False):
    @tf.function
    def train_step(x, y):
        generated_rgb_images = generator(x, training=False)
        with tf.GradientTape() as tape:
            preds_1 = discriminator([x, y])
            preds_0 = discriminator([x, generated_rgb_images])
            # BCE
            # loss = -(
            #     tf.reduce_mean(
            #         tf.math.log(preds_1 + 1e-8)
            #     ) + tf.reduce_mean(
            #         tf.math.log(1-preds_0 + 1e-8)
            #     )
            # )
            # MSE
            loss = tf.reduce_mean((1 - preds_1)**2) + tf.reduce_mean(preds_0**2)
        grads = tape.gradient(loss, discriminator.trainable_weights)
        discriminator.optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))


        with tf.GradientTape() as tape:
            generated_rgb_images = generator(x)
            preds = discriminator([x, generated_rgb_images])
            loss = generator.loss(y, generated_rgb_images) + alpha * (
                -tf.reduce_mean(
                    tf.math.log(preds + 1e-8)
                )
            )
        grads = tape.gradient(loss, generator.trainable_weights)
        generator.optimizer.apply_gradients(zip(grads, generator.trainable_weights))
        return tf.reduce_mean(preds)

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train)) \
        .map(lambda x, y: (tf.reduce_mean(x, axis=-1, keepdims=True), tf.cast(y, tf.float32) / 255.)) \
        .cache() \
        .shuffle(buffer_size=1024) \
        .batch(batch_size)

    for epoch in range(epochs):
        disc_losses = []
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            disc_losses.append(train_step(x_batch_train, y_batch_train))
            print(f"\r{step}/{train_dataset.cardinality().numpy()} - Avg disc pred: {np.mean(disc_losses[-3:])}", end="")

        print("Time taken: %.2fs" % (time.time() - start_time))


        if save_only_weights:
            generator.save_weights(f"models/{name}/{epoch}/generator")
            discriminator.save_weights(f"models/{name}/{epoch}/discriminator")
        else:
            generator.save(f"models/{name}/{epoch}/generator")
            discriminator.save(f"models/{name}/{epoch}/discriminator")


def train_pix2pix_random_crop(X_train, generator, discriminator, name, alpha=1.0, batch_size=32, epochs=30, save_only_weights=False, plot=False):
    @tf.function
    def train_step(x, y):
        generated_rgb_images = generator(x, training=False)
        with tf.GradientTape() as tape:
            preds_1 = discriminator([x, y])
            preds_0 = discriminator([x, generated_rgb_images])
            # BCE
            # loss = -(
            #     tf.reduce_mean(
            #         tf.math.log(preds_1 + 1e-8)
            #     ) + tf.reduce_mean(
            #         tf.math.log(1-preds_0 + 1e-8)
            #     )
            # )
            # MSE
            loss = tf.reduce_mean((1 - preds_1)**2) + tf.reduce_mean(preds_0**2)
        grads = tape.gradient(loss, discriminator.trainable_weights)
        discriminator.optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))


        with tf.GradientTape() as tape:
            generated_rgb_images = generator(x)
            preds = discriminator([x, generated_rgb_images])
            rec_loss = generator.loss(y, generated_rgb_images)
            loss = rec_loss + alpha * (
                -tf.reduce_mean(
                    tf.math.log(preds + 1e-8)
                )
            )
        grads = tape.gradient(loss, generator.trainable_weights)
        generator.optimizer.apply_gradients(zip(grads, generator.trainable_weights))
        return tf.reduce_mean(preds), rec_loss

    if plot:
        plt.ion()
        fig, ax = plt.subplots()

    for epoch in range(epochs):
        disc_losses = []
        rec_losses = []
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step in range(X_train.shape[0] // batch_size):
            # sample
            y_batch_train = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
            x_batch_train = y_batch_train.mean(axis=-1, keepdims=True)
            y_batch_train = y_batch_train / 255.
            # crop
            full = tf.concat((x_batch_train, y_batch_train), axis=-1)
            full_shape = list(full.shape)
            full_shape[1] = full_shape[2] = 128
            full = tf.image.random_crop(full, full_shape)
            x_batch_train = full[:,:,:,0,None]
            y_batch_train = full[:,:,:,1:]

            # train
            disc, rec = train_step(x_batch_train, y_batch_train)
            disc_losses.append(disc)
            rec_losses.append(rec)
            if plot:
                ax.clear()
                ax.imshow(generator(X_train[:1,:128,:128,:].mean(axis=-1, keepdims=True))[0])
                plt.pause(0.001)
            print(f"\r{step}/{X_train.shape[0] // batch_size} - Avg disc pred: {np.mean(disc_losses[-3:])} - Avg rec loss: {np.mean(rec_losses[-3:])}", end="")

        print("\nTime taken: %.2fs" % (time.time() - start_time))


        if save_only_weights:
            generator.save_weights(f"models/{name}/{epoch}/generator")
            discriminator.save_weights(f"models/{name}/{epoch}/discriminator")
        else:
            generator.save(f"models/{name}/{epoch}/generator")
            discriminator.save(f"models/{name}/{epoch}/discriminator")



