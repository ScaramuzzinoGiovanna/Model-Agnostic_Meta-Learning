import tensorflow as tf
from keras import optimizers, losses
import numpy as np


def train(lr, model, dataset):
    losses_ = []
    optimizer = optimizers.Adam(learning_rate=lr)
    loss_mse = losses.MeanSquaredError()
    total_loss = 0
    for i in range(len(dataset[0])):
        x = dataset[0][i]
        y = dataset[1][i]
        with tf.GradientTape(watch_accessed_variables=False) as outer_tape:
            outer_tape.watch(model.trainable_variables)
            with tf.GradientTape() as inner_tape:
                # Forward pass
                x = tf.convert_to_tensor(x)
                y = tf.convert_to_tensor(y)
                y_pred = model(x, training=True)
                # Compute the loss value
                inner_loss = loss_mse(y, y_pred)
            # Compute gradients
            trainable_vars = model.trainable_variables
            gradients_inner = inner_tape.gradient(inner_loss, trainable_vars)
            # copy model
            copied_model = tf.keras.models.clone_model(model)
            copied_model.set_weights(model.get_weights())
            # gradient descendent
            k = 0
            for j in range(1, len(copied_model.layers)):
                copied_model.layers[j].kernel = tf.subtract(model.layers[j].kernel,
                                                            tf.multiply(lr, gradients_inner[k]))
                copied_model.layers[j].bias = tf.subtract(model.layers[j].bias,
                                                          tf.multiply(lr, gradients_inner[k + 1]))
                k += 2
            # forward copied model
            y_pred_outer = copied_model(x, training=True)
            # calculate loss copied model
            loss_outer = loss_mse(y, y_pred_outer)
            # compute gradients of outer
        gradients_outer = outer_tape.gradient(loss_outer, model.trainable_variables)
        # Update weights of model
        optimizer.apply_gradients(zip(gradients_outer, model.trainable_variables))
        loss_outer = loss_outer.numpy()
        total_loss = total_loss + loss_outer
        avg_loss = total_loss / (i + 1.0)
        losses_.append(avg_loss)

        if i % 1000 == 0 and i > 0:
            print('Step {}: loss = {}'.format(i, avg_loss))

    return model, losses_
