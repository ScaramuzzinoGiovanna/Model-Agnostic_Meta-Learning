import tensorflow as tf
from keras import optimizers


def train(lr, epochs, model, dataset):
    losses_ = []
    total_loss = 0
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="mse", metrics=['accuracy'])
    for i in range(len(dataset[0])):
        x = dataset[0][i]
        y = dataset[1][i]
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        hist = model.fit(x=x, y=y, epochs=epochs, verbose=0)
        loss = hist.history['loss'][0]
        total_loss = total_loss + loss
        avg_loss = total_loss / (i + 1.0)
        losses_.append(avg_loss)

        if i % 1000 == 0 and i > 0:
            print('Step {}: loss = {}'.format(i, avg_loss))

    return model, losses_
