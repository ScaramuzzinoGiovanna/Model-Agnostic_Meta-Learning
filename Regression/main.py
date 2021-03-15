import os

import keras as k
import tensorflow as tf
from keras import optimizers, losses

import maml
import pretrained
import regressionModel as regression
from sinusoidGenerator import SinusoidGenerator
from utility import *


def eval(model, x_test, y_test, x_tab, y_tab, lr, num_steps):
    # addestro su un nuovo dataset il modello gia addestrato
    optimizer = optimizers.Adam(learning_rate=lr)
    loss_mse = losses.MeanSquaredError()
    predictions = []
    losses_ = []
    x_tab = tf.convert_to_tensor(x_tab)
    y_tab = tf.convert_to_tensor(y_tab)
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test)
    for step in range(max(num_steps) + 1):
        if step == 0:
            pred = model(x_tab, training=True)
            loss = loss_mse(y_tab, pred)
            predictions.append((step, pred))
            losses_.append(loss.numpy())
        else:
            with tf.GradientTape() as tape:
                # train
                pred_train = model(x_test, training=True)
                loss_train = loss_mse(y_test, pred_train)
            gradient = tape.gradient(loss_train, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            pred = model(x_tab, training=True)
            loss = loss_mse(y_tab, pred)
            losses_.append(loss.numpy())
            if step in num_steps:
                predictions.append((step, pred))
    return predictions, losses_


if __name__ == '__main__':
    evaluation = True
    K = 10
    lr_pretrained = 0.002
    # K = 5
    # lr_pretrained = 0.01
    lr_maml = 0.01

    trains_examples = 20000
    test_examples = 2
    # test_x_points = [0.0, 3.0, 3.5, 3.9, 5.0]
    test_x_points = [0.0, 1.0, 1.9, 2.5, 2.7, 3.0, 3.5, 3.9, 4.3, 5.0]
    epochs = 1
    n_steps = [0, 1, 10]
    folder_plots = 'plots/tasks' + str(trains_examples)

    if not os.path.isdir(folder_plots):
        print('create folder ')
        os.makedirs(folder_plots)  # create folder
    else:
        print('folder exist yet')

    model_pre = regression.regression_model()
    model_maml = regression.regression_model()
    sin_gen = SinusoidGenerator(K)
    dataset_train = sin_gen.create_dataset(trains_examples)

    if os.path.isdir('models/pretrained_K{}_task{}'.format(K, trains_examples)):
        print('load pretrained model')
        model_pretrained = k.models.load_model('models/pretrained_K{}_task{}'.format(K, trains_examples))
    else:
        print('train pretrained model')
        model_pretrained, losses_pretrained = pretrained.train(lr=lr_pretrained, epochs=epochs, model=model_pre,
                                                               dataset=dataset_train)
        plot_avgLosses(losses_pretrained, folder_plots, 'pretrained', K)
        model_pretrained.save('models/pretrained_K{}_task{}'.format(K, trains_examples))
        k.backend.clear_session()

    if os.path.isdir('models/MAML_K{}_tasks{}'.format(K, trains_examples)):
        print('load MAML model')
        model_maml = k.models.load_model('models/MAML_K{}_tasks{}'.format(K, trains_examples))
    else:
        print('train MAML model')
        model_maml, losses_maml = maml.train(lr=lr_maml, epochs=epochs, model=model_maml, dataset=dataset_train)
        plot_avgLosses(losses_maml, folder_plots, 'MAML', K)
        model_maml.save('models/MAML_K{}_tasks{}'.format(K, trains_examples))
        k.backend.clear_session()

    if evaluation:
        # dataset_test, x_tab, y_tab = sin_gen.create_dataset(test_examples, test=True)
        dataset_test, x_tab, y_tab = sin_gen.create_dataset_with_specific_point(test_x_points)
        # execute evaluation
        for i in range(len(dataset_test[0])):
            if len(dataset_test[0]) == 1:  # caso con punti x dati
                n_test = 'half points'
            else:
                n_test = i
            x_test = dataset_test[0]
            y_test = dataset_test[1]
            pred_maml, mse_maml = eval(model_maml, x_test[i], y_test[i], x_tab[i], y_tab[i], lr_maml, n_steps)
            plot_regression(pred_maml, x_tab[i], y_tab[i], x_test[i], y_test[i], K, n_test, "MAML", folder_plots)

            pred_pretrained, mse_pretrained = eval(model_pretrained, x_test[i], y_test[i], x_tab[i], y_tab[i],
                                                   lr_pretrained, n_steps)
            plot_regression(pred_pretrained, x_tab[i], y_tab[i], x_test[i], y_test[i], K, n_test, "pretrained",
                            folder_plots)

            plot_learning_curve(mse_maml, mse_pretrained, K, n_test, folder_plots)
