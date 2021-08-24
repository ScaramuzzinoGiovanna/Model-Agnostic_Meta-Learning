#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import datetime
import os
import time

import keras as k
import numpy as np
import tensorflow as tf
from keras import losses, optimizers

from model import *
from taskGenerator import TaskGenerator

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.set_visible_devices(physical_devices[0], 'GPU')


def get_accuracy(y, y_pred):
    accuracy = k.metrics.Accuracy()
    accuracy.update_state(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    return accuracy.result().numpy()


class MAML:
    def __init__(self, model_name, out_size):

        self.model_name = model_name
        self.model = model_name().build_model(out_size=out_size)
        self.loss_cross_entropy = losses.CategoricalCrossentropy()
        self.out_size = out_size
        self.best_loss = 999999
        self.best_acc = -999999

    def train(self, update_steps_train, gen_batch, lr_inner, lr_outer, n_interations, log_dir, write_log=True):
        # MAML algorithm
        if write_log == True:
            summary_writer = tf.summary.create_file_writer(logdir=log_dir)
        optimizer = optimizers.Adam(learning_rate=lr_outer)
        losses_ = []
        accuracies_ = []

        for iteration in range(n_interations):
            self.iteration_train = iteration  # for keyboard interruption
            batch = gen_batch.get_batch('train')
            tot_batch_loss = 0
            tot_batch_accuracy = 0
            with tf.GradientTape(watch_accessed_variables=False) as outer_tape:
                outer_tape.watch(self.model.trainable_variables)
                for i in range(len(batch)):
                    copied_model = self.model_name().build_model(out_size=self.out_size)
                    copied_model = self.set_weights_(copied_model, self.model)
                    support_x, support_y, query_x, query_y = batch[i]  # task
                    for step in range(update_steps_train):
                        if step == 0:
                            inner_weights = self.generate_inner_weights(copied_model)
                        with tf.GradientTape() as inner_tape:
                            inner_tape.watch(inner_weights)
                            support_y_pred = copied_model(support_x, training=True)
                            inner_loss = self.loss_cross_entropy(support_y, support_y_pred)
                        gradients_inner = inner_tape.gradient(inner_loss, inner_weights)
                        # update weights
                        copied_model, inner_weights = self.update_weights(copied_model, lr_inner,
                                                                          gradients_inner)
                    query_y_pred = copied_model(query_x, training=True)
                    outer_loss = self.loss_cross_entropy(query_y, query_y_pred)
                    accuracy = get_accuracy(query_y, query_y_pred)
                    tot_batch_loss = tot_batch_loss + outer_loss
                    tot_batch_accuracy = tot_batch_accuracy + accuracy
                avg_loss = tot_batch_loss / len(batch)
                avg_acc = tot_batch_accuracy / len(batch)
                losses_.append(avg_loss)
                accuracies_.append(avg_acc)

            # Compute second order gradients - gradient on avg loss and apply it to model
            gradients_outer = outer_tape.gradient(avg_loss,
                                                  self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients_outer, self.model.trainable_variables))

            if write_log == True:
                with summary_writer.as_default():
                    tf.summary.scalar('epoch_loss_avg', avg_loss.numpy(), step=iteration)
                    tf.summary.scalar('epoch_accuracy', avg_acc, step=iteration)

            if iteration % 1 == 0 and iteration > 0:
                eval_avg_loss, eval_avg_acc = self.validation()
                if write_log == True:
                    with summary_writer.as_default():
                        tf.summary.scalar('eval_epoch_loss_avg', eval_avg_loss.numpy(), step=iteration)
                        tf.summary.scalar('eval_epoch_accuracy', eval_avg_acc, step=iteration)

            if iteration % 1000 == 0 and iteration > 0:
                print('batch: {}------ query loss: {}    query accuracy: {}'.format(iteration, avg_loss.numpy(),
                                                                                    avg_acc))
                print('Evaluation --- batch: {}------ query loss: {}  query accuracy: {}'.format(iteration,
                                                                                                 eval_avg_loss.numpy(),
                                                                                                 eval_avg_acc))
            if iteration % 1000 == 0 and iteration > 0:
                print('model ' + str(iteration) + ' saved')
                tf.train.Checkpoint(self.model).save(folder_name + '/_model_' + str(iteration + 1) + '.ckpt')

        return losses_, accuracies_

    def validation(self):
        # validate MAML
        test_set = batch_generator.get_batch('val')
        tot_batch_loss = 0
        tot_batch_accuracy = 0
        with tf.GradientTape(watch_accessed_variables=False) as outer_tape:
            outer_tape.watch(self.model.trainable_variables)
            for i in range(len(test_set)):  # bach-size -tasks
                copied_model = self.model_name().build_model(out_size=self.out_size)
                copied_model = self.set_weights_(copied_model, self.model)
                support_x, support_y, query_x, query_y = test_set[i]  # task
                for step in range(update_steps_train):
                    if step == 0:
                        inner_weights = self.generate_inner_weights(copied_model)
                    with tf.GradientTape() as inner_tape:
                        inner_tape.watch(inner_weights)
                        support_y_pred = copied_model(support_x, training=True)
                        inner_loss = self.loss_cross_entropy(support_y, support_y_pred)
                    gradients_inner = inner_tape.gradient(inner_loss, inner_weights)
                    # update weights
                    copied_model, inner_weights = self.update_weights(copied_model, lr_inner, gradients_inner)

                query_y_pred = copied_model(query_x, training=True)
                outer_loss = self.loss_cross_entropy(query_y, query_y_pred)
                accuracy = get_accuracy(query_y, query_y_pred)
                tot_batch_loss = tot_batch_loss + outer_loss
                tot_batch_accuracy = tot_batch_accuracy + accuracy
            avg_loss = tot_batch_loss / len(test_set)  # è un tensore
            avg_acc = tot_batch_accuracy / len(test_set)
            if avg_loss < self.best_loss:
                self.model.save(folder_name + '/best_model_loss')  # save always best
                self.best_loss = avg_loss
            if avg_acc > self.best_acc:
                self.model.save(folder_name + '/best_model_acc')
                self.best_acc = avg_acc

            return avg_loss, avg_acc

    def update_weights(self, model, lr_inner, gradients_inner):
        # update the weights of the model
        copied_model = model
        k = 0
        inner_weights = []
        layers = ['input', 're_lu', 'max_pooling2d', 'flatten']
        for j in range(len(copied_model.layers)):
            if 'batch_normalization' in copied_model.layers[j].name:
                copied_model.layers[j].gamma = model.layers[j].gamma - (lr_inner * gradients_inner[k])
                copied_model.layers[j].beta = model.layers[j].beta - (lr_inner * gradients_inner[k + 1])
                inner_weights.append(copied_model.layers[j].gamma)
                inner_weights.append(copied_model.layers[j].beta)
            elif any([l in copied_model.layers[j].name for l in layers]):
                k -= 2
                pass
            else:
                copied_model.layers[j].kernel = model.layers[j].kernel - (
                        lr_inner * gradients_inner[k])
                copied_model.layers[j].bias = model.layers[j].bias - (
                        lr_inner * gradients_inner[k + 1])
                inner_weights.append(copied_model.layers[j].kernel)
                inner_weights.append(copied_model.layers[j].bias)
            k += 2
        return copied_model, inner_weights

    def set_weights_(self, copied_model, model):
        # set the copied_model weights by copying them from the model
        layers = ['input', 're_lu', 'max_pooling2d', 'flatten']

        for j in range(len(copied_model.layers)):
            if 'batch_normalization' in copied_model.layers[j].name:
                copied_model.layers[j].gamma = model.layers[j].gamma
                copied_model.layers[j].beta = model.layers[j].beta
            elif any([l in copied_model.layers[j].name for l in layers]):
                pass
            else:
                copied_model.layers[j].kernel = model.layers[j].kernel
                copied_model.layers[j].bias = model.layers[j].bias

        return copied_model

    def generate_inner_weights(self, copied_model):
        # generate the weights witch must be observed from the inner_tape
        layers = ['input', 're_lu', 'max_pooling2d', 'flatten']
        inner_weights = []
        for j, v in enumerate(copied_model.layers):
            if any([l in self.model.layers[j].name for l in layers]):
                pass
            elif 'batch_normalization' in v.name:
                a = copied_model.layers[j].gamma
                b = copied_model.layers[j].beta
                inner_weights.append(a)
                inner_weights.append(b)
            else:
                a = copied_model.layers[j].kernel
                b = copied_model.layers[j].bias
                inner_weights.append(a)
                inner_weights.append(b)
        return inner_weights

    def test(self, lr, update_steps_test, batch_generator, n_interations):
        # test the MAML trained model with new tasks
        acc_ = []

        for iteration in range(n_interations):
            test_set = batch_generator.get_batch('test')
            tot_batch_loss = 0
            tot_batch_accuracy = 0

            if iteration % 100 == 0:
                print(str(iteration) + ' batch tested')
            for i in range(len(test_set)):
                copied_model = self.model_name().build_model(out_size=self.out_size)
                copied_model = self.set_weights_(copied_model, self.model)
                support_x, support_y, query_x, query_y = test_set[i]  # task
                for step in range(update_steps_test):
                    if step == 0:
                        inner_weights = self.generate_inner_weights(copied_model)
                    with tf.GradientTape() as inner_tape:
                        inner_tape.watch(inner_weights)
                        support_y_pred = copied_model(support_x, training=True)
                        inner_loss = self.loss_cross_entropy(support_y, support_y_pred)
                    gradients_inner = inner_tape.gradient(inner_loss, inner_weights)
                    # update weights
                    copied_model, inner_weights = self.update_weights(copied_model, lr, gradients_inner)

                query_y_pred = copied_model(query_x, training=True)
                outer_loss = self.loss_cross_entropy(query_y, query_y_pred)
                accuracy = get_accuracy(query_y, query_y_pred)
                # print('step' + str(iteration) + 'batch' + str(i) + '-----------------acc' + str(accuracy))
                tot_batch_loss = tot_batch_loss + outer_loss
                tot_batch_accuracy = tot_batch_accuracy + accuracy
            avg_loss = tot_batch_loss / len(test_set)  # è un tensore
            avg_acc = tot_batch_accuracy / len(test_set)
            acc_.append(avg_acc)
        mean_acc = sum(acc_) / len(acc_)
        std = np.std(acc_)
        ci95 = 1.96 * std / np.sqrt(n_interations_test)
        print('mean accuracy', mean_acc)
        print('stds', std)
        print('ci95', ci95)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default='omniglot', choices=['omniglot', 'miniimagenet'])
    ap.add_argument("-t", "--type", default='test', choices=['train', 'test'])
    ap.add_argument("-n_way", "--n_way", default='20')  # class used for classification
    ap.add_argument("-shot_num", "--shot_num", default='5')  # example per class of support set
    ap.add_argument("-query_num", "--query_num", default='5')  # query images
    ap.add_argument("-update_steps_train", default="1")
    ap.add_argument("-update_steps_test", default="3")
    ap.add_argument("-lr", default='0.4')
    ap.add_argument("-meta_batch_size", default='32')
    ap.add_argument("-model_name", default='MiniimagenetConvModel',
                    choices=['MiniimagenetConvModel', 'OmniglotConvModel'])

    ap.add_argument("-iterations_train", "--iterations_train", default='60000')

    #                       ---------------caso 5way 5shot omniglot-------------------   (in omniglot scegliere tra conv e non conv model)
    # -d omniglot -t train -n_way 5 -shot_num 5 -query_num 5 -update_steps_train 1 -update_steps_test 3 -lr 0.4 -meta_batch_size 32 -model_name OmniglotConvModel
    # -d omniglot -t test -n_way 5 -shot_num 5 -query_num 5 -update_steps_train 1 -update_steps_test 3 -lr 0.4 -meta_batch_size 32 -model_name OmniglotConvModel

    #                       --------------caso 20way 5shot omniglot-----------------------
    # -d omniglot -t train -n_way 20 -shot_num 5 -query_num 5 -update_steps_train 5 -update_steps_test 5 -lr 0.1 -meta_batch_size 16 -model_name OmniglotConvModel
    # -d omniglot -t test -n_way 20 -shot_num 5 -query_num 5 -update_steps_train 5 -update_steps_test 5 -lr 0.1 -meta_batch_size 16 -model_name OmniglotConvModel

    #                       ---------------caso 5way 1shot omniglot-------------------
    # -d omniglot -t train -n_way 5 -shot_num 1 -query_num 1 -update_steps_train 1 -update_steps_test 3 -lr 0.4 -meta_batch_size 32 -model_name OmniglotConvModel
    # -d omniglot -t test -n_way 5 -shot_num 1 -query_num 1 -update_steps_train 1 -update_steps_test 3 -lr 0.4 -meta_batch_size 32 -model_name OmniglotConvModel

    #                       --------------caso 20way 1shot omniglot-----------------------
    # -d omniglot -t train -n_way 20 -shot_num 1 -query_num 1 -update_steps_train 5 -update_steps_test 5 -lr 0.1 -meta_batch_size 16 -model_name OmniglotConvModel
    # -d omniglot -t test -n_way 20 -shot_num 1 -query_num 1 -update_steps_train 5 -update_steps_test 5 -lr 0.1 -meta_batch_size 16 -model_name OmniglotConvModel

    #                       -----------------caso 5way 1shot miniimagenet-------------------
    # -d miniimagenet -t train -n_way 5 -shot_num 1 -query_num 15 -update_steps_train 5 -update_steps_test 10 -lr 0.01 -meta_batch_size 4 -model_name MiniimagenetConvModel
    # -d miniimagenet -t test -n_way 5 -shot_num 1 -query_num 15 -update_steps_train 5 -update_steps_test 10 -lr 0.01 -meta_batch_size 4 -model_name MiniimagenetConvModel

    #                     --------------------caso 5way 5shot miniimagenet-------------------
    # -d miniimagenet -t train -n_way 5 -shot_num 5 -query_num 15 -update_steps_train 5 -update_steps_test 10 -lr 0.01 -meta_batch_size 2 -model_name MiniimagenetConvModel
    # -d miniimagenet -t test -n_way 5 -shot_num 5 -query_num 15 -update_steps_train 5 -update_steps_test 10 -lr 0.01 -meta_batch_size 2 -model_name MiniimagenetConvModel

    args = vars(ap.parse_args())
    dataset = args['dataset']
    type = args['type']
    n_way = int(args['n_way'])
    shot_num = int(args['shot_num'])
    query_num = int(args['query_num'])
    update_steps_train = int(args['update_steps_train'])
    update_steps_test = int(args['update_steps_test'])
    lr_inner = float(args['lr'])
    n_interations_train = int(args['iterations_train'])
    meta_batch_size = int(args['meta_batch_size'])
    model_name = args['model_name']

    if model_name == 'MiniimagenetConvModel':
        model = MiniimagenetConvModel

    elif model_name == 'OmniglotConvModel':
        model = OmniglotConvModel

    out_size = n_way
    lr_outer = 0.001

    folder_name = 'models_' + str(dataset) + '/maml_' + str(n_way) + 'way_' + str(shot_num) + 'shot_' + str(
        model_name) + '_'

    batch_generator = TaskGenerator(dataset, n_way, shot_num, query_num, meta_batch_size)
    maml = MAML(model_name=model, out_size=out_size)
    try:
        if type == "train":
            log_dir = 'logs_' + str(dataset) + '/' + str(n_way) + 'way_' + str(shot_num) + 'shot_' + str(
                model_name) + '_' + str(
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + '/'
            print(' train maml model with: ' + str(n_way) + 'way-' + str(shot_num) + 'shot')

            if not os.path.isdir(folder_name):
                print('create folder ')
                os.makedirs(folder_name)  # create folder
            else:
                print('base folder exist yet, create new ')
                folder_name = folder_name + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                os.makedirs(folder_name)
            start = time.time()
            loss, acc = maml.train(update_steps_train, batch_generator, lr_inner, lr_outer, n_interations_train,
                                   log_dir, write_log=True)
            print('run time : ', time.time() - start)
    except KeyboardInterrupt:
        tf.train.Checkpoint(maml.model).save(folder_name + '/_model_' + str(maml.iteration_train) + '.ckpt')

    if type == "test":
        n_interations_test = 600
        print('test maml model with: ' + str(n_way) + 'way-' + str(shot_num) + 'shot')

        maml.model = tf.keras.models.load_model(folder_name + '/best_model_acc')
        maml.test(lr_inner, update_steps_test, batch_generator, n_interations_test)
