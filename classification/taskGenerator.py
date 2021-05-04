import os
import random
import time

import cv2
import numpy as np
import tensorflow as tf


class TaskGenerator:
    def __init__(self, dataset_name, n_way, shot_num, query_num, meta_batch_size):
        '''
        :param dataset_name: dataset name
        :param n_way: a train task contains images from different N classes
        :param shot_num: k images used for meta-train
        :param query_num: k images used for meta-test
        :param meta_batch_size: the number of tasks in a batch
        '''
        self.dataset = dataset_name
        self.meta_batch_size = meta_batch_size
        self.n_way = n_way
        self.shot_num = shot_num
        self.query_num = query_num

        if self.dataset == 'miniimagenet':
            META_TRAIN_DIR = 'datasets/MiniImagenet/train'
            META_VAL_DIR = 'datasets/MiniImagenet/val'
            META_TEST_DIR = 'datasets/MiniImagenet/test'

            self.metatrain_folders = [os.path.join(META_TRAIN_DIR, label) \
                                      for label in os.listdir(META_TRAIN_DIR) \
                                      if os.path.isdir(os.path.join(META_TRAIN_DIR, label))
                                      ]
            self.metaval_folders = [os.path.join(META_VAL_DIR, label) \
                                    for label in os.listdir(META_VAL_DIR) \
                                    if os.path.isdir(os.path.join(META_VAL_DIR, label))
                                    ]
            self.metatest_folders = [os.path.join(META_TEST_DIR, label) \
                                     for label in os.listdir(META_TEST_DIR) \
                                     if os.path.isdir(os.path.join(META_TEST_DIR, label))
                                     ]

        if self.dataset == 'omniglot':
            if self.shot_num != self.query_num:
                self.query_num = self.shot_num
            DATA_FOLDER = 'datasets/Omniglot'
            character_folders = [
                os.path.join(DATA_FOLDER, family, character) \
                for family in os.listdir(DATA_FOLDER) \
                if os.path.isdir(os.path.join(DATA_FOLDER, family)) \
                for character in os.listdir(os.path.join(DATA_FOLDER, family))
            ]
            # Shuffle dataset
            random.seed(9314)
            random.shuffle(character_folders)

            n_val = 100
            n_train = 1200 - n_val
            self.metatrain_folders = character_folders[:n_train]
            self.metaval_folders = character_folders[n_train:n_train + n_val]
            self.metatest_folders = character_folders[n_train + n_val:]

        # Record the relationship between image label and the folder name in each task
        self.label_map = []

    def get_batch(self, type):
        # return batch set for type
        if type == 'train':
            folders = self.metatrain_folders
        elif type == 'val':
            folders = self.metaval_folders
        elif type == 'test':
            folders = self.metatest_folders
        else:
            raise Exception('error type dataset split selected')

        batch_set = []
        self.label_map = []
        for i in range(self.meta_batch_size):
            folders_idx = np.array(np.random.choice(len(folders), self.n_way, False))
            sampled_folders = np.array(folders)[folders_idx].tolist()
            labels = np.arange(self.n_way).tolist()
            np.random.shuffle(labels)
            folder_with_label = list(zip(sampled_folders, labels))
            support_x, support_y, query_x, query_y = self.generate_set(folder_with_label)
            batch_set.append((support_x, support_y, query_x, query_y))
        return batch_set

    def shuffle_set(self, set_x, set_y):
        # Shuffle sets
        set_seed = random.randint(0, 100)
        random.seed(set_seed)
        random.shuffle(set_x)
        random.seed(set_seed)
        random.shuffle(set_y)
        return set_x, set_y

    def extract_images(self, image_file):
        # reads and preprocesses the images
        if self.dataset == 'omniglot':
            img = cv2.imread(image_file)
            if img.shape[0] != 28 or img.shape[1] != 28:
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            dict_cv2_rotations = {'clock_90': cv2.ROTATE_90_CLOCKWISE, 'c_clock_90': cv2.ROTATE_90_COUNTERCLOCKWISE,
                                  'rotate_180': cv2.ROTATE_180, 'no_rotation:': 0}
            random_cv2_rotation_key = np.random.choice(list(dict_cv2_rotations.keys()), 1)[0]
            if dict_cv2_rotations[random_cv2_rotation_key] != 0:
                img_rotate = cv2.rotate(img, dict_cv2_rotations[random_cv2_rotation_key])
            else:
                img_rotate = img
            return np.reshape(img_rotate, (28, 28, 1))

        if self.dataset == 'miniimagenet':
            img = cv2.imread(image_file).astype(np.float32) / 255
            return img

    def generate_set(self, folder_list):
        # generate support and query sets ( dividing images and labels)
        support_x = []
        support_y = []
        query_x = []
        query_y = []

        for i, elem in enumerate(folder_list):
            folder = elem[0]
            label = elem[1]
            tmp_image_files = [os.path.join(folder, img) for img in
                               np.random.choice(os.listdir(folder), self.shot_num + self.query_num, False)]
            tmp_support_x = random.sample(tmp_image_files, self.shot_num)
            tmp_query_x = [img_file for img_file in tmp_image_files if img_file not in tmp_support_x]

            support_x.extend([self.extract_images(img_file) for img_file in tmp_support_x])
            query_x.extend([self.extract_images(img_file) for img_file in tmp_query_x])
            support_y.extend([tf.one_hot(label, self.n_way) for _ in range(len(tmp_support_x))])
            query_y.extend([tf.one_hot(label, self.n_way) for _ in range(len(tmp_query_x))])

        # shuffle images and labels
        support_x, support_y = self.shuffle_set(support_x, support_y)
        query_x, query_y = self.shuffle_set(query_x, query_y)

        # convert to tensor
        support_x = tf.convert_to_tensor(np.array(support_x))
        support_y = tf.convert_to_tensor(np.array(support_y))
        query_x = tf.convert_to_tensor(np.array(query_x))
        query_y = tf.convert_to_tensor(np.array(query_y))

        return support_x, support_y, query_x, query_y


if __name__ == '__main__':
    tasks = TaskGenerator(dataset_name="miniimagenet", n_way=5, shot_num=1, query_num=15, meta_batch_size=4)
    start = time.time()
    for i in range(4):  # iterations
        batch_set = tasks.get_batch('train')
        x, y, _, _ = batch_set[0]
        print(len(x[0]))
    print(time.time() - start)
