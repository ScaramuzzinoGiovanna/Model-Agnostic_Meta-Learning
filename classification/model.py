from keras import Model
from keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, Input, Flatten, Dense


class OmniglotConvModel():

    def build_model(self, filters=64, input_size=(28, 28, 1), out_size=5):
        # default omniglot
        inputs = Input(shape=(input_size))
        self.c1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2),
                         padding='SAME', kernel_initializer='glorot_normal')(inputs)
        self.b1 = BatchNormalization(axis=-1)(self.c1)
        self.r1 = ReLU()(self.b1)

        self.c2 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2),
                         padding='SAME', kernel_initializer='glorot_normal')(self.r1)
        self.b2 = BatchNormalization(axis=-1)(self.c2)
        self.r2 = ReLU()(self.b2)

        self.c3 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),
                         padding='SAME', kernel_initializer='glorot_normal')(self.r2)
        self.b3 = BatchNormalization(axis=-1)(self.c3)
        self.r3 = ReLU()(self.b3)

        self.c4 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),
                         padding='SAME', kernel_initializer='glorot_normal')(self.r3)
        self.b4 = BatchNormalization(axis=-1)(self.c4)
        self.r4 = ReLU()(self.b4)

        self.f = Flatten()(self.r4)
        self.out = Dense(out_size, activation='softmax')(self.f)

        model = Model(inputs, self.out)
        return model


class MiniimagenetConvModel():

    def build_model(self, filters=32, input_size=(84, 84, 3), out_size=5):
        # default miniImagenet
        inputs = Input(shape=(input_size))
        self.c1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),
                         padding='SAME', kernel_initializer='glorot_normal')(inputs)
        self.b1 = BatchNormalization(axis=-1)(self.c1)
        self.r1 = ReLU()(self.b1)
        self.mp1 = MaxPool2D(pool_size=2)(self.r1)

        self.c2 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),
                         padding='SAME', kernel_initializer='glorot_normal')(self.mp1)
        self.b2 = BatchNormalization(axis=-1)(self.c2)
        self.r2 = ReLU()(self.b2)
        self.mp2 = MaxPool2D(pool_size=2)(self.r2)

        self.c3 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),
                         padding='SAME', kernel_initializer='glorot_normal')(self.mp2)
        self.b3 = BatchNormalization(axis=-1)(self.c3)
        self.r3 = ReLU()(self.b3)
        self.mp3 = MaxPool2D(pool_size=2)(self.r3)

        self.c4 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),
                         padding='SAME', kernel_initializer='glorot_normal')(self.mp3)
        self.b4 = BatchNormalization(axis=-1)(self.c4)
        self.r4 = ReLU()(self.b4)
        self.mp4 = MaxPool2D(pool_size=2)(self.r4)

        self.f = Flatten()(self.mp4)
        self.out = Dense(out_size, activation='softmax')(self.f)

        model = Model(inputs, self.out)
        return model


class OmniglotNoConvModel():
    # omniglot default
    def build_model(self, input_size=(28, 28, 1), out_size=5):
        inputs = Input(shape=(input_size))
        self.c1 = Dense(256, kernel_initializer='glorot_normal')(inputs)
        self.b1 = BatchNormalization(axis=-1)(self.c1)
        self.r1 = ReLU()(self.b1)

        self.c2 = Dense(128, kernel_initializer='glorot_normal')(self.r1)
        self.b2 = BatchNormalization(axis=-1)(self.c2)
        self.r2 = ReLU()(self.b2)

        self.c3 = Dense(64, kernel_initializer='glorot_normal')(self.r2)
        self.b3 = BatchNormalization(axis=-1)(self.c3)
        self.r3 = ReLU()(self.b3)

        self.c4 = Dense(64, kernel_initializer='glorot_normal')(self.r3)
        self.b4 = BatchNormalization(axis=-1)(self.c4)
        self.r4 = ReLU()(self.b4)

        self.f = Flatten()(self.r4)
        self.out = Dense(out_size, activation='softmax')(self.f)

        model = Model(inputs, self.out)

        return model
