from keras import activations, Input, Model
from keras.layers import Dense


def regression_model():
    inputs = Input(shape=(1,))
    h1 = Dense(40, activation=activations.relu)(inputs)
    h2 = Dense(40, activation=activations.relu)(h1)
    outputs = Dense(1)(h2)
    model = Model(inputs, outputs)
    return model
