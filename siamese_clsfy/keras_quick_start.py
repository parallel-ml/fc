import numpy as np
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling1D, Dense, Activation, BatchNormalization, Input, Flatten
from keras.layers.merge import Concatenate
from keras.models import load_model, Model, Sequential
from memory_profiler import profile

from util.output import title, timer, avg_timer, subtitle


def main():
    pass


def run_fc():
    model = Sequential()
    model.add(Dense(4096, input_shape=(7680,)))
    model.add(BatchNormalization(input_shape=(4096,)))
    model.add(Activation('relu', input_shape=(4096,)))

    model.add(Dense(4096, input_shape=(4096,)))
    model.add(BatchNormalization(input_shape=(4096,)))
    model.add(Activation('relu', input_shape=(4096,)))

    model.add(Dense(51, input_shape=(4096,)))
    model.add(BatchNormalization(input_shape=(51,)))
    model.add(Activation('softmax', input_shape=(51,)))
    test_x = np.random.rand(7680)
    output = model.predict(np.array([test_x]))


def run_maxpool():
    test_x = np.random.rand(16, 256)

    N = 16
    input = Input(shape=(N, 256), name='input')

    max1 = MaxPooling1D(pool_size=N, strides=N)(input)
    max2 = MaxPooling1D(pool_size=N / 2, strides=N / 2)(input)
    max3 = MaxPooling1D(pool_size=N / 4, strides=N / 4)(input)
    max4 = MaxPooling1D(pool_size=N / 8, strides=N / 8)(input)

    mrg = Concatenate(axis=1)([max1, max2, max3, max4])
    flat = Flatten()(mrg)
    model = Model(input=input, outputs=flat)
    output = model.predict(np.array([test_x]))
    print output


def run_temporal():
    model = load_temporal()
    test_x = np.random.rand(12, 16, 20)
    # pop the last three layers used by training
    for _ in range(3):
        model.pop()
    output = model.predict(np.array([test_x]))
    print output


def run_spatial():
    model = load_spatial()
    test_x = np.random.rand(12, 16, 3)
    # pop the last three layers from training
    for _ in range(3):
        model.pop()
    output = model.predict(np.array([test_x]))
    print output


def load_spatial():
    return load_cnn(nb_channel=3)


def load_temporal():
    return load_cnn(nb_channel=20)


def load_cnn(nb_class=1000, bias=True, act='relu', bn=True, dropout=False, moredense=False, nb_filter=256,
             nb_channel=3, clsfy_act='softmax'):
    # input image is 16x12, RGB(x3)
    model = Sequential()
    model.add(Conv2D(nb_filter, (5, 5), use_bias=bias, padding='same', input_shape=(12, 16, nb_channel)))
    model.add(Activation(act, name='relu_1'))
    if bn:
        model.add(BatchNormalization())

    model.add(Conv2D(nb_filter, (3, 3), use_bias=bias, padding='same'))
    model.add(Activation(act, name='relu_2'))
    if bn:
        model.add(BatchNormalization())

    model.add(Conv2D(nb_filter, (3, 3), use_bias=bias, padding='same'))
    model.add(Activation(act, name='relu_3'))
    if bn:
        model.add(BatchNormalization())

    model.add(Flatten())
    # now models.output_shape == (None, 24576)
    model.add(Dense(256, input_dim=24576, use_bias=bias, name='dense_4'))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(act, name='relu_4'))

    return model


if __name__ == '__main__':
    main()
