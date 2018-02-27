import os
import time

import numpy as np
from keras import backend as K
from keras.layers import Flatten, Dense, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Layer
from keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

img_input = None


class LRN2D(Layer):
    """ LRN2D class is from original keras but gets removed at latest version """

    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN2D, self).__init__(**kwargs)

    def get_output(self, x, mask=None):
        b, ch, r, c = x.shape
        half_n = self.n // 2
        input_sqr = K.square(x)
        extra_channels = K.zeros((b, int(ch) + 2 * half_n, r, c))
        input_sqr = K.concatenate(
            [extra_channels[:, :half_n, :, :], input_sqr, extra_channels[:, half_n + int(ch):, :, :]], axis=1)

        scale = self.k
        norm_alpha = self.alpha / self.n
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i + int(ch), :, :]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def conv2D_bn(x, nb_filter, nb_row, nb_col, activation='relu', batch_norm=True, name=''):
    if name != '':
        global img_input
        if img_input != x:
            input_shape = Model(img_input, x).output_shape
            input_shape = input_shape[1:] if input_shape[0] is None else input_shape
        else:
            input_shape = (224, 224, 3)
        temp_input = Input(shape=input_shape)

        y = Conv2D(nb_filter, kernel_size=(nb_row, nb_col), activation=activation, )(temp_input)
        y = ZeroPadding2D(padding=(1, 1))(y)
        if batch_norm:
            y = LRN2D()(y)
            y = ZeroPadding2D(padding=(1, 1))(y)
        y = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(y)
        y = ZeroPadding2D(padding=(1, 1))(y)

        temp_model = Model(temp_input, y)
        test_x = np.random.random_sample(input_shape)
        start = time.time()
        for _ in range(50):
            temp_model.predict(np.array([test_x]))
        print '{:s}: {:.3f} sec'.format(name, (time.time() - start) / 50)
        del temp_model

    x = Conv2D(nb_filter, kernel_size=(nb_row, nb_col), activation=activation, )(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    if batch_norm:
        x = LRN2D()(x)
        x = ZeroPadding2D(padding=(1, 1))(x)

    x = MaxPooling2D(strides=(2, 2), pool_size=(2, 2))(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    return x


def flatten(x):
    global img_input
    input_shape = Model(img_input, x).output_shape
    input_shape = input_shape[1:] if input_shape[0] is None else input_shape
    temp_input1 = Input(shape=input_shape)

    y = Flatten()(temp_input1)

    temp_model = Model(temp_input1, y)
    test_s1 = np.random.random_sample(input_shape)
    start = time.time()
    for _ in range(50):
        temp_model.predict(np.array([test_s1]))
    print '{:s}: {:.3f} sec'.format('flatten', (time.time() - start) / 50)
    del temp_model

    x = Flatten()(x)
    return x


def dense(x, act='relu', dim=4096, name=''):
    if name != '':
        global img_input
        input_shape = Model(img_input, x).output_shape
        input_shape = input_shape[1:] if input_shape[0] is None else input_shape
        temp_input = Input(shape=input_shape)

        y = Dense(dim, activation=act)(temp_input)

        temp_model = Model(temp_input, y)
        test_x = np.random.random_sample(input_shape)
        start = time.time()
        for _ in range(50):
            temp_model.predict(np.array([test_x]))
        print '{:s}: {:.3f} sec'.format(name, (time.time() - start) / 50)
        del temp_model

    x = Dense(dim, activation=act)(x)
    return x


def alexnet1():
    global img_input
    img_input = Input(shape=(224, 224, 3))

    stream1 = conv2D_bn(img_input, 3, 11, 11, name='conv1 utils stream')
    stream1 = conv2D_bn(stream1, 48, 5, 5, name='conv2 utils stream')
    stream1 = conv2D_bn(stream1, 128, 3, 3, name='conv3 utils stream')
    stream1 = conv2D_bn(stream1, 192, 3, 3, name='conv4 utils stream')
    stream1 = conv2D_bn(stream1, 192, 3, 3, name='conv5 utils stream')

    fc = flatten(stream1)


def alexnet2():
    fc = Input(shape=(27648,))

    fc = dense(fc, name='fc1 utils stream')
    fc = dense(fc, name='fc2 utils stream')
    fc = dense(fc, act='softmax', dim=1000, name='fc3 utils stream')


alexnet1()
alexnet2()
