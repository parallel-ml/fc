#!/usr/bin/python

import os
import sys

import numpy as np
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from util.output import title, timer, avg_timer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

N = 4096


def main():
    run_fc_1()


@title('fc layer first')
def run_fc_1():
    @timer('load')
    def load():
        model = Sequential()
        model.add(Dense(N, input_shape=(7680,)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        return model

    test_x = np.random.rand(7680)
    model = load()

    @avg_timer('inference')
    def predict():
        model.predict(np.array([test_x]))

    predict()


@title('fc layer second')
def run_fc_2():
    @timer('load')
    def load():
        model = Sequential()

        model.add(Dense(N, input_shape=(N,)))
        model.add(BatchNormalization(input_shape=(N,)))
        model.add(Activation('relu', input_shape=(N,)))

        return model

    test_x = np.random.rand(N)
    model = load()

    @avg_timer('inference')
    def predict():
        model.predict(np.array([test_x]))

    predict()


@title('fc layer third')
def run_fc_3():
    @timer('load')
    def load():
        model = Sequential()

        model.add(Dense(51, input_shape=(N,)))
        model.add(BatchNormalization(input_shape=(51,)))
        model.add(Activation('softmax', input_shape=(51,)))

        return model

    test_x = np.random.rand(N)
    model = load()

    @avg_timer('inference')
    def predict():
        model.predict(np.array([test_x]))

    predict()


if __name__ == '__main__':
    main()
