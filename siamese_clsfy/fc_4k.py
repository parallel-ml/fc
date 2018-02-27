import numpy as np
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from memory_profiler import profile

from util.output import *


def train():
    model = Sequential()
    model.add(Dense(4096, input_shape=(7680,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(51))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    test_x = np.random.rand(1, 7680)
    test_y = np.random.rand(1, 51)

    model.fit(test_x, test_y)

    model.save('/home/jiashen/weights/4k_weights/fc_4k_test_weights.h5')


@profile
def test():
    @timer('load')
    def load():
        model = load_model('/home/pi/fc_4k_test_weights.h5')
        return model

    test_x = np.random.rand(7680)
    model = load()

    @avg_timer('inference')
    def predict():
        model.predict(np.array([test_x]))

    predict()


def main():
    test()


if __name__ == '__main__':
    main()
