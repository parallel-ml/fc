import os

import numpy as np
from keras.layers import MaxPooling1D, Dense, Activation, BatchNormalization, Input, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.merge import Concatenate
from keras.models import Model, Sequential, load_model

from util.output import title, timer, avg_timer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

WEIGHT = True


def main():
    for i in range(0,1):
        global WEIGHT
        run_fc()
        if not WEIGHT:
            run_maxpool()
        run_temporal()
        run_spatial()


@title('fc layer')
def run_fc():
    @timer('load')
    def load():
        model = Sequential()
        model.add(Dense(8192, input_shape=(7680,)))
        model.add(BatchNormalization(input_shape=(4096,)))
        model.add(Activation('relu', input_shape=(4096,)))

        model.add(Dense(4096, input_shape=(4096,)))
        model.add(BatchNormalization(input_shape=(4096,)))
        model.add(Activation('relu', input_shape=(4096,)))

        model.add(Dense(51, input_shape=(4096,)))
        model.add(BatchNormalization(input_shape=(51,)))
        model.add(Activation('softmax', input_shape=(51,)))
        return model

    @timer('load')
    def load_weights():
        return load_model(
            #'/home/jiashen/weights/clsfybatch_4/0000_epoch-4.0079_loss-0.0253_acc-4.1435_val_loss-0.0266_val_acc.hdf5'
            '/MLdatasets/siamese_model/clsfybatch_300/hmdb/twostream_1/transforms_1/full_fc/split_1/4243_epoch-0.0000_loss-1.0000_acc-8.9889_val_loss-0.2479_val_acc.hdf5'
            )

    test_x = np.random.rand(7680)
    model = load() if not WEIGHT else load_weights()

    @avg_timer('inference')
    def predict():
        model.predict(np.array([test_x]))

    predict()


@title('maxpooling layer')
def run_maxpool():
    test_x = np.random.rand(100, 256)

    @timer('load')
    def load():
        N = 100
        input = Input(shape=(N, 256), name='input')

        max1 = MaxPooling1D(pool_size=N, strides=N)(input)
        max2 = MaxPooling1D(pool_size=N / 2, strides=N / 2)(input)
        max3 = MaxPooling1D(pool_size=N / 4, strides=N / 4)(input)
        max4 = MaxPooling1D(pool_size=N / 8, strides=N / 8)(input)

        mrg = Concatenate(axis=1)([max1, max2, max3, max4])
        flat = Flatten()(mrg)
        model = Model(input=input, outputs=flat)
        return model

    model = load()

    @avg_timer('inference')
    def predict():
        model.predict(np.array([test_x]))

    predict()


@title('temporal')
def run_temporal():
    @timer('load')
    def load():
        return load_temporal()

    @timer('load')
    def load_weights():
        return load_model(
            #'/home/jiashen/weights/batch_4_noaug/199_epoch-0.2510_loss-0.9403_acc-6.5269_val_loss-0.3061_val_acc.hdf5'
            '/MLdatasets/siamese_model/batch_4_noaug/hmdb/oflow/first_try/split_1/199_epoch-0.2510_loss-0.9403_acc-6.5269_val_loss-0.3061_val_acc.hdf5'
            )

    model = load() if not WEIGHT else load_weights()
    test_x = np.random.rand(12, 16, 20)
    # pop the last three layers used by training
    for _ in range(3):
        model.pop()

    @avg_timer('inference')
    def predict():
        model.predict(np.array([test_x]))

    predict()


@title('spatial')
def run_spatial():
    @timer('load')
    def load():
        return load_spatial()

    @timer('load')
    def load_weights():
        return load_model(
            #'/home/jiashen/weights/batch_4_aug/199_epoch-5.2804_loss-0.1080_acc-5.9187_val_loss-0.0662_val_acc.hdf5'
            '/MLdatasets/siamese_model/batch_100_aug/imgnet/filter_256/28Oct-original/199_epoch-4.2777_loss-0.1949_acc-4.9191_val_loss-0.1437_val_acc.hdf5'
            )

    model = load() if not WEIGHT else load_weights()
    test_x = np.random.rand(12, 16, 3)
    # pop the last three layers used by training
    for _ in range(3):
        model.pop()

    @avg_timer('inference')
    def predict():
        model.predict(np.array([test_x]))

    predict()


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
