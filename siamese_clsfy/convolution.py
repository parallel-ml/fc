from keras.layers.convolutional import Conv2D
# from keras.activations import Dense, Activation
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


# input_dim_ordering should be set to "tf"
def load_spatial():
    return load_cnn(nb_channel=3)


def load_temporal():
    return load_cnn(nb_channel=6)


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


def load_fc(input_shape=7680):
    model = Sequential()
    model.add(Dense(4096, input_shape=(input_shape,)))
    model.add(BatchNormalization(input_shape=(4096,)))
    model.add(Activation('relu', input_shape=(4096,)))

    model.add(Dense(4096, input_shape=(4096,)))
    model.add(BatchNormalization(input_shape=(4096,)))
    model.add(Activation('relu', input_shape=(4096,)))

    model.add(Dense(51, input_shape=(4096,)))
    model.add(BatchNormalization(input_shape=(51,)))
    model.add(Activation('softmax', input_shape=(51,)))
    return model


if __name__ == '__main__':
    print load_temporal().summary()