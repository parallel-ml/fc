import os
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.models import Model
from keras import layers
import numpy as np
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


img_input = None


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    global img_input
    input_shape = Model(img_input, input_tensor).output_shape
    input_shape = input_shape[1:] if input_shape[0] is None else input_shape
    temp_input = Input(shape=input_shape)

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(temp_input)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    x = layers.add([x, temp_input])
    x = Activation('relu')(x)

    model = Model(temp_input, x)
    test_x = np.random.random_sample(input_shape)
    start = time.time()
    for _ in range(50):
        model.predict(np.array([test_x]))
    print '{:s}_identity_shortcut: {:.3f} sec'.format(('stage_' + str(stage) + '_block_' + block), (time.time() - start) / 50)

    del model

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    global img_input
    input_shape = Model(img_input, input_tensor).output_shape
    input_shape = input_shape[1:] if input_shape[0] is None else input_shape
    temp_input = Input(shape=input_shape)

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(temp_input)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(temp_input)
    shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    model = Model(temp_input, x)
    test_x = np.random.random_sample(input_shape)
    start = time.time()
    for _ in range(50):
        model.predict(np.array([test_x]))
    print '{:s}_conv_shortcut: {:.3f} sec'.format(('stage_' + str(stage) + '_block_' + block), (time.time() - start) / 50)

    del model

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x


def resnet50(input_shape=(224, 224, 3)):
    global img_input
    img_input = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    model = Model(img_input, x)
    test_x = np.random.random_sample(input_shape)
    start = time.time()
    for _ in range(50):
        model.predict(np.array([test_x]))
    print 'input conv: {:.3f} sec'.format((time.time() - start) / 50)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    fc_input_shape = Model(img_input, x).output_shape
    fc_input_shape = fc_input_shape[1:] if fc_input_shape[0] is None else fc_input_shape
    temp_input = Input(shape=fc_input_shape)

    fc = AveragePooling2D((7, 7), name='avg_pool')(temp_input)
    fc = Flatten()(fc)
    fc = Dense(1000, activation='softmax', name='fc1000')(fc)

    temp_model = Model(temp_input, fc)
    test_x = np.random.random_sample(fc_input_shape)
    start = time.time()
    for _ in range(50):
        temp_model.predict(np.array([test_x]))
    print 'final fc: {:.3f} sec'.format((time.time() - start) / 50)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(1000, activation='softmax', name='fc1000')(x)

    # Create models.
    model = Model(img_input, x, name='resnet50')
    return model

model = resnet50()