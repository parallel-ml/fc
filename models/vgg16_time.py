import os
import time

import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model, Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    vgg16()


def layers(model, output_shape, name, layer_type, **kwargs):
    if len(model.layers) == 0:
        input_shape = (224, 224, 3)
    else:
        input_shape = model.output_shape

    input_shape = input_shape[1:] if input_shape[0] is None else input_shape

    if layer_type == 'dense':
        activation = kwargs['activation']
        layer = Dense(output_shape, activation=activation, input_shape=input_shape)
    elif layer_type == 'flatten':
        layer = Flatten(name=name, input_shape=input_shape)
    elif layer_type == 'conv':
        kernal = kwargs['kernal']
        activation = kwargs['activation']
        padding = kwargs['padding']
        layer = Conv2D(output_shape, kernel_size=kernal, activation=activation, padding=padding, name=name,
                       input_shape=input_shape)
    else:
        pool_size = kwargs['pool_size']
        strides = kwargs['strides']
        layer = MaxPooling2D(pool_size=pool_size, name=name, strides=strides, input_shape=input_shape)

    test_model = Sequential()
    test_model.add(layer)

    test_x = np.random.random_sample(input_shape)
    start = time.time()
    for _ in range(50):
        test_model.predict(np.array([test_x]))
    print '{:s}: {:.3f} sec'.format(name, (time.time() - start) / 50)

    model.add(layer)
    return model


def vgg16():
    x = Sequential()

    # Block 1
    x = layers(x, 64, 'block1_conv1', 'conv', kernal=(3, 3), activation='relu', padding='same')
    x = layers(x, 64, 'block1_conv2', 'conv', kernal=(3, 3), activation='relu', padding='same')
    x = layers(x, None, 'block1_pool', 'maxpool', pool_size=(2, 2), strides=(2, 2))

    # Block 2
    x = layers(x, 128, 'block2_conv1', 'conv', kernal=(3, 3), activation='relu', padding='same')
    x = layers(x, 128, 'block2_conv2', 'conv', kernal=(3, 3), activation='relu', padding='same')
    x = layers(x, None, 'block2_pool', 'maxpool', pool_size=(2, 2), strides=(2, 2))

    # Block 3
    x = layers(x, 256, 'block3_conv1', 'conv', kernal=(3, 3), activation='relu', padding='same')
    x = layers(x, 256, 'block3_conv2', 'conv', kernal=(3, 3), activation='relu', padding='same')
    x = layers(x, 256, 'block3_conv3', 'conv', kernal=(3, 3), activation='relu', padding='same')
    x = layers(x, None, 'block3_pool', 'maxpool', pool_size=(2, 2), strides=(2, 2))

    # Block 4
    x = layers(x, 512, 'block4_conv1', 'conv', kernal=(3, 3), activation='relu', padding='same')
    x = layers(x, 512, 'block4_conv2', 'conv', kernal=(3, 3), activation='relu', padding='same')
    x = layers(x, 512, 'block4_conv3', 'conv', kernal=(3, 3), activation='relu', padding='same')
    x = layers(x, None, 'block4_pool', 'maxpool', pool_size=(2, 2), strides=(2, 2))

    # Block 5
    x = layers(x, 512, 'block5_conv1', 'conv', kernal=(3, 3), activation='relu', padding='same')
    x = layers(x, 512, 'block5_conv2', 'conv', kernal=(3, 3), activation='relu', padding='same')
    x = layers(x, 512, 'block5_conv3', 'conv', kernal=(3, 3), activation='relu', padding='same')
    x = layers(x, None, 'block5_pool', 'maxpool', pool_size=(2, 2), strides=(2, 2))

    # Classification block
    x = layers(x, None, 'block6_flatten', 'flatten')
    x = layers(x, 4096, 'block6_dense1', 'dense', activation='relu')
    x = layers(x, 4096, 'block6_dense2', 'dense', activation='relu')
    x = layers(x, 1000, 'block6_dense3', 'dense', activation='softmax')


if __name__ == '__main__':
    main()
