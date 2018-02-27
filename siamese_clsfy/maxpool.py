import numpy as np
from keras.layers import Input, Flatten
from keras.layers.merge import Concatenate
from keras.layers.pooling import MaxPooling1D
from keras.models import Model

from util.output import timer, title, subtitle, avg_timer


@title('max pooling layers')
def main():
    k = 100
    while k <= 100:
        @subtitle('{:d} frames'.format(k))
        def finetune(n):
            test_x = np.random.rand(n, 256)

            @timer('load models')
            def load(*args, **kwargs):
                N = kwargs['N']

                input = Input(shape=test_x.shape, name='input')

                max1 = MaxPooling1D(pool_size=N, strides=N)(input)
                max2 = MaxPooling1D(pool_size=N / 2, strides=N / 2)(input)
                max3 = MaxPooling1D(pool_size=N / 4, strides=N / 4)(input)
                max4 = MaxPooling1D(pool_size=N / 8, strides=N / 8)(input)

                mrg = Concatenate(axis=1)([max1, max2, max3, max4])

                flat = Flatten()(mrg)

                model = Model(input=input, outputs=flat)

                print model.summary()

                return model

            model = load(N=n)
            model.predict(np.array([test_x]))

            @avg_timer('inference')
            def forward():
                return model.predict(np.array([test_x]))

            forward()

        finetune(k)
        k += 100


if __name__ == '__main__':
    main()
