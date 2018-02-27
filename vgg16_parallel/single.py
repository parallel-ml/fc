import model as ml
import numpy as np
import time


def main():
    model = ml.vgg16()
    test_x = np.random.rand(224, 224, 3)
    start = time.time()
    for _ in range(50):
        model.predict(np.array([test_x]))
    print '{:.3f} sec'.format((time.time() - start) / 50)


if __name__ == '__main__':
    main()
