import model as ml
import numpy as np
import time


def main():
    model = ml.whole()
    test_x = np.random.rand(220, 220, 3)
    start = time.time()
    for _ in range(50):
        model.predict(np.array([test_x]))
    print '{:.3f} sec'.format((time.time() - start) / 50)


if __name__ == '__main__':
    main()
