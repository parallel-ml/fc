from collections import deque
import time

import cv2
import numpy as np


def main():
    flows = deque()
    start = time.time()
    frame0 = None
    count = 1
    while True:
        # current frame
        ret, frame = 'unknown', np.random.rand(12, 16, 3) * 255
        frame = frame.astype(dtype=np.uint8)
        image = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame0 is not None:
            # append new 1-1 optical frame into deque
            flows.appendleft(cv2.calcOpticalFlowFarneback(frame0, frame, None, 0.5, 3, 4, 3, 5, 1.1, 0))
            if len(flows) == 10:
                # concatenate at axis 2
                # ex: (3, 2, 1) + (3, 2, 1) = (3, 2, 2)
                optical_flow = np.concatenate(flows, axis=2)
                flows.pop()
                count += 1

        if count > 100:
            break

        frame0 = frame

    print '{:.3f} sec'.format((time.time() - start) / 100)


if __name__ == '__main__':
    main()
