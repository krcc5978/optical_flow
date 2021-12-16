import cv2
import math
import numpy as np


class OpticalFlow:

    def __init__(self, old_gray):
        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        self.color = np.random.randint(0, 255, (100, 3))

        self.p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)

    def lucas_kanade(self, frame, mask, old_gray, new_gray):
        """
        疎なオプティカルフロー
        :return:
        """

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, self.p0, None, **self.lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            if 1 < math.sqrt((a - c) ** 2 + (b - d) ** 2):
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        self.p0 = good_new.reshape(-1, 1, 2)

        return img

    def farneback(self, hsv, old_gray, new_gray):
        """
        密なオプティカルフロー
        :return:
        """
        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return img

