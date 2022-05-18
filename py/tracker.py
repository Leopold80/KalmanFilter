import numpy as np

from .kalman_core import KFCore


"""
用于2d点的卡尔曼滤波实现，滤波时调用__call__即可
"""


class KalmanFilter(KFCore):
    dimx = 4
    dimz = 2

    def __init__(self, q, r, p, dt):
        Q = np.identity(KalmanFilter.dimx) * q
        R = np.identity(KalmanFilter.dimz) * r
        P = np.identity(KalmanFilter.dimx) * p
        A = np.identity(KalmanFilter.dimx)
        H = np.zeros((KalmanFilter.dimz, KalmanFilter.dimx))
        H[0:KalmanFilter.dimz, 0:KalmanFilter.dimz] = np.identity(KalmanFilter.dimz)
        x0 = np.zeros((KalmanFilter.dimx, 1))

        super(KalmanFilter, self).__init__(Q, R, P, A, H, x0, KalmanFilter.dimx)
        self.set_dt(dt)

    def set_dt(self, dt):
        self.A[0:KalmanFilter.dimz, KalmanFilter.dimz:KalmanFilter.dimx] = np.identity(KalmanFilter.dimz) * dt

    def set_x0(self, x0):
        self.xp = x0
