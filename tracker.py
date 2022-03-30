from kalman_core import KFCore
import numpy as np


class KFTracker:
    dimx = 8
    dimz = 4

    def __init__(self, q, r, p, dt):
        Q = np.identity(KFTracker.dimx) * q
        R = np.identity(KFTracker.dimz) * r
        P = np.identity(KFTracker.dimx) * p
        A = np.identity(KFTracker.dimx)
        H = np.zeros((KFTracker.dimz, KFTracker.dimx))
        H[0:KFTracker.dimz, 0:KFTracker.dimz] = np.identity(KFTracker.dimz)
        x0 = np.zeros((KFTracker.dimx, 1))
        self._kf = KFCore(Q=Q, R=R, P=P, A=A, H=H, x=x0, dimx=KFTracker.dimx)
        self.set_dt(dt)

    def set_dt(self, dt):
        self._kf.A[0:4, 4:8] = np.identity(KFTracker.dimz) * dt

    def init_x0(self, x0):
        self._kf.x = x0

    def predict(self):
        return self._kf.predict()

    def update(self, z, xp):
        return self._kf.update(z, xp)
