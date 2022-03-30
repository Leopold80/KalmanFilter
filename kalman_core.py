import numpy as np
from typing import Iterable
from itertools import chain


class KFCore:
    __slots__ = (
        "x",  # (x, 1) 上次卡尔曼滤波后验估计
        "K",  # (x, z) 卡尔曼增益
        "R",  # (z, z) 观测噪声协方差 R增大 动态响应变慢 收敛稳定性变好
        "Q",  # (x, x) 过程噪声协方差 Q增大 动态响应变快 收敛稳定性变坏
        "P",  # (x, x) 状态空间协方差
        "A",  # (x, x) 状态转移矩阵
        "H",  # (z, x) 观测矩阵
        "dimx"  # 状态空间维度
    )

    def __init__(self, **kwargs):
        # for x in self.__slots__:
        #     setattr(self, x, None)
        for k, v in kwargs.items():
            v = v.astype(np.float32) if isinstance(v, np.ndarray) else v
            setattr(self, k, v)

    def predict(self):
        xp = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return xp

    def update(self, z, xp):
        self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = xp + self.K @ (z - self.H @ xp)
        self.P = (np.identity(self.dimx) - self.K @ self.H) @ self.P
        return self.x

    def __str__(self):
        s = []
        for x in self.__slots__:
            m = "self.{}".format(x)
            try:
                m = str(eval(m))
            except AttributeError:
                m = "attr not init"
            s.append((x, m))
        s = chain(*s)
        return "\n".join(s)

