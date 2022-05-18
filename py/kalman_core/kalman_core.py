from itertools import chain
from typing import List

import numpy as np


class KFCore:
    __slots__ = (
        "xp",  # (x, 1) 卡尔曼滤波先验估计
        "K",  # (x, z) 卡尔曼增益
        "R",  # (z, z) 观测噪声协方差 R增大 动态响应变慢 收敛稳定性变好
        "Q",  # (x, x) 过程噪声协方差 Q增大 动态响应变快 收敛稳定性变坏
        "P",  # (x, x) 状态空间协方差
        "A",  # (x, x) 状态转移矩阵
        "H",  # (z, x) 观测矩阵
        "dimx"  # 状态空间维度
    )

    def __init__(self, Q, R, P, A, H, x0, dimx):
        self.K: np.ndarray = np.zeros((0,))
        self.Q: np.ndarray = Q
        self.R: np.ndarray = R
        self.P: np.ndarray = P
        self.A: np.ndarray = A
        self.H: np.ndarray = H
        self.xp: np.ndarray = x0
        self.dimx: int = dimx

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.xp = self.A @ x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.xp

    def update(self, z: np.ndarray) -> np.ndarray:
        self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        x = self.xp + self.K @ (z - self.H @ self.xp)
        self.P = (np.identity(self.dimx) - self.K @ self.H) @ self.P
        return x

    def __call__(self, z: np.ndarray, observing=True) -> List[np.ndarray]:
        """
        更新and预测
        :param z: 观测量
        :return: [滤波值 预测值]
        """
        x = self.update(z)
        xp = self.predict(x)
        return [self.H @ m for m in (x, xp)] if observing else [x, xp]

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
