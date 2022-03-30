#pragma once

#include <Eigen/Dense>

namespace Kalman {
	template <int _a, int _b>
	using Matf = Eigen::Matrix<float, _a, _b>;
	using MatXf = Matf<Eigen::Dynamic, Eigen::Dynamic>;

	template <typename T>
	using Ref = Eigen::Ref<T>;

	/*
	* 卡尔曼滤波核心运算模块.
	* 其他具体实现类应该继承或包含此模块，以实现具体的卡尔曼滤波算法.
	*/
	template <int dimz, int dimx>
	struct KFCore {
		Matf<dimx, 1> x;     // 上一次卡尔曼滤波后验估计
		Matf<dimx, dimz> K;  // 卡尔曼增益
		Matf<dimz, dimz> R;  // 预测过程噪声偏差的协方差
		Matf<dimx, dimx> Q;  // 测量噪声偏差
		Matf<dimx, dimx> P;  // 估计误差协方差
		Matf<dimx, dimx> A;  // 状态转移矩阵
		Matf<dimz, dimx> H;  // 观测矩阵

		Matf<dimx, 1> predict() {
			Matf<dimx, 1> xp = A * x;  // 计算先验估计
			P = A * P * A.transpose() + Q;  // 先验协方差
			return xp;
		}

		Matf<dimx, 1> update(Ref<Matf<dimz, 1>> z, Ref<Matf<dimx, 1>> xp) {
			K = P * H.transpose() * (H * P * H.transpose() + R).inverse();  // 卡尔曼增益
			x = xp + K * (z - H * xp);  // 后验估计
			P = (Matf<dimx, dimx>::Identity() - K * H) * P;  // 协方差矩阵
			return x;
		}
	};
}
