#pragma once

#include <Eigen/Dense>
#include "kfcore.hpp"


namespace Kalman {
	/*
	* 动态类型卡尔曼滤波核心部分，除动态数组外与正常卡尔曼滤波相同。
	*/
	template <>
	struct KFCore<-1, -1> {
		MatXf x, K, R, Q, P, A, H;

		MatXf predict() {
			MatXf xp = A * x;
			P = A * P * P.transpose() + Q;
			return xp;
		}

		MatXf update(Ref<MatXf> z, Ref<MatXf> xp) {
			K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
			x = xp + K * (z - H * xp);
			P = (MatXf::Identity(xp.rows(), xp.rows()) - K * H) * P;
			return x;
		}
	};
}
