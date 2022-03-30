#include <iostream>
#include <Eigen/Dense>

#include "object_tracker.hpp"


int main() {
	/*
	* 卡尔曼滤波示例程序
	* 编写了一个目标检测算法的卡尔曼滤波类
	*/
	ObjectTracker kf(1, 1, 1, 1);
	Kalman::Matf<8, 1> x0;
	x0 << 0, 0, 0, 0, 10, 100, 1000, 10000;

	kf.init_x0(x0);

	for (size_t i = 0; i < 100; i++) {
		kf.set_dt(1.);
		Kalman::Matf<8, 1> xp =  kf.predit();  // 计算先验状态向量
		Kalman::Matf<4, 1> z = xp.topLeftCorner<4, 1>();  // 将先验状态量作为观测值
		Kalman::Matf<8, 1> x0 = kf.update(z, xp);  // 计算后验状态量
		Kalman::Matf<4, 1> x = x0.topLeftCorner<4, 1>();

		std::cout << "************ " << i + 1 << " ************" << std::endl;
		std::cout << "flitted value: " << std::endl;
		std::cout << x.transpose() << "\n" << std::endl;
		std::cout << std::endl;
	}

	return 0;
}
