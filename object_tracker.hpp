#pragma once

#include <Eigen/Dense>

#include "KFCore/kfcore.hpp"


constexpr auto dimz = 4;
constexpr auto dimx = 8;


/*
* 用例
* 目标检测算法用卡尔曼滤波
* 观测量为x1 y1 x2 y2（或x y w h)
* 状态量为观测量及其对应的速度
*/
class ObjectTracker {
private:
	Kalman::KFCore<dimz, dimx>  _kf;  // 卡尔曼滤波核心
	float t_prev;  // 上一次时间
public:
	ObjectTracker(float q, float r, float p, float dt) {
		this->_kf.Q = Kalman::Matf<dimx, dimx>::Identity() * q;
		this->_kf.R = Kalman::Matf<dimz, dimz>::Identity() * r;
		this->_kf.P = Kalman::Matf<dimx, dimx>::Identity() * p;
		this->_kf.A = Kalman::Matf<dimx, dimx>::Identity();
		this->set_dt(dt);
		this->_kf.H = Kalman::Matf<dimz, dimx>::Zero();
		this->_kf.H.topLeftCorner<dimz, dimz>() = Kalman::Matf<dimz, dimz>::Identity();
	}

	/*
	* 设置状态转移矩阵中dt
	*/
	void set_dt(float dt) {
		this->_kf.A.topRightCorner<dimz, dimz>() = Kalman::Matf<dimz, dimz>::Identity() * dt;
	}

	/*
	* 初始化后验状态矩阵
	*/
	void init_x0(Kalman::Ref<Kalman::Matf<dimx, 1>> x0) {
		this->_kf.x = x0;
	}

	Kalman::Matf<dimx, 1> predit() {
		return this->_kf.predict();
	}

	Kalman::Matf<dimx, 1> update(Kalman::Ref<Kalman::Matf<dimz, 1>> z, Kalman::Ref<Kalman::Matf<dimx, 1>> xp) {
		return this->_kf.update(z, xp);
	}
};