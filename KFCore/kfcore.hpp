#pragma once

#include <Eigen/Dense>

namespace Kalman {
	template <int _a, int _b>
	//using Matf = Eigen::Matrix<float, _a, _b, Eigen::RowMajor>;
	using Matf = Eigen::Matrix<float, _a, _b>;
	using MatXf = Matf<Eigen::Dynamic, Eigen::Dynamic>;

	template <typename T>
	using Ref = Eigen::Ref<T>;

	/*
	* �������˲���������ģ��.
	* ��������ʵ����Ӧ�ü̳л������ģ�飬��ʵ�־���Ŀ������˲��㷨.
	*/
	template <int dimz, int dimx>
	struct KFCore {
		Matf<dimx, 1> x;     // ��һ�ο������˲��������
		Matf<dimx, dimz> K;  // ����������
		Matf<dimz, dimz> R;  // Ԥ���������ƫ���Э����
		Matf<dimx, dimx> Q;  // ��������ƫ��
		Matf<dimx, dimx> P;  // �������Э����
		Matf<dimx, dimx> A;  // ״̬ת�ƾ���
		Matf<dimz, dimx> H;  // �۲����

		Matf<dimx, 1> predict() {
			Matf<dimx, 1> xp = A * x;  // �����������
			P = A * P * A.transpose() + Q;  // ����Э����
			return xp;
		}

		Matf<dimx, 1> update(Ref<Matf<dimz, 1>> z, Ref<Matf<dimx, 1>> xp) {
			K = P * H.transpose() * (H * P * H.transpose() + R).inverse();  // ����������
			x = xp + K * (z - H * xp);  // �������
			P = (Matf<dimx, dimx>::Identity() - K * H) * P;  // Э�������
			return x;
		}
	};
}