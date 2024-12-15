#pragma once

#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;      // 检测结果的类型
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;   // 多个检测结果的超定矩阵

//Kalmanfilter
//typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER; // 卡尔曼滤波器的类型
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;   // 状态的类型
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;   // 状态协方差的类型
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;  // 观测的类型
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;  // 观测协方差的类型
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;                 // 状态矩阵和协方差矩阵的对组
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;              // 观测矩阵和协方差矩阵的对组




