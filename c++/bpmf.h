#ifndef BPMF_H
#define BPMF_H

#define EIGEN_DONT_PARALLELIZE

#include <Eigen/Dense>
#include <Eigen/Sparse>

Eigen::VectorXd nrandn(int n, double mean = 0, double sigma = 1);

std::pair<Eigen::VectorXd, Eigen::MatrixXd> CondNormalWishart(const Eigen::MatrixXd &U, const Eigen::VectorXd &mu, const double kappa, const Eigen::MatrixXd &T, const int nu);

#endif
