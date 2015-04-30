#ifndef BPMF_H
#define BPMF_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;

VectorXd nrandn(int n, double mean = 0, double sigma = 1);

std::pair<Eigen::VectorXd, Eigen::MatrixXd> CondNormalWishart(MatrixXd U, VectorXd mu, double kappa, MatrixXd T, int nu);

#endif
