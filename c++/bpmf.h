#ifndef BPMF_H
#define BPMF_H

#include <Eigen/Dense>
#include <Eigen/Sparse>


Eigen::VectorXd nrandn(int n, double mean = 0, double sigma = 1);

std::pair<Eigen::VectorXd, Eigen::MatrixXd> CondNormalWishart(Eigen::MatrixXd U, Eigen::VectorXd mu, double kappa, Eigen::MatrixXd T, int nu);

#endif
