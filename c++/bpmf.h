#ifndef BPMF_H
#define BPMF_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;

MatrixXd MvNormal(MatrixXd covar, VectorXd mean, int nn = 1); 

std::pair<VectorXd, MatrixXd> NormalWishart(VectorXd mu, double kappa, MatrixXd T, double nu, int nn = 1);

double rand(double mean = 0, double sigma = 1);

VectorXd randn(int n, double mean = 0, double sigma = 1);

MatrixXd WishartUnit(MatrixXd sigma, int df);

MatrixXd Wishart(MatrixXd sigma, int df);

MatrixXd CondWishart(MatrixXd sigma, int df);

#endif
