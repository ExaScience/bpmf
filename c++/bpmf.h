/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#ifndef BPMF_H
#define BPMF_H


#define EIGEN_RUNTIME_NO_MALLOC 1
#define EIGEN_DONT_PARALLELIZE 1

#include "Eigen/Dense"
#include "Eigen/Sparse"

const int num_latent = 32;

typedef Eigen::SparseMatrix<double> SparseMatrixD;
typedef Eigen::Matrix<double, num_latent, num_latent> MatrixNNd;
typedef Eigen::Matrix<double, num_latent, Eigen::Dynamic> MatrixNXd;
typedef Eigen::Matrix<double, num_latent, 1> VectorNd;
typedef Eigen::Map<MatrixNXd, Eigen::Aligned> MapNXd;
typedef Eigen::Map<Eigen::VectorXd, Eigen::Aligned> MapXd;

#if defined(_OPENMP)
#include <omp.h>
#pragma omp declare reduction (VectorPlus : VectorNd : omp_out.noalias() += omp_in) initializer(omp_priv = VectorNd::Zero())
#pragma omp declare reduction (MatrixPlus : MatrixNNd : omp_out.noalias() += omp_in) initializer(omp_priv = MatrixNNd::Zero())
#endif

#endif
