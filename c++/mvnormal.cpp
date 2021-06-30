/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


/*
 * Sampling from WishHart and Normal distributions
 */

#include <numeric>
#include <functional>
#include <iostream>
#include <random>

#include "bpmf.h"

#ifdef BPMF_RANDOM123
#include <Random123/philox.h>
#include <Random123/MicroURNG.hpp>

typedef r123::MicroURNG<r123::Philox4x32> RNG;
static thread_local RNG rng({{0}}, {{42}});
#else

typedef std::mt19937 RNG;
static thread_local RNG rng;
#endif



using namespace Eigen;

void rng_set_pos(uint32_t c)
{
#ifdef BPMF_RANDOM123
    rng.reset({{c}}, {{42}});
#endif
}

double randn() {
    return std::normal_distribution<>()(rng);
}

double randu() {
    return std::uniform_real_distribution<>(0., 1.0)(rng);
}


/* -------------------------------------------------------------------------------- */

/*
  Draw nn samples from a size-dimensional normal distribution
  with a specified mean vector and precision matrix
*/
VectorNd MvNormalChol_prec(double kappa, const MatrixNNd & Lambda_U, const VectorNd & mean)
{
  VectorNd r = nrandn(num_latent);
  Lambda_U.triangularView<Upper>().solveInPlace(r);
  return (r / sqrt(kappa)) + mean;
}


void WishartUnitChol(int df, MatrixNNd & c) {
    c.setZero();

    for ( int i = 0; i < num_latent; i++ ) {
        std::gamma_distribution<> gam(0.5*(df - i));
        c(i,i) = sqrt(2.0 * gam(rng));
        VectorXd r = nrandn(num_latent-i-1);
        for(int j=i+1;j<num_latent;j++) c.coeffRef(i,j) = randn();
    }
}

void WishartChol(const MatrixNNd &sigma, const int df, MatrixNNd & U)
{
//  Get R, the upper triangular Cholesky factor of SIGMA.
  auto chol = sigma.llt();

//  Get AU, a sample from the unit Wishart distribution.
  MatrixNNd au;
  WishartUnitChol(df, au);
  U.noalias() = au * chol.matrixU();

#ifdef TEST_MVNORMAL
    cout << "WISHART {\n" << endl;
    cout << "  sigma::\n" << sigma << endl;
    cout << "  au:\n" << au << endl;
    cout << "  df:\n" << df << endl;
    cout << "}\n" << endl;
#endif
}


// from julia package Distributions: conjugates/normalwishart.jl
std::pair<VectorNd, MatrixNNd> NormalWishart(const VectorNd & mu, double kappa, const MatrixNNd & T, double nu)
{
  MatrixNNd LamU;
  WishartChol(T, nu, LamU);
  VectorNd mu_o = MvNormalChol_prec(kappa, LamU, mu);

#ifdef TEST_MVNORMAL
    cout << "NORMAL WISHART {\n" << endl;
    cout << "  mu:\n" << mu << endl;
    cout << "  kappa:\n" << kappa << endl;
    cout << "  T:\n" << T << endl;
    cout << "  nu:\n" << nu << endl;
    cout << "  mu_o\n" << mu_o << endl;
    cout << "  Lam\n" << LamU.transpose() * LamU << endl;
    cout << "}\n" << endl;
#endif

  return std::make_pair(mu_o , LamU);
}

std::pair<VectorNd, MatrixNNd> CondNormalWishart(const int N, const MatrixNNd &S, const VectorNd &Um, const VectorNd &mu, const double kappa, const MatrixNNd &T, const int nu)
{
    auto mu_m = (mu - Um);
    VectorNd mu_c = (kappa*mu + N*Um) / (kappa + N);

    double kappa_c = kappa + N;
    double kappa_m = (kappa * N)/(kappa + N);
    auto X = ( T + N * S + kappa_m * (mu_m * mu_m.transpose()));
    MatrixNNd T_c = X.inverse();
    int nu_c = nu + N;

#ifdef TEST_MVNORMAL
    cout << "mu_c:\n" << mu_c << endl;
    cout << "kappa_c:\n" << kappa_c << endl;
    cout << "T_c:\n" << T_c << endl;
    cout << "nu_c:\n" << nu_c << endl;
#endif

    return NormalWishart(mu_c, kappa_c, T_c, nu_c);
}
