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

using namespace std;
using namespace Eigen;

/*
  We need a functor that can pretend it's const,
  but to be a good random number generator 
  it needs mutable state.
*/


thread_local struct RNG rng;

RNG::RNG(unsigned long long c)
  : generator(42), capacity(c), counter(0)
{
  Sys::cout() << " RNG: ";
  for (unsigned long long i = 0; i < c; ++i)
  {
      double d = normal_d(generator);
      if (i<10) Sys::cout() << d << " ";
      stash.push_back(d);
  }
  Sys::cout() << std::endl;
}

double &RNG::operator()()
{
  counter++;
  return stash[counter % capacity];
}

double randn() {
    return rng();
}

void RNG::set_pos(unsigned long long p)
{
  rng.counter = p;
}

/* -------------------------------------------------------------------------------- */

/*
  Draw nn samples from a size-dimensional normal distribution
  with a specified mean and covariance
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
        c(i,i) = sqrt(2.0 * gam(rng.generator));
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
