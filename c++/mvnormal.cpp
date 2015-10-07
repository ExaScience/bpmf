
/*
 * From:
 * http://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
 */


#include <iostream>


#include "bpmf.h"

using namespace std;
using namespace Eigen;

/*
  We need a functor that can pretend it's const,
  but to be a good random number generator 
  it needs mutable state.
*/

#ifndef __clang__
thread_local 
#endif 
static std::mt19937 rng;

#ifndef __clang__
thread_local 
#endif 
static normal_distribution<> nd;

double randn(double) {
  return nd(rng);
}

auto
nrandn(int n) -> decltype( VectorXd::NullaryExpr(n, ptr_fun(randn)) ) 
{
    return VectorXd::NullaryExpr(n, ptr_fun(randn));
}

MatrixXd MvNormal_prec(const MatrixXd & Lambda, const VectorXd & mean, int nn = 1)
{
  int size = mean.rows(); // Dimensionality (rows)

  LLT<MatrixXd> chol(Lambda);

  MatrixXd r = MatrixXd::NullaryExpr(size,nn,ptr_fun(randn));
	chol.matrixU().solveInPlace(r);
  return r.colwise() + mean;
}

MatrixXd MvNormalChol_prec(double kappa, const MatrixXd & Lambda_U, const VectorXd & mean, int nn = 1)
{
  int size = mean.rows(); // Dimensionality (rows)

  MatrixXd r = MatrixXd::NullaryExpr(size,nn,ptr_fun(randn));
	Lambda_U.triangularView<Upper>().solveInPlace(r);
  return (r / sqrt(kappa)).colwise() + mean;
}

/*
  Draw nn samples from a size-dimensional normal distribution
  with a specified mean and covariance
*/
MatrixXd MvNormal(const MatrixXd & covar, const VectorXd & mean, int nn = 1)
{
  int size = mean.rows(); // Dimensionality (rows)
  MatrixXd normTransform(size,size);

  LLT<MatrixXd> cholSolver(covar);
  normTransform = cholSolver.matrixL();

  auto normSamples = MatrixXd::NullaryExpr(size,nn,ptr_fun(randn));
  MatrixXd samples = (normTransform * normSamples).colwise() + mean;

  return samples;
}

void WishartUnitChol(int m, int df, MatrixXd & c) {
    c.setZero(m, m);

    for ( int i = 0; i < m; i++ ) {
        std::gamma_distribution<> gam(0.5*(df - i));
        c(i,i) = sqrt(2.0 * gam(rng));
        VectorXd r = nrandn(m-i-1);
        c.block(i,i+1,1,m-i-1) = r.transpose();
    }

#ifdef TEST_MVNORMAL
    cout << "WISHART UNIT {\n" << endl;
    cout << "  m:\n" << m << endl;
    cout << "  df:\n" << df << endl;
    cout << "}\n" << c << endl;
#endif
}

void WishartChol(const MatrixXd &sigma, const int df, MatrixXd & U)
{
//  Get R, the upper triangular Cholesky factor of SIGMA.
  auto chol = sigma.llt();

//  Get AU, a sample from the unit Wishart distribution.
  MatrixXd au;
  WishartUnitChol(sigma.cols(), df, au);
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
std::pair<VectorXd, MatrixXd> NormalWishart(const VectorXd & mu, double kappa, const MatrixXd & T, double nu)
{
  MatrixXd LamU;
 	WishartChol(T, nu, LamU);
  MatrixXd mu_o = MvNormalChol_prec(kappa, LamU, mu);

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

std::pair<VectorXd, MatrixXd> OldCondNormalWishart(const MatrixXd &U, const VectorXd &mu, const double kappa, const MatrixXd &T, const int nu)
{
  int N = U.cols();

  auto Um = U.rowwise().mean();

  // http://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
  MatrixXd C = U.colwise() - Um;
  MatrixXd S = (C * C.adjoint()) / double(N - 1);
  VectorXd mu_c = (kappa*mu + N*Um) / (kappa + N);
  double kappa_c = kappa + N;
  MatrixXd T_c = ( T + N * S.transpose() + (kappa * N)/(kappa + N) * (mu - Um) * ((mu - Um).transpose())).inverse();
  int nu_c = nu + N;

#ifdef TEST_MVNORMAL
  cout << "mu_c:\n" << mu_c << endl;
  cout << "kappa_c:\n" << kappa_c << endl;
  cout << "T_c:\n" << T_c << endl;
  cout << "nu_c:\n" << nu_c << endl;
#endif

  return NormalWishart(mu_c, kappa_c, T_c, nu_c);
}

// from bpmf.jl -- verified
std::pair<VectorXd, MatrixXd> CondNormalWishart(const MatrixXd &U, const VectorXd &mu, const double kappa, const MatrixXd &T, const int nu)
{
  int N = U.cols();

  VectorXd Um = U.rowwise().mean();

  // http://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
  auto C = U.colwise() - Um;
  MatrixXd S = (C * C.adjoint()) / double(N - 1);
  VectorXd mu_c = (kappa*mu + N*Um) / (kappa + N);
  double kappa_c = kappa + N;
  auto mu_m = (mu - Um);
  double kappa_m = (kappa * N)/(kappa + N);
  auto X = ( T + N * S + kappa_m * (mu_m * mu_m.transpose()));
  MatrixXd T_c = X.inverse();
  int nu_c = nu + N;

#ifdef TEST_MVNORMAL
  cout << "mu_c:\n" << mu_c << endl;
  cout << "kappa_c:\n" << kappa_c << endl;
  cout << "T_c:\n" << T_c << endl;
  cout << "nu_c:\n" << nu_c << endl;
#endif

  return NormalWishart(mu_c, kappa_c, T_c, nu_c);
}

#if defined(TEST_MVNORMAL) || defined (BENCH_MVNORMAL)

int main()
{

    MatrixXd U(32,32 * 1024);
    U.setOnes();

    VectorXd mu(32);
    mu.setZero();

    double kappa = 2;

    MatrixXd T(32,32);
    T.setIdentity(32,32);
    T.array() /= 4;

    int nu = 3;

    VectorXd mu_out;
    MatrixXd T_out;

#ifdef BENCH_MVNORMAL
    for(int i=0; i<300; ++i) {
        tie(mu_out, T_out) = CondNormalWishart(U, mu, kappa, T, nu);
        cout << i << "\r" << flush;
    }
    cout << endl << flush;

    for(int i=0; i<7; ++i) {
        cout << i << ": " << (int)(100.0 * acc[i] / acc[7])  << endl;
    }
    cout << "total: " << acc[7] << endl;

    for(int i=0; i<300; ++i) {
        tie(mu_out, T_out) = OldCondNormalWishart(U, mu, kappa, T, nu);
        cout << i << "\r" << flush;
    }
    cout << endl << flush;

    cout << "total: " << acc[8] << endl;


#else
#if 1
    cout << "COND NORMAL WISHART\n" << endl;

    tie(mu_out, T_out) = CondNormalWishart(U, mu, kappa, T, nu);

    cout << "mu_out:\n" << mu_out << endl;
    cout << "T_out:\n" << T_out << endl;

    cout << "\n-----\n\n";
#endif

#if 0
    cout << "NORMAL WISHART\n" << endl;

    tie(mu_out, T_out) = NormalWishart(mu, kappa, T, nu);
    cout << "mu_out:\n" << mu_out << endl;
    cout << "T_out:\n" << T_out << endl;

#endif

#if 0
    cout << "MVNORMAL\n" << endl;
    MatrixXd out = MvNormal(T, mu, 10);
    cout << "mu:\n" << mu << endl;
    cout << "T:\n" << T << endl;
    cout << "out:\n" << out << endl;
#endif
#endif
}

#endif
