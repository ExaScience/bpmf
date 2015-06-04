
/*
 * From:
 * http://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
 */


#include <iostream>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include "bpmf.h"

using namespace std;
using namespace Eigen;

/*
  We need a functor that can pretend it's const,
  but to be a good random number generator 
  it needs mutable state.
*/

#if 0
thread_local 
#endif 
static boost::mt19937 rng;

namespace Eigen {
namespace internal {
template<typename Scalar> 
struct scalar_normal_dist_op 
{
  mutable boost::normal_distribution<Scalar> norm;  // The gaussian combinator

  EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

  template<typename Index>
  inline const Scalar operator() (Index, Index = 0) const { return norm(rng); }
};


template<typename Scalar>
struct functor_traits<scalar_normal_dist_op<Scalar> >
{ enum { Cost = 50 * NumTraits<Scalar>::MulCost, PacketAccess = false, IsRepeatable = false }; };
} // end namespace internal
} // end namespace Eigen


/*
  Draw nn samples from a size-dimensional normal distribution
  with a specified mean and covariance
*/
MatrixXd MvNormal(MatrixXd covar, VectorXd mean, int nn = 1) 
{
  int size = mean.rows(); // Dimensionality (rows)
  internal::scalar_normal_dist_op<double> randN; // Gaussian functor
  MatrixXd normTransform(size,size);

  LLT<MatrixXd> cholSolver(covar);
  normTransform = cholSolver.matrixL();

  MatrixXd samples = (normTransform 
                           * MatrixXd::NullaryExpr(size,nn,randN)).colwise() 
                           + mean;

  return samples;
}

MatrixXd WishartUnit(int m, int df)
{
    MatrixXd c(m,m);
    c.setZero();

    for ( int i = 0; i < m; i++ ) {
        boost::gamma_distribution<> gam(0.5*(df - i));
        boost::variate_generator<boost::mt19937&, boost::gamma_distribution<> > gen(rng, gam);
        c(i,i) = sqrt(2.0 * gam(rng));
        c.block(i,i+1,1,m-i-1) = nrandn(m-i-1).transpose();
    }

    MatrixXd ret = c.transpose() * c;

#ifdef TEST_MVNORMAL
    cout << "WISHART UNIT {\n" << endl;
    cout << "  m:\n" << m << endl;
    cout << "  df:\n" << df << endl;
    cout << "  ret;\n" << ret << endl;
    cout << "  c:\n" << c << endl;
    cout << "}\n" << ret << endl;
#endif

    return ret;
}

MatrixXd Wishart(const MatrixXd &sigma, int df)
{
//  Get R, the upper triangular Cholesky factor of SIGMA.
  MatrixXd r = sigma.llt().matrixU();

//  Get AU, a sample from the unit Wishart distribution.
  MatrixXd au = WishartUnit(sigma.cols(), df);

//  Construct the matrix A = R' * AU * R.
  MatrixXd a = r.transpose() * au * r; 


#ifdef TEST_MVNORMAL
    cout << "WISHART {\n" << endl;
    cout << "  sigma::\n" << sigma << endl;
    cout << "  r:\n" << r << endl;
    cout << "  au:\n" << au << endl;
    cout << "  df:\n" << df << endl;
    cout << "  a:\n" << a << endl;
    cout << "}\n" << endl;
#endif


  return a;
}


// from julia package Distributions: conjugates/normalwishart.jl
std::pair<VectorXd, MatrixXd> NormalWishart(VectorXd mu, double kappa, MatrixXd T, double nu) 
{
  MatrixXd Lam = Wishart(T, nu); 
  MatrixXd mu_o = MvNormal(Lam.inverse() / kappa, mu);

#ifdef TEST_MVNORMAL
    cout << "NORMAL WISHART {\n" << endl;
    cout << "  mu:\n" << mu << endl;
    cout << "  kappa:\n" << kappa << endl;
    cout << "  T:\n" << T << endl;
    cout << "  nu:\n" << nu << endl;
    cout << "  mu_o\n" << mu_o << endl;
    cout << "  Lam\n" << Lam << endl;
    cout << "}\n" << endl;
#endif

  return std::make_pair(mu_o , Lam);
}

VectorXd nrandn(int n, double mean, double sigma)
{
    VectorXd ret(n);
    
    boost::random::normal_distribution<> dist(mean,sigma);

    for(int i=0; i<n; ++i) ret(i) = dist(rng);
        
    return ret;
}

double acc[7] = { .0 };

// from bpmf.jl -- verified
std::pair<VectorXd, MatrixXd> CondNormalWishart(const MatrixXd &U, const VectorXd &mu, const double kappa, const MatrixXd &T, const int nu)
{
         double   t[7];

  int nrows = U.rows();

  t[0] = tick();
  auto Um = U.rowwise().mean();
  t[1] = tick();
  acc[0] = t[1] - t[0];

  // http://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
  MatrixXd C = U.colwise() - Um;
  t[2] = tick();
  acc[1] = t[2] - t[1];
  auto S = (C * C.adjoint()) / double(U.cols() - 1);
  t[3] = tick();
  acc[2] = t[3] - t[2];
  VectorXd mu_c = (kappa*mu + nrows*Um) / (kappa + nrows);
  double kappa_c = kappa + nrows;
  t[4] = tick();
  acc[3] = t[4] - t[3];
  VectorXd mu_m = (mu - Um);
  double kappa_m = (kappa * nrows)/(kappa + nrows);
  auto X = ( T + nrows * S.transpose() + kappa_m * (mu_m * mu_m.transpose())); //.inverse();
  t[5] = tick();
  acc[4] = t[5] - t[4];
  MatrixXd T_c = X.inverse();
  t[6] = tick();
  acc[5] = t[6] - t[5];
  acc[6] = t[6] - t[0];
  int nu_c = nu + nrows;

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
    for(int i=0; i<30; ++i) {
        tie(mu_out, T_out) = CondNormalWishart(U, mu, kappa, T, nu);
        cout << i << "\r" << flush;
    }
    cout << endl << flush;

    for(int i=0; i<6; ++i) {
        cout << i << ": " << acc[i] / acc[6] << endl;
    }

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
