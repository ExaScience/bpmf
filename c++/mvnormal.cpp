
/*
 * From:
 * http://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
 */


#include <iostream>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <Eigen/Dense>

#include "bpmf.h"

using namespace std;
using namespace Eigen;

/*
  We need a functor that can pretend it's const,
  but to be a good random number generator 
  it needs mutable state.
*/
namespace Eigen {
namespace internal {
template<typename Scalar> 
struct scalar_normal_dist_op 
{
  static boost::mt19937 rng;    // The uniform pseudo-random algorithm
  mutable boost::normal_distribution<Scalar> norm;  // The gaussian combinator

  EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

  template<typename Index>
  inline const Scalar operator() (Index, Index = 0) const { return norm(rng); }
};

template<typename Scalar> boost::mt19937 scalar_normal_dist_op<Scalar>::rng;

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
        boost::gamma_distribution<> chi(0.5*(df - i));
        boost::mt19937 rng;
        boost::variate_generator<boost::mt19937&, boost::gamma_distribution<> > gen(rng, chi);
        c(i,i) = sqrt(2.0 * chi(gen));
        c.block(i,i+1,1,m-i-1) = nrandn(m-i-1).transpose();
    }

    MatrixXd ret = c * c.transpose();

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
  int size = mu.cols(); // Dimensionality (rows)
  
  MatrixXd Lam = Wishart(T, nu); 
  MatrixXd mu_o = MvNormal(Lam / kappa, mu);

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
    static boost::mt19937 gen;
    boost::random::normal_distribution<> dist(mean,sigma);

    for(int i=0; i<n; ++i) ret(i) = dist(gen);
        
    return ret;
}

// from bpmf.jl -- verified
std::pair<VectorXd, MatrixXd> CondNormalWishart(MatrixXd U, VectorXd mu, double kappa, MatrixXd T, int nu)
{
  int N = U.cols();
  auto Um = U.colwise().mean();
  auto Ut = Um.transpose();

  // http://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
  MatrixXd C = U.rowwise() - Um;
  MatrixXd S = (C.adjoint() * C) / double(U.rows() - 1);

  VectorXd mu_c = (kappa*mu + N*Ut) / (kappa + N);
  double kappa_c = kappa + N;
  MatrixXd T_c = ( T.inverse() + N * S + (kappa * N)/(kappa + N) * (mu - Ut) * ((mu - Ut).transpose())).inverse();
  int nu_c = nu + N;

#ifdef TEST_MVNORMAL
  cout << "mu_c:\n" << mu_c << endl;
  cout << "kappa_c:\n" << kappa_c << endl;
  cout << "T_c:\n" << T_c << endl;
  cout << "nu_c:\n" << nu_c << endl;
#endif

  return NormalWishart(mu_c, kappa_c, T_c, nu_c);
}

#ifdef TEST_MVNORMAL

int main()
{
    MatrixXd U(3,3);
    U.setOnes();

    VectorXd mu(3);
    mu.setZero();

    double kappa = 2;

    MatrixXd T(3,3);
    T.setIdentity(3,3);
    T.array() /= 4;

    int nu = 3;

    VectorXd mu_out;
    MatrixXd T_out;

#if 0
    cout << "COND NORMAL WISHART\n" << endl;

    tie(mu_out, T_out) = CondNormalWishart(U, mu, kappa, T, nu);

    cout << "mu_out:\n" << mu_out << endl;
    cout << "T_out:\n" << T_out << endl;

    cout << "\n-----\n\n";
#endif

#if 1
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
}

#endif
