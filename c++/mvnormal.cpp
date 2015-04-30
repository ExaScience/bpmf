
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
Eigen::MatrixXd MvNormal(Eigen::MatrixXd covar, Eigen::VectorXd mean, int nn = 1) 
{
  int size = mean.rows(); // Dimensionality (rows)
  Eigen::internal::scalar_normal_dist_op<double> randN; // Gaussian functor
  Eigen::internal::scalar_normal_dist_op<double>::rng.seed(1); // Seed the rng
  Eigen::MatrixXd normTransform(size,size);

  Eigen::LLT<Eigen::MatrixXd> cholSolver(covar);
  normTransform = cholSolver.matrixL();

  Eigen::MatrixXd samples = (normTransform 
                           * Eigen::MatrixXd::NullaryExpr(size,nn,randN)).colwise() 
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

    auto ret = c.transpose() * c;

#ifdef TEST_MVNORMAL
    cout << "WISHART UNIT\n" << endl;
    cout << "  m:\n" << m << endl;
    cout << "  df:\n" << df << endl;
    cout << "  WishartUnit\n" << ret << endl;
#endif

    return ret;
}

MatrixXd Wishart(LLT<MatrixXd> sigma, int df)
{
  auto u = WishartUnit(sigma.cols(), df);
  auto ret = sigma.matrixL() * u * sigma.matrixU();

#ifdef TEST_MVNORMAL
    cout << "WISHART\n" << endl;
    cout << "  Sigma:\n" << sigma.reconstructedMatrix() << endl;
    cout << "  df:\n" << df << endl;
    cout << "  Wishart\n" << ret << endl;
#endif


  return ret;
}


// from julia package Distributions: conjugates/normalwishart.jl
std::pair<Eigen::VectorXd, Eigen::MatrixXd> NormalWishart(Eigen::VectorXd mu, double kappa, Eigen::MatrixXd T, double nu) 
{
  int size = mu.cols(); // Dimensionality (rows)
  
  Eigen::LLT<Eigen::MatrixXd> cholSolver(T);
  auto Lam = Wishart(cholSolver, nu); 
  auto mu_o = MvNormal(Lam / kappa, mu);

#ifdef TEST_MVNORMAL
    cout << "BEGIN NORMAL WISHART\n" << endl;
    cout << "  mu:\n" << mu << endl;
    cout << "  kappa:\n" << kappa << endl;
    cout << "  T:\n" << T << endl;
    cout << "  nu:\n" << nu << endl;
    cout << "  mu_o\n" << mu_o << endl;
    cout << "  Lam\n" << Lam << endl;
    cout << "  Lam-1\n" << Lam_1 << endl;
    cout << "END NORMAL WISHART\n" << endl;
#endif

  return std::make_pair(mu_o , Lam);
}

VectorXd nrandn(int n, double mean, double sigma)
{
    VectorXd ret(n);
    boost::mt19937 gen;
    boost::random::normal_distribution<> dist(mean,sigma);

    for(int i=0; i<n; ++i) ret(i) = dist(gen);
        
    return ret;
}

// from bpmf.jl -- verified
std::pair<Eigen::VectorXd, Eigen::MatrixXd> CondNormalWishart(MatrixXd U, VectorXd mu, double kappa, MatrixXd T, int nu)
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

    int nu = 3;

    VectorXd mu_out;
    MatrixXd T_out;

    cout << "COND NORMAL WISHART\n" << endl;

    tie(mu_out, T_out) = CondNormalWishart(U, mu, kappa, T, nu);

    cout << "mu_out:\n" << mu_out << endl;
    cout << "T_out:\n" << T_out << endl;

    cout << "\n-----\n\n";

    cout << "NORMAL WISHART\n" << endl;

    tie(mu_out, T_out) = NormalWishart(mu, kappa, T, nu);

    cout << "mu_out:\n" << mu_out << endl;
    cout << "T_out:\n" << T_out << endl;



}

#endif
