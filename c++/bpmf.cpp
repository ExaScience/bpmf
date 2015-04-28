
#include <stdlib.h>     /* srand, rand */

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include "bpmf.h"

using namespace std;

typedef SparseMatrix<double> SparseMatrixD;

const int num_feat = 30;
unsigned num_p = 0;
unsigned num_m = 0;

const int alpha = 2;
const int nsims = 2;
const int burnin = 5;

double mean_rating;

SparseMatrixD M;
typedef Eigen::Triplet<double> T;
vector<T> probe_vec, test_vec;

MatrixXd sample_u;
MatrixXd sample_m;

VectorXd mu_u(num_feat);
VectorXd mu_m(num_feat);
MatrixXd Lambda_u(num_feat, num_feat);
MatrixXd Lambda_m(num_feat, num_feat);

// parameters of Inv-Whishart distribution (see paper for details)
MatrixXd WI_u(num_feat, num_feat);
const int b0_u = 2;
const int df_u = num_feat;
VectorXd mu0_u(num_feat);

MatrixXd WI_m(num_feat, num_feat);
const int b0_m = 2;
const int df_m = num_feat;
VectorXd mu0_m(num_feat);

void loadChemo(const char* fname)
{
    std::vector<T> lst;
    lst.reserve(100000);
    
    FILE *f = fopen(fname, "r");
    assert(f && "Could not open file");

    // skip header
    char buf[2048];
    fscanf(f, "%s\n", buf);

    // data
    unsigned i, j;
    double v;
    while (!feof(f)) {
        if (!fscanf(f, "%d,%d,%lg\n", &i, &j, &v)) continue;

        if ((rand() % 5) == 0) {
            probe_vec.push_back(T(i,j,log10(v)));
            if (v<200) 
                test_vec.push_back(T(i,j,log10(v));
        } else {
            num_p = std::max(num_p, i);
            num_m = std::max(num_m, j);
            lst.push_back(T(i,j,log10(v)));
        }
    }
    num_p++;
    num_m++;
    fclose(f);

    M = SparseMatrix<double>(num_p, num_m);
    M.setFromTriplets(lst.begin(), lst.end());
}

void init() {
    mean_rating = M.sum() / M.nonZeros();
    Lambda_u.setIdentity();
    Lambda_m.setIdentity();

    // parameters of Inv-Whishart distribution (see paper for details)
    WI_u.setIdentity();
    mu0_u.setZero();

    WI_m.setIdentity();
    mu0_m.setZero();

    sample_u = MatrixXd(num_p, num_feat);
    sample_u.setZero();
    sample_m = MatrixXd(num_m, num_feat);
    sample_m.setZero();
}


/*
double pred(VevtorXd probe_vec, sample_m, sample_u, mean_rating)
  sum(sample_m[probe_vec[:,2],:].*sample_u[probe_vec[:,1],:],2) + mean_rating
end
*/

double rand(double mean, double sigma)
{
    boost::mt19937 gen;
    boost::random::normal_distribution<> dist(mean,sigma);
    return dist(gen);
}

VectorXd randn(int n, double mean, double sigma)
{
    VectorXd ret(n);
    boost::mt19937 gen;
    boost::random::normal_distribution<> dist(mean,sigma);

    for(int i=0; i<n; ++i) ret(i) = dist(gen);
        
    return ret;
}

MatrixXd sample_movie(int mm, SparseMatrixD &mat, double mean_rating, 
    MatrixXd sample_u, int alpha, MatrixXd mu_u, MatrixXd Lambda_u)
{
    int i = 0;
    MatrixXd E(mat.col(mm).nonZeros(), num_feat);
    VectorXd rr(mat.col(mm).nonZeros());
    for (SparseMatrixD::InnerIterator it(mat,mm); it; ++it, ++i) {
        //cout << "M[" << it.row() << "," << it.col() << "] = " << it.value() << endl;
        E.row(i) = sample_u.row(it.row());
        rr(i) = it.value() - mean_rating;
    }

    auto MM = E.transpose() * E;
    MatrixXd MMs = alpha * MM.array();
    assert(MMs.cols() == num_feat && MMs.rows() == num_feat);
    auto covar = (Lambda_u + MMs).inverse();
    auto MMrr = E.transpose() * rr; 
    MMrr.array() *= alpha;
    auto U = Lambda_u * mu_u;
    auto mu = covar * (MMrr + U);

    auto chol = covar.llt().matrixL().transpose();
    auto result = chol * randn(num_feat) + mu;
    return result.transpose();
}

MatrixXd WishartUnit(MatrixXd sigma, int df)
{
    auto m = sigma.cols();
    MatrixXd c(m,m);
    c.setZero();

    for ( int i = 0; i < m; i++ ) {
        boost::gamma_distribution<> chi(0.5*(df - i));
        boost::mt19937 rng;
        boost::variate_generator<boost::mt19937&, boost::gamma_distribution<> > gen(rng, chi);
        c(i,i) = sqrt(2.0 * chi(gen));
        c.block(i,i+1,1,m-i) = randn(m-i);
    }

    return c.transpose() * c;
}

MatrixXd Wishart(MatrixXd sigma, int df)
{
  MatrixXd r = sigma.llt().matrixU();
  auto u = WishartUnit(sigma, df);
  return r.transpose() * u;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> CondNormalWishart(MatrixXd U, VectorXd mu, double kappa, MatrixXd T, int nu)
{
#if 0
  int n = U.cols();
  VectorXd mean_U = U.mean(); // FIXME: should be collumnwise!!!
  auto S = U.cov
  S = cov(U, mean=Ū)
  Ū = Ū'

  mu_c = (kappa*mu + N*Ū) / (kappa + N)
  kappa_c = kappa + N
  T_c = inv( inv(T) + N * S + (kappa * N)/(kappa + N) * (mu - Ū) * (mu - Ū)' )
  nu_c = nu + N

  NormalWishart(vec(mu_c), kappa_c, T_c, nu_c)
end


  MatrixXd r = sigma.llt().matrixU();
  auto u = WishartUnit(sigma, df);
  return r.transpose() * u;
#endif
}


void run() {
    double err_avg = 0.0;
    double err = 0.0;

    SparseMatrixD Mt = M.transpose();

    std::cout << "Sampling" << endl;
    for(int i=0; i<nsims; ++i) {

      // Sample from movie hyperparams
      tie(mu_m, Lambda_m) = CondNormalWishart(sample_m, mu0_m, b0_m, WI_m, df_m);

      // Sample from user hyperparams
      tie(mu_u, Lambda_u) = CondNormalWishart(sample_u, mu0_u, b0_u, WI_u, df_u);

      for(int mm = 1; mm < num_m; ++mm) {
        sample_m.row(mm) = sample_movie(mm, M, mean_rating, sample_u, alpha, mu_m, Lambda_m);
      }

      for(int uu = 1; uu < num_p; ++uu) {
        sample_u.row(uu) = sample_movie(uu, Mt, mean_rating, sample_m, alpha, mu_u, Lambda_u);
      }
#if 0

      probe_rat = pred(probe_vec, sample_m, sample_u, mean_rating)

      if i > burnin
        probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1)
        counter_prob = counter_prob + 1
      else
        probe_rat_all = probe_rat
        counter_prob = 1
      end

      err_avg = mean(ratings_test .== (probe_rat_all .< log10(200)))
      err = mean(ratings_test .== (probe_rat .< log10(200)))

      printf("Iteration %d:\t avg RMSE %6.4f RMSE %6.4f FU(%6.4f) FM(%6.4f)\n", i, err_avg, err, vecnorm(sample_u), vecnorm(sample_m));
#endif
    }
}

int main(int argc, char *argv[])
{
    const char *fname = argv[1];
    assert(fname && "filename missing");

    loadChemo(fname);
    init();
    run();

    return 0;
}
