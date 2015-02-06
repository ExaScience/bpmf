
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>


using namespace Eigen;
using namespace std;

typedef SparseMatrix<double> SparseMatrixD;

const int num_feat = 30;
unsigned num_p = 0;
unsigned num_m = 0;

const int alpha = 2;
const int nsims = 10;
const int burnin = 5;

double mean_rating;

SparseMatrixD M;

MatrixXd sample_u;
MatrixXd sample_m;
MatrixXd mu_u(num_feat, 1);
MatrixXd mu_m(num_feat, 1);
MatrixXd Lambda_u(num_feat, num_feat);
MatrixXd Lambda_m(num_feat, num_feat);

// parameters of Inv-Whishart distribution (see paper for details)
MatrixXd WI_u(num_feat, num_feat);
const int b0_u = 2;
const int df_u = num_feat;
MatrixXd mu0_u(num_feat, 1);

MatrixXd WI_m(num_feat, num_feat);
const int b0_m = 2;
const int df_m = num_feat;
MatrixXd mu0_m(num_feat, 1);

void loadChemo()
{
    typedef Eigen::Triplet<double> T;
    std::vector<T> lst;
    lst.reserve(100000);
    
    FILE *f = fopen("../data/chembl_19_mf1/chembl-IC50-360targets.csv", "r");
    assert(f);

    // skip header
    char buf[2048];
    fscanf(f, "%s\n", buf);

    // data
    unsigned i, j;
    double v_ij;
    while (!feof(f)) {
        if (!fscanf(f, "%d,%d,%lg\n", &i, &j, &v_ij)) continue;
        num_p = std::max(num_p, i);
        num_m = std::max(num_m, j);
        lst.push_back(T(i,j,v_ij));
    }

    fclose(f);

    M = SparseMatrix<double>(num_p+1, num_m+1);
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
function pred(probe_vec, sample_m, sample_u, mean_rating)
  sum(sample_m[probe_vec[:,2],:].*sample_u[probe_vec[:,1],:],2) + mean_rating
end
*/

double rand(double mean = 0.5, double sigma = 1)
{
    boost::mt19937 gen;
    boost::random::normal_distribution<> dist(mean,sigma);
    return dist(gen);
}

VectorXd randn(int n, double mean = 0.5, double sigma = 1)
{
    VectorXd ret(n);
    boost::mt19937 gen;
    boost::random::normal_distribution<> dist(mean,sigma);

    for(int i=0; i<n; ++i) ret << dist(gen);
        
    return ret;
}

MatrixXd sample_movie(int mm, SparseMatrixD &mat, double mean_rating, 
    MatrixXd sample_u, int alpha, MatrixXd mu_u, MatrixXd Lambda_u)
{
    auto rr = mat.col(mm).toDense();
    rr.array() -= mean_rating;

    int i = 0;

    MatrixXd E(sample_u.rows(), mat.col(mm).nonZeros());
    for (SparseMatrixD::InnerIterator it(mat,mm); it; ++it, ++i) {
        // cout << "M[" << it.row() << "," << it.col() << "] = " << it.value() << endl;
        E.col(i) = sample_u.col(it.col());
    }

    auto MM = E * E.transpose();
    MatrixXd MMs = alpha * MM.array();
    auto covar = (Lambda_u + MMs).inverse();
    auto mu = covar * (alpha * MM.transpose() * rr + Lambda_u * mu_u);

    return covar.llt().matrixL().transpose() * randn(num_feat) + mu;
}

void run() {
    double err_avg = 0.0;
    double err = 0.0;

    std::cout << "Sampling" << endl;
    for(int i=0; i<nsims; ++i) {

#if 0
      // Sample from movie hyperparams
      mu_m, Lambda_m = rand( ConditionalNormalWishart(sample_m, vec(mu0_m), b0_m, WI_m, df_m) )

      // Sample from user hyperparams
      mu_u, Lambda_u = rand( ConditionalNormalWishart(sample_u, vec(mu0_u), b0_u, WI_u, df_u) )
#endif

      for(int mm = 1; mm < num_m; ++mm) {
        auto sample = sample_movie(mm, M, mean_rating, sample_u, alpha, mu_m, Lambda_m);
      }

#if 0

      for uu = 1:num_p
        sample_u[uu, :] = sample_user(uu, Au, mean_rating, sample_m, alpha, mu_u, Lambda_u)
      end

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

int main()
{
    loadChemo();
    init();
    run();

    return 0;
}
