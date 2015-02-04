#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

using namespace Eigen;
using namespace std;

const int num_feat = 30;
unsigned num_p = 0;
unsigned num_m = 0;

const int alpha = 2;
const int nsims = 10;
const int burnin = 5;

double mean_rating;

typedef Matrix<double, Dynamic, Dynamic> MatrixD;
typedef SparseMatrix<double> SparseMatrixD;


SparseMatrix<double> loadChemo()
{
    ifstream f("chembl_19_mf1/chembl-IC50-360targets.csv");

    typedef Eigen::Triplet<double> T;
    std::vector<T> lst;
    lst.reserve(100000);
    
    while (!f.eof()) {
        unsigned i, j;
        double v_ij; 
        f >> i >> j >> v_ij;
        num_p = std::max(num_p, i);
        num_m = std::max(num_m, j);
        lst.push_back(T(i,j,v_ij));
    }

    f.close();

    SparseMatrix<double> M(num_p, num_m);
    M.setFromTriplets(lst.begin(), lst.end());

    return M;
}

void init(SparseMatrix<double> &mat) {
    mean_rating = mat.sum() / mat.nonZeros();

    MatrixD sample_u(num_p, num_feat);
    MatrixD sample_m(num_m, num_feat);
    MatrixD mu_u(num_feat, 1);
    MatrixD mu_m(num_feat, 1);
    MatrixD Lambda_u(num_feat, num_feat);
    Lambda_u.setIdentity();
    MatrixD Lambda_m(num_feat, num_feat);
    Lambda_m.setIdentity();

    // parameters of Inv-Whishart distribution (see paper for details)
    MatrixD WI_u(num_feat, num_feat);
    WI_u.setIdentity();
    const int b0_u = 2;
    const int df_u = num_feat;
    MatrixD mu0_u(num_feat, 1);
    mu0_u.setZero();

    MatrixD WI_m(num_feat, num_feat);
    WI_m.setIdentity();
    const int b0_m = 2;
    const int df_m = num_feat;
    MatrixD mu0_m(num_feat, 1);
    mu0_m.setZero();
}

/*
function pred(probe_vec, sample_m, sample_u, mean_rating)
  sum(sample_m[probe_vec[:,2],:].*sample_u[probe_vec[:,1],:],2) + mean_rating
end
*/

MatrixD randn(int n)
{
    return MatrixD(n,n);
}

MatrixD sample_movie(int mm, SparseMatrixD &mat, double mean_rating, 
    MatrixD sample_u, int alpha, MatrixD mu_u, MatrixD Lambda_u)
{
    auto rr = mat.col(mm).toDense();
    rr.array() -= mean_rating;
    MatrixD MM;

    int i = 0;
    for (SparseMatrixD::InnerIterator it(mat,mm); it; ++it, ++i) {
        MM.col(i) = sample_u.col(it.col());
    }

    auto covar = (Lambda_u + alpha * MM.transpose() * MM).inverse();
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

      for mm = 1:num_m
        sample_m[mm, :] = sample_movie(mm, Am, mean_rating, sample_u, alpha, mu_m, Lambda_m)
      end

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
