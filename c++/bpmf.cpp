
#include <stdlib.h>     /* srand, rand */

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include "bpmf.h"

using namespace std;
using namespace Eigen;

typedef SparseMatrix<double> SparseMatrixD;

const int num_feat = 30;
unsigned num_p = 0;
unsigned num_m = 0;

const int alpha = 2;
const int nsims = 50;
const int burnin = 5;

double mean_rating = .0;

SparseMatrixD M;
typedef Eigen::Triplet<double> T;
vector<T> probe_vec;

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
        } else {
            num_p = std::max(num_p, i);
            num_m = std::max(num_m, j);
            mean_rating += v;
            lst.push_back(T(i,j,log10(v)));
        }
    }
    num_p++;
    num_m++;
    mean_rating /= lst.size();
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

double eval_probe_vec(const vector<T> &probe_vec, const MatrixXd &sample_m, const MatrixXd &sample_u, double mean_rating)
{
    unsigned correct = 0;
    for(auto t : probe_vec) {
         double prediction = sample_m.row(t.col()).dot(sample_u.row(t.row())) + mean_rating;
         correct += (t.value() < log10(200)) == (prediction < log10(200));
    }
    return (double)correct / (double)probe_vec.size();
}

MatrixXd sample_movie(int mm, SparseMatrixD &mat, double mean_rating, 
    MatrixXd sample_u, int alpha, MatrixXd mu_u, MatrixXd Lambda_u)
{
    int i = 0;
    MatrixXd E(mat.col(mm).nonZeros(), num_feat);
    VectorXd rr(mat.col(mm).nonZeros());
    //cout << "movie " << endl;
    //cout << "mean rating " << mean_rating << endl;
    for (SparseMatrixD::InnerIterator it(mat,mm); it; ++it, ++i) {
        //cout << "M[" << it.row() << "," << it.col() << "] = " << it.value() << endl;
        E.row(i) = sample_u.row(it.row());
        rr(i) = it.value() - mean_rating;
    }


    auto MM = E.transpose() * E;

    //cout << "MM = " << MM << endl;
    MatrixXd MMs = alpha * MM.array();
    assert(MMs.cols() == num_feat && MMs.rows() == num_feat);
    auto covar = (Lambda_u + MMs).inverse();
    //cout << "Lambda_u = " << Lambda_u << endl;
    //cout << "covar = " << covar << endl;
    auto MMrr = E.transpose() * rr; 
    MMrr.array() *= alpha;
    auto U = Lambda_u * mu_u;
    auto mu = covar * (MMrr + U);
    //cout << "mu = " << mu << endl;

    auto chol = covar.llt().matrixL().transpose();
    auto result = chol * nrandn(num_feat) + mu;

    //cout << "movie " << mm << ":" << result.transpose() << endl;

    return result.transpose();
}

void run() {
    unsigned counter_prob = 0;
    VectorXd probe_rat_all; probe_rat_all.setZero();
    auto start = std::chrono::steady_clock::now();

    SparseMatrixD Mt = M.transpose();

    std::cout << "Sampling" << endl;
    for(int i=0; i<nsims; ++i) {

      // Sample from movie hyperparams
      tie(mu_m, Lambda_m) = CondNormalWishart(sample_m, mu0_m, b0_m, WI_m, df_m);

      // Sample from user hyperparams
      tie(mu_u, Lambda_u) = CondNormalWishart(sample_u, mu0_u, b0_u, WI_u, df_u);

#pragma omp parallel for
      for(int mm = 1; mm < num_m; ++mm) {
        sample_m.row(mm) = sample_movie(mm, M, mean_rating, sample_u, alpha, mu_m, Lambda_m);
      }

#pragma omp parallel for
      for(int uu = 1; uu < num_p; ++uu) {
        sample_u.row(uu) = sample_movie(uu, Mt, mean_rating, sample_m, alpha, mu_u, Lambda_u);
      }

      double correct_ratio = eval_probe_vec(probe_vec, sample_m, sample_u, mean_rating);
      double norm_u = sample_u.norm();
      double norm_m = sample_m.norm();
      auto end = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration<double>(end - start);
      double samples_per_sec = (i + 1) * (num_p + num_m) / elapsed.count();

      printf("Iteration %d:\t num_correct: %3.2f%%\tFU(%6.2f)\tFM(%6.2f)\tSamples/sec: %6.2f\n",
              i, 100*correct_ratio, norm_u, norm_m, samples_per_sec);
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
