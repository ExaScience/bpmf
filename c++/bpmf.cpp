
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

const int num_feat = 32;
unsigned num_p = 0;
unsigned num_m = 0;

const int alpha = 2;
const int nsims = 100;
const int burnin = 5;

double mean_rating = .0;

SparseMatrixD M;
typedef Eigen::Triplet<double> T;
vector<T> probe_vec;

double *u_data, *m_data;

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
        i--;
        j--;

        if ((rand() % 5) == 0) {
            probe_vec.push_back(T(i,j,log10(v)));
        } 
#ifndef TEST_SAMPLE
        else // if not in test case -> remove probe_vec from lst
#endif
        {
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

    u_data = new double[num_p * num_feat]();
    m_data = new double[num_m * num_feat]();
}

pair<double,double> eval_probe_vec(const vector<T> &probe_vec, const MatrixXd &sample_m, const MatrixXd &sample_u, double mean_rating)
{
    unsigned n = probe_vec.size();
    unsigned correct = 0;
    double diff = .0;
    for(auto t : probe_vec) {
         double prediction = sample_m.col(t.col()).dot(sample_u.col(t.row())) + mean_rating;
         //cout << "prediction: " << prediction - mean_rating << " + " << mean_rating << " = " << prediction << endl;
         //cout << "actual: " << t.value() << endl;
         correct += (t.value() < log10(200)) == (prediction < log10(200));
         diff += abs(t.value() - prediction);
    }
   
    return std::make_pair((double)correct / n, diff / n);
}

MatrixXd sample_movie(int mm, const SparseMatrixD &mat, double mean_rating, 
    const MatrixXd &sample_u, int alpha, const MatrixXd &mu_u, const MatrixXd &Lambda_u)
{
    int i = 0;
    MatrixXd E(mat.col(mm).nonZeros(), num_feat);
    VectorXd rr(mat.col(mm).nonZeros());
    //cout << "movie " << endl;
    for (SparseMatrixD::InnerIterator it(mat,mm); it; ++it, ++i) {
        // cout << "M[" << it.row() << "," << it.col() << "] = " << it.value() << endl;
        E.row(i) = sample_u.row(it.row());
        rr(i) = it.value() - mean_rating;
    }


    MatrixXd MM = E.transpose() * E;

    MatrixXd MMs = alpha * MM.array();
    assert(MMs.cols() == num_feat && MMs.rows() == num_feat);
    MatrixXd covar = (Lambda_u + MMs).inverse();
    MatrixXd MMrr = E.transpose() * rr; 
    MMrr.array() *= alpha;
    MatrixXd U = Lambda_u * mu_u;
    MatrixXd mu = covar * (MMrr + U);

    MatrixXd chol = covar.llt().matrixU().transpose();
#ifdef TEST_SAMPLE
    VectorXd r(num_feat); r.setConstant(0.25);
#else
    VectorXd r = nrandn(num_feat);
#endif
    MatrixXd result = chol * r + mu;

    assert(result.rows() == num_feat && result.cols() == 1);

#ifdef TEST_SAMPLE
      cout << "movie " << mm << ":" << result.cols() << " x" << result.rows() << endl;
      cout << "mean rating " << mean_rating << endl;
      cout << "E = [" << E << "]" << endl;
      cout << "rr = [" << rr << "]" << endl;
      cout << "MM = [" << MM << "]" << endl;
      cout << "Lambda_u = [" << Lambda_u << "]" << endl;
      cout << "covar = [" << covar << "]" << endl;
      cout << "mu = [" << mu << "]" << endl;
      cout << "chol = [" << chol << "]" << endl;
      cout << "rand = [" << r <<"]" <<  endl;
      cout << "result = [" << result << "]" << endl;
#endif

    return result.transpose();
}

#ifdef TEST_SAMPLE
void test() {
    typedef Map<MatrixXd> MapXd;
    MapXd sample_u(u_data, num_feat, num_p);
    MapXd sample_m(m_data, num_feat, num_m);

    mu_m.setZero();
    Lambda_m.setIdentity();
    sample_u.setConstant(2.0);
    Lambda_m *= 0.5;
    sample_m.col(0) = sample_movie(0, M, mean_rating, sample_u.transpose(), alpha, mu_m, Lambda_m).transpose();
}

#else

void run() {
    auto start = std::chrono::steady_clock::now();

    SparseMatrixD Mt = M.transpose();

    typedef Map<MatrixXd> MapXd;
    MapXd sample_u(u_data, num_feat, num_p);
    MapXd sample_m(m_data, num_feat, num_m);

    std::cout << "Sampling" << endl;
    for(int i=0; i<nsims; ++i) {

      // Sample from movie hyperparams
      tie(mu_m, Lambda_m) = CondNormalWishart(sample_m.transpose(), mu0_m, b0_m, WI_m, df_m);

      // Sample from user hyperparams
      tie(mu_u, Lambda_u) = CondNormalWishart(sample_u.transpose(), mu0_u, b0_u, WI_u, df_u);

#pragma omp parallel for
      for(int mm = 0; mm < num_m; ++mm) {
        sample_m.col(mm) = sample_movie(mm, M, mean_rating, sample_u.transpose(), alpha, mu_m, Lambda_m).transpose();
      }

#pragma omp parallel for
      for(int uu = 0; uu < num_p; ++uu) {
        sample_u.col(uu) = sample_movie(uu, Mt, mean_rating, sample_m.transpose(), alpha, mu_u, Lambda_u).transpose();
      }

      auto eval = eval_probe_vec(probe_vec, sample_m, sample_u, mean_rating);
      double norm_u = sample_u.norm();
      double norm_m = sample_m.norm();
      auto end = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration<double>(end - start);
      double samples_per_sec = (i + 1) * (num_p + num_m) / elapsed.count();

      printf("Iteration %d:\t num_correct: %3.2f%%\tavg_diff: %3.2f\tFU(%6.2f)\tFM(%6.2f)\tSamples/sec: %6.2f\n",
              i, 100*eval.first, eval.second, norm_u, norm_m, samples_per_sec);
    }
}

#endif

int main(int argc, char *argv[])
{
    const char *fname = argv[1];
    assert(fname && "filename missing");
    Eigen::initParallel();
    Eigen::setNbThreads(1);

    loadChemo(fname);
    init();
#ifdef TEST_SAMPLE
    test();
#else
    run();
#endif

    return 0;
}
