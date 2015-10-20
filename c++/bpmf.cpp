
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>

#include <unsupported/Eigen/SparseExtra>

#ifdef _OPENMP
#include <omp.h>
#else
#include <tbb/tbb.h>
#endif

#include "bpmf.h"

using namespace std;
using namespace Eigen;

const int num_feat = 64;

const double alpha = 2;
const int nsims = 200;
const int burnin = 20;

double mean_rating = .0;

typedef SparseMatrix<double> SparseMatrixD;
SparseMatrixD M, Mt, P;

typedef Matrix<double, num_feat, 1> VectorNd;
typedef Matrix<double, num_feat, num_feat> MatrixNNd;
typedef Matrix<double, num_feat, Dynamic> MatrixNXd;

MatrixNXd sample_u;
MatrixNXd sample_m;

struct HyperParams 
{
    MatrixNNd Lambda, Lambda_f;
    VectorNd mu;

    MatrixNNd WI;
    const int b0 = 2;
    const int df = num_feat;
    VectorNd mu0;


    HyperParams() {
        WI.setIdentity();
        mu0.setZero();
    }

    void sample(const MatrixNXd &s) {
      tie(mu, Lambda) = CondNormalWishart(s, mu0, b0, WI, df);
      Lambda_f = Lambda.triangularView<Upper>().transpose() * Lambda;
    }
};

void init() {
    mean_rating = M.sum() / M.nonZeros();
    sample_u = MatrixNXd(num_feat,M.rows());
    sample_m = MatrixNXd(num_feat,M.cols());
    sample_u.setZero();
    sample_m.setZero();
}

inline double sqr(double x) { return x*x; }

std::pair<double,double> eval_probe_vec(int n, VectorXd & predictions, const MatrixNXd &sample_m, const MatrixNXd &sample_u, double mean_rating)
{
    double se = 0.0, se_avg = 0.0;
    unsigned idx = 0;
    for (int k=0; k<P.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(P,k); it; ++it) {
            const double pred = sample_m.col(it.col()).dot(sample_u.col(it.row())) + mean_rating;
            //se += (it.value() < log10(200)) == (pred < log10(200));
            se += sqr(it.value() - pred);

            const double pred_avg = (n == 0) ? pred : (predictions[idx] + (pred - predictions[idx]) / n);
            //se_avg += (it.value() < log10(200)) == (pred_avg < log10(200));
            se_avg += sqr(it.value() - pred_avg);
            predictions[idx++] = pred_avg;
        }
    }

    const unsigned N = P.nonZeros();
    const double rmse = sqrt( se / N );
    const double rmse_avg = sqrt( se_avg / N );
    return std::make_pair(rmse, rmse_avg);
}

void sample(MatrixNXd &s, int mm, const SparseMatrixD &mat, double mean_rating,
    const MatrixNXd &samples, double alpha, const HyperParams &hp)
{

		int count = 0;
		for (SparseMatrixD::InnerIterator it(mat,mm); it; ++it, ++count)
			if( count > BREAKPOINT )
				break;

		VectorNd rr; rr.setZero();
		Eigen::LLT<MatrixNNd> chol;

		if( count < BREAKPOINT ) {
			const_cast<MatrixNNd&>( chol.matrixLLT() ) = hp.Lambda.transpose();
			for (SparseMatrixD::InnerIterator it(mat,mm); it; ++it) {
					auto col = samples.col(it.row());
					chol.rankUpdate(col, alpha);
					rr.noalias() += col * ((it.value() - mean_rating) * alpha);
			}
		} else {
			MatrixNNd MM; MM.setZero();
			for (SparseMatrixD::InnerIterator it(mat,mm); it; ++it) {
					auto col = samples.col(it.row());
					MM.noalias() += col * col.transpose();
					rr.noalias() += col * ((it.value() - mean_rating) * alpha);
			}

			chol = (hp.Lambda_f + alpha * MM).llt();
		}

		if(chol.info() != Eigen::Success)
			throw std::runtime_error("Cholesky Decomposition failed!");

    VectorNd tmp = rr + hp.Lambda_f * hp.mu;
    chol.matrixL().solveInPlace(tmp);
    tmp += nrandn(num_feat);
    chol.matrixU().solveInPlace(tmp);
    s.col(mm) = tmp;

#ifdef TEST_SAMPLE
      cout << "movie " << mm << ":" << result.cols() << " x" << result.rows() << endl;
      cout << "mean rating " << mean_rating << endl;
      cout << "E = [" << E << "]" << endl;
      cout << "rr = [" << rr << "]" << endl;
      cout << "MM = [" << MM << "]" << endl;
      cout << "covar = [" << covar << "]" << endl;
      cout << "mu = [" << mu << "]" << endl;
      cout << "chol = [" << chol << "]" << endl;
      cout << "rand = [" << r <<"]" <<  endl;
      cout << "result = [" << result << "]" << endl;
#endif

}

#ifdef TEST_SAMPLE
void test() {
    MatrixNXd sample_u(M.rows());
    MatrixNXd sample_m(M.cols());

    mu_m.setZero();
    Lambda_m.setIdentity();
    sample_u.setConstant(2.0);
    Lambda_m *= 0.5;
    sample_m.col(0) = sample(0, M, mean_rating, sample_u, alpha, mu_m, Lambda_m);
}

#else

void run() {
		VectorXd predictions;
		predictions = VectorXd::Zero( P.nonZeros() );

    HyperParams hp_u, hp_m;

    auto start = tick();
    std::cout << "Sampling" << endl;
    for(int i=0; i<nsims; ++i) {

      // Sample from user and movie hyperparams
      hp_u.sample(sample_u);
      hp_m.sample(sample_m);

      const int num_m = M.cols();
      const int num_u = M.rows();
#ifdef _OPENMP
#pragma omp parallel for
      for(int mm=0; mm<num_m; ++mm) {
        sample(sample_m, mm, M, mean_rating, sample_u, alpha, hp_m);
      }
#pragma omp parallel for
      for(int uu=0; uu<num_u; ++uu) {
        sample(sample_u, uu, Mt, mean_rating, sample_m, alpha, hp_u);
      }
#else
      tbb::parallel_for(0, num_m, [](int mm) {
        sample(sample_m, mm, M, mean_rating, sample_u, alpha, hp_m);
      });

      tbb::parallel_for(0, num_u, [](int uu) {
         sample(sample_u, uu, Mt, mean_rating, sample_m, alpha, hp_u);
       });
#endif

      auto eval = eval_probe_vec( (i < burnin) ? 0 : (i - burnin), predictions, sample_m, sample_u, mean_rating);
//      auto eval = std::make_pair(0.0, 0.0);
      double norm_u = sample_u.norm();
      double norm_m = sample_m.norm();
      auto end = tick();
      auto elapsed = end - start;
      double samples_per_sec = (i + 1) * (M.rows() + M.cols()) / elapsed;

      printf("Iteration %d:\t RMSE: %3.2f\tavg RMSE: %3.2f\tFU(%6.2f)\tFM(%6.2f)\tSamples/sec: %6.2f\n",
              i, eval.first, eval.second, norm_u, norm_m, samples_per_sec);
    }

  auto end = tick();
  auto elapsed = end - start;
  printf("Total time: %6.2f\n", elapsed);
}

#endif

int main(int argc, char *argv[])
{
    if(argc < 3) {
       cerr << "Usage: " << argv[0] << " <train_matrix.mtx> <test_matrix.mtx>" << endl;
       abort();
    }

    cerr << "num_feat: " << num_feat << endl;
    cerr << "nsims: " << nsims << endl;
    cerr << "burnin: " << burnin << endl;

    Eigen::initParallel();

    cerr << "Loading training matrix (" << argv[1] << ")" << endl;
    loadMarket(M, argv[1]);
    Mt = M.transpose();

    cerr << "Loading test matrix (" << argv[2] << ")" << endl;
    loadMarket(P, argv[2]);

    assert(M.nonZeros() > P.nonZeros());

    init();
#ifdef TEST_SAMPLE
    test();
#else
    run();
#endif

    return 0;
}
