/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#ifndef BPMF_H
#define BPMF_H

#include <bitset>
#include <functional>

#define EIGEN_RUNTIME_NO_MALLOC 1
#define EIGEN_DONT_PARALLELIZE 1

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "counters.h"

#ifndef BPMF_NUMLATENT
#error Define BPMF_NUMLATENT
#endif

const int num_latent = BPMF_NUMLATENT;

typedef Eigen::SparseMatrix<double> SparseMatrixD;
typedef Eigen::Map<Eigen::SparseMatrix<double>> SparseMapD;
typedef Eigen::Matrix<double, num_latent, num_latent> MatrixNNd;
typedef Eigen::Matrix<double, num_latent, Eigen::Dynamic> MatrixNXd;
typedef Eigen::Matrix<double, num_latent, 1> VectorNd;
typedef Eigen::Map<MatrixNXd, Eigen::Aligned> MapNXd;
typedef Eigen::Map<Eigen::VectorXd, Eigen::Aligned> MapXd;

void assert_same_struct(SparseMatrixD &A, SparseMatrixD &B);

std::pair< VectorNd, MatrixNNd>
CondNormalWishart(const int N, const MatrixNNd &C, const VectorNd &Um, const VectorNd &mu, const double kappa, const MatrixNNd &T, const int nu);

struct RNG
{
  std::normal_distribution<> normal_d;
  std::mt19937 generator;

  unsigned long long capacity;
  unsigned long long counter;

  std::vector<double> stash;

  RNG(unsigned long long c = 100000);
  ~RNG();
  double &operator()();
};

extern struct RNG rng;

double randn();
void rng_set_pos(unsigned long long p);
 
#define nrandn(n) (Eigen::VectorXd::NullaryExpr((n), [](double) { return randn(); }))

inline double sqr(double x) { return x*x; }

//
// sampled hyper parameters for priors
//
struct HyperParams {
    // global constants
    double alpha;
    double mean_rating;
    int num, other_num, nnz;

    // fixed params
    static const int b0 = 2;
    static const int df = num_latent;
    VectorNd mu0;
    MatrixNNd WI;

    // sampling output
    VectorNd mu;
    MatrixNNd LambdaF;
    MatrixNNd LambdaU; // triangulated upper part
    MatrixNNd LambdaL; // triangulated lower part
 
    // c'tor
    HyperParams()
    {
        WI.setIdentity();
        mu0.setZero();
    }

    void sample(const int N, const VectorNd &sum, const MatrixNNd &cov);
};

class PrecomputedLLT : public Eigen::LLT<MatrixNNd>
{
  public:
    void operator=(const MatrixNNd &m) { m_matrix = m; m_isInitialized = true; m_info = Eigen::Success; }
};

// 
// System represent all things related to the movies OR users
// Hence a classic matrix factorization always has TWO Sys objects
// for the two factors
struct Sys {
    //-- static info
    static bool verbose;
    static int burnin, nsims;
    static double alpha;
    static std::string odirname;

    static void Init();
    static void Finalize();
    static void Abort(int);
    static void sync();

    static std::ostream *os;
    static std::ostream &cout() { os->flush(); return *os; }
    
    //-- c'tor
    std::string name;
    int iter;
    Sys(std::string name, std::string fname, std::string pname);
    Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &Pavg);
    ~Sys();
    void init();
    void alloc_and_init();

    //-- sparse ratings matrix
    double* ratings_ptr;
    int *outer_ptr, *inner_ptr;
    SparseMatrixD _M;

    SparseMapD M() const { return SparseMapD(other_num(), num(), nnz(), outer_ptr, inner_ptr, ratings_ptr); }

    double mean_rating;
    int other_num() const { return hp().other_num; }
    int num() const { return hp().num; }
    int from() const { return 0; }
    int to() const { return num(); }
    int nnz() const { return hp().nnz; }
    int nnz(int i) const { return M().col(i).nonZeros(); }


    //-- factors of the MF
    double* items_ptr;
    MapNXd items() const { return MapNXd(items_ptr, num_latent, num()); }

    void sample(Sys &in);
    VectorNd sample(long idx, Sys &other);

    //-- colwise sum of U
    VectorNd sum;
    //-- colwise covariance of U
    MatrixNNd cov;
    //-- elementwise norm of U
    double norm;

    //-- hyper params
    HyperParams *hp_ptr;
    const HyperParams &hp() const { return *hp_ptr; };
    HyperParams &hp() { return *hp_ptr; };
    void sample_hp() { hp().sample(num(), sum, cov); }

    // output predictions
    SparseMatrixD T, Torig; // test matrix (input)
    SparseMatrixD Pavg, Pm2; // predictions for items in T (output)`
    double rmse, rmse_avg;
    int num_predict;
    void predict(Sys& other, bool all = false);
    void print(double, double, double, double); 

    // performance counting
    std::vector<double> sample_time;
    void register_time(int i, double t);
};

const int breakpoint1 = 24; 
const int breakpoint2 = 10500;

#endif
