/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#ifndef BPMF_H
#define BPMF_H

#include <bitset>
#include <functional>
#include <random>

#define EIGEN_RUNTIME_NO_MALLOC 1
#define EIGEN_DONT_PARALLELIZE 1

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "counters.h"
#include "thread_vector.h"

#ifndef BPMF_NUMLATENT
#error Define BPMF_NUMLATENT
#endif

#if defined(BPMF_GPI_COMM) or defined(BPMF_GPICXX_COMM)
#define BPMF_MPI_COMM
#define BPMF_HYBRID_COMM
#elif defined(BPMF_MPI_PUT_COMM)
#define BPMF_MPI_COMM
#elif defined(BPMF_MPI_BCAST_COMM)
#define BPMF_MPI_COMM
#elif defined(BPMF_MPI_REDUCE_COMM)
#ifndef BPMF_REDUCE
#define BPMF_REDUCE
#endif
#define BPMF_MPI_COMM
#elif defined(BPMF_MPI_ALLREDUCE_COMM)
#ifndef BPMF_REDUCE
#define BPMF_REDUCE
#endif
#define BPMF_MPI_COMM
#elif defined(BPMF_MPI_ISEND_COMM)
#define BPMF_MPI_COMM
#elif defined(BPMF_ARGO_COMM)
#define BPMF_MPI_COMM
#elif defined(BPMF_NO_COMM)
#else
#error no comm include
#endif


const int num_latent = BPMF_NUMLATENT;

typedef Eigen::SparseMatrix<double> SparseMatrixD;
typedef Eigen::Matrix<double, num_latent, num_latent> MatrixNNd;
typedef Eigen::Matrix<double, num_latent, Eigen::Dynamic> MatrixNXd;
typedef Eigen::Matrix<double, num_latent, 1> VectorNd;
typedef Eigen::Map<MatrixNXd, Eigen::Aligned> MapNXd;
typedef Eigen::Map<Eigen::VectorXd, Eigen::Aligned> MapXd;

void assert_same_struct(SparseMatrixD &A, SparseMatrixD &B);

std::pair< VectorNd, MatrixNNd>
CondNormalWishart(const int N, const MatrixNNd &C, const VectorNd &Um, const VectorNd &mu, const double kappa, const MatrixNNd &T, const int nu);

void rng_set_pos(uint32_t p);
double randn();
double randu();
 
#define nrandn(n) (Eigen::VectorXd::NullaryExpr((n), [](double) { return randn(); }))

inline double sqr(double x) { return x*x; }

//
// sampled hyper parameters for priors
//
struct HyperParams {
    // fixed params
    const int b0 = 2;
    const int df = num_latent;
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

    void sample(const int N, const VectorNd &sum, const MatrixNNd &cov)
    {
        std::tie(mu, LambdaU) = CondNormalWishart(N, cov, sum / N, mu0, b0, WI, df);
        LambdaF = LambdaU.triangularView<Eigen::Upper>().transpose() * LambdaU;
        LambdaL = LambdaU.transpose();
    }
};

struct Sys;

// 
// System represent all things related to the movies OR users
// Hence a classic matrix factorization always has TWO Sys objects
// for the two factors
struct Sys {
    //-- static info
    static bool permute;
    static bool verbose;
    static int nprocs, procid;
    static int burnin, nsims, update_freq;
    static double alpha;
    static std::string odirname;

    static void Init();
    static void Finalize();
    static void Abort(int);
    static void sync();

    static std::ostream *os, *dbgs;
    static std::ostream &cout() { 
      if (!os) return std::cout;
      os->flush(); return *os;
    }

    static std::ostream &dbg() {
      if (!dbgs) return std::cerr;
      dbgs->flush(); return *dbgs;
    }
    
    //-- c'tor
    std::string name;
    int iter;
    Sys(std::string name, std::string fname, std::string pname);
    Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &Pavg);
    virtual ~Sys();
    void init();
    virtual void alloc_and_init() = 0;

    //-- sparse matrix
    SparseMatrixD M;        // known ratings
    double mean_rating;
    int num() const { return M.cols(); }
    int nnz() const { return M.nonZeros(); }
    int nnz(int i) const { return M.col(i).nonZeros(); }

    // assignment and connectivity
    typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> PermMatrix;
    void permuteCols(const PermMatrix &, Sys &other); 
    void unpermuteCols(Sys &other); 
    PermMatrix col_permutation;
    void assign(Sys &);
    bool assigned;

    std::vector<int> dom;
    int proc(int pos) const {
        int proc = 0;
        while (dom[proc+1] <= pos) proc++;
        return proc;
    }

    // assignment domain of users/movies to nodes
    // assignment is continues: node i is assigned items from(i) until to(i)
    int num(int i) const { return to(i) - from(i); } // number of items on node i
    int from(int i = procid) const { return dom.at(i); } 
    int to(int i = procid) const { return dom.at(i+1); }
    void print_dom(std::ostream &os) {
        for (int i = 0; i < nprocs; ++i)
            os << i << ": [" << from(i) << ":" << to(i) << "[" << std::endl;
    }

    // connectivity matrix tells what what items need to be sent to what nodes
    void opt_conn(Sys& to);
    void update_conn(Sys& to);
    void build_conn(Sys& to);
    static const unsigned max_procs = 1024;
    typedef std::bitset<max_procs> bm;
    const bm &conn(unsigned idx) const { assert(nprocs>1); return conn_map.at(idx); }
    bool conn(unsigned from, int to) { return (nprocs>1) && conn_map.at(from).test(to); }
    std::vector<bm> conn_map;
    std::map<std::pair<unsigned, unsigned>, unsigned> conn_count_map;
    unsigned conn_count(int from, int to) { assert(nprocs>1);  return conn_count_map[std::make_pair(from, to)]; }
    unsigned send_count(int to) { return conn_count(Sys::procid, to); }
    unsigned recv_count(int from) { return conn_count(from, Sys::procid); }

    //-- factors of the MF
    double* items_ptr;
    MapNXd items() const { return MapNXd(items_ptr, num_latent, num()); }
    VectorNd sample(long idx, Sys &in);
    void preComputeMuLambda(const Sys &other);
    void computeMuLambda(long idx, const Sys &other, VectorNd &rr, MatrixNNd &MM, bool local_only) const;

    //-- to pre-compute Lambda/Mu from other side
    Eigen::MatrixXd precMu, precLambda;
    Eigen::Map<MatrixNNd> precLambdaMatrix(int idx) 
    {
        return Eigen::Map<MatrixNNd>(precLambda.col(idx).data());
    }

    //-- for propagated posterior
    Eigen::MatrixXd propMu, propLambda;
    void add_prop_posterior(std::string);
    bool has_prop_posterior() const;

    //-- for aggregated posterior
    Eigen::MatrixXd aggrMu, aggrLambda;
    void finalize_mu_lambda();
    
    // virtual functions will be overriden based on COMM: NO_COMM, MPI, or GASPI
    virtual void send_item(int i) = 0;
    void bcast();
    void reduce_sum_cov_norm();
    virtual void sample(Sys &in);

    VectorNd sum;  //-- sum of all U-vectors
    MatrixNNd cov; //-- covariance
    double norm; 

    //-- hyper params
    HyperParams hp;

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

inline void argo_data_init(Sys& movies, Sys& users)
{
#ifdef BPMF_ARGO_COMM
    VectorNd zz = VectorNd::Zero();

    int i;
    for (i = movies.from(); i < movies.to(); ++i)
        movies.items().col(i) = zz;
    for (i = users.from();  i < users.to();  ++i)
        users.items().col(i)  = zz;
#endif
}


const int breakpoint1 = 24; 
const int breakpoint2 = 10500;

#endif
