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
#include "thread_vector.h"

#ifndef BPMF_NUMLATENT
#error Define BPMF_NUMLATENT
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

double randn(double);
auto nrandn(int n) -> decltype( Eigen::VectorXd::NullaryExpr(n, std::ptr_fun(randn)) ); 

inline auto nrandn() -> decltype( VectorNd::NullaryExpr(std::ptr_fun(randn)) ) { 
    return VectorNd::NullaryExpr(std::ptr_fun(randn)); 
}

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

    void sample(const int N, const  VectorNd &sum, const  MatrixNNd &cov) {
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

    static std::ostream *os;
    static std::ostream &cout() { os->flush(); return *os; }
    
    //-- c'tor
    std::string name;
    int iter;
    Sys(std::string name, std::string fname, std::string pname);
    Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &Pavg);
    virtual ~Sys();
    void init();
    virtual void alloc_and_init() = 0;

    //-- sparse matrix
    SparseMatrixD M; // known ratings
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
    VectorNd sample(long idx, const MapNXd in);

    //-- for propagated posterior
    Eigen::MatrixXd propMu, propLambda;
    void add_prop_posterior(std::string);
    bool has_prop_posterior() const;

    //-- for aggregated posterior
    Eigen::MatrixXd aggrMu, aggrLambda;
    void finalize_mu_lambda();
    
    // virtual functions will be overriden based on COMM: NO_COMM, MPI, or GASPI
    virtual void send_items(int, int) = 0;
    void bcast();
    virtual void sample(Sys &in);
    static unsigned grain_size;

    //-- covariance
    double *sum_ptr;
    MapNXd sum_map() const { return MapNXd(sum_ptr, num_latent, Sys::nprocs); }
    MapNXd::ColXpr sum(int i)  const { return sum_map().col(i); }
    MapNXd::ColXpr local_sum() const { return sum(Sys::procid); }
    VectorNd aggr_sum() const { return sum_map().rowwise().sum(); }

    double *cov_ptr;
    MapNXd cov_map() const { return MapNXd(cov_ptr, num_latent, Sys::nprocs * num_latent); }
    MapNXd cov(int i) const { return MapNXd(cov_ptr + i*num_latent*num_latent, num_latent, num_latent); }
    MapNXd local_cov() { return cov(Sys::procid); }
    MatrixNNd aggr_cov() const { 
        MatrixNNd ret(MatrixNNd::Zero());
        for(int i=0; i<Sys::nprocs; ++i) ret += cov(i);
        return ret;
    }

    // norm
    double *norm_ptr;
    MapXd norm_map() const { return MapXd(norm_ptr, Sys::nprocs); }
    double &norm(int i) const { return norm_ptr[i]; }
    double &local_norm() const { return norm(Sys::procid); }
    double aggr_norm() const { return norm_map().sum(); }

    //-- hyper params
    HyperParams hp;
    virtual void sample_hp() { hp.sample(num(), aggr_sum(), aggr_cov()); }

    // output predictions
    SparseMatrixD T, Torig; // test matrix (input)
    SparseMatrixD Pavg, Pm2; // predictions for items in T (output)`
    double rmse, rmse_avg;
    void predict(Sys& other, bool all = false);
    void print(double, double, double, double); 

    // performance counting
    std::vector<double> sample_time;
    void register_time(int i, double t);
};

#endif
