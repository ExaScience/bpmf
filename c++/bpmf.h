#ifndef BPMF_H
#define BPMF_H

#include <bitset>

#define EIGEN_RUNTIME_NO_MALLOC 1
#define EIGEN_DONT_PARALLELIZE 1

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "counters.h"

const int num_feat = 100;

typedef Eigen::SparseMatrix<double> SparseMatrixD;
typedef Eigen::Matrix<double, num_feat, num_feat> MatrixNNd;
typedef Eigen::Matrix<double, num_feat, Eigen::Dynamic> MatrixNXd;
typedef Eigen::Matrix<double, num_feat, 1> VectorNd;
typedef Eigen::Map<MatrixNXd> MapNXd;
typedef Eigen::Map<Eigen::VectorXd> MapXd;

std::pair< VectorNd, MatrixNNd>
CondNormalWishart(const int N, const MatrixNNd &C, const VectorNd &Um, const VectorNd &mu, const double kappa, const MatrixNNd &T, const int nu);

double randn(double);
auto nrandn(int n) -> decltype( Eigen::VectorXd::NullaryExpr(n, std::ptr_fun(randn)) ); 

inline auto nrandn() -> decltype( VectorNd::NullaryExpr(std::ptr_fun(randn)) ) { 
    return VectorNd::NullaryExpr(std::ptr_fun(randn)); 
}

double tick();
inline double sqr(double x) { return x*x; }

struct HyperParams {
    // fixed params
    const int b0 = 2;
    const int df = num_feat;
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

struct Eval {
    Eval(std::string probename, double mean_rating, int burnin);

    const double mean_rating;
    const int burnin;
    SparseMatrixD T, P;
   
    // state 
    Eigen::VectorXd predictions;
    double rmse, rmse_avg;
    int iter;

    // funcs
    void predict(int n, Sys& movies, Sys& users);
    void print(double, double, double); 
};

struct Sys {
    //-- static info
    static int nprocs, procid, nthrds;
    static void Init();
    static void Finalize();
    static void Abort(int);
    static void SetupThreads(int);
    static void sync();
    

    //-- c'tor
    std::string name;
    Sys(std::string name, std::string fname);
    Sys(std::string name, const SparseMatrixD &M);
    virtual ~Sys() {}
    void init(const Sys &);
    virtual void alloc_and_init(const Sys &);

    //-- sparse matrix
    SparseMatrixD M;
    double mean_rating;
    int num() const { return M.cols(); }

    // assignment and connectivity
    std::vector<std::vector<unsigned>> proc_to_item;
    std::vector<unsigned> item_to_proc;
    int proc(unsigned pos) const { return item_to_proc.at(pos); }
    const std::vector<unsigned> &items_at(unsigned proc) const { return proc_to_item.at(proc); }
    const std::vector<unsigned> &my_items() const { return proc_to_item.at(procid); }
    bool assigned() const { return !item_to_proc.empty(); }

    // connectivity
    void build_conn(const Sys& to);
    static const unsigned max_procs = 64;
    typedef std::bitset<max_procs> bm;
    const bm &conn(unsigned idx) const { return conn_map.at(idx); }
    std::vector<bm> conn_map;

    //-- factors
    double* items_ptr;
    MapNXd items() const { return MapNXd(items_ptr, num_feat, num()); }
    VectorNd sample(long idx, const MapNXd in);
    virtual void send_items(int, int) {};
    virtual void send_item(int) {};
    virtual void sample(Sys &in);

    //-- covariance
    double *sum_ptr;
    MapNXd sum_map() const { return MapNXd(sum_ptr, num_feat, Sys::nprocs); }
    MapNXd::ColXpr sum(int i)  const { return sum_map().col(i); }
    MapNXd::ColXpr local_sum() const { return sum(Sys::procid); }
    VectorNd aggr_sum() const { return sum_map().rowwise().sum(); }

    double *cov_ptr;
    MapNXd cov_map() const { return MapNXd(cov_ptr, num_feat, Sys::nprocs * num_feat); }
    MapNXd cov(int i) const { return MapNXd(cov_ptr + i*num_feat*num_feat, num_feat, num_feat); }
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
};

#endif
