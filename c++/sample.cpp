/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#include <memory>
#include <cstdio>
#include <iostream>
#include <climits>
#include <stdexcept>
#include <cmath>

#include "error.h"
#include "bpmf.h"
#include "io.h"

static const bool measure_perf = false;

std::ostream *Sys::os = 0;
std::ostream *Sys::dbgs = 0;

int Sys::procid = -1;
int Sys::nprocs = -1;

int Sys::nsims;
int Sys::burnin;
int Sys::update_freq;
double Sys::alpha = 2.0;

std::string Sys::odirname = "";

bool Sys::permute = true;
bool Sys::verbose = false;

// verifies that A has the same non-zero structure as B
void assert_same_struct(SparseMatrixD &A, SparseMatrixD &B)
{
    assert(A.cols() == B.cols());
    assert(A.rows() == B.rows());
    for(int i=0; i<A.cols(); ++i) assert(A.col(i).nonZeros() == B.col(i).nonZeros());
}

//
// Does predictions for prediction matrix T
// Computes RMSE (Root Means Square Error)
//
void Sys::predict(Sys& other, bool all)
{
    int n = (iter < burnin) ? 0 : (iter - burnin);
   
    double se(0.0); // squared err
    double se_avg(0.0); // squared avg err
    int nump = 0; // number of predictions

    int lo = from();
    int hi = to();
    if (all) {
#ifdef BPMF_REDUCE
        Sys::cout() << "WARNING: predict all items in test set not available in BPMF_REDUCE mode" << std::endl;
#else
        lo = 0;
        hi = num();
#endif
    }

    #pragma omp parallel for reduction(+:se,se_avg,nump)
    for(int k = lo; k<hi; k++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(T,k); it; ++it)
        {
#ifdef BPMF_REDUCE
            if (it.row() >= other.to() || it.row() < other.from())
                continue;
#endif
            auto m = items().col(it.col());
            auto u = other.items().col(it.row());

            const double pred = m.dot(u) + mean_rating;
            se += sqr(it.value() - pred);

            // update average prediction
            double &avg = Pavg.coeffRef(it.row(), it.col());
            double delta = pred - avg;
            avg = (n == 0) ? pred : (avg + delta/n);
            double &m2 = Pm2.coeffRef(it.row(), it.col());
            m2 = (n == 0) ? 0 : m2 + delta * (pred - avg);
            se_avg += sqr(it.value() - avg);

            nump++;
        }
    }

    num_predict = nump;
    rmse = sqrt( se / num_predict );
    rmse_avg = sqrt( se_avg / num_predict );
}

//
// Prints sampling progress
//
void Sys::print(double items_per_sec, double ratings_per_sec, double norm_u, double norm_m) {
  char buf[1024];
  std::string phase = (iter < Sys::burnin) ? "Burnin" : "Sampling";
  sprintf(buf, "%d: %s iteration %d:\t RMSE: %3.4f\tavg RMSE: %3.4f\tFU(%6.2f)\tFM(%6.2f)\titems/sec: %6.2f\tratings/sec: %6.2fM\n",
                    Sys::procid, phase.c_str(), iter, rmse, rmse_avg, norm_u, norm_m, items_per_sec, ratings_per_sec / 1e6);
  Sys::cout() << buf;
}

//
// Constructor with that reads MTX files
// 
Sys::Sys(std::string name, std::string fname, std::string probename)
    : name(name), iter(-1), assigned(false), dom(nprocs+1)
{

    read_matrix(fname, M);
    read_matrix(probename, T);

    auto rows = std::max(M.rows(), T.rows());
    auto cols = std::max(M.cols(), T.cols());
    M.conservativeResize(rows,cols);
    T.conservativeResize(rows,cols);
    Pm2 = Pavg = Torig = T; // reference ratings and predicted ratings
    assert(M.rows() == Pavg.rows());
    assert(M.cols() == Pavg.cols());
    assert(Sys::nprocs <= (int)Sys::max_procs);
}

//
// Constructs Sys as transpose of existing Sys
//
Sys::Sys(std::string name, const SparseMatrixD &Mt, const SparseMatrixD &Pt) : name(name), iter(-1), assigned(false), dom(nprocs+1) {
    M = Mt.transpose();
    Pm2 = Pavg = T = Torig = Pt.transpose(); // reference ratings and predicted ratings
    assert(M.rows() == Pavg.rows());
    assert(M.cols() == Pavg.cols());
}

Sys::~Sys() 
{
    if (measure_perf) {
        Sys::cout() << " --------------------\n";
        Sys::cout() << name << ": sampling times on " << procid << "\n";
        for(int i = from(); i<to(); ++i) 
        {
            Sys::cout() << "\t" << nnz(i) << "\t" << sample_time.at(i) / nsims  << "\n";
        }
        Sys::cout() << " --------------------\n\n";
    }
}

bool Sys::has_prop_posterior() const
{
    return propMu.nonZeros() > 0;
}

void Sys::add_prop_posterior(std::string fnames)
{
    if (fnames.empty()) return;

    std::size_t pos = fnames.find_first_of(",");
    std::string mu_name = fnames.substr(0, pos);
    std::string lambda_name = fnames.substr(pos+1);

    read_matrix(mu_name, propMu);
    read_matrix(lambda_name, propLambda);

    assert(propMu.cols() == num());
    assert(propLambda.cols() == num());

    assert(propMu.rows() == num_latent);
    assert(propLambda.rows() == num_latent * num_latent);

}

//
// Intializes internal Matrices and Vectors
//
void Sys::init()
{
    //-- M
    assert(M.rows() > 0 && M.cols() > 0);
    mean_rating = M.sum() / M.nonZeros();
#ifndef BPMF_ARGO_COMM
    items().setZero();
#endif
    sum.setZero();
    cov.setZero();
    norm = 0.;
    col_permutation.setIdentity(num());

#ifdef BPMF_REDUCE
    precMu = MatrixNXd::Zero(num_latent, num());
    precLambda = Eigen::MatrixXd::Zero(num_latent * num_latent, num());
#endif

    if (Sys::odirname.size())
    {
        aggrMu = Eigen::MatrixXd::Zero(num_latent, num());
        aggrLambda = Eigen::MatrixXd::Zero(num_latent * num_latent, num());
    }

    int count_larger_bp1 = 0;
    int count_larger_bp2 = 0;
    int count_sum = 0;
    for(int k = 0; k<M.cols(); k++) {
        int count = M.col(k).nonZeros();
        count_sum += count;
        if (count > breakpoint1) count_larger_bp1++;
        if (count > breakpoint2) count_larger_bp2++;
    }

    Sys::cout() << "mean rating: " << mean_rating << std::endl;
    Sys::cout() << "total number of ratings in train: " << M.nonZeros() << std::endl;
    Sys::cout() << "total number of ratings in test: " << T.nonZeros() << std::endl;
    Sys::cout() << "average ratings per row: " << (double)count_sum / (double)M.cols() << std::endl;
    Sys::cout() << "rows > break_point1: " << 100. * (double)count_larger_bp1 / (double)M.cols() << std::endl;
    Sys::cout() << "rows > break_point2: " << 100. * (double)count_larger_bp2 / (double)M.cols() << std::endl;
    Sys::cout() << "num " << name << ": " << num() << std::endl;
    if (has_prop_posterior())
    {
        Sys::cout() << "with propagated posterior" << std::endl;
    }

    if (measure_perf) sample_time.resize(num(), .0);
}

class PrecomputedLLT : public Eigen::LLT<MatrixNNd>
{
  public:
    void operator=(const MatrixNNd &m) { m_matrix = m; m_isInitialized = true; m_info = Eigen::Success; }
};

void Sys::preComputeMuLambda(const Sys &other)
{
    BPMF_COUNTER("preComputeMuLambda");
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < num(); ++i)
    {
        VectorNd mu = VectorNd::Zero();
        MatrixNNd Lambda = MatrixNNd::Zero();
        computeMuLambda(i, other, mu, Lambda, true);
        precLambdaMatrix(i) = Lambda;
        precMu.col(i) = mu;
    }
}

void Sys::computeMuLambda(long idx, const Sys &other, VectorNd &rr, MatrixNNd &MM, bool local_only) const
{
    BPMF_COUNTER("computeMuLambda");
    for (SparseMatrixD::InnerIterator it(M, idx); it; ++it)
    {
        if (local_only && (it.row() < other.from() || it.row() >= other.to())) continue;
        auto col = other.items().col(it.row());
        MM.triangularView<Eigen::Upper>() += col * col.transpose();
        rr.noalias() += col * ((it.value() - mean_rating) * alpha);
    }
}

//
// Update ONE movie or one user
//
VectorNd Sys::sample(long idx, Sys &other)
{
    auto start = tick();
    rng_set_pos((idx+1) * num_latent * (iter+1));
    //Sys::dbg() << "-- original start name: " << name << " iter: " << iter << " idx: " << idx << ": " << rng.counter << std::endl;

    VectorNd hp_mu;
    MatrixNNd hp_LambdaF; 
    MatrixNNd hp_LambdaL; 
    if (has_prop_posterior())
    {
        hp_mu = propMu.col(idx);
        hp_LambdaF = Eigen::Map<MatrixNNd>(propLambda.col(idx).data()); 
        hp_LambdaL =  hp_LambdaF.llt().matrixL();
    }
    else
    {
        hp_mu = hp.mu;
        hp_LambdaF = hp.LambdaF; 
        hp_LambdaL = hp.LambdaL; 
    }

    VectorNd rr = hp_LambdaF * hp.mu;                // vector num_latent x 1, we will use it in formula (14) from the paper
    MatrixNNd MM(MatrixNNd::Zero());
    PrecomputedLLT chol;                             // matrix num_latent x num_latent, chol="lambda_i with *" from formula (14)

#ifdef BPMF_REDUCE
    rr += precMu.col(idx);
    MM += precLambdaMatrix(idx);
#else
    computeMuLambda(idx, other, rr, MM, false);
#endif

    // copy upper -> lower part, matrix is symmetric.
    MM.triangularView<Eigen::Lower>() = MM.transpose();
    MM = hp_LambdaF + alpha * MM;

#ifdef BPMF_NO_COVARIANCE
    // only keep diagonal -- 
    MatrixNNd MM1 = MM.diagonal().asDiagonal();
    MM = MM1;
#endif

    chol.compute(MM);

    if(chol.info() != Eigen::Success) THROWERROR("Cholesky failed");

    // now we should calculate formula (14) from the paper
    // u_i for k-th iteration = Gaussian distribution N(u_i | mu_i with *, [lambda_i with *]^-1) =
    //                        = mu_i with * + s * [U]^-1, 
    //                        where 
    //                              s is a random vector with N(0, I),
    //                              mu_i with * is a vector num_latent x 1, 
    //                              mu_i with * = [lambda_i with *]^-1 * rr,
    //                              lambda_i with * = L * U       

    // Expression u_i = U \ (s + (L \ rr)) in Matlab looks for Eigen library like: 

    chol.matrixL().solveInPlace(rr);                    // L*Y=rr => Y=L\rr, we store Y result again in rr vector  
    rr += nrandn(num_latent);                           // rr=s+(L\rr), we store result again in rr vector
    chol.matrixU().solveInPlace(rr);                    // u_i=U\rr 
    items().col(idx) = rr;                              // we save rr vector in items matrix (it is user features matrix)

    auto stop = tick();
    register_time(idx, 1e6 * (stop - start));
    //Sys::cout() << "  " << count << ": " << 1e6*(stop - start) << std::endl;

    assert(rr.norm() > .0);

    //SHOW(rr.transpose());
    //Sys::dbg() << "-- original done name: " << name << " iter: " << iter << " idx: " << idx << ": " << rng.counter << std::endl;

    return rr;
}

// 
// update ALL movies / users in parallel
//
void Sys::sample(Sys &other) 
{
    BPMF_COUNTER("compute");
    iter++;
    thread_vector<VectorNd>  sums(VectorNd::Zero()); // sum
    thread_vector<double>    norms(0.0); // squared norm
    thread_vector<MatrixNNd> prods(MatrixNNd::Zero()); // outer prod

    rng_set_pos(iter); // make this consistent
    hp.sample(num(), sum, cov);

#pragma omp parallel for schedule(guided)
    for (int i = from(); i < to(); ++i)
    {
#pragma omp task
        {
            auto r = sample(i, other);

            MatrixNNd cov = (r * r.transpose());
            prods.local() += cov;
            sums.local() += r;
            norms.local() += r.squaredNorm();

            if (iter >= burnin && Sys::odirname.size())
            {
                aggrMu.col(i) += r;
                aggrLambda.col(i) += Eigen::Map<Eigen::VectorXd>(cov.data(), num_latent * num_latent);
            }

            send_item(i);
        }
    }
#pragma omp taskwait

#ifdef BPMF_REDUCE
    other.preComputeMuLambda(*this);
#endif

    VectorNd sum = sums.combine();
    MatrixNNd prod = prods.combine();   
    norm = norms.combine();

    const int N = num();
    cov = (prod - (sum * sum.transpose() / N)) / (N-1);
}

void Sys::register_time(int i, double t)
{
    if (measure_perf) sample_time.at(i) += t;
}
