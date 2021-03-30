/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#include <random>
#include <memory>
#include <cstdio>
#include <iostream>
#include <climits>
#include <stdexcept>
#include <cmath>

#include "error.h"
#include "bpmf.h"
#include "io.h"
#include "ompss.h"

static const bool measure_perf = false;

std::ostream *Sys::os = 0;
std::ostream *Sys::db = 0;

int Sys::nsims;
int Sys::burnin;
double Sys::alpha = 2.0;

std::string Sys::odirname = "";

bool Sys::verbose = false;
bool Sys::redirect = false;

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

    for(int k = lo; k<hi; k++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(T,k); it; ++it)
        {
            auto m = items().col(it.col());
            auto u = other.items().col(it.row());

            //SHOW(it.col());
            //SHOW(it.row());
            //SHOW(m.norm());
            //SHOW(u.norm());

            //assert(m.norm() > 0.0);
            //assert(u.norm() > 0.0);

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

    rmse = sqrt( se / num_predict );
    rmse_avg = sqrt( se_avg / num_predict );
    num_predict = nump;
}

//
// Prints sampling progress
//
void Sys::print(double items_per_sec, double ratings_per_sec, double norm_u, double norm_m) {
  char buf[1024];
  std::string phase = (iter < Sys::burnin) ? "Burnin" : "Sampling";
  sprintf(buf, "%s iteration %d:\t RMSE: %3.4f\tavg RMSE: %3.4f\tFU(%6.2f)\tFM(%6.2f)\titems/sec: %6.2f\tratings/sec: %6.2fM\n",
                    phase.c_str(), iter, rmse, rmse_avg, norm_u, norm_m, items_per_sec, ratings_per_sec / 1e6);
  Sys::cout() << buf;
}

//
// Constructor with that reads MTX files
// 
Sys::Sys(std::string name, std::string fname, std::string probename)
    : name(name), iter(-1)
{
    read_matrix(fname, _M);
    read_matrix(probename, T);

    auto rows = std::max(_M.rows(), T.rows());
    auto cols = std::max(_M.cols(), T.cols());
    _M.conservativeResize(rows,cols);
    T.conservativeResize(rows,cols);
    Pm2 = Pavg = Torig = T; // reference ratings and predicted ratings
    assert(_M.rows() == Pavg.rows());
    assert(_M.cols() == Pavg.cols());
    //SHOW(sizeof(*this));
    //SHOW(sizeof(_M));
    //SHOW(&(_M));
}

//
// Constructs Sys as transpose of existing Sys
//
Sys::Sys(std::string name, const SparseMatrixD &Mt, const SparseMatrixD &Pt)
   : name(name), iter(-1)
{
    _M = Mt.transpose();
    Pm2 = Pavg = T = Torig = Pt.transpose(); // reference ratings and predicted ratings
    assert(_M.rows() == Pavg.rows());
    assert(_M.cols() == Pavg.cols());
    SHOW(sizeof(*this));
}

Sys::~Sys() 
{
    if (measure_perf) {
        Sys::cout() << " --------------------\n";
        Sys::cout() << name << ": sampling times\n";
        for(int i = from(); i<to(); ++i) 
        {
            Sys::cout() << "\t" << nnz(i) << "\t" << sample_time.at(i) / nsims  << "\n";
        }
        Sys::cout() << " --------------------\n\n";
    }
}


void Sys::alloc_and_init()
{
    // shouldn't be needed --> on stack
    hp_ptr = (HyperParams *)lmalloc(sizeof(HyperParams));
    hp() = HyperParams();
    hp().alpha = alpha;
    hp().num = _M.cols();
    hp().other_num = _M.rows();
    hp().nnz = _M.nonZeros();


    ratings_ptr = (double *)lmalloc(sizeof(double) * _M.nonZeros());
    inner_ptr   = (int *)lmalloc(sizeof(int) * _M.nonZeros());
    outer_ptr   = (int *)lmalloc(sizeof(int) * ( _M.outerSize() + 1));

    std::memcpy(M().valuePtr(),      _M.valuePtr(),      sizeof(double) * M().nonZeros());
    std::memcpy(M().innerIndexPtr(), _M.innerIndexPtr(), sizeof(int) * M().nonZeros());
    std::memcpy(M().outerIndexPtr(), _M.outerIndexPtr(), sizeof(int) * ( M().outerSize() + 1));

    for(int k = 0; k<M().cols(); k++) 
        assert(M().col(k).nonZeros() == _M.col(k).nonZeros());

    items_ptr = (double *)dmalloc(sizeof(double) * num_latent * num());

    init();

    hp().mean_rating = mean_rating;
}     


void Sys::Init() { }

void Sys::Finalize() { } 

void Sys::sync() {}

void Sys::Abort(int) { abort();  }

//
// Intializes internal Matrices and Vectors
//
void Sys::init()
{
    //-- M
    assert(M().rows() > 0 && M().cols() > 0);
    mean_rating = M().sum() / M().nonZeros();
    items().setZero();
    sum.setZero();
    cov.setZero();
    norm = .0;

    int count_larger_bp1 = 0;
    int count_larger_bp2 = 0;
    int count_sum = 0;
    for(int k = 0; k<M().cols(); k++) {
        int count = M().col(k).nonZeros();
        count_sum += count;
        if (count > breakpoint1) count_larger_bp1++;
        if (count > breakpoint2) count_larger_bp2++;
    }

    Sys::cout() << "mean rating: " << mean_rating << std::endl;
    Sys::cout() << "total number of ratings in train: " << M().nonZeros() << std::endl;
    Sys::cout() << "total number of ratings in test: " << T.nonZeros() << std::endl;
    Sys::cout() << "average ratings per row: " << (double)count_sum / (double)M().cols() << std::endl;
    Sys::cout() << "rows > break_point1: " << 100. * (double)count_larger_bp1 / (double)M().cols() << std::endl;
    Sys::cout() << "rows > break_point2: " << 100. * (double)count_larger_bp2 / (double)M().cols() << std::endl;
    Sys::cout() << "num " << name << ": " << num() << std::endl;

    if (measure_perf) sample_time.resize(num(), .0);
}



void HyperParams::sample(const int N, const VectorNd &sum, const MatrixNNd &cov)
{
    //SHOW(N);
    //SHOW(sum);
    //SHOW(cov);
    //SHOW(mu0);
    //SHOW(b0);
    //SHOW(df);
    //SHOW(WI);

    std::tie(mu, LambdaU) = CondNormalWishart(N, cov, sum / N, mu0, b0, WI, df);
    LambdaF = LambdaU.triangularView<Eigen::Upper>().transpose() * LambdaU;
    LambdaL = LambdaU.transpose();

    //SHOW(LambdaF);
}
//
// Update ONE movie or one user
//
VectorNd Sys::sample(long idx, Sys &other)
{
    rng.counter = (idx+1) * num_latent * (iter+1);
    Sys::dbg() << "-- original start name: " << name << " iter: " << iter << " idx: " << idx << ": " << rng.counter << std::endl;

    auto start = tick();

    VectorNd rr = hp().LambdaF * hp().mu;                // vector num_latent x 1, we will use it in formula (14) from the paper
    MatrixNNd MM(MatrixNNd::Zero());
    PrecomputedLLT chol;                             // matrix num_latent x num_latent, chol="lambda_i with *" from formula (14)

    //SHOW(hp().mu);
    //SHOW(hp().LambdaF);

    SHOW(hp().other_num);
    SHOW(hp().num);
    SHOW(hp().nnz);
    SHOW(M().outerSize());
    SHOW(M().innerSize());
    SHOW(M().nonZeros());

    SHOW("before computeMuLambda");
    SHOW(MM);
    SHOW(rr.transpose());

    //computeMuLambda(idx, other, rr, MM);
    for (SparseMapD::InnerIterator it(M(), idx); it; ++it)
    {
        auto col = other.items().col(it.row());
        MM.triangularView<Eigen::Upper>() += col * col.transpose();
        rr.noalias() += col * ((it.value() - hp().mean_rating) * hp().alpha);
    }
    
    // copy upper -> lower part, matrix is symmetric.
    MM.triangularView<Eigen::Lower>() = MM.transpose();

    SHOW(M());
    SHOW(other.items());
    SHOW(hp().mu.transpose());
    SHOW(hp().LambdaF);
    SHOW("after computeMuLambda");
    SHOW(MM);
    SHOW(rr.transpose());

    chol.compute(hp().LambdaF + hp().alpha * MM);

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

    //SHOW(rr.norm());
    //SHOW(items().col(idx).norm());

    SHOW(rr.transpose());
    Sys::dbg() << "-- original done name: " << name << " iter: " << iter << " idx: " << idx << ": " << rng.counter << std::endl;

    return rr;
}


void sample_task(
    long iter,
    long idx,
    const void *hp_void_ptr, 
    const double *other_ptr,
    const double *ratings_ptr,
    const int *inner_ptr,
    const int *outer_ptr,
    double *items_ptr)
{
    const HyperParams *hp_ptr = (const HyperParams *)hp_void_ptr;
    rng.counter = (idx+1) * num_latent * (iter+1);
    Sys::dbg() << "-- oss start iter " << iter << " idx: " << idx << ": " << rng.counter << std::endl;

    const Eigen::Map<const SparseMatrixD> M( hp_ptr->other_num, hp_ptr->num, hp_ptr->nnz, outer_ptr, inner_ptr, ratings_ptr);
    const Eigen::Map<const MatrixNXd> other(other_ptr, num_latent, hp_ptr->other_num);
    Eigen::Map<MatrixNXd> items(items_ptr, num_latent, hp_ptr->num);

    SHOW(hp_ptr->other_num);
    SHOW(hp_ptr->num);
    SHOW(hp_ptr->nnz);
    SHOW(M.outerSize());
    SHOW(M.innerSize());
    SHOW(M.nonZeros());

    VectorNd rr = hp_ptr->LambdaF * hp_ptr->mu;
    MatrixNNd MM(MatrixNNd::Zero());
    PrecomputedLLT chol;

    SHOW("before computeMuLambda");
    SHOW(MM);
    SHOW(rr.transpose());

    //computeMuLambda(idx, other, rr, MM);
    for (Eigen::Map<const SparseMatrixD>::InnerIterator it(M,idx); it; ++it)
    {
        auto col = other.col(it.row());
        MM.triangularView<Eigen::Upper>() += col * col.transpose();
        rr.noalias() += col * ((it.value() - hp_ptr->mean_rating) * hp_ptr->alpha);
    }
    
    // copy upper -> lower part, matrix is symmetric.
    MM.triangularView<Eigen::Lower>() = MM.transpose();

    SHOW(M);
    SHOW(other);
    SHOW(hp_ptr->mu.transpose());
    SHOW(hp_ptr->LambdaF);
    SHOW("after computeMuLambda");
    SHOW(MM);
    SHOW(rr.transpose());

    chol.compute(hp_ptr->LambdaF + hp_ptr->alpha * MM);

    if(chol.info() != Eigen::Success) THROWERROR("Cholesky failed");

    chol.matrixL().solveInPlace(rr);                    // L*Y=rr => Y=L\rr, we store Y result again in rr vector  
    rr += nrandn(num_latent);                           // rr=s+(L\rr), we store result again in rr vector
    chol.matrixU().solveInPlace(rr);                    // u_i=U\rr 
    items.col(idx) = rr;                              // we save rr vector in items matrix (it is user features matrix)

    SHOW(rr.transpose());
    Sys::dbg() << "-- oss done iter " << iter << " idx: " << idx << ": " << rng.counter << std::endl;
}


// 
// update ALL movies / users in parallel
//
void Sys::sample(Sys &other) 
{
    iter++;
    VectorNd  local_sum(VectorNd::Zero()); // sum
    double    local_norm(0.0); // squared norm
    MatrixNNd local_prod(MatrixNNd::Zero()); // outer prod

    int num_ratings = M().nonZeros();
    int outer_size_plus_one = M().outerSize() + 1;
    int num_items = num();
    int other_num_items = other_num();
    const double *other_ptr = other.items_ptr;

    const int this_iter = this->iter;
    const void *this_hp_ptr = (void *)this->hp_ptr;
    const int *this_inner_ptr = this->inner_ptr;
    const int *this_outer_ptr = this->outer_ptr;
    const double *this_ratings_ptr = this->ratings_ptr;
    double *this_items_ptr = this->items_ptr;

    Sys::dbg() << name << " -- Start scheduling oss tasks - iter " << iter << std::endl;

    sample_task_scheduler(
        from(),
        to(),
        num_latent,
        num_ratings,
        outer_size_plus_one,
        num_items,
        other_num_items,
        other_ptr,

        this_iter,
        this_hp_ptr,
        sizeof(HyperParams),
        this_inner_ptr,
        this_outer_ptr,
        this_ratings_ptr,
        this_items_ptr);

    Sys::dbg() << name << " -- Finished taskwait oss tasks - iter " << iter << std::endl;

    for (int i = from(); i < to(); ++i)
    {
        const VectorNd r1 = items().col(i);
        const VectorNd r2 = Sys::sample(i, other);

        if ((r1-r2).norm() > 0.0001) {
            Sys::dbg() << " Error at " << i << ":"
                << "\noriginal: " << r2.transpose()
                << "\noss     : " << r1.transpose()
                << std::endl;
        }

        local_prod += (r1 * r1.transpose());
        local_sum += r1;
        local_norm += r1.squaredNorm();
    }

    const int N = num();
    sum = local_sum;
    cov = (local_prod - (sum * sum.transpose() / N)) / (N-1);
    norm = local_norm;
}

void Sys::register_time(int i, double t)
{
    if (measure_perf) sample_time.at(i) += t;
}