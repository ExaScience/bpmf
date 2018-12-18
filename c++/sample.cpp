/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include "error.h"
#include "bpmf.h"

#include <random>
#include <memory>
#include <cstdio>
#include <iostream>
#include <climits>
#include <stdexcept>

#include "io.h"

#if defined(_OPENMP)
#include "omp.h"

#pragma omp declare reduction (VectorPlus : VectorNd : omp_out += omp_in) initializer(omp_priv = VectorNd::Zero())
#pragma omp declare reduction (MatrixPlus : MatrixNNd : omp_out += omp_in) initializer(omp_priv = MatrixNNd::Zero())
#endif

static const bool measure_perf = false;

std::ostream *Sys::os;
int Sys::procid = -1;
int Sys::nprocs = -1;

int Sys::nsims;
int Sys::burnin;
double Sys::alpha;

bool Sys::permute = true;

bool Sys::verbose = false;

unsigned Sys::grain_size;

void calc_upper_part(MatrixNNd &m, VectorNd v);         // function for calcutation of an upper part of a symmetric matrix: m = v * v.transpose(); 
void copy_lower_part(MatrixNNd &m);                     // function to copy an upper part of a symmetric matrix to a lower part

// verifies that A is transpose of B
void assert_transpose(SparseMatrixD &A, SparseMatrixD &B)
{
    SparseMatrixD At = A.transpose();
    SparseMatrixD Bt = B.transpose();
    assert(At.cols() == B.cols());
    assert(A.cols() == Bt.cols());

    for(int i=0; i<B.cols(); ++i) assert(At.col(i).nonZeros() == B.col(i).nonZeros());
    for(int i=0; i<A.cols(); ++i) assert(Bt.col(i).nonZeros() == A.col(i).nonZeros());
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
    unsigned nump(0); // number of predictions

    int lo = all ? 0 : from();
    int hi = all ? num() : to();
    #pragma omp parallel for reduction(+:se,se_avg,nump)
    for(int k = lo; k<hi; k++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(T,k); it; ++it)
        {
            auto m = items().col(it.col());
            auto u = other.items().col(it.row());

            assert(m.norm() > 0.0);
            assert(u.norm() > 0.0);

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

    rmse = sqrt( se / nump );
    rmse_avg = sqrt( se_avg / nump );
}

//
// Prints sampling progress
//
void Sys::print(double items_per_sec, double ratings_per_sec, double norm_u, double norm_m) {
  char buf[1024];
  std::string phase = (iter < Sys::burnin) ? "Burnin" : "Sampling";
  sprintf(buf, "%d: %s iteration %d:\t RMSE: %3.2f\tavg RMSE: %3.2f\tFU(%6.2f)\tFM(%6.2f)\titems/sec: %6.2f\tratings/sec: %6.2fM\n",
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
    Pm2 = Pavg = T; // reference ratings and predicted ratings
    assert(M.rows() == Pavg.rows());
    assert(M.cols() == Pavg.cols());
    assert(Sys::nprocs <= (int)Sys::max_procs);
}

//
// Constructs Sys as transpose of existing Sys
//
Sys::Sys(std::string name, const SparseMatrixD &Mt, const SparseMatrixD &Pt) : name(name), iter(-1), assigned(false), dom(nprocs+1) {
    M = Mt.transpose();
    Pm2 = Pavg = T = Pt.transpose(); // reference ratings and predicted ratings
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
    items().setZero();
    sum_map().setZero();
    cov_map().setZero();
    norm_map().setZero();
    col_permutation.setIdentity(num());


    aggrMu = Eigen::MatrixXd::Zero(num_latent, num());
    aggrLambda = Eigen::MatrixXd::Zero(num_latent * num_latent, num());

    Sys::cout() << "mean rating = " << mean_rating << std::endl;
    Sys::cout() << "total number of ratings in train = " << M.nonZeros() << std::endl;
    Sys::cout() << "total number of ratings in test = " << T.nonZeros() << std::endl;
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


//
// Update ONE movie or one user
//
VectorNd Sys::sample(long idx, const MapNXd in)
{
    auto start = tick();

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


    int breakpoint1 = 24; 
    int breakpoint2 = 10500; 
    
    const int count = M.innerVector(idx).nonZeros(); // count of nonzeros elements in idx-th row of M matrix 
                                                     // (how many movies watched idx-th user?).

    VectorNd rr = hp_LambdaF * hp.mu;                 // vector num_latent x 1, we will use it in formula (14) from the paper
    PrecomputedLLT chol;                             // matrix num_latent x num_latent, chol="lambda_i with *" from formula (14) 
    
    // if this user movie has less than 1K ratings,
    // we do a serial rank update
    if( count < breakpoint1 ) {

        chol = hp_LambdaL;
        for (SparseMatrixD::InnerIterator it(M,idx); it; ++it) {
            auto col = in.col(it.row());
            chol.rankUpdate(col, alpha);
            rr.noalias() += col * ((it.value() - mean_rating) * alpha);
        }

    // else we do a serial full cholesky decomposition
    // (not used if breakpoint1 == breakpoint2)
    } else if (count < breakpoint2) {

        MatrixNNd MM(MatrixNNd::Zero());
        for (SparseMatrixD::InnerIterator it(M,idx); it; ++it) {
            auto col = in.col(it.row());
            
            //MM.noalias() += col * col.transpose();
            calc_upper_part(MM, col);
            
            rr.noalias() += col * ((it.value() - mean_rating) * alpha);
        }

        // Here, we copy a triangular upper part to a triangular lower part, because the matrix is symmetric.
        copy_lower_part(MM);

        chol.compute(hp_LambdaF + alpha * MM);
    // for > 1K ratings, we have additional thread-level parallellism
    } else {
        auto from = M.outerIndexPtr()[idx];   // "from" belongs to [1..m], m - number of movies in M matrix 
        auto to = M.outerIndexPtr()[idx+1];   // "to"   belongs to [1..m], m - number of movies in M matrix
        MatrixNNd MM(MatrixNNd::Zero());               // matrix num_latent x num_latent 
 
        // #pragma omp parallel for reduction(VectorPlus:rr) reduction(MatrixPlus:MM)
        #pragma omp parallel for reduction(VectorPlus:rr) reduction(MatrixPlus:MM) schedule(dynamic,200)
        for(int j = from; j<to; ++j) {                 // for each nonzeros elemen in the i-th row of M matrix
            auto val = M.valuePtr()[j];                // value of the j-th nonzeros element from idx-th row of M matrix
            auto idx = M.innerIndexPtr()[j];           // index "j" of the element [i,j] from M matrix in compressed M matrix 
            auto col = in.col(idx);                    // vector num_latent x 1 from V matrix: M[i,j] = U[i,:] x V[idx,:] 

            //MM.noalias() += col * col.transpose();     // outer product
            calc_upper_part(MM, col);
            rr.noalias() += col * ((val - mean_rating) * alpha); // vector num_latent x 1
        }

        copy_lower_part(MM);

        chol.compute(hp_LambdaF + alpha * MM);         // matrix num_latent x num_latent
                                                       // chol="lambda_i with *" from formula (14)
                                                       // lambda_i with * = LambdaU + alpha * MM
    }

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
    rr += nrandn();                                     // rr=s+(L\rr), we store result again in rr vector
    chol.matrixU().solveInPlace(rr);                    // u_i=U\rr 
    items().col(idx) = rr;                              // we save rr vector in items matrix (it is user features matrix)

    auto stop = tick();
    register_time(idx, 1e6 * (stop - start));
    //Sys::cout() << "  " << count << ": " << 1e6*(stop - start) << std::endl;

    assert(rr.norm() > .0);

    return rr;
}

// 
// update ALL movies / users in parallel
//
void Sys::sample(Sys &in) 
{
    iter++;
    VectorNd  sum(VectorNd::Zero()); // sum
    double    norm(0.0); // squared norm
    MatrixNNd prod(MatrixNNd::Zero()); // outer prod

//#pragma omp parallel for reduction(VectorPlus:sum) reduction(MatrixPlus:prod) reduction(+:norm) schedule(dynamic, 1)
#pragma omp parallel for reduction(VectorPlus:sum) reduction(MatrixPlus:prod) reduction(+:norm) schedule(dynamic,1) 
    for(int i = from(); i<to(); ++i) {
        auto r = sample(i,in.items());

        MatrixNNd cov = (r * r.transpose());
        prod += cov;
        sum += r;
        norm += r.squaredNorm();

        if (iter >= burnin)
        {
            aggrMu.col(i) += r;
            aggrLambda.col(i) += Eigen::Map<Eigen::VectorXd>(cov.data(), num_latent * num_latent);
        }

        send_items(i, i + 1);
    }

    const int N = num();
    local_sum() = sum;
    local_cov() = (prod - (sum * sum.transpose() / N)) / (N-1);
    local_norm() = norm;

}

void Sys::register_time(int i, double t)
{
    if (measure_perf) sample_time.at(i) += t;
}

void calc_upper_part(MatrixNNd &m, VectorNd v)
{
  // we use the formula: m = m + v * v.transpose(), but we calculate only an upper part of m matrix
  for (int j=0; j<num_latent; j++)          // columns
  {
    for(int i=0; i<=j; i++)              // rows
    {
      m(i,j) = m(i,j) + v[j] * v[i];
    }
  }
}

void copy_lower_part(MatrixNNd &m)
{
  // Here, we copy a triangular upper part to a triangular lower part, because the matrix is symmetric.
  for (int j=1; j<num_latent; j++)          // columns
  {
    for(int i=0; i<=j-1; i++)            // rows
    {
      m(j,i) = m(i,j);
    }
  }
}
