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
#include <cstring>

#include "error.h"
#include "bpmf.h"
#include "io.h"


#pragma oss declare reduction (+: VectorNd: omp_out=omp_out+omp_in) \
     initializer(omp_priv=VectorNd::Zero(omp_orig.size()))

#pragma oss declare reduction (+: MatrixNNd: omp_out=omp_out+omp_in) \
     initializer(omp_priv=MatrixNNd::Zero(omp_orig.rows(), omp_orig.cols()))


void Sys::Init() { }

void Sys::Finalize() { } 

void Sys::sync() {}

void Sys::Abort(int) { abort();  }

void Sys::alloc_and_init()
{
    hp_ptr = (HyperParams *)nanos6_lmalloc(sizeof(HyperParams));
    hp() = HyperParams();
    hp().alpha = alpha;
    hp().num = _M.cols();
    hp().other_num = _M.rows();
    hp().nnz = _M.nonZeros();

    ratings_ptr = (double *)nanos6_lmalloc(sizeof(double) * _M.nonZeros());
    inner_ptr   = (int *)nanos6_lmalloc(sizeof(double) * _M.nonZeros());
    outer_ptr   = (int *)nanos6_lmalloc(sizeof(double) * ( _M.outerSize() + 1));

    std::memcpy(M().valuePtr(),      _M.valuePtr(),      sizeof(double) * M().nonZeros());
    std::memcpy(M().innerIndexPtr(), _M.innerIndexPtr(), sizeof(int) * M().nonZeros());
    std::memcpy(M().outerIndexPtr(), _M.outerIndexPtr(), sizeof(int) * ( M().outerSize() + 1));

    for(int k = 0; k<M().cols(); k++) 
        assert(M().col(k).nonZeros() == _M.col(k).nonZeros());

    items_ptr = (double *)nanos6_dmalloc(sizeof(double) * num_latent * num(), nanos6_equpart_distribution, 0, NULL);

    init();

    hp().mean_rating = mean_rating;
}     

void sample_task(
    long idx,
    const HyperParams *hp_ptr, 
    const double *other_ptr,
    const double *ratings_ptr,
    const int *inner_ptr,
    const int *outer_ptr,
    double *items_ptr)
{
    const Eigen::Map<const SparseMatrixD> M( hp_ptr->other_num, hp_ptr->num, hp_ptr->nnz, outer_ptr, inner_ptr, ratings_ptr);
    const Eigen::Map<const MatrixNXd> other(other_ptr, num_latent, hp_ptr->other_num);
    Eigen::Map<MatrixNXd> items(items_ptr, num_latent, hp_ptr->num);

    VectorNd rr = hp_ptr->LambdaF * hp_ptr->mu;
    MatrixNNd MM(MatrixNNd::Zero());
    PrecomputedLLT chol;

    //computeMuLambda(idx, other, rr, MM);
    for (Eigen::Map<const SparseMatrixD>::InnerIterator it(M,idx); it; ++it)
    {
        auto col = other.col(it.row());
        MM.triangularView<Eigen::Upper>() += col * col.transpose();
        rr.noalias() += col * ((it.value() - hp_ptr->mean_rating) * hp_ptr->alpha);
    }
    
    // copy upper -> lower part, matrix is symmetric.
    MM.triangularView<Eigen::Lower>() = MM.transpose();

    chol.compute(hp_ptr->LambdaF + hp_ptr->alpha * MM);

    if(chol.info() != Eigen::Success) THROWERROR("Cholesky failed");

    chol.matrixL().solveInPlace(rr);                    // L*Y=rr => Y=L\rr, we store Y result again in rr vector  
    rr += nrandn(num_latent);                           // rr=s+(L\rr), we store result again in rr vector
    chol.matrixU().solveInPlace(rr);                    // u_i=U\rr 
    items.col(idx) = rr;                              // we save rr vector in items matrix (it is user features matrix)
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
    int outer_size = M().outerSize();
    int num_items = num();
    int other_num_items = other_num();

    for (int i = from(); i < to(); ++i)
    {
        #pragma oss task \
            in(hp_ptr[0]) \
            in(ratings_ptr[0;num_ratings]) \
            in(inner_ptr[0;num_ratings]) \
            in(outer_ptr[0;outer_size]) \
            in(other.items_ptr[0;other_num_items*num_latent]) \
            out(items_ptr[i*num_latent;num_latent])
        sample_task(i, hp_ptr, other.items_ptr, ratings_ptr, inner_ptr, outer_ptr, items_ptr);
    }
#pragma oss taskwait

    for (int i = from(); i < to(); ++i)
    {
        const auto &r = items().col(i);
        local_prod += (r * r.transpose());
        local_sum += r;
        local_norm += r.squaredNorm();
    }

    const int N = num();
    sum = local_sum;
    cov = (local_prod - (sum * sum.transpose() / N)) / (N-1);
    norm = local_norm;
}