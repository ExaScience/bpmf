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
#ifndef _OMPSS
#define nanos6_lmalloc(size) malloc(size)
#define nanos6_dmalloc(size, A, B, C) malloc(size)
#endif

    // shouldn't be needed --> on stack
    hp_ptr = (HyperParams *)nanos6_lmalloc(sizeof(HyperParams));
    hp() = HyperParams();
    hp().alpha = alpha;
    hp().num = _M.cols();
    hp().other_num = _M.rows();
    hp().nnz = _M.nonZeros();


    ratings_ptr = (double *)nanos6_lmalloc(sizeof(double) * _M.nonZeros());
    inner_ptr   = (int *)nanos6_lmalloc(sizeof(int) * _M.nonZeros());
    outer_ptr   = (int *)nanos6_lmalloc(sizeof(int) * ( _M.outerSize() + 1));


    std::memcpy(M().valuePtr(),      _M.valuePtr(),      sizeof(double) * M().nonZeros());
    std::memcpy(M().innerIndexPtr(), _M.innerIndexPtr(), sizeof(int) * M().nonZeros());
    std::memcpy(M().outerIndexPtr(), _M.outerIndexPtr(), sizeof(int) * ( M().outerSize() + 1));

    for(int k = 0; k<M().cols(); k++) 
        assert(M().col(k).nonZeros() == _M.col(k).nonZeros());

    items_ptr = (double *)nanos6_dmalloc(sizeof(double) * num_latent * num(), nanos6_equpart_distribution, 0, NULL);

#ifndef _OMPSS
#undef nanos6_lmalloc
#undef nanos6_dmalloc
#endif

    init();

    hp().mean_rating = mean_rating;
}     


void sample_task(
    long iter,
    long idx,
    const HyperParams *hp_ptr, 
    const double *other_ptr,
    const double *ratings_ptr,
    const int *inner_ptr,
    const int *outer_ptr,
    double *items_ptr)
{
    rng.counter = (idx+1) * num_latent * (iter+1);
    Sys::cout() << "-- oss start iter " << iter << " idx: " << idx << ": " << rng.counter << std::endl;

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
    Sys::cout() << "-- oss done iter " << iter << " idx: " << idx << ": " << rng.counter << std::endl;
}

void sample_task_wrapper(
    long iter,
    long idx,
    const HyperParams *hp_ptr, 
    const double *other_ptr,
    const double *ratings_ptr,
    const int *inner_ptr,
    const int *outer_ptr,
    double *items_ptr)
{
    sample_task(iter, idx, hp_ptr, other_ptr, ratings_ptr, inner_ptr, outer_ptr, items_ptr);
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
    const HyperParams *this_hp_ptr = this->hp_ptr;
    const int *this_inner_ptr = this->inner_ptr;
    const int *this_outer_ptr = this->outer_ptr;
    const double *this_ratings_ptr = this->ratings_ptr;
    double *this_items_ptr = this->items_ptr;

    Sys::cout() << name << " -- Start scheduling oss tasks - iter " << iter << std::endl;

    for (int i = from(); i < to(); ++i)
    {
        #pragma oss task \
            in(this_hp_ptr[0]) \
            in(this_ratings_ptr[0;num_ratings]) \
            in(this_inner_ptr[0;num_ratings]) \
            in(this_outer_ptr[0;outer_size_plus_one]) \
            in(other_ptr[0;other_num_items*num_latent]) \
            out(this_items_ptr[i*num_latent;num_latent])
        sample_task_wrapper(this_iter, i, this_hp_ptr, other_ptr, this_ratings_ptr, this_inner_ptr, this_outer_ptr, this_items_ptr);
    }

    Sys::cout() << name << " -- Finished scheduling oss tasks - iter " << iter << std::endl;
    // taskwait copies outputs from sample_task to this task
#pragma oss taskwait

    Sys::cout() << name << " -- Finished taskwait oss tasks - iter " << iter << std::endl;
    for (int i = from(); i < to(); ++i)
    {
        const VectorNd r1 = items().col(i);
        const VectorNd r2 = Sys::sample(i, other);

        if ((r1-r2).norm() > 0.0001) {
            Sys::cout() << " Error at " << i << ":"
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