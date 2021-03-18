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

    hp().alpha = alpha;
    hp().num = _M.cols();
    hp().other_num = _M.rows();
    hp().nnz = _M.nonZeros();

    ratings_ptr = (double *)nanos6_lmalloc(sizeof(double) * _M.nonZeros());
    inner_ptr   = (int *)nanos6_lmalloc(sizeof(double) * _M.nonZeros());
    outer_ptr   = (int *)nanos6_lmalloc(sizeof(double) * _M.outerSize());

    std::memcpy(M().valuePtr(), _M.valuePtr(), M().nonZeros());
    std::memcpy(M().innerIndexPtr(), _M.innerIndexPtr(), M().nonZeros());
    std::memcpy(M().outerIndexPtr(), _M.outerIndexPtr(), M().outerSize());

    for(int k = 0; k<M().cols(); k++) 
        assert(M().col(k).nonZeros() == _M.col(k).nonZeros());

    items_ptr = (double *)nanos6_lmalloc(sizeof(double) * num_latent * num());

    init();

    hp().mean_rating = mean_rating;
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

    for (int i = from(); i < to(); ++i)
    {
//#pragma oss task \
//    out(out_ptr[0;num_latent]) \
//    shared(other) private(i)
        {
            auto r = sample(i, other);
            local_prod += (r * r.transpose());
            local_sum += r;
            local_norm += r.squaredNorm();
        }
    }
#pragma oss taskwait

    const int N = num();
    sum = local_sum;
    cov = (local_prod - (sum * sum.transpose() / N)) / (N-1);
    norm = local_norm;
}