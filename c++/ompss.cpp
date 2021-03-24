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
#include "ompss.h"


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

void sample_task_scheduler(
    int from,
    int to,
    int num_ratings,
    int outer_size_plus_one,
    int num_items,
    int other_num_items,
    const double *other_ptr,

    const int this_iter,
    const HyperParams *this_hp_ptr,
    const int *this_inner_ptr,
    const int *this_outer_ptr,
    const double *this_ratings_ptr,
    double *this_items_ptr
)
{
    for (int i = from; i < to; ++i)
    {
        #pragma oss task \
            in(this_hp_ptr[0]) \
            in(this_ratings_ptr[0;num_ratings]) \
            in(this_inner_ptr[0;num_ratings]) \
            in(this_outer_ptr[0;outer_size_plus_one]) \
            in(other_ptr[0;other_num_items*num_latent]) \
            out(this_items_ptr[i*num_latent;num_latent])
        sample_task(this_iter, i, this_hp_ptr, other_ptr, this_ratings_ptr, this_inner_ptr, this_outer_ptr, this_items_ptr);
    }

#pragma oss taskwait
}
