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


std::vector<const double *> Sys::collectColumns(long idx, const double *other) const
{
    std::vector<const double *> columns;
    for (SparseMatrixD::InnerIterator it(M, idx); it; ++it)
    {
#pragma oss task in(other[it.row() * num_latent;num_latent])
        const auto col = &other[it.row() * num_latent];
        columns.push_back(col);
    }
#pragma oss taskwait

    return columns;
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

#pragma oss declare reduction (+: VectorNd: omp_out=omp_out+omp_in) \
     initializer(omp_priv=VectorNd::Zero(omp_orig.size()))

#pragma oss declare reduction (+: MatrixNNd: omp_out=omp_out+omp_in) \
     initializer(omp_priv=MatrixNNd::Zero(omp_orig.rows(), omp_orig.cols()))

    for (int i = from(); i < to(); ++i)
    {
        const double *out_ptr = items().col(i).data();

#pragma oss task reduction(+:local_sum, local_norm, local_prod) \
    out(out_ptr[0;num_latent]) \
    shared(other) private(i)
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