/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include <cstdlib>

#include "ompss.h"

void *lmalloc(unsigned long size)
{
#ifdef _OMPSS
    return nanos6_lmalloc(size);
#else
    return malloc(size);
#endif
}

void *dmalloc(unsigned long size)
{
#ifdef _OMPSS
    return nanos6_dmalloc(size, nanos6_equpart_distribution, 0, NULL);
#else
    return malloc(size);
#endif
}

void sample_task_scheduler(
    int from,
    int to,
    int num_latent,
    int num_ratings,
    int outer_size_plus_one,
    int num_items,
    int other_num_items,
    const double *other_ptr,

    const int this_iter,
    const void *this_hp_ptr,
    const int hp_size,
    const int *this_inner_ptr,
    const int *this_outer_ptr,
    const double *this_ratings_ptr,
    double *this_items_ptr
)
{
    for (int i = from; i < to; ++i)
    {
        #pragma oss task \
            in(this_hp_ptr[0;hp_size]) \
            in(this_ratings_ptr[0;num_ratings]) \
            in(this_inner_ptr[0;num_ratings]) \
            in(this_outer_ptr[0;outer_size_plus_one]) \
            in(other_ptr[0;other_num_items*num_latent]) \
            out(this_items_ptr[i*num_latent;num_latent])
        sample_task(this_iter, i, this_hp_ptr, other_ptr, this_ratings_ptr, this_inner_ptr, this_outer_ptr, this_items_ptr);
    }

#pragma oss taskwait
}
