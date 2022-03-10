/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

/* C   libs */
#include <assert.h>

#include "ompss.h"

#ifndef OMPSS
#include <stdlib.h>
#endif
 
void *lmalloc(unsigned long size)
{
#ifdef OMPSS
    return nanos6_lmalloc(size);
#else
    return malloc(size);
#endif
}

void *dmalloc(unsigned long size)
{
#ifdef OMPSS
    return nanos6_dmalloc(size, nanos6_equpart_distribution, 0, NULL);
#else
    return malloc(size);
#endif
}

void oss_taskwait()
{
    #pragma oss taskwait
}

void oss_reset_stats()
{
#ifdef CLUSTER_ARGO
   nanos6_argo_reset_stats();
#endif
}

static void node_chunk(
    int *chunk,
    const int node_id,
    const int nodes,
    const int to,
    const int index,
    const int bsize
)
{
    *chunk = (node_id != nodes-1) ? bsize : to-index;
}

static void task_chunk(
    int *chunk,
    const int to,
    const int index,
    const int bsize
)
{
#define MIN(a,b) (((a)<(b))?(a):(b))

    *chunk = MIN(bsize, to-index-1);

#undef MIN
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
    int nodes = nanos6_get_num_cluster_nodes();

    const int num_tasks_per_node = 100;
    const int num_items_per_node = to / nodes;

    for (int node_id = 0; node_id < nodes; ++node_id)
    {
        int i = node_id*num_items_per_node, num_items_this_node;
        node_chunk(&num_items_this_node, node_id, nodes, to, i, num_items_per_node);

        #pragma oss task                                                          \
            weakin(this_hp_ptr     [0;hp_size])                                   \
            weakin(this_ratings_ptr[0;num_ratings])                               \
            weakin(this_inner_ptr  [0;num_ratings])                               \
            weakin(this_outer_ptr  [0;outer_size_plus_one])                       \
            weakin(other_ptr       [0;other_num_items*num_latent])                \
            weakout(this_items_ptr [i*num_latent;num_items_this_node*num_latent]) \
            firstprivate(i, num_tasks_per_node, num_items_this_node)              \
            node(node_id)                                                         \
            label("sample_weak_task")
        {
            const int num_items_per_task = num_items_this_node / num_tasks_per_node;
            assert(num_items_per_task > 0);

            for (int j = i; j < i+num_items_this_node; j += num_items_per_task)
            {
                int num_items_this_task;
                task_chunk(&num_items_this_task, i+num_items_this_node, j, num_items_per_task);

                #pragma oss task                                                      \
                    in(this_hp_ptr     [0;hp_size])                                   \
                    in(this_ratings_ptr[0;num_ratings])                               \
                    in(this_inner_ptr  [0;num_ratings])                               \
                    in(this_outer_ptr  [0;outer_size_plus_one])                       \
                    in(other_ptr       [0;other_num_items*num_latent])                \
                    out(this_items_ptr [j*num_latent;num_items_this_task*num_latent]) \
                    firstprivate(j, num_items_this_task)                              \
                    node(nanos6_cluster_no_offload)                                   \
                    label("sample_strong_task")
                {
                    for (int k = j; k < j+num_items_this_task; k++)
                        sample_task(this_iter, k, this_hp_ptr, other_ptr, this_ratings_ptr, this_inner_ptr, this_outer_ptr, this_items_ptr);
                }
            }
        }
    }
}
