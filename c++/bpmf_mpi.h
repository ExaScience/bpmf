/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#include "bpmf_mpi_common.h"

struct MPI_Sys : public Sys 
{
    //-- c'tor
    MPI_Sys(std::string name, std::string fname) : Sys(name,fname) {}
    MPI_Sys(std::string name, const SparseMatrixD &M) : Sys(name,M) {}
    virtual void alloc_and_init(const Sys &);

    virtual void send_items(int from, int to);
    virtual void send_item(int i);
    virtual void sample(Sys &in);

    MPI_Win items_win;
    MPI_Win sum_win;
    MPI_Win cov_win;
    MPI_Win norm_win;
};

void MPI_Sys::alloc_and_init(const Sys &other)
{
 
    const int items_size = sizeof(double) * num_feat * num();
    const int sum_size   = sizeof(double) * num_feat * Sys::nprocs;
    const int cov_size   = sizeof(double) * num_feat * num_feat * Sys::nprocs;
    const int norm_size  = sizeof(double) * Sys::nprocs;

    MPI_Alloc_mem(items_size, MPI_INFO_NULL, &items_ptr);
    MPI_Alloc_mem(sum_size,   MPI_INFO_NULL, &sum_ptr);
    MPI_Alloc_mem(cov_size,   MPI_INFO_NULL, &cov_ptr);
    MPI_Alloc_mem(norm_size,  MPI_INFO_NULL, &norm_ptr);

    MPI_Win_create(items_ptr, items_size, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &items_win); 
    MPI_Win_create(sum_ptr,   sum_size,   sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &sum_win); 
    MPI_Win_create(cov_ptr,   cov_size,   sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &cov_win); 
    MPI_Win_create(norm_ptr,  norm_size,  sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &norm_win); 

    MPI_Win_fence(0,items_win);
    MPI_Win_fence(0,sum_win);
    MPI_Win_fence(0,cov_win);
    MPI_Win_fence(0,norm_win);

    init(other);
}

void MPI_Sys::send_items(int from, int to)
{
    for(int k = 0; k < Sys::nprocs; k++) {
        if (k == Sys::procid) continue;
        auto offset = from * num_feat;
        auto size = num_feat * (to -  from);
        MPI_Put(items_ptr+offset, size, MPI_DOUBLE, k, offset, size, MPI_DOUBLE, items_win); 
    }
}

void MPI_Sys::send_item(int i)
{
    for(int k = 0; k < Sys::nprocs; k++) {
        if (k == Sys::procid) continue;
        auto offset = i * num_feat;
        auto size = num_feat;
        MPI_Put(items_ptr+offset, size, MPI_DOUBLE, k, offset, size, MPI_DOUBLE, items_win); 
    }
}

void MPI_Sys::sample(Sys &in)
{
    Sys::sample(in);

    for(int k = 0; k < Sys::nprocs; k++) {
        if (k == Sys::procid) continue;
        auto base = Sys::procid;
        {
            //-- sum
            auto offset = base * num_feat;
            auto size = num_feat;
            MPI_Put(sum_ptr+offset, size, MPI_DOUBLE, k, offset, size, MPI_DOUBLE, sum_win); 
        }
        {
            //-- cov
            auto offset = base * num_feat * num_feat;
            auto size = num_feat * num_feat;
            MPI_Put(cov_ptr+offset, size, MPI_DOUBLE, k, offset, size, MPI_DOUBLE, cov_win); 
        }
        {
            //-- norm
            auto offset = base;
            auto size = 1;
            MPI_Put(norm_ptr+offset, size, MPI_DOUBLE, k, offset, size, MPI_DOUBLE, norm_win); 
        }
    }

    MPI_Win_fence(0,items_win);
    MPI_Win_fence(0,sum_win);
    MPI_Win_fence(0,cov_win);
    MPI_Win_fence(0,norm_win);
}
