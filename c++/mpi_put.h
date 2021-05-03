/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include <mutex>
#include <mpi.h>


struct MPI_Sys : public Sys 
{
    //-- c'tor
    MPI_Sys(std::string name, std::string fname, std::string pname) : Sys(name,fname,pname) {}
    MPI_Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &P) : Sys(name,M,P) {}
    virtual void alloc_and_init();

    virtual void send_item(int i);
    virtual void sample(Sys &in);

    void reduce_sum_cov_norm();

    MPI_Win items_win;
    MPI_Win cov_win;
    MPI_Win norm_win;
};


void MPI_Sys::alloc_and_init()
{
 
    const int items_size = sizeof(double) * num_latent * num();

    MPI_Alloc_mem(items_size, MPI_INFO_NULL, &items_ptr);
    MPI_Win_create(items_ptr, items_size, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &items_win); 
    MPI_Win_fence(0,items_win);

    init();
}

void MPI_Sys::send_item(int i)
{
    BPMF_COUNTER("send_items");
    static std::mutex m;

    m.lock();
    for(int k = 0; k < Sys::nprocs; k++) {
        if (k == Sys::procid) continue;
        auto offset = i * num_latent;
        auto size = num_latent;
        MPI_Put(items_ptr+offset, size, MPI_DOUBLE, k, offset, size, MPI_DOUBLE, items_win); 
    }
    m.unlock();
}

void MPI_Sys::sample(Sys &in)
{
    {
        BPMF_COUNTER("sample");
        Sys::sample(in);
    }

    reduce_sum_cov_norm();

    {
        BPMF_COUNTER("fence");
        MPI_Win_fence(0,items_win);
    }
}

#include "mpi_common.h"
