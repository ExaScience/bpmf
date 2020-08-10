/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#define SYS MPI_Sys

void Sys::Init()
{
    int provided;
    MPI_Init_thread(0, 0, MPI_THREAD_SERIALIZED, &provided);
    assert(provided == MPI_THREAD_SERIALIZED);
    MPI_Comm_rank(MPI_COMM_WORLD, &Sys::procid);
    MPI_Comm_size(MPI_COMM_WORLD, &Sys::nprocs);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
}

void Sys::Finalize()
{
    MPI_Finalize();
}

void Sys::sync()
{
    MPI_Barrier(MPI_COMM_WORLD);
}

void Sys::Abort(int err)
{
    MPI_Abort(MPI_COMM_WORLD, err);
}
void MPI_Sys::alloc_and_init()
{
    items_ptr = (double *)malloc(sizeof(double) * num_latent * num());
    sum_ptr = (double *)malloc(sizeof(double) * num_latent * MPI_Sys::nprocs);
    cov_ptr = (double *)malloc(sizeof(double) * num_latent * num_latent * MPI_Sys::nprocs);
    norm_ptr = (double *)malloc(sizeof(double) * MPI_Sys::nprocs);

    init();
}

void MPI_Sys::bcast_sum_cov_norm()
{
    BPMF_COUNTER("bcast");
    for (int k = 0; k < Sys::nprocs; k++)
    {
        //sum, cov, norm
        MPI_Bcast(sum(k).data(), sum(k).size(), MPI_DOUBLE, k, MPI_COMM_WORLD);
        MPI_Bcast(cov(k).data(), cov(k).size(), MPI_DOUBLE, k, MPI_COMM_WORLD);
        MPI_Bcast(&norm(k), 1, MPI_DOUBLE, k, MPI_COMM_WORLD);
    }
}
