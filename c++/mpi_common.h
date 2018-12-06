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
#ifdef BPMF_PROFILING
    perf_data.print();
#endif
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

void MPI_Sys::bcast_items()
{
    for(int i = 0; i < num(); i++) {
        MPI_Bcast(items().col(i).data(), num_latent, MPI_DOUBLE, proc(i), MPI_COMM_WORLD);
    }
}


void MPI_Sys::alloc_and_init()
{
    items_ptr = (double *)malloc(sizeof(double) * num_latent * num());
    sum_ptr = (double *)malloc(sizeof(double) * num_latent * MPI_Sys::nprocs);
    cov_ptr = (double *)malloc(sizeof(double) * num_latent * num_latent * MPI_Sys::nprocs);
    norm_ptr = (double *)malloc(sizeof(double) * MPI_Sys::nprocs);

    init();
}

