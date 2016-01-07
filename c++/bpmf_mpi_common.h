#include <mpi.h>

#define SYS MPI_Sys

void Sys::Init()
{
    MPI_Init(0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &Sys::procid);
    MPI_Comm_size(MPI_COMM_WORLD, &Sys::nprocs);
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
