/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include <mpi.h>
#include "error.h"

struct MPI_Sys : public Sys 
{
    //-- c'tor
    MPI_Sys(std::string name, std::string fname, std::string probename) : Sys(name, fname, probename) {}
    MPI_Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &P) : Sys(name, M, P) {}

    virtual void sample(Sys &in);
    virtual void send_item(int) {}
    virtual void alloc_and_init();

    void bcast_sum_cov_norm();

};

void MPI_Sys::sample(Sys &in)
{
    { BPMF_COUNTER("compute"); Sys::sample(in); }

    {
        BPMF_COUNTER("communicate");


        for (int i = 0; i < nprocs; i++)
        {
            auto col = in.from(i);
            auto num_cols = in.num(i);
            
            MPI_Allreduce(MPI_IN_PLACE, in.precMu.col(col).data(), num_cols*num_latent, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, in.precLambda.col(col).data(), num_cols*num_latent*num_latent, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        bcast_sum_cov_norm();
    }
}

#include "mpi_common.h"
