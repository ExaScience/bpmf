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

    void reduce_sum_cov_norm();

};


void MPI_Sys::sample(Sys &in)
{
    {
        BPMF_COUNTER("communicate");

        // 1. -- Worker sends updates to PS
        //     - sends norm, cov and sum 
        //     - sends precMu, precLambda

        /* CODE HERE */

        // 2. -- PS combines precMu, precLambda
        //       Reduction for each movie/user accross workers

        for (int i = 0; i < nprocs; i++)
        {
            auto col = from(i);
            auto num_cols = num(i);

            {
                auto recv_buf = precMu.col(col).data();
                auto send_buf = (i == procid) ? MPI_IN_PLACE : recv_buf;
                MPI_Reduce(send_buf, recv_buf, num_cols * num_latent, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
            }

            {
                auto recv_buf = precLambda.col(col).data();
                auto send_buf = (i == procid) ? MPI_IN_PLACE : recv_buf;
                MPI_Reduce(send_buf, recv_buf, num_cols * num_latent * num_latent, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
            }
        }

        // 3. -- PS sends combined precMu, precLambda
        //          sends cov/norm/sum

        // 4. -- Worker reduces sum/norm/cov
        reduce_sum_cov_norm();
    }

    { BPMF_COUNTER("compute"); Sys::sample(in); }


}

#include "mpi_common.h"
