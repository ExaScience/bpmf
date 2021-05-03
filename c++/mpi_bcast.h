/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include <mpi.h>

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
    { BPMF_COUNTER("compute"); Sys::sample(in); }

    { 
        BPMF_COUNTER("communicate"); 
        bcast();
        reduce_sum_cov_norm();
    }
}

#include "mpi_common.h"
