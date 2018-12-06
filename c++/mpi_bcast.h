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
    virtual void send_items(int,int) {}
    virtual void alloc_and_init();
};

void MPI_Sys::sample(Sys &in)
{
    { BPMF_COUNTER("compute"); Sys::sample(in); }

    { 
        BPMF_COUNTER("communicate"); 
        bcast_items();

        for(int k = 0; k < Sys::nprocs; k++) {
            //sum, cov, norm
            MPI_Bcast(sum(k).data(), sum(k).size(), MPI_DOUBLE, k, MPI_COMM_WORLD);
            MPI_Bcast(cov(k).data(), cov(k).size(), MPI_DOUBLE, k, MPI_COMM_WORLD);
            MPI_Bcast(&norm(k),      1,             MPI_DOUBLE, k, MPI_COMM_WORLD);
        }
    }
}

#include "mpi_common.h"
