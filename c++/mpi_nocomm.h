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

    virtual void send_items(int,int) {}
    virtual void bcast_items();
    virtual void alloc_and_init();
};

#include "mpi_common.h"
