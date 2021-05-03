/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#define SYS NC_Sys

struct NC_Sys : public Sys 
{
    //-- c'tor
    NC_Sys(std::string name, std::string fname, std::string probename) : Sys(name, fname, probename) {}
    NC_Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &P) : Sys(name, M, P) {}
    virtual void alloc_and_init();

    virtual void send_item(int) {}
};


void Sys::Init()
{
    NC_Sys::procid = 0;
    NC_Sys::nprocs = 1;
}

void Sys::Finalize()
{
}

void NC_Sys::alloc_and_init()
{
    items_ptr = (double *)malloc(sizeof(double) * num_latent * num());
    init();
}

void Sys::sync() {}

void Sys::Abort(int) { abort(); }
