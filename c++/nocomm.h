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

    virtual void send_items(int, int) {}
    virtual void bcast_items() {}
};


void Sys::Init()
{
    NC_Sys::procid = 0;
    NC_Sys::nprocs = 1;
}

void Sys::Finalize()
{
#ifdef BPMF_PROFILING
        perf_data.print();
#endif

}

void NC_Sys::alloc_and_init()
{
    items_ptr = (double *)malloc(sizeof(double) * num_latent * num());
    sum_ptr = (double *)malloc(sizeof(double) * num_latent * NC_Sys::nprocs);
    cov_ptr = (double *)malloc(sizeof(double) * num_latent * num_latent * NC_Sys::nprocs);
    norm_ptr = (double *)malloc(sizeof(double) * NC_Sys::nprocs);

    init();
}

void Sys::sync() {}

void Sys::Abort(int) { abort(); }
