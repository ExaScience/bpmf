/*
 * Copyright (c) 2021-2022, etascale
 * All rights reserved.
 */

#include <mpi.h>
#include "argo.hpp"

#define SYS ARGO_Sys

struct ARGO_Sys : public Sys
{
    //-- c'tor
    ARGO_Sys(std::string name, std::string fname, std::string probename) : Sys(name, fname, probename) {}
    ARGO_Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &P) : Sys(name, M, P) {}
    ~ARGO_Sys();

    //-- helpers
    bool is_local(void*);

    //-- virtuals
    virtual void send_item(int);
    virtual void sample(Sys &in);
    virtual void alloc_and_init();

    //-- process_queue queue with protecting mutex
    std::mutex m;
    std::list<int> queue;
    void process_queue();
    void commit_index(int);
};

ARGO_Sys::~ARGO_Sys()
{
    argo::codelete_array(items_ptr);
}

void ARGO_Sys::alloc_and_init()
{
    items_ptr = argo::conew_array<double>(num_latent * num());

    init();
}

void ARGO_Sys::send_item(int i)
{
#if defined(ARGO_NODE_WIDE_RELEASE) || \
    defined(ARGO_SELECTIVE_RELEASE)
    commit_index(i);
    process_queue();
#endif
}

bool ARGO_Sys::is_local(void *addr)
{
    return (argo::get_homenode(addr) == procid);
}

void ARGO_Sys::commit_index(int i)
{
#ifdef ARGO_SELECTIVE_RELEASE
    auto offset = i * num_latent;
    if (is_local(items_ptr+offset)) return;

    m.lock(); queue.push_back(i); m.unlock();
#endif
}

void ARGO_Sys::process_queue()
{
    int main_thread;
    MPI_Is_thread_main(&main_thread);
    if (!main_thread) return;

    {
        BPMF_COUNTER("process_queue");

#if   defined(ARGO_NODE_WIDE_RELEASE)
        argo::backend::release();
#elif defined(ARGO_SELECTIVE_RELEASE)
        int q = queue.size();
        while (q--) {
            m.lock();
            int i = queue.front();
            queue.pop_front();
            m.unlock();

            auto offset = i * num_latent;
            auto size = num_latent;
            argo::backend::selective_release(items_ptr+offset, size*sizeof(double));
        }
#endif
    }
}

void ARGO_Sys::sample(Sys &in)
{
    { BPMF_COUNTER("compute"); Sys::sample(in); }

    // send remaining
    process_queue();

    { BPMF_COUNTER("sync_sample"); Sys::sync(); }

    // reduce small structs
    reduce_sum_cov_norm();
}

void Sys::Init()
{
    // global address space size - 50GiB
    argo::init(50*1024*1024*1024UL);

    Sys::procid = argo::node_id();
    Sys::nprocs = argo::number_of_nodes();
}

void Sys::Finalize()
{
    argo::finalize();
}

void Sys::sync()
{
    argo::barrier();
}

void Sys::Abort(int err)
{
    MPI_Abort(MPI_COMM_WORLD, err);
}

void Sys::reduce_sum_cov_norm()
{
    BPMF_COUNTER("reduce_sum_cov_norm");
    MPI_Allreduce(MPI_IN_PLACE, sum.data(), num_latent, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, cov.data(), num_latent * num_latent, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}
