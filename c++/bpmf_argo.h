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

    //-- virtuals
    virtual void send_item(int);
    virtual void sample(Sys &in);
    virtual void alloc_and_init();

    //-- process_queue queue with protecting mutex
    std::mutex m;
    std::list<int> queue;
    void process_queue();
    void commit_index(int);

    //-- helpers
    bool are_items_adjacent(int, int);
    void pop_front(int);
    void release();
    void acquire();
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

void ARGO_Sys::commit_index(int i)
{
#if defined(ARGO_SELECTIVE_RELEASE)
    bool push = 0;
    for(int k = 0; k < nprocs; ++k)
        if (conn(i, k)) {
            push = 1;
            break;
        }
    // if no-one needs it, don't push
    if (!push) return;

    m.lock(); queue.push_back(i); m.unlock();
#endif
}

void ARGO_Sys::process_queue()
{
    int main_thread;
    MPI_Is_thread_main(&main_thread);
    if (!main_thread) return;

    // send items in buffer
    {
        BPMF_COUNTER("release");
        release();
    }
}

void ARGO_Sys::sample(Sys &in)
{
    // compute my own chunk
    {
        BPMF_COUNTER("compute");
        Sys::sample(in);
    }

    // send remaining items
    process_queue();

    // reduce small structs
    reduce_sum_cov_norm();

    // recv necessary items
    {
        BPMF_COUNTER("acquire");
        acquire();
    }
}

void ARGO_Sys::release()
{
#if   defined(ARGO_NODE_WIDE_RELEASE)
    argo::backend::release();
#elif defined(ARGO_SELECTIVE_RELEASE)

#ifdef FINE_GRAINED_SELECTIVE_RELEASE
    int q = queue.size();
    while (q--) {
        auto i = queue.front();
        m.lock(); queue.pop_front(); m.unlock();

        auto offset = i * num_latent;
        auto size = num_latent;
        argo::backend::selective_release(
                items_ptr+offset, size*sizeof(double));
    }
#else
    m.lock();
    int q = queue.size();
    queue.sort();
    m.unlock();

    while (q) {
        auto i = queue.front();
        auto l = queue.begin();
        auto r = std::next(l);

        int c;
        for (c = 1; c < q; ++c, ++l, ++r)
            if (!are_items_adjacent(*l, *r))
                break;
        m.lock(); pop_front(c); m.unlock();

        auto offset = i * num_latent;
        auto size = c * num_latent;
        argo::backend::selective_release(
                items_ptr+offset, size*sizeof(double));

        q -= c;
    }
#endif

#endif
}

void ARGO_Sys::acquire()
{
#if   defined(ARGO_NODE_WIDE_ACQUIRE)
    argo::backend::acquire();
#elif defined(ARGO_SELECTIVE_ACQUIRE)

#ifndef ARGO_LOCALITY
    std::vector<int> regions(4, 0);
    int iters;

    if (nprocs == 1) {
        regions.at(0) = from(0);
        regions.at(1) =   to(0);
        iters = 1;
    } else
        if (procid == 0) {
            regions.at(0) = from(1);
            regions.at(1) =   to(nprocs-1);
            iters = 1;
        } else if (procid == nprocs-1) {
            regions.at(0) = from(0);
            regions.at(1) =   to(procid-1);
            iters = 1;
        } else {
            regions.at(0) = from(0);
            regions.at(1) =   to(procid-1);
            regions.at(2) = from(procid+1);
            regions.at(3) =   to(nprocs-1);
            iters = 2;
        }
#endif

#ifdef FINE_GRAINED_SELECTIVE_ACQUIRE

#ifndef ARGO_LOCALITY
    for (int it = 0; it < iters; ++it) {
        int lo = regions.at(it*2);
        int hi = regions.at(it*2+1);

        for(int i = lo; i < hi; ++i)
            if (conn(i, procid)) {
                auto offset = i * num_latent;
                auto size = num_latent;
                argo::backend::selective_acquire(
                        items_ptr+offset, size*sizeof(double));
            }
    }
#else
    for (int i : items_remote) {
        auto offset = i * num_latent;
        auto size = num_latent;
        argo::backend::selective_acquire(
                items_ptr+offset, size*sizeof(double));
    }
#endif

#else

#ifndef ARGO_LOCALITY
    for (int it = 0; it < iters; ++it) {
        int lo = regions.at(it*2);
        int hi = regions.at(it*2+1);

        for(int i = lo, c, b; i < hi; i += c+b) {
            c = 0;
            b = 0;
            for (int k = i; k < hi; ++k)
                if (conn(k, procid))
                    ++c;
                else {
                    ++b;
                    break;
                }

            if (c > 0) {
                auto offset = i * num_latent;
                auto size = c * num_latent;
                argo::backend::selective_acquire(
                        items_ptr+offset, size*sizeof(double));
            }
        }
    }
#else
    int lo = 0;
    int hi = items_remote.size();

    for (int i = lo, c; i < hi; i += c) {
        c = 1;
        for (int k = i; k < hi-1; ++k, ++c)
            if (!are_items_adjacent(
                        items_remote.at(k), items_remote.at(k+1)))
                break;

        auto offset = items_remote.at(i) * num_latent;
        auto size = c * num_latent;
        argo::backend::selective_acquire(
                items_ptr+offset, size*sizeof(double));
    }
#endif

#endif

#else
    Sys::sync();
#endif
}

bool ARGO_Sys::are_items_adjacent(int l, int r)
{
    return (l == r-1);
}

void ARGO_Sys::pop_front(int elems)
{
    auto beg = queue.begin();
    auto end = std::next(beg, elems);
    queue.erase(beg, end);
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
