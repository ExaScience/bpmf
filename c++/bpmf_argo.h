/*
 * Copyright (c) 2021-2022, etascale
 * All rights reserved.
 */

#include <mpi.h>
#include "argo.hpp"

#if !defined(ARGO_LOCALITY)   && \
     defined(ARGO_NO_RELEASE) && \
    (defined(ARGO_NODE_WIDE_ACQUIRE) || \
     defined(ARGO_SELECTIVE_ACQUIRE))
#error This coherence operations combination violates correctness.
#endif

#define SYS ARGO_Sys

struct ARGO_Sys : public Sys
{
    //-- c'tor
    ARGO_Sys(std::string name, std::string fname, std::string probename) : Sys(name, fname, probename) {}
    ARGO_Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &P) : Sys(name, M, P) {}
    ~ARGO_Sys();

    //-- virtuals
    virtual void sample(Sys&);
    virtual void send_item(int);
    virtual void alloc_and_init();

    //-- process_queue queue with protecting mutex
    std::mutex m;
    std::list<int> queue;
    void commit_index(int);
    void process_queue();

    //-- helpers
    bool are_items_adjacent(int, int);
    void pop_front(int);
    void init_vectors();

    //-- release
    void release();
    void node_wide_release();
    void ssd_fine_grained_conn();
    void ssd_coarse_grained_conn();

    //-- acquire
    void acquire(Sys&);
    void node_wide_acquire();
    void ssi_fine_grained_conn(Sys&);
    void ssi_coarse_grained_conn(Sys&);
    void ssi_fine_grained_remote(Sys&);
    void ssi_coarse_grained_remote(Sys&);

    int region_chunks;
    std::vector<int> regions;
    void setup_regions(Sys&);
};

ARGO_Sys::~ARGO_Sys()
{
    argo::codelete_array(items_ptr);
}

void ARGO_Sys::alloc_and_init()
{
    items_ptr = argo::conew_array<double>(num_latent * num());

    init();
    init_vectors();
}

void ARGO_Sys::send_item(int i)
{
#ifdef ARGO_SELECTIVE_RELEASE
    commit_index(i);
    process_queue();
#endif
}

void ARGO_Sys::commit_index(int i)
{
#ifdef ARGO_SELECTIVE_RELEASE
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

void ARGO_Sys::sample(Sys& in)
{
    // because of the order
    // movies.sample(users)
    // users.sample(movies)
#ifdef ARGO_VERIFICATION
    if(!name.compare("movs"))
        MPI_Barrier(MPI_COMM_WORLD);
#endif

    // recv necessary items
    {
        BPMF_COUNTER("acquire");
        acquire(in);
    }

    // compute my own chunk
    {
        BPMF_COUNTER("compute");
        Sys::sample(in);
    }

    // send remaining items
    process_queue();

    // w8 release to finish
    MPI_Barrier(MPI_COMM_WORLD);

    // reduce small structs
    reduce_sum_cov_norm();

    // because of the order
    // movies.sample(users)
    // users.sample(movies)
#ifdef ARGO_VERIFICATION
    if(!name.compare("users"))
        acquire(*this);
#endif
}

void ARGO_Sys::release()
{
#if   defined(ARGO_NODE_WIDE_RELEASE)
    node_wide_release();
#elif defined(ARGO_SELECTIVE_RELEASE)

#ifdef FINE_GRAINED_SELECTIVE_RELEASE
    ssd_fine_grained_conn();
#else
    ssd_coarse_grained_conn();
#endif

#endif
}

void ARGO_Sys::acquire(Sys& in)
{
#if   defined(ARGO_NODE_WIDE_ACQUIRE)
    node_wide_acquire();
#elif defined(ARGO_SELECTIVE_ACQUIRE)

#ifndef ARGO_LOCALITY
    setup_regions(in);
#endif

#ifdef FINE_GRAINED_SELECTIVE_ACQUIRE
#ifndef ARGO_LOCALITY
    ssi_fine_grained_conn(in);
#else
    ssi_fine_grained_remote(in);
#endif
#else
#ifndef ARGO_LOCALITY
    ssi_coarse_grained_conn(in);
#else
    ssi_coarse_grained_remote(in);
#endif
#endif

#else
    Sys::sync();
#endif
}

void ARGO_Sys::node_wide_release()
{
    argo::backend::release();
}

void ARGO_Sys::ssd_fine_grained_conn()
{
    int q = queue.size();
    while (q--) {
        auto i = queue.front();
        m.lock(); queue.pop_front(); m.unlock();

        auto offset = i * num_latent;
        auto size = num_latent;
        argo::backend::selective_release(
                items_ptr+offset, size*sizeof(double));
    }
}

void ARGO_Sys::ssd_coarse_grained_conn()
{
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
}

void ARGO_Sys::node_wide_acquire()
{
    argo::backend::acquire();
}

void ARGO_Sys::ssi_fine_grained_conn(Sys& in)
{
    for (int it = 0; it < region_chunks; ++it) {
        int lo = regions.at(it*2);
        int hi = regions.at(it*2+1);

        for(int i = lo; i < hi; ++i)
            if (in.conn(i, procid)) {
                auto offset = i * num_latent;
                auto size = num_latent;
                argo::backend::selective_acquire(
                        in.items_ptr+offset, size*sizeof(double));
            }
    }
}

void ARGO_Sys::ssi_coarse_grained_conn(Sys& in)
{
    for (int it = 0; it < region_chunks; ++it) {
        int lo = regions.at(it*2);
        int hi = regions.at(it*2+1);

        for(int i = lo, c, b; i < hi; i += c+b) {
            c = 0;
            b = 0;
            for (int k = i; k < hi; ++k)
                if (in.conn(k, procid))
                    ++c;
                else {
                    ++b;
                    break;
                }

            if (c > 0) {
                auto offset = i * num_latent;
                auto size = c * num_latent;
                argo::backend::selective_acquire(
                        in.items_ptr+offset, size*sizeof(double));
            }
        }
    }
}

void ARGO_Sys::ssi_fine_grained_remote(Sys& in)
{
    for (int i : in.items_remote) {
        auto offset = i * num_latent;
        auto size = num_latent;
        argo::backend::selective_acquire(
                in.items_ptr+offset, size*sizeof(double));
    }
}

void ARGO_Sys::ssi_coarse_grained_remote(Sys& in)
{
    int lo = 0;
    int hi = in.items_remote.size();

    for (int i = lo, c; i < hi; i += c) {
        c = 1;
        for (int k = i; k < hi-1; ++k, ++c)
            if (!are_items_adjacent(
                        in.items_remote.at(k), in.items_remote.at(k+1)))
                break;

        auto offset = in.items_remote.at(i) * num_latent;
        auto size = c * num_latent;
        argo::backend::selective_acquire(
                in.items_ptr+offset, size*sizeof(double));
    }
}

void ARGO_Sys::setup_regions(Sys& in)
{
    if (nprocs == 1) {
        region_chunks = 1;
        regions.resize(2*region_chunks);
        regions.at(0) = in.from(0);
        regions.at(1) =   in.to(0);
    } else
        if (procid == 0) {
            region_chunks = 1;
            regions.resize(2*region_chunks);
            regions.at(0) = in.from(1);
            regions.at(1) =   in.to(nprocs-1);
        } else if (procid == nprocs-1) {
            region_chunks = 1;
            regions.resize(2*region_chunks);
            regions.at(0) = in.from(0);
            regions.at(1) =   in.to(procid-1);
        } else {
            region_chunks = 2;
            regions.resize(2*region_chunks);
            regions.at(0) = in.from(0);
            regions.at(1) =   in.to(procid-1);
            regions.at(2) = in.from(procid+1);
            regions.at(3) =   in.to(nprocs-1);
        }
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

void ARGO_Sys::init_vectors()
{
#ifdef ARGO_LOCALITY
    for (int k = 0; k < num(); ++k)
        if (argo::get_homenode(
                    items_ptr+k*num_latent) == procid)
            items_local.push_back(k);
        else
            items_remote.push_back(k);

    std::sort(items_local.begin(), items_local.end());
    std::sort(items_remote.begin(), items_remote.end());
#endif
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
