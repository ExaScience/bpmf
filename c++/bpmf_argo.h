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
    bool is_item_local(void*);
    bool are_items_adjacent(int, int);
    void pop_front(int);
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
    auto offset = i * num_latent;

    // push to queue only non-local items
    if (!is_item_local(items_ptr+offset))
    {
        bool push = 0;
        for(int k = 0; k < Sys::nprocs; ++k)
            // can filter ourselves out?
            if (conn(i, k)) {
                push = 1;
                break;
            }
        // if no-one needs it, don't push
        if (!push) return;
    } else
        return;

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

#ifdef FINE_GRAINED_SELECTIVE_RELEASE
        int q = queue.size();
        while (q--) {
            auto i = queue.front();
            m.lock(); queue.pop_front(); m.unlock();

            auto offset = i * num_latent;
            auto size = num_latent;
            argo::backend::selective_release(items_ptr+offset, size*sizeof(double));
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
                if (!are_items_adjacent(*l, *r)) break;
            m.lock(); pop_front(c); m.unlock();

            auto offset = i * num_latent;
            auto size = c * num_latent;
            argo::backend::selective_release(items_ptr+offset, size*sizeof(double));

            q -= c;
        }
#endif

#endif
    }
}

void ARGO_Sys::sample(Sys &in)
{
    {
        BPMF_COUNTER("compute");
        Sys::sample(in);
    }

    // send remaining
    process_queue();

    // reduce small structs
    reduce_sum_cov_norm();

    // argodsm node-acquire
    {
        BPMF_COUNTER("acquire");
        argo::backend::acquire;
    }
}

bool ARGO_Sys::is_item_local(void *addr)
{
    return (argo::get_homenode(addr) == procid);
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
    argo::init(128*1024*1024UL,
               128*1024*1024UL);

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
