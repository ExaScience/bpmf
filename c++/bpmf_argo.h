/*
 * Copyright (c) 2021-2022, etascale
 * All rights reserved.
 */

#include <mpi.h>
#include "argo.hpp"

#if defined(ARGO_NO_RELEASE) && \
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

    //-- virtual functions
    virtual void sample(Sys&);
    virtual void send_item(int);
    virtual void alloc_and_init();

    //-- process_queue queue with protecting mutex
    std::mutex m;
    std::list<int> queue;
    void commit_index(int);
    void process_queue();

    //-- release functions
    void release();
    void node_wide_release();
    void fine_grained_release();
    void coarse_grained_release();

    //-- acquire functions
    void acquire();
    void node_wide_acquire();
    void fine_grained_acquire();
    void coarse_grained_acquire();

    //-- helper functions
    void pop_front(int);
    static bool are_items_adjacent(int, int);
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
#ifdef ARGO_SELECTIVE_RELEASE
    commit_index(i);
    process_queue();
#endif
}

void ARGO_Sys::commit_index(int i)
{
    bool push = 0;
    for(int k = 0; k < nprocs; ++k)
        if (conn(i, k)) {
            push = 1;
            break;
        }
    // if no-one needs it, don't push
    if (!push) return;

    m.lock(); queue.push_back(i); m.unlock();
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

    // compute my own chunk
    {
        BPMF_COUNTER("compute");
        Sys::sample(in);
    }

    // send remaining items
    process_queue();

    // w8 release to finish
    {
        BPMF_COUNTER("barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }

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
    node_wide_release();
#elif defined(ARGO_SELECTIVE_RELEASE)

#ifdef FINE_GRAINED_SELECTIVE_RELEASE
    fine_grained_release();
#else
    coarse_grained_release();
#endif

#endif
}

void ARGO_Sys::acquire()
{
#if   defined(ARGO_NODE_WIDE_ACQUIRE)
    node_wide_acquire();
#elif defined(ARGO_SELECTIVE_ACQUIRE)

#ifdef FINE_GRAINED_SELECTIVE_ACQUIRE
    fine_grained_acquire();
#else
    coarse_grained_acquire();
#endif

#else
    Sys::sync();
#endif
}

void ARGO_Sys::node_wide_release()
{
    argo::backend::release();
}

void ARGO_Sys::fine_grained_release()
{
    int q = queue.size();
    while (q--) {
        const auto i = queue.front();
        m.lock(); queue.pop_front(); m.unlock();

        const auto offset = i * num_latent;
        const auto size = num_latent;
        argo::backend::selective_release(
                items_ptr+offset, size*sizeof(double));
    }
}

void ARGO_Sys::coarse_grained_release()
{
    m.lock();
    int q = queue.size();
    queue.sort();
    m.unlock();

    while (q) {
        const auto i = queue.front();
              auto l = queue.begin();
              auto r = std::next(l);

        int c;
        for (c = 1; c < q; ++c, ++l, ++r)
            if (!are_items_adjacent(*l, *r))
                break;
        m.lock(); pop_front(c); m.unlock();

        const auto offset = i * num_latent;
        const auto size = c * num_latent;
        argo::backend::selective_release(
                items_ptr+offset, size*sizeof(double));

        q -= c;
    }
}

void ARGO_Sys::node_wide_acquire()
{
    argo::backend::acquire();
}

void ARGO_Sys::fine_grained_acquire()
{
    const int lo = from(0);
    const int hi =   to(nprocs-1);
    const int my_lo = from(procid);
    const int my_hi =   to(procid);

    for(int i = lo; i < hi; ++i) {
        // if currently in my region
        if (i >= my_lo && i < my_hi) {
            // jump to next region
            i += my_hi - my_lo - 1;
            continue;
        }

        if (conn(i, procid)) {
            const auto offset = i * num_latent;
            const auto size = num_latent;
            argo::backend::selective_acquire(
                    items_ptr+offset, size*sizeof(double));
        }
    }
}

void ARGO_Sys::coarse_grained_acquire()
{
    const int lo = from(0);
    const int hi =   to(nprocs-1);
    const int my_lo = from(procid);
    const int my_hi =   to(procid);

    auto ssi = [&](
            const auto idx, const auto num) {
        const auto offset = idx * num_latent;
        const auto size = num * num_latent;
        argo::backend::selective_acquire(
                items_ptr+offset, size*sizeof(double));
    };

    int s{-1}, c{0};
    for(int i = lo; i < hi; ++i) {
        // if currently in my region
        if (i >= my_lo && i < my_hi) {
            // ssi leftover items
            if (c)
                ssi(s, c);
            // reset vars
            s = -1;
            c =  0;
            // jump to next region
            i += my_hi - my_lo - 1;
            continue;
        }

        // if conn between item-node
        if (conn(i, procid)) {
            // set stamp
            if (s == -1)
                s = i;
            // inc count
            ++c;
            // corner case
            if (i == hi-1)
                ssi(s, c);
        } else {
            // streak broke
            if (c > 0)
                ssi(s, c);
            // reset vars
            s = -1;
            c =  0;
        }
    }
}

bool ARGO_Sys::are_items_adjacent(int l, int r)
{
    return (l == r-1);
}

void ARGO_Sys::pop_front(int elems)
{
    const auto beg = queue.begin();
    const auto end = std::next(beg, elems);
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
