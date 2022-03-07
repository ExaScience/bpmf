/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include <cmath>
#include <mpi.h>
#include <mutex>

#include <GASPI.h>
#include <GASPI_Ext.h>

#include "thread_vector.h"

#define SYS GASPI_Sys

static int gaspi_free(int k = 0) {
    gaspi_number_t queue_size, queue_max;
    gaspi_queue_size_max(&queue_max); 
    gaspi_queue_size(k, &queue_size); 
    assert(queue_size <= queue_max);
    //Sys::cout() << "used/max == " << queue_size << "/" << queue_max << std::endl;
    return (queue_max - queue_size);
}

#define SUCCESS_OR_DIE(f...) \
do  { \
  const gaspi_return_t r = f; \
  if (r != GASPI_SUCCESS) { \
    Sys::cout() << "Error: " << #f << "[" << __FILE__ << ":" << __LINE__ << "]: " << r << std::endl;  \
    abort(); \
  } \
} while (0)


static void gaspi_checked_wait(int k = 0)
{
    BPMF_COUNTER("gaspi_wait");
    SUCCESS_OR_DIE(gaspi_wait(k, GASPI_BLOCK));
}

static int gaspi_wait_for_queue(int k = 0) {
    BPMF_COUNTER("wait4queue");
    int free = gaspi_free(k);
    assert(free >= 0);
    while ((free = gaspi_free(k)) == 0) gaspi_checked_wait(k); 
    assert(free > 0);
    return free;
}

#define SUCCESS_OR_RETRY(f...) \
do  { \
  gaspi_return_t r; \
  do { \
      r = f; \
           if (r == GASPI_SUCCESS) ; \
      else if(r == GASPI_QUEUE_FULL) gaspi_wait_for_queue(); \
      else { \
        Sys::cout() << "Error: " << #f << "[" << __FILE__ << ":" << __LINE__ << "]: " << r << std::endl;  \
        sleep(1); \
        abort(); \
      } \
  } while (r == GASPI_QUEUE_FULL); \
} while (0)
 
static double* gaspi_malloc(gaspi_segment_id_t seg, size_t size) {
	// Sys::cout() << "alloc id " << (int)seg << " with size " << (int)size << std::endl;
        SUCCESS_OR_DIE(gaspi_segment_create(seg, size, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));
        void *ptr;
	//Sys::cout() << "ptr: " << &ptr << std::endl;
        SUCCESS_OR_DIE(gaspi_segment_ptr(seg, &ptr));
        return (double*)ptr;
}

struct GASPI_Sys : public Sys 
{
    //-- c'tor
    GASPI_Sys(std::string name, std::string fname, std::string probename) : Sys(name, fname, probename) {}
    GASPI_Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &P) : Sys(name, M, P) {}
    ~GASPI_Sys();
    virtual void alloc_and_init();

    virtual void send_item(int);
    virtual void sample(Sys &in);

    gaspi_segment_id_t items_seg = (gaspi_segment_id_t)-1;

    std::vector<double> sync_time;

    // -- related to gaspi_write throttling
    double send_prob = 1.0; // probability to actually send

    bool do_comm(int from, int to)
    {
        if(from == to) return false;
        if(iter == 0) return true;
        if(send_prob >= 0.9999) return true;
        return randu() < send_prob;
    }

    bool do_send(int to)
    {
        return do_comm(Sys::procid, to);
    }

    //-- process_queue queue with protecting mutex
    std::mutex m;
    std::list<int> queue;
    void process_queue();

};

GASPI_Sys::~GASPI_Sys()
{
    for(int k = 0; k < Sys::nprocs; k++) {
        Sys::cout() << name << "@" << Sys::procid << ": sync_time from " << k << ": " << sync_time[k] << std::endl;
    }
}

void GASPI_Sys::alloc_and_init()
{

    gaspi_number_t max;
    gaspi_queue_size_max(&max); 
    Sys::cout() << "gaspi queue depth: " << max << std::endl;
    gaspi_rw_list_elem_max (&max); 
    Sys::cout() << "gaspi rw list max: " << max << std::endl;

    sync_time.resize(Sys::nprocs);

    static gaspi_segment_id_t seg_id_cnt = 0;
    items_ptr = gaspi_malloc(seg_id_cnt, sizeof(double) * num_latent * num());
    items_seg = seg_id_cnt++;

    sync();

    init();
}

void GASPI_Sys::send_item(int i)
{
    BPMF_COUNTER("send_item");
    m.lock(); queue.push_back(i); m.unlock();

    process_queue();
}

void GASPI_Sys::process_queue() 
{

    int main_thread;
    MPI_Is_thread_main(&main_thread);
    if (!main_thread) return;

    {
        BPMF_COUNTER("process_queue");


        int q = queue.size();
        while (q--) {
            m.lock();
            int i = queue.front();
            queue.pop_front();
            m.unlock();

            for (int k = 0; k < Sys::nprocs; k++)
            {
                if (!do_send(k)) continue;
                if (!conn(i, k)) continue;

                auto offset = i * num_latent * sizeof(double);
                auto size = num_latent * sizeof(double);
                SUCCESS_OR_RETRY(gaspi_write(items_seg, offset, k, items_seg, offset, size, 0, GASPI_BLOCK));
            }
        }
    }
}

void GASPI_Sys::sample(Sys &in)
{
    {
        BPMF_COUNTER("compute");
        Sys::sample(in);
    }

    process_queue();

    {
        BPMF_COUNTER("notify");

        for (int k = 0; k < Sys::nprocs; k++)
        {
            SUCCESS_OR_RETRY(gaspi_notify(items_seg, k, Sys::procid, iter+1, 0, GASPI_BLOCK));
        }

    }

    reduce_sum_cov_norm();

    {
        BPMF_COUNTER("sync");
        auto start = tick();

        for (int k = 0; k < Sys::nprocs; k++)
        {
            gaspi_notification_id_t id;
            gaspi_notification_t val = 0;
            SUCCESS_OR_DIE(gaspi_notify_waitsome(items_seg, 0, Sys::nprocs, &id, GASPI_BLOCK));
            SUCCESS_OR_DIE(gaspi_notify_reset(items_seg, id, &val));
            auto stop = tick();
            sync_time[id] += stop - start;
        }
    }
}

void Sys::Init()
{
#ifdef BPMF_HYBRID_COMM
    int provided;
    MPI_Init_thread(0, 0, MPI_THREAD_SERIALIZED, &provided);
    assert(provided == MPI_THREAD_SERIALIZED);
    MPI_Comm_rank(MPI_COMM_WORLD, &Sys::procid);
    MPI_Comm_size(MPI_COMM_WORLD, &Sys::nprocs);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
#endif

    gaspi_rank_t rank;
    SUCCESS_OR_DIE(gaspi_proc_init(GASPI_BLOCK));
    gaspi_proc_rank(&rank);
    if (Sys::procid >= 0)
    {
        assert(rank == Sys::procid);
    }
    else
    {
        Sys::procid = rank;
    }

    gaspi_number_t size;
    gaspi_group_size(GASPI_GROUP_ALL, &size);
    if (Sys::nprocs > 0)
    {
        assert(Sys::nprocs == (int)size);
    }
    else
    {
        assert(size > 0);
        Sys::nprocs = size;
    }
}

void Sys::Finalize()
{
    gaspi_proc_term(GASPI_BLOCK);

#ifdef BPMF_HYBRID_COMM
    MPI_Finalize();
#endif
}

void Sys::sync()
{
#ifdef BPMF_HYBRID_COMM
    MPI_Barrier(MPI_COMM_WORLD);
#else
    gaspi_barrier(GASPI_GROUP_ALL,GASPI_BLOCK);
#endif
}

void Sys::Abort(int err)
{
    gaspi_proc_term(GASPI_BLOCK);
    exit(err);
}

void Sys::reduce_sum_cov_norm()
{
    BPMF_COUNTER("reduce_sum_cov_norm");
    MPI_Allreduce(MPI_IN_PLACE, sum.data(), num_latent, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, cov.data(), num_latent * num_latent, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}