/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include <cmath>
#include <mpi.h>

#include <GASPI.h>
#include <GASPI_Ext.h>

#include "thread_vector.h"

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

namespace GASPI
{
    struct Context {
        Context() 
        {
            SUCCESS_OR_DIE(gaspi_proc_init(GASPI_BLOCK));
        }
        Context(const Context &) = delete;
        Contex operator=(const Context &) = delete;
        ~Context()
        {
            gaspi_proc_term(GASPI_BLOCK);
        }

        gaspi_segment_id_t m_segment_id = gaspi_segment_id_t();

        gaspi_rank_t rank() const
        {
            gaspi_rank_t r;
            gaspi_proc_rank(&r);
            return r;
        }


        gaspi_number_t size() const
        {
            gaspi_number_t s;
            gaspi_group_size(GASPI_GROUP_ALL, &s);
            return s
        }

        gaspi_segment_id_t next_segment_id()
        {
            return m_segment_id++; 
        }
    };

    static std::unique_ptr<Context> m_context = 0;
    static Context &context()
    {
        if (!m_context) m_context = std::make_unique<Context>();
        return *m_context;
    }

    template<typename T>
    struct SharedSegment
    {
        gaspi_segment_id m_segment_id;
        size_t m_num_elem;
        T* m_data;
        gaspi_notify_value_t phase = 1;

        SharedSegment(size_t num_elem)
        : m_segment_id(contex().next_segment_id())
        , m_num_elem(num_elem)
        {
            m_data = (T *)gaspi_malloc(seg_id_cnt, sizeof(T) * num_elem);
        }

        T* data()
        {
            return m_data;
        }

        SharedRef<T>& elem(size_t i)
        {
            return SharedRef<T>{m_segment_id, i};
        }

        const T& elem(size_t i) const
        {
            return m_data[i];
        }

        void notify() const
        {
            for (int k = 0; k < context()::nprocs; k++)
            {
                SUCCESS_OR_RETRY(gaspi_notify(m_segment_id, k, Sys::procid, phase, 0, GASPI_BLOCK));
            }
        }

        void wait() const
        {
            gaspi_notification_id_t id;
            gaspi_notification_t val = 0;
            for (int k = 0; k < context()::nprocs; k++)
            {
                SUCCESS_OR_DIE(gaspi_notify_waitsome(m_segment_id, 0, Sys::nprocs, &id, GASPI_BLOCK));
                SUCCESS_OR_DIE(gaspi_notify_reset(m_segment_id, id, &val));
            }
        }

        virtual bool do_send(size_t, int) = 0;

    };

    template<typename T> 
    struct SharedRef 
    {
        SharedSegment &m_seg;
        size_t m_elem_id;

        operator=(const T val&)
        {
            m_data[i] = val;
            auto offset = m_elem_id * sizeof(T);
            auto size = sizeof(T);
            auto seg_id = seg.m_segment_id;
            for (int k = 0; k < Sys::nprocs; k++)
            {
                if (!m_seg.do_send(m_elem_id, k)) continue;
                SUCCESS_OR_RETRY(gaspi_write(seg_id, offset, k, seg_id, offset, size, 0, GASPI_BLOCK));
            }
        }
    };
}                



struct GASPI_Sys : public Sys, public GASPI::SharedSegment<VectorNd>
{
    //-- c'tor
    GASPI_Sys(std::string name, std::string fname, std::string probename)
        : Sys(name, fname, probename) , GASPI::SharedSegment<VectorNd> {}

    GASPI_Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &P) : Sys(name, M, P) {}
    ~GASPI_Sys();
    virtual void alloc_and_init();

    virtual void send_item(int);
    virtual void sample(Sys &in);
    virtual void sample_hp();

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

};

GASPI_Sys::~GASPI_Sys()
{
    for(int k = 0; k < Sys::nprocs; k++) {
        Sys::cout() << name << "@" << Sys::procid << ": sync_time from " << k << ": " << sync_time[k] << std::endl;
    }
}

void GASPI_Sys::alloc_and_init()
{
    sync();
    init();
}

void GASPI_Sys::send_item(int i)
{
    BPMF_COUNTER("send_item");
    for (int k = 0; k < Sys::nprocs; k++)
    {
        if (!do_send(k)) continue;
        if (!conn(i, k)) continue;
        auto offset = i * num_latent * sizeof(double);
        auto size = num_latent * sizeof(double);
        SUCCESS_OR_RETRY(gaspi_write(items_seg, offset, k, items_seg, offset, size, 0, GASPI_BLOCK));
    }
}

void GASPI_Sys::sample(Sys &in)
{
    {
        BPMF_COUNTER("compute");
        Sys::sample(in);
    }

    {
        BPMF_COUNTER("notify");
        notify();
    }

    reduce_sum_cov_norm();

    {
        BPMF_COUNTER("sync");
        wait();
    }
}

void GASPI_Sys::sample_hp()
{
    { BPMF_COUNTER("compute"); Sys::sample_hp(); }
}



void Sys::Finalize()
{
    gaspi_proc_term(GASPI_BLOCK);
    MPI_Finalize();
}

void Sys::sync()
{
    MPI_Barrier(MPI_COMM_WORLD);
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