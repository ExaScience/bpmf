/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#include <GASPI.h>

#define SYS GASPI_Sys

#define SUCCESS_OR_DIE(f...) \
do  { \
  const gaspi_return_t r = f; \
  if (r == GASPI_ERROR) { \
    gaspi_printf ("Error: '%s' [%s:%i]: %i\n", #f, __FILE__, __LINE__, r); \
    sleep(1); \
    abort(); \
  } \
} while (0);

static double* gaspi_malloc(gaspi_segment_id_t seg, size_t size) {
        SUCCESS_OR_DIE(gaspi_segment_create(seg, size, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));
        void *ptr;
        SUCCESS_OR_DIE(gaspi_segment_ptr(seg, &ptr));
        return (double*)ptr;
}

struct GASPI_Sys : public Sys 
{
    //-- c'tor
    GASPI_Sys(std::string name, std::string fname) : Sys(name, fname) {}
    GASPI_Sys(std::string name, const SparseMatrixD &M) : Sys(name, M) {}
    ~GASPI_Sys();
    virtual void alloc_and_init(const Sys &);

    virtual void send_items(int from, int to);
    virtual void send_item(int i);
    virtual void sample(Sys &in);
    virtual void sample_hp();

    gaspi_segment_id_t items_seg = -1;
    gaspi_segment_id_t sum_seg = -1;
    gaspi_segment_id_t cov_seg = -1;
    gaspi_segment_id_t norm_seg = -1;

    std::vector<double> sync_time;
    unsigned nsim;
};

GASPI_Sys::~GASPI_Sys()
{
    for(int k = 0; k < Sys::nprocs; k++) {
        std::cout << "@" << Sys::procid << ": sync_time from " << k << ": " << sync_time[k] << std::endl;
    }
}

void GASPI_Sys::alloc_and_init(const Sys &other)
{
    nsim = 1;
    sync_time.resize(Sys::nprocs);

    static gaspi_segment_id_t seg_id_cnt = 0;
    items_ptr = gaspi_malloc(seg_id_cnt, sizeof(double) * num_feat * num());
    items_seg = seg_id_cnt++;
    sum_ptr = gaspi_malloc(seg_id_cnt, sizeof(double) * num_feat * Sys::nprocs);
    sum_seg = seg_id_cnt++;
    cov_ptr = gaspi_malloc(seg_id_cnt, sizeof(double) * num_feat * num_feat * Sys::nprocs);
    cov_seg = seg_id_cnt++;
    norm_ptr = gaspi_malloc(seg_id_cnt, sizeof(double) * Sys::nprocs);
    norm_seg = seg_id_cnt++;

    world().sync();

    init(other);
}

static void gaspi_checked_wait(int k = 0)
{
    BPMF_COUNTER("gaspi_wait");
    SUCCESS_OR_DIE(gaspi_wait(k, GASPI_BLOCK));
}

static void gaspi_wait_for_queue(int k = 0) {
    BPMF_COUNTER("wait4queue");
    gaspi_number_t queue_size, queue_max;
    gaspi_queue_size_max(&queue_max); 
    gaspi_queue_size(k, &queue_size); 
    if(queue_size > queue_max - 1) gaspi_checked_wait(k); 
}

void GASPI_Sys::send_items(int from, int to)
{
    BPMF_COUNTER("gaspi_write");
    gaspi_wait_for_queue(0);
    for(int k = 0; k < Sys::nprocs; k++) {
        if (k == Sys::procid) continue;
        auto offset = from * num_feat * sizeof(double);
        auto size = num_feat * sizeof(double) * (to -  from);
        SUCCESS_OR_DIE(gaspi_write(items_seg, offset, k, items_seg, offset, size, 0, GASPI_BLOCK));
    }
}

void GASPI_Sys::send_item(int i)
{
    gaspi_wait_for_queue(0);
    for(int k = 0; k < Sys::nprocs; k++) {
        if (!conn(i).test(k)) continue;
        auto offset = i * num_feat * sizeof(double);
        auto size = num_feat * sizeof(double);
        SUCCESS_OR_DIE(gaspi_write(items_seg, offset, k, items_seg, offset, size, 0, GASPI_BLOCK));
    }
}


static void gaspi_bcast(int seg, int offset, int size) {
    offset *= sizeof(double);
    size *= sizeof(double);
    for(int k = 0; k < Sys::nprocs; k++) {
        if (k == Sys::procid) continue;
        SUCCESS_OR_DIE(gaspi_write(seg, offset, k, seg, offset, size, 0, GASPI_BLOCK));
    }
}

void GASPI_Sys::sample(Sys &in)
{
   { BPMF_COUNTER("compute"); Sys::sample(in); }

   {
       BPMF_COUNTER("bcast");
       auto base = Sys::procid;
       gaspi_bcast(sum_seg,  base * num_feat, num_feat);
       gaspi_bcast(cov_seg,  base * num_feat * num_feat, num_feat * num_feat);
       gaspi_bcast(norm_seg, base, 1);

       for(int k = 0; k < Sys::nprocs; k++) {
           if (k == Sys::procid) continue;
           SUCCESS_OR_DIE(gaspi_notify(norm_seg, k, Sys::procid, nsim, 0, GASPI_BLOCK));
       }  
   }
}


void GASPI_Sys::sample_hp()
{
    { BPMF_COUNTER("compute"); Sys::sample_hp(); }


    auto start = tick();
    if (nsim > 1) { 
        BPMF_COUNTER("sync");
        for(int k = 0; k < Sys::nprocs - 1; k++) {
            gaspi_notification_id_t first_id;
            gaspi_notification_t val = 0;
            SUCCESS_OR_DIE(gaspi_notify_waitsome(norm_seg, 0, Sys::nprocs, &first_id, GASPI_BLOCK));
            SUCCESS_OR_DIE(gaspi_notify_reset (norm_seg, first_id, &val));
            auto stop = tick();
            sync_time[first_id] += stop - start;
        }
    }

    nsim++;
}
