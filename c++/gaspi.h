/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#ifdef BPMF_HYBRID_COMM
#include <mpi.h>
#endif

#include <GASPI.h>
#include <GASPI_Ext.h>

#include <thread>

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
  assert(gaspi_free(0) >= 0); \
  const gaspi_return_t r = f; \
  if (r == GASPI_ERROR) { \
    Sys::cout() << "Error: " << #f << "[" << __FILE__ << ":" << __LINE__ << "]: " << r << std::endl;  \
    sleep(1); \
    abort(); \
  } \
} while (0);

static double* gaspi_malloc(gaspi_segment_id_t seg, size_t size) {
	Sys::cout() << "alloc id " << (int)seg << " with size " << (int)size << std::endl;
        SUCCESS_OR_DIE(gaspi_segment_create(seg, size, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));
        void *ptr;
	Sys::cout() << "ptr = " << &ptr << std::endl;
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

    virtual void send_items(int from, int to);
    virtual void bcast_items();
    virtual void actual_send(int from, int to);
    virtual void sample(Sys &in);
    virtual void sample_hp();

    gaspi_segment_id_t items_seg = -1;
    gaspi_segment_id_t sum_seg = -1;
    gaspi_segment_id_t cov_seg = -1;
    gaspi_segment_id_t norm_seg = -1;

    std::vector<double> sync_time;
    unsigned nsim;

    //-- process_queue queue with protecting mutex
    working_mutex m;
    std::list<std::pair<int,int>> queue;
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

    sync();

    init();
}

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

void GASPI_Sys::send_items(int from, int to)
{
    m.lock(); queue.push_back(std::make_pair(from,to)); m.unlock();
    process_queue();
}

void GASPI_Sys::actual_send(int from, int to)
{
    BPMF_COUNTER("send_items");

    int free = gaspi_wait_for_queue(0);

    for(int i = from; i < to; ++i) for(int k = 0; k < Sys::nprocs; k++) {
        auto offset = i * num_feat * sizeof(double);
        auto size = num_feat * sizeof(double);
        SUCCESS_OR_DIE(gaspi_write(items_seg, offset, k, items_seg, offset, size, 0, GASPI_BLOCK));
        assert((free - 1) == gaspi_free(0));
        if (--free <= 0) free = gaspi_wait_for_queue(0);
    }
}

void GASPI_Sys::process_queue() 
{
    if (!Sys::isMasterThread()) return;

    {
        BPMF_COUNTER("process_queue");

        int q = queue.size();
        int from, to;
        while (q--) {
            m.lock();
            std::tie(from,to) = queue.front();
            queue.pop_front();
            m.unlock();
            actual_send(from,to);
	}
    }
}

static void gaspi_bcast(int seg, int offset, int size) {
    offset *= sizeof(double);
    size *= sizeof(double);
    int free = gaspi_wait_for_queue(0);
    for(int k = 0; k < Sys::nprocs; k++) {
        if (k == Sys::procid) continue;
        SUCCESS_OR_DIE(gaspi_write(seg, offset, k, seg, offset, size, 0, GASPI_BLOCK));
	assert((free - 1) == gaspi_free(0));
	if (--free <= 0) free = gaspi_wait_for_queue(0);
    }
}

void GASPI_Sys::sample(Sys &in)
{
   { BPMF_COUNTER("compute"); Sys::sample(in); }

   process_queue();

   {
       BPMF_COUNTER("bcast");
       auto base = Sys::procid;
       gaspi_bcast(sum_seg,  base * num_feat, num_feat);
       gaspi_bcast(cov_seg,  base * num_feat * num_feat, num_feat * num_feat);
       gaspi_bcast(norm_seg, base, 1);

       int free = gaspi_wait_for_queue(0);
       for(int k = 0; k < Sys::nprocs; k++) {
           if (k == Sys::procid) continue;
           SUCCESS_OR_DIE(gaspi_notify(norm_seg, k, Sys::procid, nsim, 0, GASPI_BLOCK));
           assert((free - 1) == gaspi_free(0));
           if (--free <= 0) free = gaspi_wait_for_queue(0);
       }  
   }

   { 
       BPMF_COUNTER("sync");
       auto start = tick();

       for(int k = 0; k < Sys::nprocs - 1; k++) {
           gaspi_notification_id_t first_id;
           gaspi_notification_t val = 0;
           SUCCESS_OR_DIE(gaspi_notify_waitsome(norm_seg, 0, Sys::nprocs, &first_id, GASPI_BLOCK));
           SUCCESS_OR_DIE(gaspi_notify_reset (norm_seg, first_id, &val));
           auto stop = tick();
           sync_time[first_id] += stop - start;
       }
   }
}

void GASPI_Sys::bcast_items()
{
#ifdef BPMF_HYBRID_COMM
    for(int i = 0; i < num(); i++) {
        MPI_Bcast(items().col(i).data(), num_feat, MPI_DOUBLE, proc(i), MPI_COMM_WORLD);
    }
#endif
}

void GASPI_Sys::sample_hp()
{
    { BPMF_COUNTER("compute"); Sys::sample_hp(); }
    nsim++;
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
    gaspi_config_t c;
    gaspi_config_get(&c);
    c.queue_size_max = 4096;
    c.queue_num = 1;
    gaspi_config_set(c);
    gaspi_proc_init(GASPI_BLOCK);
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
    gaspi_group_size(GASPI_GROUP_ALL,&size);
    if (Sys::nprocs > 0)
    {
	    assert(Sys::nprocs == (int)size);
    }
    else
    {
	    Sys::nprocs = size;
    }
}

void Sys::Finalize()
{
#ifdef BPMF_PROFILING
    perf_data.print();
#endif

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
