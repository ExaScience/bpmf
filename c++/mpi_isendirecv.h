/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include <mpi.h>

#include <list>
#include <array>
#include <sstream>
#include <mutex>

const static int CS = 100;
const static int NC  = 5;

template<typename T>
struct SendRecvBuffer
{
    const int peer;
    const int total;

    int num;
    int pos;
    int posted;

    
    std::array<std::vector<T>, NC> data;
    std::array<MPI_Request, NC> req;

    typedef std::list<int> lst;
    lst outstanding, empty, avail;

    int has(const lst &s) const { return !s.empty(); }
    int first(const lst &s) const { return s.front(); }

    int pop(lst &s) { assert(!s.empty()); int i = first(s); s.pop_front(); return i; }
    int push(lst &s, int v) { s.push_back(v); return v;}

    int mark_post()   { return push(outstanding, pop(empty)); } 
    int mark_arrive() { return push(avail, pop(outstanding)); } 
    int mark_sent()   { return push(empty, pop(outstanding)); } 
    int mark_free()   { return push(empty, pop(avail)); } 

    SendRecvBuffer(int p, int t) : peer(p), total(t), num(0), pos(0), posted(0)
    {
        for(int i=0; i<NC; ++i) {
            data[i].resize(CS);
            empty.push_back(i);
        }
    }

    virtual ~SendRecvBuffer() {
        assert(num == total);
    }

    bool has_data() {
        if (done()) return false;
        if (has(avail)) return true;
        test();
        return has(avail);
    }

    bool done() { return num == total; }

    virtual void wait() = 0;
    void mpi_wait() {
        BPMF_COUNTER("MPI_Wait");
        assert(has(outstanding));
        MPI_Wait(&req.at(first(outstanding)), MPI_STATUS_IGNORE);
    }

    virtual bool test() = 0;
    bool mpi_test() {
        int flag;
        if (!has(outstanding)) return false;
        BPMF_COUNTER("MPI_Test");
        MPI_Test(&req.at(first(outstanding)), &flag, MPI_STATUS_IGNORE);
        return flag;
    }

    void wait_all() {
        while (has(outstanding)) this->wait();
    }

    void put(const T &d) {
        assert(has(empty));
//        log() <<  ":put: num: " << num << " of " << total << std::endl;
        data[first(empty)].at(pos) = d;
        pos++;
        assert(num < total);
        num++;
        if (pos == CS || num == total) mpi_isend();
    }

    void mpi_isend() {
        assert(has(empty));
        BPMF_COUNTER("MPI_Isend");
//        log() <<  ": mpi_isend: num: " << num << " of " << total << std::endl;
        auto p = data.at(first(empty)).data();
        auto s = &req.at(first(empty));
        MPI_Isend(p, CS * sizeof(T), MPI_BYTE, peer, 0, MPI_COMM_WORLD, s);
        mark_post();
        pos = 0;
        if (!has(empty)) wait(); 
        else test();
    }

    const T get() {
        assert(has(avail));
        assert(num < total);
        if (!has(avail)) wait();
        auto p = data[first(avail)].at(pos);
        pos = (pos + 1) % CS;
        if (pos == 0) { mark_free(); mpi_irecv(); }
//        log() << ":get: num: " << num << " of " << total << std::endl;
        num++;
        return p;
    }

    void mpi_irecv() {
        BPMF_COUNTER("MPI_Irecv");
        int total_chunks  = (int) ceil( (float)total / CS );
        while (has(empty) && posted < total_chunks) {
            int c = mark_post();
            auto p = data.at(c).data();
            auto s = &req.at(c);
            MPI_Irecv(p, CS * sizeof(T), MPI_BYTE, peer, 0, MPI_COMM_WORLD, s);
            posted++;
        }
//        log() << ": posted " << posted << " irecv\n";
    }


};

template<typename T>
struct SendBuffer : public SendRecvBuffer<T>
{
    SendBuffer(int p, int t) : SendRecvBuffer<T>(p,t) {}
    ~SendBuffer() { this->wait_all(); }
    void wait() { this->mpi_wait(); this->mark_sent(); }
    bool test() { bool b = this->mpi_test(); if (b) this->mark_sent(); return b;}
};

template<typename T>
struct RecvBuffer : public SendRecvBuffer<T>
{
    RecvBuffer(int p, int t) : SendRecvBuffer<T>(p,t) { this->mpi_irecv(); }
    ~RecvBuffer() { this->wait_all(); }
    void wait() { this->mpi_wait(); this->mark_arrive(); }
    bool test() { bool b = (this->mpi_test()); if (b) this->mark_arrive(); return b;}
};

struct MPI_Sys : public Sys 
{
    //-- c'tor
    MPI_Sys(std::string name, std::string fname, std::string probename) : Sys(name, fname, probename) {}
    MPI_Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &P) : Sys(name, M, P) {}

    //-- virtuals
    virtual void sample(Sys &in);
    virtual void send_item(int);
    virtual void alloc_and_init();

    // common 
    void reduce_sum_cov_norm();

    //-- local status
    typedef std::pair<int,VectorNd> ElBuf;
    std::vector<SendBuffer<ElBuf> *> sb;
    std::vector<RecvBuffer<ElBuf> *> rb;

    //-- process_queue queue with protecting mutex
    std::mutex m;
    std::list<int> queue;
    void process_queue();
};

void MPI_Sys::sample(Sys &in)
{
    //Sys::cout() << Sys::procid << ": ------------ creating buffers --------------\n";
    for(int i=0; i<Sys::nprocs; ++i) {
        sb.push_back(new SendBuffer<ElBuf>(i, send_count(i)));
        rb.push_back(new RecvBuffer<ElBuf>(i, recv_count(i)));
    }

    //Sys::cout() << Sys::procid << ": ------------ start compute --------------\n";

    { BPMF_COUNTER("compute"); Sys::sample(in); }


    //Sys::cout() << Sys::procid << ": ------------ compute done --------------\n";
    //Sys::cout() << Sys::procid << ": compute done, still " << queue.size() << " items in queue\n";
    process_queue();
    //Sys::cout() << Sys::procid << ": process queue done, queue empty (" << queue.size() << ")\n";

    {
        BPMF_COUNTER("waitall"); 
        // do all remaining recvs
        bool all_done;
        do {
            all_done = true;
            for(auto b : rb) {
                while (b->has_data()) {
                    auto el = b->get();
                    items().col(el.first) = el.second;
                }
                all_done = all_done && b->done();
            }
        } while (!all_done);
    }

    //Sys::cout() << Sys::procid << ": ------------ cleaning buffers --------------\n";

    for(auto b : sb) { delete b; } sb.clear();
    for(auto b : rb) { delete b; } rb.clear();
    
    //Sys::cout() << Sys::procid << ": ------------ doing bcast --------------\n";
    reduce_sum_cov_norm();
}

void MPI_Sys::send_item(int i)
{
    m.lock(); queue.push_back(i); m.unlock();

    process_queue();
}

void MPI_Sys::process_queue() 
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

            // do some sends...
            for(int k = 0; k < Sys::nprocs; k++) {
                if (!conn(i, k)) continue;
                sb.at(k)->put(std::make_pair(i, items().col(i)));
            }

            // do some recvs
            for(auto b : rb) {
                while (b->has_data()) {
                    auto el = b->get();
                    items().col(el.first) = el.second;
                }
            }
        }
    }
}


#include "mpi_common.h"
