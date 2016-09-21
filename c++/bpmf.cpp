/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <unistd.h>

#include <unsupported/Eigen/SparseExtra>

#include "bpmf.h"

using namespace std;
using namespace Eigen;

#ifdef BPMF_GPI_COMM
#include "gaspi.h"
#elif defined(BPMF_MPI_PUT_COMM)
#define BPMF_MPI_COMM
#include "mpi_put.h"
#elif defined(BPMF_MPI_BCAST_COMM)
#define BPMF_MPI_COMM
#include "mpi_bcast.h"
#elif defined(BPMF_MPI_ISENDIRECV_COMM)
#define BPMF_MPI_COMM
#include "mpi_isendirecv.h"
#elif defined(BPMF_NO_COMM)
#include "nocomm.h"
#else
#error no comm include
#endif

const char *usage = "Usage: bpmf [-t <threads>] [-i <niters>] -n <samples.mtx> -p <probe.mtx> [-u <u.mtx>] [-v <v.mtx>] [-o <pred.mtx>] [-s <m2.mtx>]\n";

int main(int argc, char *argv[])
{
  Sys::Init();
  {
    int ch;
    string fname, probename, uname, vname, oname, sname;
    int nthrds = -1;
    bool redirect = false;
    Sys::nsims = 20;
    Sys::burnin = 5;
    Sys::grain_size = 1;
    
 
    while((ch = getopt(argc, argv, "krn:t:p:i:g:w:u:v:o:s:")) != -1)
    {
        switch(ch)
        {
            case 'i': Sys::nsims = atoi(optarg); break;
            case 'g': Sys::grain_size = atoi(optarg); break;
            case 't': nthrds = atoi(optarg); break;
            case 'n': fname = optarg; break;
            case 'p': probename = optarg; break;
            case 'u': uname = optarg; break;
            case 'v': vname = optarg; break;
            case 'o': oname = optarg; break;
            case 's': sname = optarg; break;

            case 'r': redirect = true; break;
            case 'k': Sys::permute = false; break;
            case '?':
            default : std::cout << usage; Sys::Abort(1);
        }
    }

    if (Sys::nprocs >1 || redirect) {
        std::stringstream ofname;
        ofname << "bpmf_" << Sys::procid << ".out";
        Sys::os = new std::ofstream(ofname.str());
    } else {
        Sys::os = &std::cout;
    }

    if (fname.empty() || probename.empty()) { 
        Sys::cout() << usage;
        Sys::Abort(1);
    }

    SYS movies("movs", fname, probename);
    SYS users("users", movies.M, movies.Pavg);

    movies.alloc_and_init();
    users.alloc_and_init();

    movies.assign(users);
    users.assign(movies);
    movies.assign(users);
    users.assign(movies);

    users.build_conn(movies);
    movies.build_conn(users);
    assert(movies.nnz() == users.nnz());

    Sys::SetupThreads(nthrds);

    long double average_items_sec = .0;
    long double average_ratings_sec = .0;
    
    char name[1024];
    gethostname(name, 1024);
    Sys::cout() << "hostname: " << name << endl;
    Sys::cout() << "pid: " << getpid() << endl;
    if (getenv("PBS_JOBID")) Sys::cout() << "jobid: " << getenv("PBS_JOBID") << endl;
 
    if(Sys::procid == 0)
    {
        Sys::cout() << "num_feat: " << num_feat<<endl;
        Sys::cout() << "nprocs: " << Sys::nprocs << endl;
        Sys::cout() << "nthrds: " << Sys::nthrds << endl;
        Sys::cout() << "nsims: " << Sys::nsims << endl;
        Sys::cout() << "burnin: " << Sys::burnin << endl;
        Sys::cout() << "grain_size: " << Sys::grain_size << endl;
    }

    Sys::sync();

    auto begin = tick();

    for(int i=0; i<Sys::nsims; ++i) {
        BPMF_COUNTER("main");
        auto start = tick();

        movies.sample_hp();
        { BPMF_COUNTER("movies"); movies.sample(users); }
        users.sample_hp();
        { BPMF_COUNTER("users");  users.sample(movies); }

#ifndef BPMF_ONLY_COMM
        { BPMF_COUNTER("eval");   movies.predict(users); }
#endif

        auto stop = tick();
        double items_per_sec = (users.num() + movies.num()) / (stop - start);
        double ratings_per_sec = (users.nnz()) / (stop - start);
        movies.print(items_per_sec, ratings_per_sec, sqrt(users.aggr_norm()), sqrt(movies.aggr_norm()));
        average_items_sec += items_per_sec;
        average_ratings_sec += ratings_per_sec;
    }

    Sys::sync();

    auto end = tick();
    auto elapsed = end - begin;

    //-- if we need to generate output files, collect all data on proc 0
    if (sname.size() || oname.size() || uname.size() || vname.size()) {
        users.bcast_items();
        movies.bcast_items();
        movies.predict(users, true);
    }

    if (Sys::procid == 0) {
        Sys::cout() << "Total time: " << elapsed <<endl <<flush;
        Sys::cout() << "Average items/sec: " << average_items_sec / movies.iter << endl <<flush;
        Sys::cout() << "Average ratings/sec: " << average_ratings_sec / movies.iter << endl <<flush;

        if (oname.size()) { saveMarket(movies.Pavg, oname); }
        if (sname.size()) { saveMarket(movies.Pm2, sname); }
        if (uname.size()) { std::ofstream os(uname); os << users.items(); }
        if (vname.size()) { std::ofstream os(vname); os << movies.items(); }


    }
  }
  Sys::Finalize();
  if (Sys::nprocs >1) delete Sys::os;


   return 0;
}
