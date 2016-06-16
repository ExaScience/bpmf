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

#include "bpmf.h"

using namespace std;
using namespace Eigen;

#ifdef BPMF_GPI_COMM
#include "bpmf_gaspi.h"
#elif defined(BPMF_MPI_COMM)
#include "bpmf_mpi.h"
//#include "bpmf_mpi_bcast.h"
#else 
#include "bpmf_nocomm.h"
#endif

int main(int argc, char *argv[])
{
    Sys::Init();

    int ch;
    string fname;
    string probename;
    int nthrds = -1;
    int nsims = 20;
    int burnin = 5;

    while((ch = getopt(argc, argv, "n:t:r:p:i:")) != -1)
    {
        switch(ch)
        {
            case 'i': nsims = atoi(optarg); break;
            case 't': nthrds = atoi(optarg); break;
            case 'n': fname = optarg; break;
            case 'p': probename = optarg; break;
            case '?':
            default:
                      cout << "Usage: " << argv[0] << " [-t <threads>] " 
                          << "[ -i <niters> ] -n <samples.mtx> -p <probe.mtx>"
                          << endl;
                      Sys::Abort(1);
        }
    }

    if (fname.empty() || probename.empty()) { 
        cout << "Usage: " << argv[0] << " [-t <threads>] [-i <iterations>]" << " -n <samples.mtx> -p <probe.mtx>" << endl;
        Sys::Abort(1);
    }

    SYS movies("movs", fname);
    SYS users("users", movies.M);
    movies.alloc_and_init(users);
    users.alloc_and_init(movies);
    users.build_conn(movies);
    movies.build_conn(users);

    Eval eval(probename, movies.mean_rating, burnin);
    Sys::SetupThreads(nthrds);

    long double average_sampling_sec =0;
    if(Sys::procid == 0)
    {
        cout << "num_feat: " << num_feat<<endl;
        cout << "nprocs: " << Sys::nprocs << endl;
        cout << "nthrds: " << Sys::nthrds << endl;
    }


    auto begin = tick();

    for(int i=0; i<nsims; ++i) {
        BPMF_COUNTER("main");
        auto start = tick();

        movies.sample_hp();
        { BPMF_COUNTER("movies"); movies.sample(users); }
        users.sample_hp();
        { BPMF_COUNTER("users");  users.sample(movies); }
        { BPMF_COUNTER("eval");   eval.predict(i, movies, users); }

        auto stop = tick();
        double samples_per_sec = (users.num() + movies.num()) / (stop - start);
        eval.print(samples_per_sec, sqrt(users.aggr_norm()), sqrt(movies.aggr_norm()));
        average_sampling_sec += samples_per_sec;
    }

    Sys::sync();

    auto end = tick();
    auto elapsed = end - begin;

    if (Sys::procid == 0) {
        cout << "Total time: " << elapsed <<endl <<flush;
        cout << "Average Samples/sec: " << average_sampling_sec / nsims << endl <<flush;
    }

    Sys::Finalize();

    return 0;
}
