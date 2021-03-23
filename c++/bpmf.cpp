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

#include "error.h"
#include "io.h"
#include "bpmf.h"

void usage() 
{
    std::cout << "Usage: bpmf -n <MTX> -p <MTX> [-o DIR/] [-i N] [-b N] [-f N] [-krv] [-t N] [-m MTX,MTX] [-l MTX,MTX]\n"
                << "\n"
                << "Paramaters: \n"
                << "  -n MTX: Training input data\n"
                << "  -p MTX: Test input data\n"
                << "  [-o DIR]: Output directory for model and predictions\n"
                << "  [-i N]: Number of total iterations\n"
                << "  [-b N]: Number of burnin iterations\n"
                << "  [-f N]: Frequency to send model other nodes (in #iters)\n"
                << "  [-a F]: Noise precision (alpha)\n"
                << "\n"
                << "  [-k]: Do not optimize item to node assignment\n"
                << "  [-r]: Redirect stdout to file\n"
                << "  [-v]: Output all samples\n"
                << "  [-t N]: Number of OpenMP threads to use.\n"
                << "\n"
                << "  [-l MTX,MTX]: propagated posterior mu and Lambda matrices for U\n"
                << "  [-m MTX,MTX]: propagated posterior mu and Lambda matrices for V\n"
                << "\n"
                << "Matrix Formats:\n"
                << "  *.mtx: Sparse or dense Matrix Market format\n"
                << "  *.sdm: Sparse binary double format\n"
                << "  *.ddm: Dense binary double format\n"
                << std::endl;
}

std::ostream &Sys::cout()
{
    if (!Sys::os)
    {
        char name[1024];
        char output_filename[256];

        gethostname(name, 1024);
        snprintf(output_filename, 256, "bpmf_%s_%d.out", name, getpid());
        Sys::os = new std::ofstream(output_filename);
        std::cout << "output file: " << output_filename << std::endl;
    }

    return *Sys::os;
}

int main(int argc, char *argv[])
{
    char name[1024];
    gethostname(name, 1024);

    Sys::Init();
    int ch;
    std::string fname, probename;
    std::string mname, lname;
    Sys::nsims = 20;
    Sys::burnin = 5;
    
 
    while((ch = getopt(argc, argv, "krvn:t:p:i:b:f:g:w:u:v:o:s:m:l:a:d:")) != -1)
    {
        switch(ch)
        {
            case 'i': Sys::nsims = atoi(optarg); break;
            case 'b': Sys::burnin = atoi(optarg); break;
            case 'a': Sys::alpha = atof(optarg); break;
            case 'd': assert(num_latent == atoi(optarg)); break;
            case 'n': fname = optarg; break;
            case 'p': probename = optarg; break;

            //output directory matrices
            case 'o': Sys::odirname = optarg; break;

            case 'm': mname = optarg; break;
            case 'l': lname = optarg; break;

            case 'v': Sys::verbose = true; break;
            case '?':
            case 'h': 
            default : usage(); Sys::Abort(1);
        }
    }

    if (fname.empty() || probename.empty()) { 
        usage();
        Sys::Abort(1);
    }


    //SHOW(sizeof(Sys));
    Sys movies("movs", fname, probename);
    //SHOW(sizeof(movies._M));
    //SHOW(&(movies._M));
    //SHOW(sizeof(Sys));
    Sys users("users", movies._M, movies.Pavg);
    //SHOW(sizeof(Sys));

    movies.alloc_and_init();
    users.alloc_and_init();

    perf_data_init();

    long double average_items_sec = .0;
    long double average_ratings_sec = .0;
    
    Sys::cout() << "hostname: " << name << std::endl;
    Sys::cout() << "pid: " << getpid() << std::endl;
    if (getenv("PBS_JOBID")) Sys::cout() << "jobid: " << getenv("PBS_JOBID") << std::endl;
 
    Sys::cout() << "num_latent: " << num_latent<< std::endl;
    Sys::cout() << "nsims: " << Sys::nsims << std::endl;
    Sys::cout() << "burnin: " << Sys::burnin << std::endl;
    Sys::cout() << "alpha: " << Sys::alpha << std::endl;

    Sys::sync();

    auto begin = tick();

    for(int i=0; i<Sys::nsims; ++i) {
        BPMF_COUNTER("main");
        auto start = tick();

        {
            BPMF_COUNTER("movies");
            movies.sample_hp();
            movies.sample(users);
        }
        {
            BPMF_COUNTER("users");
            users.sample_hp();
            users.sample(movies);
        }

        { 
            BPMF_COUNTER("eval");
            movies.predict(users); 
            users.predict(movies); 
        }

        auto stop = tick();
        double items_per_sec = (users.num() + movies.num()) / (stop - start);
        double ratings_per_sec = (users.nnz()) / (stop - start);
        movies.print(items_per_sec, ratings_per_sec, sqrt(users.norm), sqrt(movies.norm));
        average_items_sec += items_per_sec;
        average_ratings_sec += ratings_per_sec;
    }

    Sys::sync();

    auto end = tick();
    auto elapsed = end - begin;

    // predict all
    movies.predict(users, true);

    Sys::cout() << "Total time: " << elapsed << std::endl  << std::flush;
    Sys::cout() << "Final Avg RMSE: " << movies.rmse_avg  << std::endl  << std::flush;
    Sys::cout() << "  computed on " << movies.num_predict << " items ("
                << int(100. * movies.num_predict / movies.T.nonZeros()) 
                << "% of total items in test set)" << std::endl  << std::flush;
    Sys::cout() << "Average items/sec: " << average_items_sec / movies.iter << std::endl  << std::flush;
    Sys::cout() << "Average ratings/sec: " << average_ratings_sec / movies.iter << std::endl  << std::flush;

    perf_data_print();
    Sys::Finalize();

    delete Sys::os;

    return 0;
}