/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <unistd.h>

#include "io.h"
#include "bpmf.h"

using namespace std;
using namespace Eigen;

#ifdef BPMF_GPI_COMM
#include "bpmf_gaspi.h"
#elif defined(BPMF_GPICXX_COMM)
#include "bpmf_gaspicxx.h"
#elif defined(BPMF_MPI_PUT_COMM)
#include "mpi_put.h"
#elif defined(BPMF_MPI_BCAST_COMM)
#include "mpi_bcast.h"
#elif defined(BPMF_MPI_REDUCE_COMM)
#include "mpi_reduce.h"
#elif defined(BPMF_MPI_ALLREDUCE_COMM)
#include "mpi_allreduce.h"
#elif defined(BPMF_MPI_ISEND_COMM)
#include "mpi_isendirecv.h"
#elif defined(BPMF_ARGO_COMM)
#include "bpmf_argo.h"
#elif defined(BPMF_NO_COMM)
#include "nocomm.h"
#else
#error no comm include
#endif

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

int main(int argc, char *argv[])
{
  Sys::Init();
  {
    int ch;
    string fname, probename;
    string mname, lname;
    int nthrds = -1;
    bool redirect = false;
    Sys::nsims = 20;
    Sys::burnin = 5;
    Sys::update_freq = 1;
    
 
    while((ch = getopt(argc, argv, "krvn:t:p:i:b:f:g:w:u:v:o:s:m:l:a:d:")) != -1)
    {
        switch(ch)
        {
            case 'i': Sys::nsims = atoi(optarg); break;
            case 'b': Sys::burnin = atoi(optarg); break;
            case 'f': Sys::update_freq = atoi(optarg); break;
            case 't': nthrds = atoi(optarg); break;
            case 'a': Sys::alpha = atof(optarg); break;
            case 'd': assert(num_latent == atoi(optarg)); break;
            case 'n': fname = optarg; break;
            case 'p': probename = optarg; break;

            //output directory matrices
            case 'o': Sys::odirname = optarg; break;

            case 'm': mname = optarg; break;
            case 'l': lname = optarg; break;

            case 'r': redirect = true; break;
            case 'k': Sys::permute = false; break;
            case 'v': Sys::verbose = true; break;
            case '?':
            case 'h': 
            default : usage(); Sys::Abort(1);
        }
    }

    if (Sys::nprocs >1 || redirect) {
        std::stringstream ofname;
        ofname << "bpmf_" << Sys::procid << ".out";
        Sys::os = new std::ofstream(ofname.str());
    } else {
        Sys::os = &std::cout;
    }

#ifndef NDEBUG
    Sys::dbgs = Sys::os;
#else
    Sys::dbgs = new std::ofstream("/dev/null");
#endif

    if (fname.empty() || probename.empty()) { 
        usage();
        Sys::Abort(1);
    }


    SYS movies("movs", fname, probename);
    SYS users("users", movies.M, movies.Pavg);

    movies.add_prop_posterior(mname);
    users.add_prop_posterior(lname);

    movies.alloc_and_init();
    users.alloc_and_init();

    // assign users and movies to the computation nodes
    movies.assign(users);
    users.assign(movies);
    movies.assign(users);
    users.assign(movies);

    // build connectivity matrix
    // contains what items needs to go to what nodes
    users.build_conn(movies);
    movies.build_conn(users);
    assert(movies.nnz() == users.nnz());

    threads::init(nthrds);
    perf_data_init();
    argo_data_init(movies, users);

    long double average_items_sec = .0;
    long double average_ratings_sec = .0;
    
    char name[1024];
    gethostname(name, 1024);
    Sys::cout() << "hostname: " << name << endl;
    Sys::cout() << "pid: " << getpid() << endl;
    if (getenv("PBS_JOBID")) Sys::cout() << "jobid: " << getenv("PBS_JOBID") << endl;
 
    if(Sys::procid == 0)
    {
        Sys::cout() << "num_latent: " << num_latent<<endl;
        Sys::cout() << "nprocs: " << Sys::nprocs << endl;
        Sys::cout() << "nthrds: " << threads::get_max_threads() << endl;
        Sys::cout() << "nsims: " << Sys::nsims << endl;
        Sys::cout() << "burnin: " << Sys::burnin << endl;
        Sys::cout() << "alpha: " << Sys::alpha << endl;
        Sys::cout() << "update_freq: " << Sys::update_freq << endl;
    }

    Sys::sync();

    auto begin = tick();

    for(int i=0; i<Sys::nsims; ++i) {
        BPMF_COUNTER("main");
        auto start = tick();

        { BPMF_COUNTER("movies"); movies.sample(users); }
        { BPMF_COUNTER("users"); users.sample(movies); }

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

        if (Sys::verbose)
        {
            users.bcast();
            movies.bcast();
            if (Sys::procid == 0)
            {
                write_matrix(Sys::odirname + "/U-" + std::to_string(i) + ".ddm", users.items());
                write_matrix(Sys::odirname + "/V-" + std::to_string(i) + ".ddm", movies.items());
            }
        }
    }

    Sys::sync();

    auto end = tick();
    auto elapsed = end - begin;

    users.bcast();
    movies.bcast();

    //-- if we need to generate output files, collect all data on proc 0
    if (Sys::odirname.size()) {
        // restore original order
        users.unpermuteCols(movies);
        movies.unpermuteCols(users);
        movies.predict(users, true);

        if (Sys::procid == 0) {
            // sparse
            write_matrix(Sys::odirname + "/Pavg.sdm", movies.Pavg);
            write_matrix(Sys::odirname + "/Pm2.sdm", movies.Pm2);

            // dense
            users.finalize_mu_lambda();
            write_matrix(Sys::odirname + "/U-mu.ddm", users.aggrMu);
            write_matrix(Sys::odirname + "/U-Lambda.ddm", users.aggrLambda);

            movies.finalize_mu_lambda();
            write_matrix(Sys::odirname + "/V-mu.ddm", movies.aggrMu);
            write_matrix(Sys::odirname + "/V-Lambda.ddm", movies.aggrLambda);
        }
    } else {
        movies.predict(users, true);
    }

    if (Sys::procid == 0) {
        Sys::cout() << "Total time: " << elapsed <<endl <<flush;
        Sys::cout() << "Final Avg RMSE: " << movies.rmse_avg <<endl <<flush;
        Sys::cout() << "  computed on " << movies.num_predict << " items ("
                    << int(100. * movies.num_predict / movies.T.nonZeros()) 
                    << "% of total items in test set)" << endl << flush;
        Sys::cout() << "Average items/sec: " << average_items_sec / movies.iter << endl <<flush;
        Sys::cout() << "Average ratings/sec: " << average_ratings_sec / movies.iter << endl <<flush;
    }

  }
  perf_data_print();
  Sys::Finalize();

   return 0;
}


void Sys::bcast()
{
    for(int i = 0; i < num(); i++) {
#ifdef BPMF_MPI_COMM
#ifndef BPMF_ARGO_COMM
        MPI_Bcast(items().col(i).data(), num_latent, MPI_DOUBLE, proc(i), MPI_COMM_WORLD);
#endif
        if (aggrMu.nonZeros())
            MPI_Bcast(aggrMu.col(i).data(), num_latent, MPI_DOUBLE, proc(i), MPI_COMM_WORLD);
        if (aggrLambda.nonZeros())
            MPI_Bcast(aggrLambda.col(i).data(), num_latent*num_latent, MPI_DOUBLE, proc(i), MPI_COMM_WORLD);
#else
        assert(Sys::nprocs == 1);
#endif
    }
}


void Sys::finalize_mu_lambda()
{
    assert(aggrLambda.nonZeros());
    assert(aggrMu.nonZeros());
    // calculate real mu and Lambda
    for(int i = 0; i < num(); i++) {
        int nsamples = Sys::nsims - Sys::burnin;
        auto sum = aggrMu.col(i);
        auto prod = Eigen::Map<MatrixNNd>(aggrLambda.col(i).data());
        MatrixNNd cov = (prod - (sum * sum.transpose() / nsamples)) / (nsamples - 1);
        MatrixNNd prec = cov.inverse(); // precision = covariance^-1
        aggrLambda.col(i) = Eigen::Map<Eigen::VectorXd>(prec.data(), num_latent * num_latent);
        aggrMu.col(i) = sum / nsamples;
    }
}
