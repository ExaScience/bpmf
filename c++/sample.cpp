/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#include <random>
#include <memory>
#include <cstdio>
#include <iostream>
#include <climits>
#include <stdexcept>
#include <cmath>

#include "error.h"
#include "bpmf.h"
#include "io.h"

static const bool measure_perf = false;

std::ostream *Sys::os;

int Sys::nsims;
int Sys::burnin;
double Sys::alpha = 2.0;

std::string Sys::odirname = "";

bool Sys::verbose = false;

// verifies that A has the same non-zero structure as B
void assert_same_struct(SparseMatrixD &A, SparseMatrixD &B)
{
    assert(A.cols() == B.cols());
    assert(A.rows() == B.rows());
    for(int i=0; i<A.cols(); ++i) assert(A.col(i).nonZeros() == B.col(i).nonZeros());
}

//
// Does predictions for prediction matrix T
// Computes RMSE (Root Means Square Error)
//
void Sys::predict(Sys& other, bool all)
{
    int n = (iter < burnin) ? 0 : (iter - burnin);
   
    double se(0.0); // squared err
    double se_avg(0.0); // squared avg err
    int nump = 0; // number of predictions

    int lo = from();
    int hi = to();
    if (all) {
#ifdef BPMF_REDUCE
        Sys::cout() << "WARNING: predict all items in test set not available in BPMF_REDUCE mode" << std::endl;
#else
        lo = 0;
        hi = num();
#endif
    }

    for(int k = lo; k<hi; k++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(T,k); it; ++it)
        {
            auto m = items().col(it.col());
            auto u = other.items().col(it.row());

            //SHOW(it.col());
            //SHOW(it.row());
            //SHOW(m.norm());
            //SHOW(u.norm());

            //assert(m.norm() > 0.0);
            //assert(u.norm() > 0.0);

            const double pred = m.dot(u) + mean_rating;
            se += sqr(it.value() - pred);

            // update average prediction
            double &avg = Pavg.coeffRef(it.row(), it.col());
            double delta = pred - avg;
            avg = (n == 0) ? pred : (avg + delta/n);
            double &m2 = Pm2.coeffRef(it.row(), it.col());
            m2 = (n == 0) ? 0 : m2 + delta * (pred - avg);
            se_avg += sqr(it.value() - avg);

            nump++;
        }
    }

    rmse = sqrt( se / num_predict );
    rmse_avg = sqrt( se_avg / num_predict );
    num_predict = nump;
}

//
// Prints sampling progress
//
void Sys::print(double items_per_sec, double ratings_per_sec, double norm_u, double norm_m) {
  char buf[1024];
  std::string phase = (iter < Sys::burnin) ? "Burnin" : "Sampling";
  sprintf(buf, "%s iteration %d:\t RMSE: %3.4f\tavg RMSE: %3.4f\tFU(%6.2f)\tFM(%6.2f)\titems/sec: %6.2f\tratings/sec: %6.2fM\n",
                    phase.c_str(), iter, rmse, rmse_avg, norm_u, norm_m, items_per_sec, ratings_per_sec / 1e6);
  Sys::cout() << buf;
}

//
// Constructor with that reads MTX files
// 
Sys::Sys(std::string name, std::string fname, std::string probename)
    : name(name), iter(-1)
{
    read_matrix(fname, _M);
    read_matrix(probename, T);

    auto rows = std::max(_M.rows(), T.rows());
    auto cols = std::max(_M.cols(), T.cols());
    _M.conservativeResize(rows,cols);
    T.conservativeResize(rows,cols);
    Pm2 = Pavg = Torig = T; // reference ratings and predicted ratings
    assert(_M.rows() == Pavg.rows());
    assert(_M.cols() == Pavg.cols());
    //SHOW(sizeof(*this));
    //SHOW(sizeof(_M));
    //SHOW(&(_M));
}

//
// Constructs Sys as transpose of existing Sys
//
Sys::Sys(std::string name, const SparseMatrixD &Mt, const SparseMatrixD &Pt)
   : name(name), iter(-1)
{
    _M = Mt.transpose();
    Pm2 = Pavg = T = Torig = Pt.transpose(); // reference ratings and predicted ratings
    assert(_M.rows() == Pavg.rows());
    assert(_M.cols() == Pavg.cols());
    SHOW(sizeof(*this));
}

Sys::~Sys() 
{
    if (measure_perf) {
        Sys::cout() << " --------------------\n";
        Sys::cout() << name << ": sampling times\n";
        for(int i = from(); i<to(); ++i) 
        {
            Sys::cout() << "\t" << nnz(i) << "\t" << sample_time.at(i) / nsims  << "\n";
        }
        Sys::cout() << " --------------------\n\n";
    }
}

//
// Intializes internal Matrices and Vectors
//
void Sys::init()
{
    //-- M
    assert(M().rows() > 0 && M().cols() > 0);
    mean_rating = M().sum() / M().nonZeros();
    items().setZero();
    sum.setZero();
    cov.setZero();
    norm = .0;

    int count_larger_bp1 = 0;
    int count_larger_bp2 = 0;
    int count_sum = 0;
    for(int k = 0; k<M().cols(); k++) {
        int count = M().col(k).nonZeros();
        count_sum += count;
        if (count > breakpoint1) count_larger_bp1++;
        if (count > breakpoint2) count_larger_bp2++;
    }

    Sys::cout() << "mean rating: " << mean_rating << std::endl;
    Sys::cout() << "total number of ratings in train: " << M().nonZeros() << std::endl;
    Sys::cout() << "total number of ratings in test: " << T.nonZeros() << std::endl;
    Sys::cout() << "average ratings per row: " << (double)count_sum / (double)M().cols() << std::endl;
    Sys::cout() << "rows > break_point1: " << 100. * (double)count_larger_bp1 / (double)M().cols() << std::endl;
    Sys::cout() << "rows > break_point2: " << 100. * (double)count_larger_bp2 / (double)M().cols() << std::endl;
    Sys::cout() << "num " << name << ": " << num() << std::endl;

    if (measure_perf) sample_time.resize(num(), .0);
}



void HyperParams::sample(const int N, const VectorNd &sum, const MatrixNNd &cov)
{
    //SHOW(N);
    //SHOW(sum);
    //SHOW(cov);
    //SHOW(mu0);
    //SHOW(b0);
    //SHOW(df);
    //SHOW(WI);

    std::tie(mu, LambdaU) = CondNormalWishart(N, cov, sum / N, mu0, b0, WI, df);
    LambdaF = LambdaU.triangularView<Eigen::Upper>().transpose() * LambdaU;
    LambdaL = LambdaU.transpose();

    //SHOW(LambdaF);
}

void Sys::register_time(int i, double t)
{
    if (measure_perf) sample_time.at(i) += t;
}