/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include <mpi.h>
#include "error.h"

struct MPI_Sys : public Sys 
{
    //-- c'tor
    MPI_Sys(std::string name, std::string fname, std::string probename) : Sys(name, fname, probename) {}
    MPI_Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &P) : Sys(name, M, P) {}

    virtual void sample(Sys &in);
    virtual void send_item(int) {}
    virtual void alloc_and_init();

    void reduce_sum_cov_norm();

};


void MPI_Sys::sample(Sys &in)
{
    {
        BPMF_COUNTER("communicate");

        // 1. -- Worker sends updates to PS
        //     - sends norm, cov and sum 
        //     - sends precMu, precLambda

        /* CODE HERE */

        // 2. -- PS combines precMu, precLambda
        //       Reduction for each movie/user accross workers

        for (int i = 0; i < nprocs; i++)
        {
            auto col = from(i);
            auto num_cols = num(i);

            {
                auto recv_buf = precMu.col(col).data();
                auto send_buf = (i == procid) ? MPI_IN_PLACE : recv_buf;
                MPI_Reduce(send_buf, recv_buf, num_cols * num_latent, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
            }

            {
                auto recv_buf = precLambda.col(col).data();
                auto send_buf = (i == procid) ? MPI_IN_PLACE : recv_buf;
                MPI_Reduce(send_buf, recv_buf, num_cols * num_latent * num_latent, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
            }
        }

        // 3. -- PS sends combined precMu, precLambda
        //          sends cov/norm/sum

        // 4. -- Worker reduces sum/norm/cov
        reduce_sum_cov_norm();
    }

    { BPMF_COUNTER("compute"); Sys::sample(in); }

}

struct Block
{
    std::vector<double> all_data;

    double *sum_ptr;
    double *cov_ptr;
    double *norm_ptr;
    double *precMu_ptr;
    unsigned precMu_rows, precMu_cols;
    double *precLambda_ptr;
    unsigned precLambda_rows, precLambda_cols;

    Eigen::Map<VectorNd> sum_map() { return Eigen::Map<VectorNd>(sum_ptr); }
    Eigen::Map<MatrixNNd> cov_map() { return Eigen::Map<MatrixNNd>(cov_ptr); }
    double &norm_ref() { return *norm_ptr; }
    Eigen::Map<Eigen::MatrixXd> precMu_map() { return Eigen::Map<Eigen::MatrixXd>(precMu_ptr, precMu_rows, precMu_cols); }
    Eigen::Map<Eigen::MatrixXd> precLambda_map() { return Eigen::Map<Eigen::MatrixXd>(precLambda_ptr, precLambda_rows, precLambda_cols); }

    Block(
        const VectorNd &sum,  //-- sum of all U-vectors
        const MatrixNNd &cov, //-- covariance
        double norm,
        const Eigen::MatrixXd &precMu,
        const Eigen::MatrixXd &precLambda
    ) : all_data(
            sum.nonZeros() +
            cov.nonZeros() +
            1 +
            precMu.nonZeros() +
            precLambda.nonZeros()
        )
    {
        double *p = all_data.data();

        sum_ptr = p;
        p += sum.nonZeros();

        cov_ptr = p;
        p += cov.nonZeros();

        norm_ptr = p;
        p += 1;

        precMu_ptr = p;
        p += precMu.nonZeros();
        precMu_rows = precMu.rows();
        precMu_cols = precMu.cols();

        precLambda_ptr = p;
        p += precLambda.nonZeros();
        precLambda_rows = precLambda.rows();
        precLambda_cols = precLambda.cols();

        assert((p - all_data.data()) == all_data.size());

        sum_map() = sum;
        cov_map() = cov;
        norm_ref() = norm;
        precMu_map() = precMu;
        precLambda_map() = precLambda;

    }
}; 

/*
        Block b; // layout sum, cov, norm, mu, Lambda
        b << sum << cov << norm << precMu << precLambda;

        MPI_Allreduce(MPI_IN_PLACE, block.data(), block.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        b >> sum >> cov >> norm >> precMu >> precLambda;
*/

#include "mpi_common.h"
