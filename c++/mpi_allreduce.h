/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include <mpi.h>
#include "error.h"

struct AllReduceBlock
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

    AllReduceBlock(
        unsigned int sum_size,
        unsigned int cov_size,
        unsigned int precMu_size,
        unsigned int precMu_rows,
        unsigned int precMu_cols,
        unsigned int precLambda_size,
        unsigned int precLambda_rows,
        unsigned int precLambda_cols
    ) : all_data(sum_size + cov_size + 1 + precMu_size + precLambda_size)
    {
        double *p = all_data.data();
        sum_ptr = p;
        p += sum_size;

        cov_ptr = p;
        p += cov_size;

        norm_ptr = p;
        p += 1;

        precMu_ptr = p;
        p += precMu_size;
        this->precMu_rows = precMu_rows;
        this->precMu_cols = precMu_cols;

        precLambda_ptr = p;
        p += precLambda_size;
        this->precLambda_rows = precLambda_rows;
        this->precLambda_cols = precLambda_cols;

        assert((p - all_data.data()) == all_data.size());
        sum_map().setZero();
        cov_map().setZero();
        norm_ref() = .0;
        precMu_map().setZero();
        precLambda_map().setZero();
    }

    AllReduceBlock(
        const VectorNd &sum,  //-- sum of all U-vectors
        const MatrixNNd &cov, //-- covariance
        double norm,
        const Eigen::MatrixXd &precMu,
        const Eigen::MatrixXd &precLambda
    ) : all_data(sum.nonZeros() + cov.nonZeros() + 1 + precMu.nonZeros() + precLambda.nonZeros())
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

struct MPI_Sys : public Sys 
{
    //-- c'tor
    MPI_Sys(std::string name, std::string fname, std::string probename) : Sys(name, fname, probename) {}
    MPI_Sys(std::string name, const SparseMatrixD &M, const SparseMatrixD &P) : Sys(name, M, P) {}

    virtual void sample(Sys &in);
    virtual void send_item(int) {}
    virtual void alloc_and_init();

    void reduce_sum_cov_norm();

    std::vector<AllReduceBlock> blocks;
};

 

void MPI_Sys::sample(Sys &in)
{
    if (0) {
        BPMF_COUNTER("communicate");
        MPI_Allreduce(MPI_IN_PLACE, sum.data(), sum.nonZeros(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, cov.data(), cov.nonZeros(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, precMu.data(), precMu.nonZeros(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, precLambda.data(), precLambda.nonZeros(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    } else {
        BPMF_COUNTER("allreduce");
        blocks.push_back(AllReduceBlock(sum, cov, norm, precMu, precLambda));
        const unsigned num_blocks = blocks.size();
        const unsigned slack = 16;

        static std::default_random_engine generator;
        static std::uniform_int_distribution<int> distribution(0,slack);
        const unsigned random_slack = distribution(generator);

        auto &block = (num_blocks > slack) ? blocks.at(num_blocks-1-random_slack) : blocks.back();
        Sys::cout() << "Using slack " << random_slack << std::endl;

        AllReduceBlock new_block(
            sum.nonZeros(),
            cov.nonZeros(),
            precMu.nonZeros(),
            precMu.rows(),
            precMu.cols(),
            precLambda.nonZeros(),
            precLambda.rows(),
            precLambda.cols()
        );
        MPI_Allreduce(new_block.all_data.data(), block.all_data.data(), block.all_data.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        sum        = new_block.sum_map();
        cov        = new_block.cov_map();
        norm       = new_block.norm_ref();
        precMu     = new_block.precMu_map();
        precLambda = new_block.precLambda_map();

    }

    { BPMF_COUNTER("compute"); Sys::sample(in); }

}
/*
        Block b; // layout sum, cov, norm, mu, Lambda
        b << sum << cov << norm << precMu << precLambda;

        MPI_Allreduce(MPI_IN_PLACE, block.data(), block.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        b >> sum >> cov >> norm >> precMu >> precLambda;
*/

#include "mpi_common.h"
