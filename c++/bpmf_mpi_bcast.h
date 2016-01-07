#include "bpmf_mpi_common.h"

struct MPI_Sys : public Sys 
{
    //-- c'tor
    MPI_Sys(std::string name, std::string fname) : Sys(name, fname) {}
    MPI_Sys(std::string name, const SparseMatrixD &M) : Sys(name, M) {}

    virtual void sample(Sys &in);
};

void MPI_Sys::sample(Sys &in)
{
    Sys::sample(in);

    for(int k = 0; k < Sys::nprocs; k++) {
        //items
        unsigned count = dom().local(k).count();
        unsigned start = k * count;
        MPI_Bcast(items().col(start).data(), count*num_feat, MPI_DOUBLE, k, MPI_COMM_WORLD);

        //sum, cov, norm
        MPI_Bcast(sum(k).data(), sum(k).size(), MPI_DOUBLE, k, MPI_COMM_WORLD);
        MPI_Bcast(cov(k).data(), cov(k).size(), MPI_DOUBLE, k, MPI_COMM_WORLD);
        MPI_Bcast(&norm(k),      1,             MPI_DOUBLE, k, MPI_COMM_WORLD);
    }
}
