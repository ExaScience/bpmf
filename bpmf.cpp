/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#include "Eigen/Dense"

const int num_latent = 32;

typedef Eigen::Matrix<double, num_latent, num_latent> MatrixNNd;
typedef Eigen::Matrix<double, num_latent, 1> VectorNd;

#pragma omp declare reduction (VectorPlus : VectorNd : omp_out.noalias() += omp_in) initializer(omp_priv = VectorNd::Zero())
#pragma omp declare reduction (MatrixPlus : MatrixNNd : omp_out.noalias() += omp_in) initializer(omp_priv = MatrixNNd::Zero())

void computeMuLambda_2lvls_standalone(long idx) 
{
    const unsigned count = idx; 

    VectorNd rr_local(VectorNd::Zero());
    MatrixNNd MM_local(MatrixNNd::Zero());

#pragma omp parallel
#pragma omp taskloop default(none) \
            shared(count) \
            reduction(VectorPlus:rr_local) reduction(MatrixPlus:MM_local)
    for (unsigned j = 0; j < count; j++)
    {
        // for each nonzeros elemen in the i-th row of M matrix
        auto col = VectorNd::Zero(); // vector num_latent x 1 from V matrix: M[i,j] = U[i,:] x V[idx,:]
    }

}

int main(int argc, char *argv[])
{
    for(int i=0; i<10000; ++i)
    {
        printf("%i\n", i);
        computeMuLambda_2lvls_standalone(i);
    }
    printf("Success!\n");
}