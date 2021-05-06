/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#include "bpmf.h"

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

        MM_local.triangularView<Eigen::Upper>() += col * col.transpose(); // outer product
        rr_local.noalias() += col;        // vector num_latent x 1
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