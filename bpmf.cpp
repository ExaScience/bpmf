/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */


#include "Eigen/Dense"

const int num_latent = 32;

typedef Eigen::Matrix<double, num_latent, 1> VectorNd;

#pragma omp declare reduction (VectorPlus : VectorNd : omp_out.noalias() += omp_in) initializer(omp_priv = VectorNd::Zero())

void computeMuLambda_2lvls_standalone(long c) 
{
    VectorNd rr_local(VectorNd::Zero());

#pragma omp taskloop default(none) shared(c) reduction(VectorPlus:rr_local)
    for (unsigned j = 0; j < c; j++)
    {
        auto col = VectorNd::Zero();
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