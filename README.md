# BPMF

Julia and C++ implementations of Bayesian Probabilistic Matrix Factorization using Markov Chain Monte Carlo. BPMF is a 
recommender method that allows to predict for example movie ratings. The method is described here: http://www.cs.toronto.edu/~rsalakhu/papers/bpmf.pdf

Input is in the form of a sparse matrix of values (e.g. movie ratings) `R` . The outputs are two smaller matrices `U` and `V` such that `U * V` forms a prediction matrix.

## Julia version

The Julia version takes two command line arguments:

`./bpmf <train_matrix.mtx> <test_matrix.mtx>`

Matrices should be in the MatrixMarket format. Other options need to be changed in the source code itself. E.g. the number of iterations, the size of the features vectors, ...

## Building the C++ version

A sample Makefile is provided in the c++/build directory. Eigen, including the unsupported directory to read MatirxMarket files is required. To use multiple cores, Threading Building Blocks (TBB) or OpenMP can be used. 

## Running the C++ version

The C++ version takes more arguments:

`bpmf [-t <threads>] [ -i <niters> ] -n <samples.mtx> [-p <probe.mtx>] [-u <u.mtx>] [-v <v.mtx>] [-o <pred.mtx>] [-s <m2.mtx>]`

Where
 - `[-t <threads>]`: Number of OpenMP or TBB threads to used.
 - `[-i <niters> ]`: Number of sampling iterations
 - `-n <samples.mtx>`: Training input data
 - `-p <probe.mtx>`: Test input data
 - `[-u <u.mtx>]`: Model output U matrix
 - `[-v <v.mtx>]`: Model output V matrix
 - `[-o <pred.mtx>]`: Predictions for test input data
 - `[-s <m2.mtx>]`: Full `UxV` matrix


## Input matrices
The `data/` directory contains preprocessed input data from the movielens database to predict movie ratings and
from the chembl database to predict compound-on-protein activity.

