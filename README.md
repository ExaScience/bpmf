# BPMF

Julia and C++ implementations of Bayesian Probabilistic Matrix Factorization using Markov Chain Monte Carlo. BPMF is a 
recommender method that allows to predict for example movie ratings. The method is described here: http://www.cs.toronto.edu/~rsalakhu/papers/bpmf.pdf

Input is in the form of a sparse matrix of values (e.g. movie ratings) `R` . The outputs are two smaller matrices `U` and `V` such that `U * V` forms a prediction matrix.

## Usage

Both the julia and the c++ version take two command line arguments:

`./bpmf <train_matrix.mtx> <test_matrix.mtx>`

Matrices should be in the MatrixMarket format. Other options need to be changed in the source code itself. E.g. the number of iterations, the size of the features vectors, ...

## Building the c++ version

A sample Makefile is provided in the c++/build directory. Eigen, including the unsupported directory to read MatirxMarket files is required. To use multiple cores, Threading Building Blocks (TBB) or OpenMP can be used. 

## Input matrices
The `data/` directory provides README files on how to obtain input data from the movielens database to predict movie ratings 
from the chembl database to predict compound-on-protein activity

