# bpmf

Julia and C++ implementations of Bayesian Probabilistic Matrix Factorization using Markov Chain Monte Carlo. BPMF is a 
recommender method that allows to predict for example movie ratings. The method is described here:http://www.cs.toronto.edu/~rsalakhu/papers/bpmf.pdf

Input is in the form of a sparse matrix of values (e.g. movie ratings) `R` . The outputs are two smaller matrices `U` and `V` such that `U * V` forms a prediction matrix.

## julia version

Several julia versions exists. The most basic and tested is in julia/bpmf.jl. Have a look in the source code to know how to ca

## c++ version



## input data
The `data/` directory provides README files on how to obtain input data from the movielens database to predict movie ratings 
from the chembl database to predict compound-on-protein activity

