BPMF
====

|Travis Build Status| 

Julia and C++ implementations of Bayesian Probabilistic Matrix Factorization
using Markov Chain Monte Carlo. BPMF is a recommender method that allows to
predict for example movie ratings. The BPMF method is described here: http://www.cs.toronto.edu/~rsalakhu/papers/bpmf.pdf

The implementation in this repo is described in:
"Distributed Matrix Factorization using Asynchrounous Communication", Tom Vander Aa, Imen Chakroun, Tom Haber, https://arxiv.org/pdf/1705.10633

Input is in the form of a sparse matrix of values (e.g. movie ratings) ``R``.
The outputs are two smaller matrices ``U`` and ``V`` such that ``U * V``
forms a prediction matrix.

Background information
----------------------

Tutorial slides in `tutorial_slides/ <tutorial_slides/>`__

Installation using Conda
------------------------

The single node OpenMP version of BPMF can be installed using Conda::

   conda install -c vanderaa bpmf

Building the C++ version
------------------------

See the `README <build/makefiles/README.rst>`_ in `build/makefiles`


Running the C++ version
-----------------------

The C++ version takes these arguments::

  Usage: bpmf -n <MTX> -p <MTX> [-o DIR/] [-i N] [-b N] [-krv] [-t N] [-m MTX,MTX] [-l MTX,MTX]
  
  Paramaters:
    -n MTX: Training input data
    -p MTX: Test input data
    [-o DIR]: Output directory for model and predictions
    [-i N]: Number of total iterations
    [-b N]: Number of burnin iterations
    [-a F]: Noise precision (alpha)
  
    [-k]: Do not optimize item to node assignment
    [-r]: Redirect stdout to file
    [-v]: Output all samples
    [-t N]: Number of OpenMP threads to use.
  
    [-m MTX,MTX]: propagated posterior mu and Lambda matrices for U
    [-l MTX,MTX]: propagated posterior mu and Lambda matrices for V
  
  Matrix Formats:
    *.mtx: Sparse or dense Matrix Market format
    *.sdm: Sparse binary double format
    *.ddm: Dense binary double format

Input matrices
--------------

The ``data/`` directory contains preprocessed input data from the movielens
database to predict movie ratings and from the chembl database to predict
compound-on-protein activity.

Julia version
-------------

The Julia version takes two command line arguments:

``./bpmf <train_matrix.mtx> <test_matrix.mtx>``

Matrices should be in the MatrixMarket format. Other options need to be changed in the source code itself. E.g. the number of iterations, the size of the features vectors, ...


.. |Travis Build Status| image:: https://travis-ci.org/ExaScience/bpmf.svg?branch=master
   :target: https://travis-ci.org/ExaScience/bpmf