Compiling Distributed BPMF
##########################

This document contains instructions on how to compile
BPMF.

Requirements
^^^^^^^^^^^^

For the single node version:

- BPMF source from - https://github.com/ExaScience/bpmf
- Eigen 3.3.x - https://eigen.tuxfamily.org/
- A modern C++11 compiler, with OpenMP support

For the MPI versions:

- All of the above, plus
- MPI 3.0 - any flavor should work

For the GASPI version:

- All of the above, plus
- GASPI/GPI 1.3.0 - https://github.com/cc-hpc-itwm/GPI-2/

**Make sure to compile GASPI/GPI with MPI support**


Compilation
^^^^^^^^^^^

The different subdirectories here each contain a `Makefile`, these
subdirectories contain different compilation options:

- nocomm-ser/: single node, single core
- nocomm-omp/: single node, OpenMP
- mpi-bcast-omp/: MPI+OpenMP using MPI_Bcast
- mpi-omp/: MPI+OpenMP using MPI_Isend and MPI_Irecv
- mpi-pure/: Pure MPI version using MPI_Isend and MPI_Irecv
- gpi-omp/: GASPI/GPI+OpenMP version

The number of latent variables is fixed during compilation
time. You specify this using the ``BPMF_NUMLATENT`` make variable::

    make BPMF_NUMLATENT=32

The default is 16.

Test
^^^^

A small test dataset is available in the directory ``data/tiny``.
Run the following command in this directory to test bpmf::