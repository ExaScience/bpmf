## Introduction

In this section:

 - Obtaining the tutorial and BPMF code 
 - Contents of the package

### Downloading the tutorial documentation and code

The code and documents are available on GitHub:

https://github.com/ExaScience/bpmf/tree/EuroMPI16

and also has been preloaded on ARCHER.

### Logging in to ARCHER

Run this command on ARCHER to expand the tutorial in a subdirectory called `bpmf`:

`tar xzf /work/y14/shared/eurompi/bpmf.tar.gz`

### Compiling on ARCHER

We will use the GNU compilers and not the CRAY compilers. Hence everytime
you log in, you need to execute this command:

`module switch PrgEnv-cray PrgEnv-gnu`

### Contents of the package

Please have a look at the README.md file if you want general information
on how to run BPMF and how obtain additional data files.

 - c++: C++ source code of BPMF
 - data: MovieLens and Chembl20 input files
 - build/archer:  Makefiles and job submission scripts for ARCHER
 - build/generic: Generic Makefiles. E.g. for your own machine
 - docs: Tutorial slides and documents 
 - julia: Julia source code of BPMF
