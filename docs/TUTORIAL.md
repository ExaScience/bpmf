# Performance analysis of BPMF

This tutorial aims to give the users some insights in the performance behavior of BPMF. 
The tutorial is split in several parts:

 1. the analysis of the single-node multi-threaded version of BPMF, and 
 2. analysis of the distributed version of BPMF. 


## Prerequisites

In this section:

 - Downloading the tutorial and BPMF code 
 - Logging in on ARCHER


### Downloading the tutorial documentation and code

The code and documents are available on GitHub:

`https:///

and also has been preloaded on ARCHER.


### Logging in to ARCHER

,,,

Run this command on ARCHER to expand the tutorial in your home directory:

`tar xzf ....`

### Contents of the package

 - c++: C++ source code of BPMF
 - data: MovieLens and Chembl20 input files
 - build/archer:  Makefiles and job submission scripts for ARCHER
 - build/generic: Generic Makefiles. E.g. for your own machine
 - docs: Tutorial slides and documents 
 - julia: Julia source code of BPMF

### Data Files

The input of BPMF is a set of two sparse matrices. These matrices

## Single Node, Multi-threaded BPMF

This part can be performed on your own machine (laptop) or on the ARCHER
supercomputer of EPCC


## Distributed BPMF
