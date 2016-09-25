# Single Node, Multi-threaded BPMF

This part can be performed on your own machine (laptop) or on the ARCHER
supercomputer of EPCC.

When you compile on your own machine you need to have a recent GCC installed (>= 4.9)
and you will have to have TBB installed (https://www.threadingbuildingblocks.org/).
Additionally you might to have to adapt the makefile to point to the correct compilers
and TBB directory.

## Compile and test

### Compiling the TBB version on ARCHER

Login on ARCHER.

Do not forget to switch to the GNU programming environment:

`module switch PrgEnv-cray PrgEnv-gnu`

Go to `bpmf/build/archer/nocomm-tbb` and type `make`

### Compiling the TBB version on your machine 

Go to `bpmf/build/generic/nocomm-tbb` and type `make`

### Test

In the same directory, type `make test` to verify that the code runs on a small example.

### Measure performance

Also run on a larger example:

`./bpmf -i 20 -n ../../../data/chembl_20/chembl-train-IC50.mtx -p ../../../data/chembl_20/chembl-test-IC50.mtx`

Note down the performance results, namely these lines of the output:

```Average items/sec: 732060
Average ratings/sec: 1.2253e+06

Totals on 0:
>> eval:	     0.0192	(1%) in	2
>> main:	     2.6766	(100%) in	2
>> movies:	     0.4702	(18%) in	2
>> users:	     2.1796	(81%) in	2```

And calculate the time per single user and the time per single movie. Explain why this is different.

## Load Balancing

Open `sample.cpp` and look for breakpoint1 and breakpoint2. Examine how these settings influence 
the nested thread-level parallelism of TBB. Play with the values of breakpoint1 and breakpoint2
to influence this parallelism and see how this influences performance. 

## OpenMP

Compile and run the OpenMP version in `bpmf/build/[generic,archer]/nocomm-tbb` in the same way
you did for the TBB version. Examine and explain the performance difference. Improve the OpenMP version :)

