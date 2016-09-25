# Distributed BPMF

This part can only be performed on the ARCHER supercomputer of EPCC.

## Compile and test

### Compiling the MPI + TBB version on ARCHER

Login on ARCHER.

Do not forget to switch to the GNU programming environment:

`module switch PrgEnv-cray PrgEnv-gnu`

Go to `bpmf/build/archer/mpiisend-tbb` and type `make`

### Submit a small test job to ARCHER

Generate a set of job description jobs: `./gen_cmd.sh`. These cmd-files are generated based on 
the `bpmf.cmd.tmpl` template. Have a look a the template if you want.

Submit a small job to test: `qsub -q R3941496 ./bpmf_2_mpiisend.cmd`

Verify that the job is running, is queued or has finished:

```
$> qstat -u $USER

sdb:
                                                            Req'd  Req'd   Elap
Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
--------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
3953089.sdb     vanderaa standard bpmf_mpiis    --    2  48    --  00:15 Q   --
```

Once the job has finished, go to the output directory and look at the output:

```
cd /work/y14/y14/$USER/eurompi/mpi-tbb/<jobid>
cat bpmf_0.out
```

## Test strong scaling

Submit a set of jobs with 1, 2 and 4 nodes. And see how performance scales.

## Effect of item to node assignment

Adapt the `bpmf.cmd.tmpl` such that bpmf runs with the additional option `-k`.
This option make that item to node assignment is not optimized. 

Verify the effect of this on performance.

