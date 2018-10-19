#!/bin/bash

#SBATCH --partition=thunder
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --out=bpmf-%j.out
#SBATCH --err=bpmf-%j.err
#SBATCH --time=01:00:00

set -e

BASEDIR=$SLURM_SUBMIT_DIR

cd $BASEDIR

. modules.sh

BPMF=mpi-omp/bpmf
ARGS="-n $BASEDIR/data/movielens/ml-train.mtx -p $BASEDIR/data/movielesn/ml-test.mtx"

ldd $BPMF

CMD="map --profile srun $BPMF $ARGS"

$CMD
