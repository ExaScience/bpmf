#!/bin/bash

#SBATCH --partition=thunder
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=96
#SBATCH --out=bpmf-%j.out
#SBATCH --err=bpmf-%j.err
#SBATCH --time=01:00:00

set -e

BASEDIR=$SLURM_SUBMIT_DIR

CONFIG=$1
test -z "$CONFIG" && CONFIG=gpi-omp
 
BPMF=$CONFIG/bpmf
ARGS="-n $BASEDIR/data/movielens/ml-train.mtx -p $BASEDIR/data/movielesn/ml-test.mtx"

ldd $BPMF

MACHINEFILE=$(mktemp /tmp/machinefile.XXXXX)
srun -l /bin/hostname | sort -n | awk '{print $2}' > $MACHINEFILE

GASPIRUN=$HOME/euroexa/unimem-gpi/bin/gaspi_run

RUNME=$BASEDIR/gaspi_run_me.sh

CMD="$GASPIRUN -m $MACHINEFILE $RUNME $BASEDIR"

$CMD
