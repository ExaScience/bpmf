module load THUNDER/ARM_PL/18.3_armhpc18.3 HPC_COMPILER/18.3 eigen 
module load mpich/3.2
module load arm_forge

export ARM_HPC_COMPILER_LICENSE_SEARCH_PATH=$HOME/euroexa/arm/license
export ALLINEA_LICENSE_DIR=$HOME/euroexa/arm/license
export MPICC=armclang
