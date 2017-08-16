###################################################### 
#               INSTRUMENTATION                      #
######################################################

#PREP         = scorep --nocompiler #--thread=none #--user 
#PREP         = scalasca --instrument
USER_FUNC    = #-finstrument-functions

###################################################### 
#           SETTING PATHS AND NAMES                  #
######################################################
ROOT=../../../c++

#MPdir        = /opt/cray/mpt/7.3.1/gni/mpich-gnu/5.1/
#MPinc        = -I$(MPdir)/include
#MPlib        = -L$(MPdir)/lib -lmpich_gnu_51

#CFLAGS  += $(MPinc)
#LDFLAGS += $(MPlib)

###################################################### 
#                COMPILING                           #
######################################################

COMPILER_FLAGS=-g -dynamic

#CXX=mpicxx -fopenmp
CXX=$(PREP) CC $(COMPILER_FLAGS)
#CXX=$(PREP) CC -craype-verbose $(COMPILER_FLAGS)
#CXX=$(PREP) $(SCOREP_WRAPPER_INSTRUMENTER_FLAGS) g++ $(SCOREP_WRAPPER_COMPILER_FLAGS) $(COMPILER_FLAGS)  # SCORE-P

CFLAGS+=-DBPMF_MPI_ISENDIRECV_COMM  -DBPMF_SER_SCHED -DBPMF_PROFILING #$(HPL_INCLUDES)

###################################################### 
#             ADDITIONAL FILES                       #
######################################################

include ../../Makefile.common
