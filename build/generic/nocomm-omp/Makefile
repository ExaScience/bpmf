###################################################### 
#               INSTRUMENTATION                      #
######################################################

#PREP         = scorep --user 
#PREP         = scalasca --instrument
USER_FUNC    = #-finstrument-functions

###################################################### 
#           SETTING PATHS AND NAMES                  #
######################################################

ROOT=../../../c++

SCOREP_WRAPPER_INSTRUMENTER_FLAGS= #--instrument-filter=/univ_1/ws1/ws/hpcasham-bpmf-openmp-0/SCOREP/filter_chembl_100_iter_func_add # --verbose #--keep-files
SCOREP_WRAPPER_COMPILER_FLAGS    =

###################################################### 
#                COMPILING                           #
######################################################

COMPILER_FLAGS=-fopenmp -g          # for g++
#COMPILER_FLAGS=-openmp -g -dynamic # for CC

#CXX=$(PREP) $(SCOREP_WRAPPER_INSTRUMENTER_FLAGS) icpc $(SCOREP_WRAPPER_COMPILER_FLAGS) $(COMPILER_FLAGS)
#CXX=$(PREP) $(SCOREP_WRAPPER_INSTRUMENTER_FLAGS) CC $(SCOREP_WRAPPER_COMPILER_FLAGS) $(COMPILER_FLAGS)
CXX=$(PREP) $(SCOREP_WRAPPER_INSTRUMENTER_FLAGS) g++ $(SCOREP_WRAPPER_COMPILER_FLAGS) $(COMPILER_FLAGS)  # SCORE-P
#CXX=g++ $(COMPILER_FLAGS) $(USER_FUNC)     
                                                               # EXTRA-E
CFLAGS+=-DBPMF_NO_COMM  -DBPMF_OMP_SCHED -DBPMF_PROFILING

###################################################### 
#             ADDITIONAL FILES                       #
######################################################

include ../../Makefile.common
