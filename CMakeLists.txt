cmake_minimum_required (VERSION 3.5.0)

project (bpmf)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# OPTIONS

OPTION(ENABLE_PROFILING "Enable performance counters" OFF)
OPTION(ENABLE_ASAN "Enable address sanitizer" OFF)
OPTION(ENABLE_TSAN "Enable thread sanitizer" OFF)

OPTION(ENABLE_OPENMP "Enable OpenMP Support" ON)
OPTION(ENABLE_REDUCE "Reduce Mu/Lambda version" OFF)

SET(BPMF_NUMLATENT 32 CACHE INT "Number of latent dimensions") 
OPTION(ONLY_VARIANCE "Only compute variance, not covariance" OFF)

SET(BPMF_COMM "NO_COMM" CACHE STRING "Communication library used")
SET_PROPERTY(CACHE BPMF_COMM PROPERTY STRINGS 
  GPI_COMM 
  MPI_ISEND_COMM 
  MPI_PUT_COMM 
  MPI_ALLREDUCE_COMM 
  MPI_BCAST_COMM 
  NO_COMM)


set(CMAKE_CXX_STANDARD 14)

# suppress some pesky Eigen 3.3.7 warnings
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.0)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-in-bool-context")
endif()

if (ENABLE_ASAN)
  add_compile_options(-fno-omit-frame-pointer -fsanitize=address)
  add_link_options(-fno-omit-frame-pointer -fsanitize=address)
endif()

if (ENABLE_TSAN)
  add_compile_options(-fno-omit-frame-pointer -fsanitize=thread)
  add_link_options(-fno-omit-frame-pointer -fsanitize=thread)
endif()

execute_process(
  COMMAND git describe
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE BPMF_VERSION_ERROR
  OUTPUT_VARIABLE BPMF_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT "${BPMF_VERSION_ERROR}" STREQUAL "0")
   set(BPMF_VERSION "Unknown")
endif()

message(STATUS "Version from git: ${BPMF_VERSION}")
message(STATUS "num-latent: ${BPMF_NUMLATENT}")
message(STATUS "comm: ${BPMF_COMM}")

if (ONLY_VARIANCE)
  add_definitions(-DBPMF_NO_COVARIANCE)
endif()

add_definitions(-DBPMF_VERSION=\"${BPMF_VERSION}\")
add_definitions(-DBPMF_NUMLATENT=${BPMF_NUMLATENT})
add_definitions(-DBPMF_${BPMF_COMM})
if (ENABLE_REDUCE)
  add_definitions(-DBPMF_REDUCE)
endif()

find_package(Random123)
if (Random123_FOUND)
  add_definitions(-DBPMF_RANDOM123)
endif()

add_definitions(-DEIGEN_DONT_PARALLELIZE)

find_package(Eigen3 REQUIRED)
add_definitions(-DEIGEN_DONT_PARALLELIZE)

find_package(ZLIB REQUIRED)

if(${ENABLE_PROFILING})
    add_definitions(-DBPMF_PROFILING)
endif()

include_directories(c++)

FILE (GLOB SOURCE_FILES "c++/*.cpp" "c++/*.h")

add_executable (bpmf ${SOURCE_FILES})
target_link_libraries (bpmf Eigen3::Eigen ZLIB::ZLIB)

if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(bpmf OpenMP::OpenMP_CXX)
endif()

if(BPMF_COMM MATCHES ^.PI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  set(MPIRUN mpirun -np 4)
  target_link_libraries (bpmf MPI::MPI_CXX)
  if(BPMF_COMM MATCHES ^GPI)
    find_path(GASPI_INCLUDE_DIR GASPI.h REQUIRED)
    include_directories(${GASPI_INCLUDE_DIR})
    find_library(GASPI_REL_LIBRARY GPI2 REQUIRED)
    find_library(GASPI_DEBUG_LIBRARY GPI2-dbg REQUIRED)
    find_library(IBVERBS_LIBRARY ibverbs REQUIRED)
    target_link_libraries (bpmf debug ${GASPI_DEBUG_LIBRARY} ${IBVERBS_LIBRARY})
    target_link_libraries (bpmf optimized ${GASPI_REL_LIBRARY} ${IBVERBS_LIBRARY})
  endif()
endif()

enable_testing()

add_test(bpmf_uncompressed ${MPIRUN} bpmf -i 4 
  -n ${CMAKE_CURRENT_SOURCE_DIR}/data/movielens/ml-train.mtx
  -p ${CMAKE_CURRENT_SOURCE_DIR}/data/movielens/ml-test.mtx
)

add_test(bpmf_compressed ${MPIRUN} bpmf -i 4 
  -n ${CMAKE_CURRENT_SOURCE_DIR}/data/movielens/ml-train.mtx.gz
  -p ${CMAKE_CURRENT_SOURCE_DIR}/data/movielens/ml-test.mtx.gz
)

#  add_test(NAME bpmf_tiny_rmse 
#    COMMAND ./run_test.sh "${MPIRUN} ${CMAKE_CURRENT_BINARY_DIR}/bpmf" 4.1
#    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/tiny)
