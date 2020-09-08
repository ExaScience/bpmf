cmake_minimum_required (VERSION 3.5.0)

project (bpmf)

# OPTIONS

OPTION(ENABLE_PROFILING "Enable performance counters" OFF)
OPTION(ENABLE_ASAN "Enable address sanitizer" OFF)
OPTION(ENABLE_TSAN "Enable thread sanitizer" OFF)

OPTION(SCHEDULING "Enable OpenMP Support" ON)

SET(BPMF_NUMLATENT 32 CACHE INT "Number of latent dimensions") 

SET(BPMF_COMM "NO_COMM" CACHE STRING "Communication library used")
SET_PROPERTY(CACHE BPMF_COMM PROPERTY STRINGS 
  GPI_COMM 
  MPI_ISENDIRECV_COMM 
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

add_definitions(-DBPMF_VERSION=\"${BPMF_VERSION}\")
add_definitions(-DBPMF_NUMLATENT=${BPMF_NUMLATENT})
add_definitions(-DBPMF_${BPMF_COMM})



if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

find_package(Eigen3 REQUIRED)
find_library(z REQUIRED)

if(${ENABLE_PROFILING})
    add_definitions(-DBPMF_PROFILING)
endif()

include_directories(cpp)

FILE (GLOB SOURCE_FILES "c++/*.cpp" "c++/*.h")

add_executable (bpmf ${SOURCE_FILES})
target_link_libraries (bpmf Eigen3::Eigen z)

if(BPMF_COMM MATCHES ^.PI)
  find_package(MPI REQUIRED)
  set(MPIRUN mpirun -np 4)
  target_link_libraries (bpmf MPI::MPI_C)
  if(BPMF_COMM MATCHES ^GPI)
    find_path(GASPI_INCLUDE_DIR GASPI.h REQUIRED)
    include_directories(${GASPI_INCLUDE_DIR})
    find_library(GASPI_LIBRARY GPI2 REQUIRED)
    target_link_libraries (bpmf $GASPI_LIBRARY)
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