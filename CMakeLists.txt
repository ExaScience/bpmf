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

#get GIT describe
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

message(STATUS "GIT describe ${BPMF_VERSION}")

add_definitions(-DBPMF_VERSION=\"${BPMF_VERSION}\")
add_definitions(-DBPMF_NUMLATENT=${BPMF_NUMLATENT})
add_definitions(-DBPMF_${BPMF_COMM})

if(BPMF_COMM MATCHES ^MPI)
  find_package(MPI REQUIRED)
elif(BPMF_COMM MATCHES ^GPI)
  find_package(GASPI REQUIRED)
endif()

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

#SETUP OUTPUT
add_executable (bpmf ${SOURCE_FILES})
target_link_libraries (bpmf Eigen3::Eigen z)

