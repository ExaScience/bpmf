cmake_minimum_required (VERSION 3.5.0)

project (bpmf)

# OPTIONS

OPTION(ENABLE_PROFILING "Enable performance counters" OFF)
OPTION(ENABLE_ASAN "Enable address sanitizer" OFF)
OPTION(ENABLE_TSAN "Enable thread sanitizer" OFF)

OPTION(ENABLE_OPENMP "Enable OpenMP Support" OFF)
OPTION(ENABLE_OMPSS "Enable OmpSS Support" ON)

SET(BPMF_NUMLATENT 32 CACHE INT "Number of latent dimensions") 
set(CMAKE_CXX_STANDARD 11)

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

add_definitions(-DBPMF_VERSION=\"${BPMF_VERSION}\")
add_definitions(-DBPMF_NUMLATENT=${BPMF_NUMLATENT})

find_package(Eigen3 REQUIRED)
find_package(ZLIB REQUIRED)

if(${ENABLE_PROFILING})
    add_definitions(-DBPMF_PROFILING)
endif()

if(${ENABLE_OMPSS})
    add_definitions(-DOMPSS)
endif()

include_directories(c++)

FILE (GLOB SOURCE_FILES "c++/*.cpp" "c++/*.h")

add_executable (bpmf ${SOURCE_FILES})
target_link_libraries (bpmf Eigen3::Eigen z)

if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(bpmf OpenMP::OpenMP_CXX)
endif()

if(ENABLE_OMPSS)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --ompss-2")
  find_package (MCC REQUIRED)
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
