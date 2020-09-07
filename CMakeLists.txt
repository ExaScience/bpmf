cmake_minimum_required (VERSION 3.5.0)

project (bpmf)

# OPTIONS

OPTION(ENABLE_PROFILING "Enable performance counters" OFF)
OPTION(ENABLE_ASAN "Enable address sanitizer" OFF)
OPTION(ENABLE_TSAN "Enable thread sanitizer" OFF)

OPTION(ENABLE_OPENMP "Enable OpenMP Support" ON)
OPTION(ENABLE_MPI "Enable MPI Support" OFF)
OPTION(ENABLE_GASPI "Enable GASPI Support" OFF)

# INIT CMAKE

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -fstrict-aliasing")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -fPIC")

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

string(REGEX REPLACE "-[^-]+$"
    "" BPMF_VERSION_CLEAN
    ${BPMF_VERSION})

add_definitions(-DBPMF_VERSION=\"${BPMF_VERSION}\")

message(STATUS "Cleaned version: ${BPMF_VERSION_CLEAN}")

# CONFIGURE DEPENDENCIES
if(ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()

if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()
find_package(Eigen3 REQUIRED)

if(${ENABLE_PROFILING})
    add_definitions(-DPROFILING)
endif()

include_directories(cpp)

FILE (GLOB SOURCE_FILES "c++/*.cpp" "c++/*.h")

#SETUP OUTPUT
add_executable (bpmf ${SOURCE_FILES})
target_link_libraries (bpmf Eigen3::Eigen)

