# Require a recent version of cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# This project is C++ based.
project(hello_world_app CXX)

# Instruct cmake to find the HPX settings
find_package(HPX COMPONENTS)

# hpx_executable that uses hpx::iostreams
add_hpx_executable(hello_world
    SOURCES hello_world.cpp
    COMPONENT_DEPENDENCIES iostreams)
