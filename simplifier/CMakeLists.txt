cmake_minimum_required(VERSION 3.10)
project(CadSimplifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use the path suffix that worked last time
find_package(SymEngine REQUIRED CONFIG PATH_SUFFIXES lib/cmake/symengine)

if(SymEngine_FOUND)
    message(STATUS "Found SymEngine version: ${SymEngine_VERSION}")

    add_executable(simplifier main.cpp)

    # Use the link names that worked in your old project: symengine and gmp
    target_link_libraries(simplifier PRIVATE symengine gmp)

    # If symengine includes aren't found automatically, add this:
    target_include_directories(simplifier PRIVATE ${SYMENGINE_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "SymEngine not found.")
endif()