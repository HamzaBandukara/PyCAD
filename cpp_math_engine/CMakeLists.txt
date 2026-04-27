cmake_minimum_required(VERSION 3.15)
project(pycad_cpp_engine)

# 1. Find PyBind11 (using your Python venv)
find_package(pybind11 REQUIRED)

# 2. Find your custom-compiled SymEngine
find_package(SymEngine REQUIRED CONFIG PATH_SUFFIXES lib/cmake/symengine)

if(SymEngine_FOUND)
    message(STATUS "Found SymEngine version: ${SymEngine_VERSION}")

    # 3. Create the Python Shared Module (instead of an executable)
    pybind11_add_module(pycad_cpp_engine engine_core.cpp)

    # 4. Link SymEngine and GMP
    target_link_libraries(pycad_cpp_engine PRIVATE symengine gmp)
    target_include_directories(pycad_cpp_engine PRIVATE ${SYMENGINE_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "SymEngine not found.")
endif()