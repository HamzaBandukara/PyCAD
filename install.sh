#!/bin/bash

# Exit immediately if any command fails
set -e

echo "========================================================"
echo "  PyCAD Environment Installer                           "
echo "========================================================"

echo -e "\n---> [1/7] Updating Package Lists..."
sudo apt-get update

echo -e "\n---> [2/7] Installing System Dependencies..."
# build-essential/cmake: For C++ compilation
# libgmp-dev/libmpfr-dev: Backends for SymEngine (exact arithmetic)
# qepcad: Cylindrical Algebraic Decomposition engine
# maxima: Symbolic integration CAS
# python3-dev: Required for pybind11 C++ compilation
sudo apt-get install -y build-essential cmake git qepcad maxima \
                        python3-venv python3-pip python3-dev libgmp-dev libmpfr-dev

echo -e "\n---> [3/7] Compiling SymEngine C++ Library from Source..."
if [ ! -d "symengine" ]; then
    echo "Cloning SymEngine repository..."
    git clone --depth 1 https://github.com/symengine/symengine.git
    cd symengine
    mkdir build && cd build

    echo "Configuring CMake with GMP and MPFR backends..."
    cmake -DWITH_GMP=ON -DWITH_MPFR=ON -DBUILD_SHARED_LIBS=ON \
          -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF ..

    echo "Compiling using all available CPU cores..."
    make -j$(nproc)

    echo "Installing system-wide..."
    sudo make install
    sudo ldconfig
    cd ../..

    echo "Cleaning up SymEngine source (headers and libs are now installed)..."
    rm -rf symengine

    echo "SymEngine installed successfully."
else
    echo "SymEngine directory already exists. Skipping source build."
fi

echo -e "\n---> [4/7] Recreating Python Virtual Environment..."
VENV_DIR="$HOME/.venvs/pycad"
mkdir -p "$HOME/.venvs"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created virtual environment at $VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR. Upgrading..."
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

echo -e "\n---> [5/7] Installing Python Packages..."
pip install --upgrade pip
pip install sympy scipy numpy z3-solver matplotlib pybind11 pyyaml

echo -e "\n---> [6/7] Compiling Simplifier (Radical Oracle)..."
if [ -d "simplifier" ]; then
    cd simplifier
    chmod +x rebuild.sh
    ./rebuild.sh
    cd ..
    echo "Simplifier compiled successfully."
else
    echo -e "\033[0;31m[ERROR] simplifier/ directory not found. Skipping.\033[0m"
fi

echo -e "\n---> [7/7] Compiling C++ Math Engine..."
if [ -d "cpp_math_engine" ]; then
    cd cpp_math_engine
    chmod +x build.sh
    ./build.sh
    cd ..
    echo "C++ Math Engine compiled successfully."
else
    echo -e "\033[0;31m[ERROR] cpp_math_engine/ directory not found. Skipping.\033[0m"
fi

echo -e "\n========================================================"
echo "  Installation Complete!                                "
echo "========================================================"
echo "To run:       ./pycad.sh --test 25"
echo "              ./pycad.sh --file examples/my_test.yaml"
echo "========================================================"