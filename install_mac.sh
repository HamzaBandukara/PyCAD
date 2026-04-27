#!/bin/bash

# Exit immediately if any command fails
set -e

echo "========================================================"
echo "  PyCAD Environment Installer (macOS Version)           "
echo "========================================================"

# 0. Ensure Xcode Command Line Tools are installed (provides clang/clang++, make, etc.)
if ! xcode-select -p &> /dev/null; then
    echo -e "\n---> [0/7] Xcode Command Line Tools not found. Installing..."
    xcode-select --install
    echo "Please complete the Xcode CLT installation dialog, then re-run this script."
    exit 1
else
    echo -e "\n---> [0/7] Xcode Command Line Tools found at $(xcode-select -p)"
fi

# 1. Bootstrapping Homebrew if missing
if ! command -v brew &> /dev/null; then
    echo -e "\n---> [1/7] Homebrew not found! Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo -e "\n---> [1/7] Updating Homebrew..."
    brew update
fi

echo -e "\n---> [2/7] Installing System Dependencies..."
# cmake: For C++ compilation
# gmp/mpfr: Backends for SymEngine (exact arithmetic)
# maxima: Symbolic integration CAS
# python3 (via Homebrew) includes dev headers needed for pybind11
brew install cmake git maxima gmp mpfr python3

# macOS specific warning/handler for QEPCAD
if ! command -v qepcad &> /dev/null; then
    echo -e "\n\033[0;33m[WARNING] QEPCAD is not natively available in Homebrew.\033[0m"
    echo "To install QEPCAD on macOS, it is highly recommended to use MacPorts in a separate terminal:"
    echo "  sudo port install qepcad"
    echo "The installer will continue, but ensure 'qepcad' is in your PATH before running PyCAD."
fi

echo -e "\n---> [3/7] Compiling SymEngine C++ Library from Source..."
if [ ! -d "symengine" ]; then
    echo "Cloning SymEngine repository..."
    git clone --depth 1 https://github.com/symengine/symengine.git
    cd symengine
    mkdir build && cd build

    echo "Configuring CMake with GMP and MPFR backends..."
    # CMAKE_PREFIX_PATH ensures CMake finds Homebrew-installed GMP/MPFR,
    # which live under /opt/homebrew (Apple Silicon) or /usr/local (Intel).
    cmake -DWITH_GMP=ON -DWITH_MPFR=ON -DBUILD_SHARED_LIBS=ON \
          -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF \
          -DCMAKE_PREFIX_PATH="$(brew --prefix)" ..

    echo "Compiling using all available CPU cores..."
    # sysctl is the macOS equivalent to Linux's nproc
    make -j$(sysctl -n hw.ncpu)

    echo "Installing system-wide..."
    sudo make install

    # Note: macOS uses 'dyld' which dynamically links automatically (no ldconfig needed).

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