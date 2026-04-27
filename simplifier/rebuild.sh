#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "--- 🛠️  Cleaning old build artifacts... ---"
rm -rf build
mkdir build

echo "--- ⚙️  Configuring CMake... ---"
cd build
cmake ..

echo "--- 🏗️  Compiling Simplifier C++ Engine... ---"
make

echo "--- ✅ Build Complete! Testing simple expansion... ---"
./simplifier "x*x + 2*x + 1"