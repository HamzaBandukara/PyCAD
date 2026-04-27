#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting PyCAD C++ Microservice Build..."

# ── Find Python (works in venv, system, and Docker) ─────────────────────────
if command -v python &> /dev/null; then
    PYTHON_EXE=$(which python)
elif command -v python3 &> /dev/null; then
    PYTHON_EXE=$(which python3)
else
    echo "❌ ERROR: No Python interpreter found."
    exit 1
fi

echo "📦 Python Executable: $PYTHON_EXE"

# ── Find pybind11 CMake config ──────────────────────────────────────────────
PYBIND11_CMAKE_DIR=$($PYTHON_EXE -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null || true)

if [ -z "$PYBIND11_CMAKE_DIR" ]; then
    echo "❌ ERROR: pybind11 not found. Install it with: pip install pybind11"
    exit 1
fi

echo "📍 pybind11 CMake dir: $PYBIND11_CMAKE_DIR"

# Create build directory and navigate into it
mkdir -p build
cd build

# Configure CMake
echo "⚙️  Configuring CMake..."
cmake -DPYTHON_EXECUTABLE="$PYTHON_EXE" \
      -Dpybind11_DIR="$PYBIND11_CMAKE_DIR" \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
      ..

# Compile using all available CPU cores
echo "🔨 Compiling with $(nproc) cores..."
make -j$(nproc)

# Copy the compiled .so file up to the main py-cad directory
echo "🚚 Copying compiled module to main directory..."
cp pycad_cpp_engine*.so ../../

echo "✅ Build and copy complete! Ready for Python."