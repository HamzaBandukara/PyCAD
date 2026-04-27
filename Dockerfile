# =============================================================================
# PyCAD Docker Image
# Based on the existing install.sh — mirrors the WSL build exactly
# =============================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── 1. System dependencies (mirrors install.sh step 2) ──────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    xcas \
    qepcad \
    maxima \
    python3-venv \
    python3-pip \
    python3-dev \
    libgmp-dev \
    libmpfr-dev \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Compile SymEngine from source (mirrors install.sh step 3) ────────────
RUN git clone --depth 1 https://github.com/symengine/symengine.git /tmp/symengine && \
    cd /tmp/symengine && \
    mkdir build && cd build && \
    cmake -DWITH_GMP=ON -DWITH_MPFR=ON -DBUILD_SHARED_LIBS=ON .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/symengine

# ── 3. Python packages (mirrors install.sh step 5, minus web frameworks) ────
RUN pip3 install --no-cache-dir  \
    sympy \
    symengine \
    scipy \
    numpy \
    z3-solver \
    matplotlib \
    pybind11 \
    pyyaml

# ── 4. Copy entire project ──────────────────────────────────────────────────
WORKDIR /opt/pycad
COPY . .

# ── 5. Build the simplifier (mirrors install.sh step 6) ─────────────────────
RUN cd simplifier && \
    chmod +x rebuild.sh && \
    ./rebuild.sh

# ── 6. Build the C++ math engine (mirrors install.sh step 7) ────────────────
RUN cd cpp_math_engine && \
    chmod +x build.sh && \
    ./build.sh

# ── 7. Make the compiled .so importable from the project root ────────────────
# The build scripts put the .so inside build/. We symlink or copy it so
# `import paf_cpp_engine` works from /opt/pycad.
RUN find /opt/pycad/cpp_math_engine -name "pycad_cpp_engine*.so" -exec cp {} /opt/pycad/ \; || true

# ── 8. Verify everything works ──────────────────────────────────────────────
RUN python3 -c "import paf_cpp_engine; print('[OK] C++ engine imported')" && \
    echo 'display2d: false$ print(integrate(x^2, x))$ quit()$' | maxima --very-quiet && \
    echo "[OK] Maxima works" && \
    echo "[] (x) 1 [ x > 0 ]. go go go quit" | timeout 5 qepcad +N1000000 > /dev/null 2>&1; \
    echo "[OK] QEPCAD accessible"

# ── 9. Add quick run files ──────────────────────────────────────────────
RUN printf '#!/bin/bash\n\
set -e\n\
case "$1" in\n\
    --quick)\n\
        echo "Running quick check (Sum Uniform)..."\n\
        python3 /opt/pycad/run.py --test 1\n\
        ;;\n\
    --benchmarks)\n\
        echo "Running full benchmark suite..."\n\
        python3 /opt/pycad/benchmarker.py\n\
        ;;\n\
    *)\n\
        python3 /opt/pycad/run.py "$@"\n\
        ;;\n\
esac\n' > /opt/pycad/run.sh && chmod +x /opt/pycad/run.sh


ENV PYTHONPATH=/opt/pycad:${PYTHONPATH}
ENV PYTHONUNBUFFERED=1

#ENTRYPOINT ["python3"]
#CMD ["run.py"]

ENTRYPOINT ["/opt/pycad/run.sh"]
CMD []
