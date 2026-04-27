# PyCAD Docker Image

## Prerequisites
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))

## Project Structure

The Dockerfile expects the standard PyCAD layout:

```
py-cad/
├── Dockerfile              
├── .dockerignore           
├── run.py                  
├── benchmarker.py
├── cpp_math_engine/
│   ├── build.sh
│   ├── CMakeLists.txt
│   ├── engine_core.cpp
│   └── exprtk/
│       └── exprtk.hpp
├── simplifier/
│   ├── rebuild.sh
│   ├── CMakeLists.txt
│   ├── main.cpp
│   └── test_oracle.py
├── py_cad_modules/
│   ├── __init__.py
│   ├── cad_core.py
│   ├── calculus.py
│   ├── compositional_cad.py
│   ├── cpp_oracle.py
│   ├── preprocessing.py
│   ├── recursive_orchestrator.py
│   ├── utils.py
│   └── z3_engine.py
└── benchmark_plots/        ← excluded from image
```

## Building the Image

Place the `Dockerfile` and `.dockerignore` in your project root (`py-cad/`), then:

```bash
docker build --platform linux/amd64 -t pycad .
```

For convenience, we provide a script to do this:
```bash
./builddocker.sh
```

## Running

### Run the default test
```bash
docker run --rm pycad
```

## Running with Docker

### Building the image

```bash
docker build -t pycad .
```

### Basic usage

Run a built-in test case:
```bash
docker run --rm pycad run.py --test 25
```

Run from a YAML test file:
```bash
docker run --rm pycad run.py --file examples/bspline3_uniform.yaml
```

Run all YAML test files in a directory:
```bash
docker run --rm pycad run.py --dir examples/
```

### Options

```bash
docker run --rm pycad run.py --test 25 --no-plot      # skip plotting
docker run --rm pycad run.py --test 25 --no-z3        # disable Z3 simplification
docker run --rm pycad run.py --test 25 --silent        # suppress debug output
```

### Saving output files to your machine

Plots and CSV results are saved to `outputs/` inside the container. Mount this directory to retrieve them on your host:

**Linux / macOS:**
```bash
docker run --rm -v $(pwd)/outputs:/opt/pycad/outputs pycad run.py --test 4
```

**Windows PowerShell:**
```powershell
docker run --rm -v ${PWD}/outputs:/opt/pycad/outputs pycad run.py --test 4
```

**Windows CMD:**
```cmd
docker run --rm -v %cd%/outputs:/opt/pycad/outputs pycad run.py --test 4
```

Output files will appear in the `outputs/` folder in your current directory.

### Running benchmarks with output

**Linux / macOS:**
```bash
docker run --rm -v $(pwd)/outputs:/opt/pycad/outputs pycad benchmarker.py
```

**Windows PowerShell:**
```powershell
docker run --rm -v ${PWD}/outputs:/opt/pycad/outputs pycad benchmarker.py
```

**Windows CMD:**
```cmd
docker run --rm -v %cd%/outputs:/opt/pycad/outputs pycad benchmarker.py
```

### Interactive usage

Open a Python shell inside the container:
```bash
docker run --rm -it pycad
```

Open a bash shell for debugging:
```bash
docker run --rm -it --entrypoint bash pycad
```

## Troubleshooting

### QEPCAD memory errors
Increase container memory:
```bash
docker run --rm -m 4g pycad run.py
```

### Build fails at simplifier or cpp_math_engine
Debug interactively:
```bash
docker build --target builder -t pycad-debug .  # if using multi-stage
# or just:
docker run --rm -it ubuntu:22.04 bash
# then manually run install steps
```

### Verify individual components inside the container
```bash
docker run --rm -it --entrypoint bash pycad
# then:
python3 -c "import paf_cpp_engine; print('OK')"
echo "display2d: false$ print(integrate(x^2, x))$ quit()$" | maxima --very-quiet
echo "[] (x) 1 [ x > 0 ]. go go go quit" | qepcad +N500000000
```
