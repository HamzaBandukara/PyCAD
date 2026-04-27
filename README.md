# PyCAD: A Prototype Tool For Exact Distribution of Probabilistic Programs using CADs 

---
### About

PyCAD is an open-source tool for computing exact distributions of transformed random variables. Given an expression 
`Y = f(X₁, ..., Xₙ)` where each input has a known distribution, PyCAD derives the CDF and PDF of Y using Cylindrical 
Algebraic Decomposition — allowing engineers and researchers to obtain exact output distributions for nonlinear 
functions of uncertain inputs without resorting to Monte Carlo simulation. The pipeline combines QEPCAD B, Maxima, and a
custom C++ engine (built on ExprTk) within a Python framework, with Monte Carlo validation for verification.


## Getting Started

This section provides instructions for getting started with PyCAD. We recommend using the provided Docker image for the
fastest setup.

> **Platform requirement:** PyCAD runs natively on Linux, macOS, and Windows via WSL2. Docker on Windows
> requires a WSL2 backend.

---
### Option 1: Using the Provided Docker Image

We provide a Docker image that includes all prerequisites bundled and ready to use. The image is built on Ubuntu 22.04.
After pulling the Docker image, tag it locally for convenience:
```bash
docker pull ghcr.io/hamzabandukara/pycad:0.1.0
docker tag ghcr.io/hamzabandukara/pycad:0.1.0 pycad
```

#### Using `rundocker.sh` (Recommended)

The `rundocker.sh` wrapper handles volume mounts and argument forwarding automatically:
```bash
./rundocker.sh --test 4                          # Run a built-in test
./rundocker.sh --file my_test.yaml               # Run a YAML test file
./rundocker.sh --dir examples/                   # Run all YAMLs in a directory
./rundocker.sh --quick                           # Quick smoke test (case 1)
./rundocker.sh --benchmarks                      # Full benchmark suite
./rundocker.sh --shell                           # Open an interactive shell
```

Output files will appear in the `outputs/` folder in your current directory. PyCAD runs natively on Linux, macOS,
and Windows via WSL2. Docker on Windows requires a WSL2 backend.

#### Manual `docker run`

If you prefer to run Docker directly, mount the `outputs/` directory to retrieve plots and CSV results:

```bash
docker run --rm -v $(pwd)/outputs:/opt/pycad/outputs pycad <OPTIONS>
```

The image also accepts `--quick` (runs test case 1) and `--benchmarks` (runs the full benchmark suite):
```bash
docker run --rm -v $(pwd)/outputs:/opt/pycad/outputs pycad --quick
docker run --rm -v $(pwd)/outputs:/opt/pycad/outputs pycad --benchmarks
```

Benchmark plots will appear in `outputs/benchmark_plots/`.

---
### Option 2: Building Your Own Docker Image

To build the Docker image from source:
```bash
docker build -t pycad .
```

This will install all dependencies and compile the C++ components. Once built, use the commands from Option 1 to run.

---
### Option 3: Native Installation

For users who prefer not to use Docker. The installer creates a virtual environment at `$HOME/.venvs/pycad`.

**Linux / WSL2:**
```bash
./install.sh
```
Tested on Ubuntu 22.04 and multiple WSL2 environments. A pure Windows build is not available at this time.

**macOS (experimental):**
```bash
./install_mac.sh
```
> **Note:** The macOS installer is experimental and has not been fully tested. It requires a separate QEPCAD
> installation via MacPorts (`sudo port install qepcad`). If you encounter issues, we recommend using the
> Docker image (Option 1), which has been tested and confirmed working on macOS.

To run PyCAD after installation:
```bash
./pycad.sh <OPTIONS>
```

To run the benchmarker:
```bash
./pycadbenchmarker.sh
```

Output plots appear in `outputs/` and `outputs/benchmark_plots/` respectively.

**To uninstall:** delete the `py-cad` directory and the virtual environment at `$HOME/.venvs/pycad`.

---

## Usage

### Input Modes

| Flag | Description |
|------|-------------|
| `--test <i>` | Run a built-in test case by number (see table below) |
| `--file <path>` | Run a test case from a YAML file (see YAML section below) |
| `--dir <path>` | Run all YAML test cases in a directory |

### Logging

| Flag | Description |
|------|-------------|
| `--silent` | Suppress all output; only display execution times and K-S distance (default) |
| `--log` | Show basic output at each pipeline stage |
| `--verbose` | Show all output including debug traces. Consider redirecting to a file: `> log.txt` |

### General Options

| Flag | Description |
|------|-------------|
| `--no-z3` | Disable Z3 Boolean simplification |
| `--no-plot` | Skip plotting |
| `--plot-points <n>` | Number of evaluation points for plotting (default: 400) |

---
## YAML Structure

Test cases can be defined as YAML files and run with `--file my_test.yaml`. The formula syntax is as follows:

| Syntax | Meaning |
|--------|---------|
| `^` | Exponent (not `**`) |
| `&&` | Logical AND |
| `\|\|` | Logical OR |
| `rt(n, expr)` | nth root, e.g. `rt(2, x^2+y^2)` = √(x²+y²) |
| `If[cond, true, false]` | Piecewise expression |
| `pi` | π |

Note: although the parser accepts roots other than square roots, they are not yet supported.

### Example YAML File

```yaml
# The name of the experiment.
name: "x/(x+y) with Triangular inputs"

# The semi-algebraic formula with bounds included inline.
formula: "x/(x+y) < t && 1 < x < 4 && 1 < y < 4"

# Variable ordering. The first variable MUST always be 't'.
vars: [t, x, y]

# Output filename (without extension). Plots saved as <filename>.png
filename: "x_by_xy_triangular"

# (Optional) Integrand mode. Usually "auto" (derives the joint PDF).
integrand: "auto"

# (Optional) Override the automatic t-axis range for plots.
t_min: 0.1
t_max: 0.9

# (Optional) Variable distributions.
# If omitted, all variables are assumed Uniform over their formula bounds.
# Supported types: uniform, triangular, beta
distributions:
  x:
    type: triangular
    params: [1, 4, 2.5]
  y:
    type: triangular
    params: [1, 4, 2.5]

# (Optional) Monte Carlo validation.
# If provided, a Monte Carlo histogram is overlaid and a K-S distance is computed.
# If omitted, PyCAD attempts to auto-derive it from the formula (recommended to specify manually).
mc_formula: "x / (x + y)"

# (Optional) Monte Carlo bounds. Inferred from the formula if not specified.
mc_bounds:
  x: [1, 4]
  y: [1, 4]
```

### Running YAML Files with Docker

YAML files do not need to be included in the Docker image. The `rundocker.sh` wrapper automatically mounts files and
directories into the container:

```bash
./rundocker.sh --file my_test.yaml
./rundocker.sh --dir my_tests/
```

If running Docker manually, mount the YAML file or directory alongside the outputs volume:

**Single file:**
```bash
docker run --rm \
  -v $(pwd)/my_test.yaml:/opt/pycad/my_test.yaml \
  -v $(pwd)/outputs:/opt/pycad/outputs \
  pycad --file my_test.yaml
```

**Directory of files:**
```bash
docker run --rm \
  -v $(pwd)/my_tests:/opt/pycad/my_tests \
  -v $(pwd)/outputs:/opt/pycad/outputs \
  pycad --dir my_tests/
```

The built-in examples in `examples/` are already included in the image and can be run without mounting.

---

## Built-in Test Cases

Use `--test <i>` to run a built-in test case, where `<i>` is the test number from the table below.

| # | Name | Distribution | Vars |
|---|------|-------------|------|
| 1 | Sum (Standard Uniform) | Uniform | 3 |
| 2 | Sum (Triangular) | Triangular | 3 |
| 3 | Sum (Squared Uniform) | Uniform | 3 |
| 4 | eˣ Expansion — Uniform | Uniform | 1 |
| 5 | eˣ Expansion — Triangular | Triangular | 1 |
| 6 | Cos Approximation — Uniform | Uniform | 1 |
| 7 | Cos Approximation — Triangular | Triangular | 1 |
| 8 | Ln Approximation — Uniform | Uniform | 1 |
| 9 | Ln Approximation — Triangular | Triangular | 1 |
| 10 | Sin Approximation — Uniform | Uniform | 1 |
| 11 | Sin Approximation — Triangular | Triangular | 1 |
| 12 | x_by_xy (Uniform) | Uniform | 2 |
| 13 | x_by_xy (Triangular) | Triangular | 2 |
| 14 | nonlin1 (Uniform) | Uniform | 1 |
| 15 | nonlin1 (Triangular) | Triangular | 1 |
| 16 | bspline3 (Uniform) | Uniform | 1 |
| 17 | bspline3 (Triangular) | Triangular | 1 |
| 18 | If/Else Disjoint Example (Uniform) | Uniform | 1 |
| 19 | If/Else Disjoint Example (Triangular) | Triangular | 1 |
| 20 | Doppler Effect (Test 1) | Uniform | 3 |
| 21 | Doppler Effect (Test 2) | Uniform | 3 |
| 22 | Doppler Effect (Test 3) | Uniform | 3 |
| 23 | Gröbner Basis Reduction (Sphere ∩ Plane) | Uniform | 3 |
| 24 | Spherical | Uniform | 3 |
| 25 | IMU Complementary Filter (AST Root Elim) | Uniform | 3 |
| 26 | IMU Comp Filter (Root-Free Polynomial) | Uniform | 3 |
| 27 | IMU Vector Norm (Divide & Conquer) | Uniform | 3 |
| 28 | Compositional Example | Uniform | 3 |
| 29 | Rigid Body (Test 1) | Uniform | 3 |
| 30 | Sum (Beta 2,2) | Beta(2,2) | 3 |
| 31 | x/(x+y) (Beta 2,2) | Beta(2,2) | 2 |
| 32 | z/(z+1) (Beta 2,2) | Beta(2,2) | 1 |
| 33 | BSpline3 (Beta 2,2) | Beta(2,2) | 1 |
| 34 | If/Else Disjoint (Beta 2,2) | Beta(2,2) | 1 |
| 35 | Doppler 1 (Beta 2,2) | Beta(2,2) | 3 |
| 36 | Doppler 2 (Beta 2,2) | Beta(2,2) | 3 |
| 37 | Doppler 3 (Beta 2,2) | Beta(2,2) | 3 |

The "Vars" column indicates the number of random variables (excluding the threshold parameter `t`).

---

## Note to Artifact Reviewers

The K-S distances reported in our paper were computed using Mathematica (except for `get_meas`, which was computed by 
PyCAD). Since PyCAD's K-S distance is evaluated via Monte Carlo sampling against a numerical PDF, results may vary by 
approximately ±0.001 between runs. The PASSED/MARGINAL/FAILED classification should remain consistent across runs.

---

## Acknowledgements

This work was supported by the Research Institute on Verified Trustworthy Software (VeTSS) grant "*Probabilistic Precision Tuning*".
****
