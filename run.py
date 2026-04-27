# =================================================================================
# TEST SUITE CONFIGURATION
# =================================================================================

from sympy.core.relational import Relational

from py_cad_modules.cad_core import get_cad
from py_cad_modules.calculus import integrate_cad, calculate_pdf, plot_distributions
from py_cad_modules.utils import apiprint, format_clean_dnf, dbprint
from py_cad_modules.z3_engine import z3_minimize_for_qepcad
from py_cad_modules.preprocessing import expand_ifs_to_list, auto_split_distribution_modes, parse_mathematica_string, \
    ast_to_cad_string
from py_cad_modules.recursive_orchestrator import RecursiveOrchestrator
import sympy as sp
import numpy as np
import argparse
import sys
import os
import time
import datetime
from test_loader import load_test_file, load_test_dir

os.makedirs("outputs", exist_ok=True)


TEST_CASES = {
    # ---------------------------------------------------------
    # SUM DISTRIBUTIONS
    # ---------------------------------------------------------
    1: {
        "name": "Sum (Standard Uniform)",
        "formula": "1 < x < 2 && 1 < y < 2 && 1 < z < 2 && x + y + z < t",
        "vars": ["t", "x", "y", "z"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "sum_standard",
        "mc_func": lambda x, y, z: x + y + z,
        "mc_bounds": {'x': (1, 2), 'y': (1, 2), 'z': (1, 2)}
    },
    2: {
        "name": "Sum (Triangular)",
        "formula": "1 < x < 2 && 1 < y < 2 && 1 < z < 2 && x + y + z < t",
        "vars": ["t", "x", "y", "z"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "sum_triangular",
        "distributions": {
            'x': ('triangular', 1, 2, 1.5),
            'y': ('triangular', 1, 2, 1.5),
            'z': ('triangular', 1, 2, 1.5)
        },
        "mc_func": lambda x, y, z: x + y + z,
        "mc_bounds": {'x': (1, 2), 'y': (1, 2), 'z': (1, 2)},
        "mc_generators": {
            'x': lambda n: np.random.triangular(1, 1.5, 2, n),
            'y': lambda n: np.random.triangular(1, 1.5, 2, n),
            'z': lambda n: np.random.triangular(1, 1.5, 2, n)
        }
    },
    3: {
        "name": "Sum (Squared Uniform)",
        "formula": "1 < x < 2 && 1 < y < 2 && 1 < z < 2 && x^2 + y^2 + z^2 < t",
        "vars": ["t", "x", "y", "z"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "sum_squared",
        "mc_func": lambda x, y, z: x**2 + y**2 + z**2,
        "mc_bounds": {'x': (1, 2), 'y': (1, 2), 'z': (1, 2)}
    },

    # ---------------------------------------------------------
    # EXPANSIONS & APPROXIMATIONS
    # ---------------------------------------------------------
    4: {
        "name": "e^x Expansion - Uniform",
        "formula": "1 + x + ((x^2)/2) + ((x^3)/6) + ((x^4)/24) + ((x^5)/120) < t && 0 < x < 1",
        "vars": ["t", "x"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "exp_2_uni",
        "mc_func": lambda x: 1 + x + (x**2)/2 + (x**3)/6 + (x**4)/24 + (x**5)/120,
        "mc_bounds": {'x': (0, 1)}
    },
    5: {
        "name": "e^x Expansion - Triangular",
        "formula": "1 + x + ((x^2)/2) + ((x^3)/6) + ((x^4)/24) + ((x^5)/120) < t && 0 < x < 1",
        "vars": ["t", "x"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "exp_2_tri",
        "distributions": {'x': ('triangular', 0, 1, 0.5)},
        "mc_func": lambda x: 1 + x + (x**2)/2 + (x**3)/6 + (x**4)/24 + (x**5)/120,
        "mc_bounds": {'x': (0, 1)},
        "mc_generators": {'x': lambda n: np.random.triangular(0, 0.5, 1, n)}
    },
    6: {
        "name": "Cos Approximation - Uniform",
        "formula": "((x^4)/24) - ((x^2)/2) + 1 < t && 0 < x < pi/2",
        "vars": ["t", "x"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "cos_approx_3_uni",
        "mc_func": lambda x: (x**4)/24 - (x**2)/2 + 1,
        "mc_bounds": {'x': (0, float(np.pi/2))}
    },
    7: {
        "name": "Cos Approximation - Triangular",
        "formula": "((x^4)/24) - ((x^2)/2) + 1 < t && 0 < x < pi/2",
        "vars": ["t", "x"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "cos_approx_3_tri",
        "distributions": {'x': ('triangular', 0, float(np.pi/2), float(np.pi/4))},
        "mc_func": lambda x: (x**4)/24 - (x**2)/2 + 1,
        "mc_bounds": {'x': (0, float(np.pi/2))},
        "mc_generators": {'x': lambda n: np.random.triangular(0, float(np.pi/4), float(np.pi/2), n)}
    },
    8: {
        "name": "Ln Approximation - Uniform",
        "formula": "(1/5)*((x - 1)^5) - (1/4)*((x - 1)^4) + (1/3)*((x - 1)^3) - (1/2)*((x - 1)^2) + x - 1 < t && 0 < x < 1",
        "vars": ["t", "x"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "ln_approx_5_uni",
        "mc_func": lambda x: (1/5)*((x - 1)**5) - (1/4)*((x - 1)**4) + (1/3)*((x - 1)**3) - (1/2)*((x - 1)**2) + x - 1,
        "mc_bounds": {'x': (0, 1)}
    },
    9: {
        "name": "Ln Approximation - Triangular",
        "formula": "(1/5)*((x - 1)^5) - (1/4)*((x - 1)^4) + (1/3)*((x - 1)^3) - (1/2)*((x - 1)^2) + x - 1 < t && 0 < x < 1",
        "vars": ["t", "x"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "ln_approx_5_tri",
        "distributions": {'x': ('triangular', 0, 1, 0.5)},
        "mc_func": lambda x: (1/5)*((x - 1)**5) - (1/4)*((x - 1)**4) + (1/3)*((x - 1)**3) - (1/2)*((x - 1)**2) + x - 1,
        "mc_bounds": {'x': (0, 1)},
        "mc_generators": {'x': lambda n: np.random.triangular(0, 0.5, 1, n)}
    },
    10: {
        "name": "Sin Approximation - Uniform",
        "formula": "((x^5)/120) - ((x^3)/6) + x < t && 0 < x < pi/2",
        "vars": ["t", "x"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "sin_approx_uni",
        "mc_func": lambda x: (x**5)/120 - (x**3)/6 + x,
        "mc_bounds": {'x': (0, float(np.pi/2))}
    },
    11: {
        "name": "Sin Approximation - Triangular",
        "formula": "((x^5)/120) - ((x^3)/6) + x < t && 0 < x < pi/2",
        "vars": ["t", "x"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "sin_approx_tri",
        "distributions": {'x': ('triangular', 0, float(np.pi/2), float(np.pi/4))},
        "mc_func": lambda x: (x**5)/120 - (x**3)/6 + x,
        "mc_bounds": {'x': (0, float(np.pi/2))},
        "mc_generators": {'x': lambda n: np.random.triangular(0, float(np.pi/4), float(np.pi/2), n)}
    },

    # ---------------------------------------------------------
    # NON-LINEAR & SPLINES
    # ---------------------------------------------------------
    12: {
        "name": "x_by_xy (Uniform)",
        "formula": "x/(x+y) < t && 1 < x < 4 && 1 < y < 4",
        "vars": ["t", "x", "y"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "x_by_xy_uni",
        "mc_func": lambda x, y: x / (x + y),
        "mc_bounds": {'x': (1, 4), 'y': (1, 4)}
    },
    13: {
        "name": "x_by_xy (Triangular)",
        "formula": "x/(x+y) < t && 1 < x < 4 && 1 < y < 4",
        "vars": ["t", "x", "y"],
        "integrand": "auto",
        "t_min": 0.1, "t_max": 0.9,
        "filename": "x_by_xy_tri",
        "distributions": {
            'x': ('triangular', 1, 4, 2.5),
            'y': ('triangular', 1, 4, 2.5)
        },
        "mc_func": lambda x, y: x / (x + y),
        "mc_bounds": {'x': (1, 4), 'y': (1, 4)},
        "mc_generators": {
            'x': lambda n: np.random.triangular(1, 2.5, 4, n),
            'y': lambda n: np.random.triangular(1, 2.5, 4, n)
        }
    },
    14: {
        "name": "nonlin1 (Uniform)",
        "formula": "z/(z+1) < t && 0 < z < 999",
        "vars": ["t", "z"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "nonlin1_uni",
        "mc_func": lambda z: z / (z + 1),
        "mc_bounds": {'z': (0, 999)}
    },
    15: {
        "name": "nonlin1 (Triangular)",
        "formula": "z/(z+1) < t && 0 < z < 999",
        "vars": ["t", "z"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "nonlin1_tri",
        "distributions": {'z': ('triangular', 0, 999, 499.5)},
        "mc_func": lambda z: z / (z + 1),
        "mc_bounds": {'z': (0, 999)},
        "mc_generators": {'z': lambda n: np.random.triangular(0, 499.5, 999, n)}
    },
    16: {
        "name": "bspline3 (Uniform)",
        "formula": "-(u^3)/6 < t && 0 < u < 1",
        "vars": ["t", "u"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "bspline3_uni",
        "mc_func": lambda u: -(u**3) / 6,
        "mc_bounds": {'u': (0, 1)}
    },
    17: {
        "name": "bspline3 (Triangular)",
        "formula": "-(u^3)/6 < t && 0 < u < 1",
        "vars": ["t", "u"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "bspline3_tri",
        "distributions": {'u': ('triangular', 0, 1, 0.5)},
        "mc_func": lambda u: -(u**3) / 6,
        "mc_bounds": {'u': (0, 1)},
        "mc_generators": {'u': lambda n: np.random.triangular(0, 0.5, 1, n)}
    },

    # ---------------------------------------------------------
    # DISJOINT / PIECEWISE EXAMPLES
    # ---------------------------------------------------------
    18: {
        "name": "If/Else Disjoint Example (Uniform)",
        "formula": "If[x^2 - x >= 0, x/10, x^2 + 2] < t && 0 < x < 10",
        "vars": ["t", "x"],
        "integrand": "auto",
        "t_min": 0, "t_max": 5,
        "filename": "cav10",
        "mc_func": lambda x: np.where(x**2 - x >= 0, x/10, x**2 + 2),
        "mc_bounds": {'x': (0, 10)}
    },
    19: {
        "name": "If/Else Disjoint Example (Triangular)",
        "formula": "If[x^2 - x >= 0, x/10, x^2 + 2] < t && 0 < x < 10",
        "vars": ["t", "x"],
        "integrand": "auto",
        "t_min": 0, "t_max": 5,
        "filename": "cav10_triangular",
        "distributions": {'x': ('triangular', 0, 10, 5)},
        "mc_func": lambda x: np.where(x**2 - x >= 0, x/10, x**2 + 2),
        "mc_bounds": {'x': (0, 10)},
        "mc_generators": {'x': lambda n: np.random.triangular(0, 5, 10, n)}
    },

    # ---------------------------------------------------------
    # DOPPLER EFFECT & COMPLEX GEOMETRIES
    # ---------------------------------------------------------
    20: {
        "name": "Doppler Effect (Test 1)",
        "formula": "(-(3314/10 + (6/10)*T) * v) / ((3314/10 + (6/10)*T + u)^2) < t && -100 < u < 100 && 20 < v < 20000 && -30 < T < 50",
        "vars": ["t", "u", "v", "T"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "doppler_test_1",
        "mc_func": lambda u, v, T: (-(331.4 + 0.6*T) * v) / ((331.4 + 0.6*T + u)**2),
        "mc_bounds": {'u': (-100, 100), 'v': (20, 20000), 'T': (-30, 50)}
    },
    21: {
        "name": "Doppler Effect (Test 2)",
        "formula": "(-(3314/10 + (6/10)*T) * v) / ((3314/10 + (6/10)*T + u)^2) < t && -125 < u < 125 && 15 < v < 25000 && -40 < T < 60",
        "vars": ["t", "u", "v", "T"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "doppler_test_2",
        "mc_func": lambda u, v, T: (-(331.4 + 0.6*T) * v) / ((331.4 + 0.6*T + u)**2),
        "mc_bounds": {'u': (-125, 125), 'v': (15, 25000), 'T': (-40, 60)}
    },
    22: {
        "name": "Doppler Effect (Test 3)",
        "formula": "(-(3314/10 + (6/10)*T) * v) / ((3314/10 + (6/10)*T + u)^2) < t && -30 < u < 120 && 320 < v < 20300 && -50 < T < 30",
        "vars": ["t", "u", "v", "T"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "doppler_test_3",
        "mc_func": lambda u, v, T: (-(331.4 + 0.6*T) * v) / ((331.4 + 0.6*T + u)**2),
        "mc_bounds": {'u': (-30, 120), 'v': (320, 20300), 'T': (-50, 30)}
    },
    23: {
        "name": "Gröbner Basis Reduction (Sphere Intersecting Plane)",
        "formula": "x^2 + y^2 + z^2 < t && x + y == 1 && -2 < z < 2",
        "vars": ["t", "x", "y", "z"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "groebner_sphere",
        "mc_func": lambda x, z: x ** 2 + (1 - x) ** 2 + z ** 2,
        "mc_bounds": {'x': (-3, 4), 'z': (-2, 2)}
    },

    # ---------------------------------------------------------
    # IMU / SENSOR TOPOLOGIES
    # ---------------------------------------------------------
    24: {
        "name": "Spherical",
        "formula": "x/(rt(2, x^2+y^2+z^2)) < t && 0 < x < 1 && 0 < y < 1 && 0 < z < 1",
        "vars": ["t", "x", "y", "z"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "v_norm",
        "mc_func": lambda x, y, z: x / np.sqrt(x ** 2 + y ** 2 + z ** 2),
        "mc_bounds": {'x': (0, 1), 'y': (0, 1), 'z': (0, 1)}
    },
    25: {
        "name": "IMU Complementary Filter (AST Root Elim)",
        "formula": "If[c >= 0, rt(2, (c/rt(2, a^2 + b^2 + c^2) + 1) * 0.5), (-b/rt(2, a^2 + b^2 + c^2)) / (2.0 * rt(2, (1 - c/rt(2, a^2 + b^2 + c^2)) * 0.5))] < t && -1 < a < 1 && -1 < b < 1 && -1 < c < 1",
        "vars": ["t", "a", "b", "c"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "imu_comp_filter",
        "mc_func": lambda a, b, c: np.where(
            c >= 0,
            np.sqrt((c / np.sqrt(a ** 2 + b ** 2 + c ** 2) + 1) * 0.5),
            (-b / np.sqrt(a ** 2 + b ** 2 + c ** 2)) / (
                        2.0 * np.sqrt((1 - c / np.sqrt(a ** 2 + b ** 2 + c ** 2)) * 0.5))
        ),
        "mc_bounds": {'a': (-1, 1), 'b': (-1, 1), 'c': (-1, 1)}
    },
    26: {
        "name": "IMU Comp Filter (Root-Free Polynomial)",
        "formula": "[ [ c >= 0 && t > 0 && 1 - 2*t^2 < 0 && c^2 < (a^2 + b^2 + c^2) * (1 - 2*t^2)^2 ] || [ c < 0 && [ t >= 0 || [ t < 0 && 2*t^2 - b^2 < 0 && 4*t^4*c^2 < (a^2 + b^2 + c^2)*(b^2 - 2*t^2)^2 ] ] ] ] && 0 < a < 1 && 0 < b < 1 && -1 < c < 1",
        "vars": ["t", "a", "b", "c"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "imu_comp_filter_rf",
        "mc_func": lambda a, b, c: np.where(
            c >= 0,
            np.sqrt((c / np.sqrt(a ** 2 + b ** 2 + c ** 2) + 1) * 0.5),
            (-b / np.sqrt(a ** 2 + b ** 2 + c ** 2)) / (
                        2.0 * np.sqrt((1 - c / np.sqrt(a ** 2 + b ** 2 + c ** 2)) * 0.5))
        ),
        "mc_bounds": {'a': (0, 1), 'b': (0, 1), 'c': (-1, 1)}
    },
    27: {
        "name": "IMU Vector Norm (Divide & Conquer)",
        "formula": [
            "t < 0 && x < 0 && (x^2 + y^2 + z^2) * t^2 < x^2 && 0 < x < 1 && 0 < y < 1 && 0 < z < 1",
            "t > 0 && x > 0 && (x^2 + y^2 + z^2) * t^2 > x^2 && 0 < x < 1 && 0 < y < 1 && 0 < z < 1"
        ],
        "vars": ["t", "x", "y", "z"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "imu_vector_norm_split",
        "mc_func": lambda x, y, z: x / np.sqrt(x ** 2 + y ** 2 + z ** 2),
        "mc_bounds": {'x': (0, 1), 'y': (0, 1), 'z': (0, 1)}
    },
    28: {
        "name": "Compositional Example",
        "formula": "(x / (x**2 + y**2 + z**2) < t) && (0 < x < 1) && (0 < y < 1) && (0 < z < 1)",
        # "formula": "(x / (x**2 + y**2 + z**2) < t) & (x > 0) & (x < 1) & (y > 0) & (y < 1) & (z > 0) & (z < 1)",
        "vars": ["t", "x", "y", "z"],
        "integrand": "auto",
        "t_min": 0.0, "t_max": 2.0,
        "filename": "compositional_example",
        "mc_func": lambda x, y, z: x / (x**2 + y**2 + z**2),
        "mc_bounds": {'x': (0, 1), 'y': (0, 1), 'z': (0, 1)}
    },
    29: {
        "name": "Rigid Body (Test 1)",
        "formula": "-x1*x2 - 2*x2*x3 - x1 - x3 < t && -15 < x1 < 15 && -15 < x2 < 15 && -15 < x3 < 15",
        "vars": ["t", "x1", "x2", "x3"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "rigidbody1",
        "mc_func": lambda x1, x2, x3: -x1*x2 - 2*x2*x3 - x1 - x3,
        "mc_bounds": {'x1': (-15, 15), 'x2': (-15, 15), 'x3': (-15, 15)}
    },
    # ---------------------------------------------------------
    # BETA(2,2) DISTRIBUTION MIRRORS (same formulas as uniform)
    # ---------------------------------------------------------
    30: {
        "name": "Sum (Beta 2,2)",
        "formula": "1 < x < 2 && 1 < y < 2 && 1 < z < 2 && x + y + z < t",
        "vars": ["t", "x", "y", "z"],
        "integrand": "auto",
        "distributions": {
            'x': ('beta', 2, 2, 1, 2),
            'y': ('beta', 2, 2, 1, 2),
            'z': ('beta', 2, 2, 1, 2)
        },
        "mc_func": lambda x, y, z: x + y + z,
        "mc_bounds": {'x': (1, 2), 'y': (1, 2), 'z': (1, 2)},
        "mc_generators": {
            'x': lambda n: 1 + np.random.beta(2, 2, n),
            'y': lambda n: 1 + np.random.beta(2, 2, n),
            'z': lambda n: 1 + np.random.beta(2, 2, n)
        }
    },
    31: {
        "name": "x/(x+y) (Beta 2,2)",
        "formula": "x/(x+y) < t && 1 < x < 4 && 1 < y < 4",
        "vars": ["t", "x", "y"],
        "integrand": "auto",
        "distributions": {
            'x': ('beta', 2, 2, 1, 4),
            'y': ('beta', 2, 2, 1, 4)
        },
        "mc_func": lambda x, y: x / (x + y),
        "mc_bounds": {'x': (1, 4), 'y': (1, 4)},
        "mc_generators": {
            'x': lambda n: 1 + 3 * np.random.beta(2, 2, n),
            'y': lambda n: 1 + 3 * np.random.beta(2, 2, n)
        }
    },
    32: {
        "name": "z/(z+1) (Beta 2,2)",
        "formula": "z/(z+1) < t && 0 < z < 999",
        "vars": ["t", "z"],
        "integrand": "auto",
        "distributions": {
            'z': ('beta', 2, 2, 0, 999)
        },
        "mc_func": lambda z: z / (z + 1),
        "mc_bounds": {'z': (0, 999)},
        "mc_generators": {
            'z': lambda n: 999 * np.random.beta(2, 2, n)
        }
    },
    33: {
        "name": "BSpline3 (Beta 2,2)",
        "formula": "-(u^3)/6 < t && 0 < u < 1",
        "vars": ["t", "u"],
        "integrand": "auto",
        "distributions": {
            'u': ('beta', 2, 2, 0, 1)
        },
        "mc_func": lambda u: -(u ** 3) / 6,
        "mc_bounds": {'u': (0, 1)},
        "mc_generators": {
            'u': lambda n: np.random.beta(2, 2, n)
        }
    },
    34: {
        "name": "If/Else Disjoint (Beta 2,2)",
        "formula": "If[x^2 - x >= 0, x/10, x^2 + 2] < t && 0 < x < 10",
        "vars": ["t", "x"],
        "t_min": 0, "t_max": 5,
        "integrand": "auto",
        "distributions": {
            'x': ('beta', 2, 2, 0, 10)
        },
        "mc_func": lambda x: np.where(x ** 2 - x >= 0, x / 10, x ** 2 + 2),
        "mc_bounds": {'x': (0, 10)},
        "mc_generators": {
            'x': lambda n: 10 * np.random.beta(2, 2, n)
        }
    },
    35: {
        "name": "Doppler 1 (Beta 2,2)",
        "formula": "(-(3314/10 + (6/10)*T) * v) / ((3314/10 + (6/10)*T + u)^2) < t && -100 < u < 100 && 20 < v < 20000 && -30 < T < 50",
        "vars": ["t", "u", "v", "T"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "doppler1_beta",
        "distributions": {
            'u': ('beta', 2, 2, -100, 100),
            'v': ('beta', 2, 2, 20, 20000),
            'T': ('beta', 2, 2, -30, 50)
        },
        "mc_func": lambda u, v, T: (-(331.4 + 0.6 * T) * v) / ((331.4 + 0.6 * T + u) ** 2),
        "mc_bounds": {'u': (-100, 100), 'v': (20, 20000), 'T': (-30, 50)},
        "mc_generators": {
            'u': lambda n: -100 + 200 * np.random.beta(2, 2, n),
            'v': lambda n: 20 + 19980 * np.random.beta(2, 2, n),
            'T': lambda n: -30 + 80 * np.random.beta(2, 2, n)
        }
    },
    36: {
        "name": "Doppler 2 (Beta 2,2)",
        "formula": "(-(3314/10 + (6/10)*T) * v) / ((3314/10 + (6/10)*T + u)^2) < t && -125 < u < 125 && 15 < v < 25000 && -40 < T < 60",
        "vars": ["t", "u", "v", "T"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "doppler2_beta",
        "distributions": {
            'u': ('beta', 2, 2, -125, 125),
            'v': ('beta', 2, 2, 15, 25000),
            'T': ('beta', 2, 2, -40, 60)
        },
        "mc_func": lambda u, v, T: (-(331.4 + 0.6 * T) * v) / ((331.4 + 0.6 * T + u) ** 2),
        "mc_bounds": {'u': (-125, 125), 'v': (15, 25000), 'T': (-40, 60)},
        "mc_generators": {
            'u': lambda n: -125 + 250 * np.random.beta(2, 2, n),
            'v': lambda n: 15 + 24985 * np.random.beta(2, 2, n),
            'T': lambda n: -40 + 100 * np.random.beta(2, 2, n)
        }
    },
    37: {
        "name": "Doppler 3 (Beta 2,2)",
        "formula": "(-(3314/10 + (6/10)*T) * v) / ((3314/10 + (6/10)*T + u)^2) < t && -30 < u < 120 && 320 < v < 20300 && -50 < T < 30",
        "vars": ["t", "u", "v", "T"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "doppler3_beta",
        "distributions": {
            'u': ('beta', 2, 2, -30, 120),
            'v': ('beta', 2, 2, 320, 20300),
            'T': ('beta', 2, 2, -50, 30)
        },
        "mc_func": lambda u, v, T: (-(331.4 + 0.6 * T) * v) / ((331.4 + 0.6 * T + u) ** 2),
        "mc_bounds": {'u': (-30, 120), 'v': (320, 20300), 'T': (-50, 30)},
        "mc_generators": {
            'u': lambda n: -30 + 150 * np.random.beta(2, 2, n),
            'v': lambda n: 320 + 19980 * np.random.beta(2, 2, n),
            'T': lambda n: -50 + 80 * np.random.beta(2, 2, n)
        }
    },

}


class MuteOutput:
    """Context manager to instantly silence all terminal output when benchmarking."""

    def __init__(self, mute: bool = False):
        self.mute = mute


    def __enter__(self):
        if self.mute:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mute:
            sys.stdout.close()
            sys.stdout = self._original_stdout



def run_test_orchestrator(tc: dict, z3_flg: bool = False, silent: bool = False, plot: bool = True):
    """
    Executes the full CAD pipeline for a given test configuration.
    Returns: (pdf_expr, timings_dict)
    """
    timings = {"cad": 0.0, "cdf": 0.0, "pdf": 0.0, "plot": 0.0, "total": 0.0, "KS": -1.0}
    t_start_total = time.perf_counter()

    with MuteOutput(mute=silent):
        print(f"\n{'=' * 60}")
        print(f"--- RUNNING TEST: {tc.get('name', 'Unknown')} ---")
        print(f"{'=' * 60}\n")
        print(f"--- Process Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

        # Grab formula(s)
        raw_formulas = tc["formula"] if isinstance(tc["formula"], list) else [tc["formula"]]

        # Expand 'If' statements and shatter Distributions
        independent_branches = []
        for f in raw_formulas:
            independent_branches.extend(expand_ifs_to_list(f))

        distributions = tc.get("distributions", None)
        if distributions is not None:
            independent_branches = auto_split_distribution_modes(independent_branches, distributions)

        # =========================================================================
        # RECURSIVE ORCHESTRATOR
        # =========================================================================
        apiprint("\n[API] Initializing Compositional CAD Engine...")
        base_bounds = tc.get("mc_bounds", {})
        recursive_engine = RecursiveOrchestrator(base_bounds=base_bounds, distributions=distributions)

        processed_branches = []
        custom_distributions = {}

        for branch_formula in independent_branches:
            sym_vars = list(sp.symbols(' '.join(tc["vars"]), real=True))

            # Parse the input via the Front-Door Translator
            local_syms = {str(s): s for s in sym_vars + [sp.Symbol('t', real=True)]}
            formula_expr = parse_mathematica_string(branch_formula, local_syms)

            # Evaluate subtrees
            new_expr, new_vars, new_dists, new_bounds, sub_map = recursive_engine.evaluate_formula(
                formula_expr, sym_vars
            )

            dbprint("[RUN-ROOT] New Expr, vars, dists, bounds:\n", new_expr, "\n", new_vars, "\n", new_dists, "\n",
                     new_bounds)

            # Reconstruct the QEPCAD boolean string from the pure AST
            new_formula_str = ast_to_cad_string(new_expr)

            # Inject localized bounds generated by the subtrees
            for v_str, (b_min, b_max) in new_bounds.items():
                new_formula_str += f" && [ {v_str} > {b_min} ] && [ {v_str} < {b_max} ]"

            # Only inject bounds for variables whose bounds are NOT already in the formula.
            # The original formula (from auto_split or the user) already contains bounds
            # like "1 < x < 4".
            active_var_names = [str(v) for v in new_vars]
            formula_vars = [str(s) for s in new_expr.free_symbols]

            for v_str, bounds in base_bounds.items():
                if v_str in active_var_names and v_str not in new_bounds and v_str in formula_vars:
                    # Check if this variable's bounds are already in the formula string
                    # (from the original formula or auto_split)
                    b_low, b_high = bounds
                    already_has_lower = f"{v_str} - {b_low} > 0" in new_formula_str or f"{b_low} - {v_str} < 0" in new_formula_str or f"{v_str} > {b_low}" in new_formula_str or f"{b_low} < {v_str}" in new_formula_str
                    already_has_upper = f"{v_str} - {b_high} < 0" in new_formula_str or f"{b_high} - {v_str} > 0" in new_formula_str or f"{v_str} < {b_high}" in new_formula_str or f"{b_high} > {v_str}" in new_formula_str

                    if not already_has_lower and b_low != -sp.oo and b_low is not None:
                        new_formula_str += f" && [ {v_str} - {b_low} > 0 ]"
                    if not already_has_upper and b_high != sp.oo and b_high is not None:
                        new_formula_str += f" && [ {v_str} - {b_high} < 0 ]"

            # Fracture generated Sub-CAD piecewises (u0, u1, etc.) at their
            # internal mode kinks — but NOT at their base bound conditions.

            active_sym_names = set([str(v) for v in new_vars] + ['t'])
            unique_fractures = set()

            for v_str, dist_expr in new_dists.items():
                if isinstance(dist_expr, sp.Expr) and dist_expr.has(sp.Piecewise):
                    # Collect the base bounds for this variable so we can skip them
                    v_bounds_vals = set()
                    if v_str in new_bounds:
                        v_bounds_vals.add(float(new_bounds[v_str][0]))
                        v_bounds_vals.add(float(new_bounds[v_str][1]))

                    for pw in dist_expr.find(sp.Piecewise):
                        for branch, cond in pw.args:
                            if cond in (True, False, sp.S.true, sp.S.false): continue
                            for rel in cond.atoms(Relational):
                                rel_vars = set([str(s) for s in rel.free_symbols])
                                if not rel_vars.issubset(active_sym_names):
                                    continue

                                poly = rel.lhs - rel.rhs
                                frac_expr = sp.cancel(poly)
                                numer, denom = frac_expr.as_numer_denom()
                                if denom.is_number and denom < 0:
                                    numer = -numer

                                # Skip if this is just a base-bound condition
                                # (e.g., u0 >= 2 or u0 <= 4 for u0 ∈ [2, 4])
                                try:
                                    roots = sp.solve(numer, sp.Symbol(v_str, real=True))
                                    if roots and len(roots) == 1 and roots[0].is_number:
                                        if float(roots[0]) in v_bounds_vals:
                                            continue  # This is a base bound, not a mode kink
                                except Exception:
                                    pass

                                poly_str = str(sp.expand(numer)).replace('**', '^')
                                unique_fractures.add(f"[ [ {poly_str} < 0 ] || [ {poly_str} > 0 ] ]")

            for frac_str in unique_fractures:
                new_formula_str += f" && {frac_str}"

            # 2. Fracture active global base distributions (e.g., triangular)
            # Not needed for e.g., normal, uniform, etc.
            if distributions:
                active_var_names = [str(v) for v in new_vars]
                for v_str, dist_info in distributions.items():
                    if v_str in active_var_names and isinstance(dist_info, tuple) and dist_info[
                        0].lower() == 'triangular':
                        # Skip — auto_split already added x < mode or x > mode
                        continue
            dbprint("[RUN-ROOT] Fractured Formula sent to CAD:", new_formula_str)
            processed_branches.append((new_formula_str, [str(v) for v in new_vars]))
            custom_distributions.update(new_dists)

        # Merge any newly generated compositional distributions into the global dict
        if distributions is None:
            distributions = {}
        distributions = {str(k): v for k, v in distributions.items()}
        custom_distributions = {str(k): v for k, v in custom_distributions.items()}
        distributions.update(custom_distributions)
        for v_str, bounds in base_bounds.items():
            v_str = str(v_str)  # Ensure it's a string
            if v_str not in distributions:
                b_min, b_max = bounds
                if b_min != -sp.oo and b_max != sp.oo:
                    v_sym = sp.Symbol(v_str, real=True)
                    # Force exact Rational math to prevent float drift
                    b_min_rat = sp.Rational(str(b_min))
                    b_max_rat = sp.Rational(str(b_max))
                    dbprint("b_min_rat: ", b_min_rat)
                    dbprint("b_max_rat: ", b_max_rat)
                    prob_val = 1 / (b_max_rat - b_min_rat)

                    distributions[v_str] = sp.Piecewise(
                        (prob_val, sp.And(v_sym > b_min_rat, v_sym < b_max_rat)),
                        (0, True)
                    )
        # 3. Z3 COMPRESSION (Optional, recommended, especially for the large IMU Comp Filter Example)
        if z3_flg:
            compressed_branches = []
            for branch_formula, branch_vars in processed_branches:
                compressed_branches.append((z3_minimize_for_qepcad(branch_formula, branch_vars), branch_vars))
            processed_branches = compressed_branches

        total_cdf_expr = sp.S.Zero
        t_var_sym = None
        cad_successes = 0

        # 4. Iterate over the branches
        time_cad = 0.0
        time_cdf = 0.0
        for i, (branch_formula, branch_vars) in enumerate(processed_branches):
            if len(processed_branches) > 1:
                apiprint(f"\n>>> ORCHESTRATOR: Processing Independent Branch {i + 1}/{len(processed_branches)}")
                dbprint(f"Branch Logic: {branch_formula[:100]}...\n")

            # A. Generate CAD (Timed)
            t_cad0 = time.perf_counter()
            cad_result = get_cad(branch_formula, branch_vars)
            time_cad += (time.perf_counter() - t_cad0)

            if not cad_result or not cad_result.get("success"):
                dbprint(f"Branch {i + 1} CAD Generation failed. Skipping.")
                continue

            active_dists = {}
            if distributions:
                active_dists.update(distributions)
            if custom_distributions:
                active_dists.update(custom_distributions)

            # B. Integrate CAD (Timed)
            t_cdf0 = time.perf_counter()
            integration_result = integrate_cad(
                cad_data=cad_result,
                num_params=1,
                force_closed_form=False,
                integrand=tc.get("integrand", "auto"),
                global_bounds=tc.get("mc_bounds", None),
                distributions=active_dists,
            )
            time_cdf += (time.perf_counter() - t_cdf0)

            # C. Accumulate the CDFs
            if integration_result and "cdf_expr" in integration_result:
                total_cdf_expr = sp.Add(total_cdf_expr, integration_result["cdf_expr"])
                t_var_sym = integration_result["t_var"]

                outer_dnf = integration_result.get("cad_dnf", sp.S.false)
                full_dnf = outer_dnf.subs(sub_map)

                sub_var_conds = []
                for orig_var in formula_expr.free_symbols:
                    if str(orig_var) in base_bounds and str(orig_var) not in [str(v) for v in new_vars]:
                        b_min, b_max = base_bounds[str(orig_var)]
                        if b_min != -sp.oo: sub_var_conds.append(orig_var > b_min)
                        if b_max != sp.oo: sub_var_conds.append(orig_var < b_max)

                if sub_var_conds:
                    full_dnf = sp.And(full_dnf, *sub_var_conds)

                dbprint("\n" + "=" * 80)
                dbprint("FINAL CLOSED-FORM 4D CAD (MINIMAL DNF):")
                dbprint(format_clean_dnf(full_dnf))
                dbprint("=" * 80 + "\n")
                cad_successes += 1

        timings["cad"] = time_cad
        timings["cdf"] = time_cdf

        if cad_successes == 0 or t_var_sym is None:
            dbprint("Fatal Error: All branches failed integration.")
            timings["total"] = time.perf_counter() - t_start_total
            return None, timings

        # =========================================================================
        # Probability Normalization
        # =========================================================================
        try:
            t_max_raw = tc.get("t_max")
            if t_max_raw is None:
                t_max_raw = 10.0

            t_eval_point = float(t_max_raw) + 10.0

            raw_eval = total_cdf_expr.evalf(subs={t_var_sym: t_eval_point})
            c_val = complex(raw_eval)
            max_val = c_val.real

            # 3. Calculate the integer multiplier
            scaling_factor = int(round(max_val))

            # 4. Normalize
            if scaling_factor > 1:
                apiprint(f"\n[GEOMETRY] Normalizing Probability Space. Branch Multiplier: {scaling_factor}")
                normalized_cdf = sp.expand(total_cdf_expr / scaling_factor)
            else:
                normalized_cdf = total_cdf_expr

        except Exception as e:
            dbprint(f"\n[NORMALIZATION WARNING] Could not verify asymptotic CDF limit: {e}")
            normalized_cdf = total_cdf_expr

        combined_result = {
            "cdf_expr": normalized_cdf,
            "t_var": t_var_sym
        }

        # 6. Derive PDF (Timed)
        t_pdf0 = time.perf_counter()
        pdf_expr = calculate_pdf(combined_result)
        timings["pdf"] = time.perf_counter() - t_pdf0
        timings["pdf"] = time.perf_counter() - t_pdf0

        # 6. Evaluate and Plot (Timed)
        k_dist = -1
        if plot:
            t_plot0 = time.perf_counter()
            k_dist = plot_distributions(
                cad_integration_result=combined_result,
                pdf_expr=pdf_expr,
                t_min=tc.get("t_min"),
                t_max=tc.get("t_max"),
                filename=tc.get("filename", "distribution"),
                mc_func=tc.get("mc_func"),
                mc_bounds=tc.get("mc_bounds"),
                mc_generators=tc.get("mc_generators")
            )
            timings["plot"] = time.perf_counter() - t_plot0

    timings["total"] = time.perf_counter() - t_start_total
    timings["KS"] = k_dist
    return pdf_expr, timings


"""
Supports three input modes:
  1. ./pycad.sh --test 25             Run a built-in test case by number
  2. ./pycad.sh --file my_test.yaml   Run from a YAML file
  3. ./pycad.sh --dir examples/       Run all YAML files in a directory

All modes accept:
  --no-plot    Skip plotting
  --silent     Suppress all output (default)
  --log        Show API-level output (normal)
  --verbose    Show all output including debug traces (loud)

Docker users: replace ./pycad.sh with ./rundocker.sh (or docker run ... pycad).
"""

# =================================================================================
# EXECUTION
# =================================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PyCAD — Cylindrical Algebraic Decomposition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (native):
  ./pycad.sh --test 25
  ./pycad.sh --test 25 --log
  ./pycad.sh --test 25 --verbose
  ./pycad.sh --file examples/x_by_xy_triangular.yaml
  ./pycad.sh --file examples/x_by_xy_triangular.yaml --no-z3 --no-plot
  ./pycad.sh --dir examples/

Examples (Docker):
  ./rundocker.sh --test 25
  ./rundocker.sh --file my_test.yaml --log
  ./rundocker.sh --quick
  ./rundocker.sh --benchmarks
        """
    )

    # Input source (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--test", type=int,
                        help="Run a built-in test case by number")
    source.add_argument("--file", type=str,
                        help="Run a test case from a YAML file")
    source.add_argument("--dir", type=str,
                        help="Run all YAML test cases in a directory")

    # Options
    parser.add_argument("--no-z3", action="store_true",
                        help="Disable Z3 Boolean simplification")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting")
    parser.add_argument("--plot-points", type=int, default=400,
                        help="Choose how many points to plot (Default=400)")

    # Logging levels (mutually exclusive): silent (default) / log / verbose
    logging_group = parser.add_mutually_exclusive_group(required=False)
    logging_group.add_argument("--silent", action="store_true",
                               help="Suppress all output (default)")
    logging_group.add_argument("--log", action="store_true",
                               help="Show API-level output")
    logging_group.add_argument("--verbose", action="store_true",
                               help="Show all output including debug traces")

    args = parser.parse_args()
    z3_flag = not args.no_z3
    plot_flag = not args.no_plot
    from py_cad_modules.calculus import set_plot_points

    set_plot_points(args.plot_points)

    # Logging: silent is the default when neither --log nor --verbose is given
    from py_cad_modules.utils import update_Debug, update_APIPrint

    if args.verbose:
        update_Debug()
        update_APIPrint()
        silent_flag = False
    elif args.log:
        update_APIPrint()
        silent_flag = False
    else:
        # Default: silent (no dbprint, no apiprint)
        silent_flag = True

    # Mode 1: YAML file
    if args.file:
        tc = load_test_file(args.file)
        print(f"\n[LOADER] Running: {tc['name']} (from {args.file})")
        pdf, times = run_test_orchestrator(
            tc, z3_flg=z3_flag, silent=silent_flag, plot=plot_flag
        )
        print(f"\nExecution Times: {times}")

    # Mode 2: Directory of YAML files
    elif args.dir:
        cases = load_test_dir(args.dir)
        for test_id, tc in cases.items():
            print(f"\n{'=' * 60}")
            print(f"[LOADER] Running test {test_id}: {tc['name']}")
            print(f"{'=' * 60}")
            try:
                pdf, times = run_test_orchestrator(
                    tc, z3_flg=z3_flag, silent=silent_flag, plot=plot_flag
                )
                print(f"Execution Times: {times}")
            except Exception as e:
                print(f"[ERROR] Test {test_id} failed: {e}")

    # Mode 3: Built-in test case
    elif args.test is not None:
        if args.test in TEST_CASES:
            pdf, times = run_test_orchestrator(
                TEST_CASES[args.test], z3_flg=z3_flag, silent=silent_flag, plot=plot_flag
            )
            print(f"\nExecution Times: {times}")
        else:
            print(f"Test {args.test} not found! Available: {sorted(TEST_CASES.keys())}")

    # Default: show usage information
    else:
        print(r"""
╔══════════════════════════════════════════════════════════════════════╗
║                      PyCAD — Usage Guide                             ║
╚══════════════════════════════════════════════════════════════════════╝

  An exact distribution tool using Cylindrical Algebraic Decomposition.

─── HOW TO RUN ────────────────────────────────────────────────────────

  Native (after ./install.sh):
    ./pycad.sh <OPTIONS>
    ./pycadbenchmarker.sh                  (run full benchmark suite)

  Docker (convenience wrapper — handles volume mounts automatically):
    ./rundocker.sh <OPTIONS>
    ./rundocker.sh --quick                 (quick smoke test: case 1)
    ./rundocker.sh --benchmarks            (run full benchmark suite)
    ./rundocker.sh --shell                 (open interactive shell)

  Docker (manual):
    docker run --rm -v $(pwd)/outputs:/opt/pycad/outputs pycad <OPTIONS>
    docker run --rm -v $(pwd)/outputs:/opt/pycad/outputs pycad --quick
    docker run --rm -v $(pwd)/outputs:/opt/pycad/outputs pycad --benchmarks

  Note: On Windows PowerShell use ${PWD}, on CMD use %cd%, in place
  of $(pwd). Output files appear in the outputs/ folder.

─── OPTIONS ───────────────────────────────────────────────────────────

  Input modes (pick one):
    --test <i>        Run a built-in test case by number (1–37)
    --file <path>     Run a test case from a YAML file
    --dir  <path>     Run all YAML test cases in a directory

  Logging (pick one, default: --silent):
    --silent           Suppress all output; show only timings & K-S distance
    --log              Show basic output at each pipeline stage
    --verbose          Show all output including debug traces

  General:
    --no-z3            Disable Z3 Boolean simplification
    --no-plot          Skip plotting
    --plot-points <n>  Number of evaluation points for plotting (default: 400)

─── EXAMPLES ──────────────────────────────────────────────────────────

  Native:
    ./pycad.sh --test 25
    ./pycad.sh --test 25 --log
    ./pycad.sh --file examples/x_by_xy_triangular.yaml --no-plot
    ./pycad.sh --dir examples/ --verbose

  Docker:
    ./rundocker.sh --test 25
    ./rundocker.sh --file my_test.yaml --log
    ./rundocker.sh --dir my_tests/ --verbose

─── BUILT-IN TESTS ────────────────────────────────────────────────────

  Tests 1–3     Sum distributions (Uniform / Triangular / Squared)
  Tests 4–11    Taylor expansions (e^x, cos, ln, sin)
  Tests 12–13   x/(x+y) ratio
  Tests 14–17   Nonlinear & B-spline benchmarks
  Tests 18–19   If/Else piecewise
  Tests 20–22   Doppler effect
  Tests 23–29   Gröbner, spherical, IMU, compositional, rigid body
  Tests 30–37   Beta(2,2) distribution variants

  For YAML file format and full documentation, see README.md.
""")
