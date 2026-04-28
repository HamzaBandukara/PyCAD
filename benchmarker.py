import os
import csv
import time
import sys
import warnings
import numpy as np
from run import run_test_orchestrator

BENCHMARK_SUITE = {
    # ---------------------------------------------------------
    # SUM DISTRIBUTIONS
    # ---------------------------------------------------------
    "sum_uni": {
        "name": "Sum (Uniform)",
        "formula": "1 < x < 2 && 1 < y < 2 && 1 < z < 2 && x + y + z < t",
        "vars": ["t", "x", "y", "z"],
        "integrand": "auto",
        "mc_func": lambda x, y, z: x + y + z,
        "mc_bounds": {'x': (1, 2), 'y': (1, 2), 'z': (1, 2)}
    },
    "sum_tri": {
        "name": "Sum (Triangular)",
        "formula": "1 < x < 2 && 1 < y < 2 && 1 < z < 2 && x + y + z < t",
        "vars": ["t", "x", "y", "z"],
        "integrand": "auto",
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
    "sum_sq": {
        "name": "Sum (Sq. Uni)",
        "formula": "1 < x < 2 && 1 < y < 2 && 1 < z < 2 && x^2 + y^2 + z^2 < t",
        "vars": ["t", "x", "y", "z"],
        "integrand": "auto",
        "mc_func": lambda x, y, z: x**2 + y**2 + z**2,
        "mc_bounds": {'x': (1, 2), 'y': (1, 2), 'z': (1, 2)}
    },

    # ---------------------------------------------------------
    # EXPANSIONS & APPROXIMATIONS
    # ---------------------------------------------------------
    "exp_uni": {
        "name": "e^x Expansion (Uni)",
        "formula": "1 + x + ((x^2)/2) + ((x^3)/6) + ((x^4)/24) + ((x^5)/120) < t && 0 < x < 1",
        "vars": ["t", "x"],
        "integrand": "auto",
        "mc_func": lambda x: 1 + x + (x**2)/2 + (x**3)/6 + (x**4)/24 + (x**5)/120,
        "mc_bounds": {'x': (0, 1)}
    },
    "exp_tri": {
        "name": "e^x Expansion (Tri)",
        "formula": "1 + x + ((x^2)/2) + ((x^3)/6) + ((x^4)/24) + ((x^5)/120) < t && 0 < x < 1",
        "vars": ["t", "x"],
        "integrand": "auto",
        "distributions": {'x': ('triangular', 0, 1, 0.5)},
        "mc_func": lambda x: 1 + x + (x**2)/2 + (x**3)/6 + (x**4)/24 + (x**5)/120,
        "mc_bounds": {'x': (0, 1)},
        "mc_generators": {'x': lambda n: np.random.triangular(0, 0.5, 1, n)}
    },
    "cos_approx_uni": {
        "name": "Cos Approx (Uni)",
        "formula": "((x^4)/24) - ((x^2)/2) + 1 < t && 0 < x < pi/2",
        "vars": ["t", "x"],
        "integrand": "auto",
        "mc_func": lambda x: (x**4)/24 - (x**2)/2 + 1,
        "mc_bounds": {'x': (0, float(np.pi / 2))}
    },
    "cos_approx_tri": {
        "name": "Cos Approx (Tri)",
        "formula": "((x^4)/24) - ((x^2)/2) + 1 < t && 0 < x < pi/2",
        "vars": ["t", "x"],
        "integrand": "auto",
        "distributions": {'x': ('triangular', 0, float(np.pi / 2), float(np.pi / 4))},
        "mc_func": lambda x: (x**4)/24 - (x**2)/2 + 1,
        "mc_bounds": {'x': (0, float(np.pi / 2))},
        "mc_generators": {'x': lambda n: np.random.triangular(0, float(np.pi / 4), float(np.pi / 2), n)}
    },
    "log_uni": {
        "name": "Ln Approx (Uni)",
        "formula": "(1/5)*((x - 1)^5) - (1/4)*((x - 1)^4) + (1/3)*((x - 1)^3) - (1/2)*((x - 1)^2) + x - 1 < t && 0 < x < 1",
        "vars": ["t", "x"],
        "integrand": "auto",
        "mc_func": lambda x: (1/5)*((x - 1)**5) - (1/4)*((x - 1)**4) + (1/3)*((x - 1)**3) - (1/2)*((x - 1)**2) + x - 1,
        "mc_bounds": {'x': (0, 1)}
    },
    "log_tri": {
        "name": "Ln Approx (Tri)",
        "formula": "(1/5)*((x - 1)^5) - (1/4)*((x - 1)^4) + (1/3)*((x - 1)^3) - (1/2)*((x - 1)^2) + x - 1 < t && 0 < x < 1",
        "vars": ["t", "x"],
        "integrand": "auto",
        "distributions": {'x': ('triangular', 0, 1, 0.5)},
        "mc_func": lambda x: (1/5)*((x - 1)**5) - (1/4)*((x - 1)**4) + (1/3)*((x - 1)**3) - (1/2)*((x - 1)**2) + x - 1,
        "mc_bounds": {'x': (0, 1)},
        "mc_generators": {'x': lambda n: np.random.triangular(0, 0.5, 1, n)}
    },
    "sin_approx_uni": {
        "name": "Sin Approx (Uni)",
        "formula": "((x^5)/120) - ((x^3)/6) + x < t && 0 < x < pi/2",
        "vars": ["t", "x"],
        "integrand": "auto",
        "mc_func": lambda x: (x**5)/120 - (x**3)/6 + x,
        "mc_bounds": {'x': (0, float(np.pi / 2))}
    },
    "sin_approx_tri": {
        "name": "Sin Approx (Tri)",
        "formula": "((x^5)/120) - ((x^3)/6) + x < t && 0 < x < pi/2",
        "vars": ["t", "x"],
        "integrand": "auto",
        "distributions": {'x': ('triangular', 0, float(np.pi / 2), float(np.pi / 4))},
        "mc_func": lambda x: (x**5)/120 - (x**3)/6 + x,
        "mc_bounds": {'x': (0, float(np.pi / 2))},
        "mc_generators": {'x': lambda n: np.random.triangular(0, float(np.pi / 4), float(np.pi / 2), n)}
    },

    # ---------------------------------------------------------
    # NON-LINEAR & SPLINES
    # ---------------------------------------------------------
    "x_by_xy_uni": {
        "name": "x/(x+y) (Uniform)",
        "formula": "x/(x+y) < t && 1 < x < 4 && 1 < y < 4",
        "vars": ["t", "x", "y"],
        "integrand": "auto",
        "mc_func": lambda x, y: x / (x + y),
        "mc_bounds": {'x': (1, 4), 'y': (1, 4)}
    },
    "x_by_xy_tri": {
        "name": "x/(x+y) (Triangular)",
        "formula": "x/(x+y) < t && 1 < x < 4 && 1 < y < 4",
        "vars": ["t", "x", "y"],
        "integrand": "auto",
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
    "nonlin1_uni": {
        "name": "z/(z+1) (Uniform)",
        "formula": "z/(z+1) < t && 0 < z < 999",
        "vars": ["t", "z"],
        "integrand": "auto",
        "mc_func": lambda z: z / (z + 1),
        "mc_bounds": {'z': (0, 999)}
    },
    "nonlin1_tri": {
        "name": "z/(z+1) (Triangular)",
        "formula": "z/(z+1) < t && 0 < z < 999",
        "vars": ["t", "z"],
        "integrand": "auto",
        "distributions": {'z': ('triangular', 0, 999, 499.5)},
        "mc_func": lambda z: z / (z + 1),
        "mc_bounds": {'z': (0, 999)},
        "mc_generators": {'z': lambda n: np.random.triangular(0, 499.5, 999, n)}
    },
    # NOTE: the below case DOES WORK, and succeeds, it is mathematically correct.
    # The benchmarker states "fail" as the tail end is infinity (thus giving a high K-S distance!)
    "bspline3_uni": {
        "name": "BSpline3 (Uniform)",
        "formula": "-(u^3)/6 < t && 0 < u < 1",
        "vars": ["t", "u"],
        "integrand": "auto",
        "mc_func": lambda u: -(u**3) / 6,
        "mc_bounds": {'u': (0, 1)}
    },
    "bspline3_tri": {
        "name": "BSpline3 (Triangular)",
        "formula": "-(u^3)/6 < t && 0 < u < 1",
        "vars": ["t", "u"],
        "integrand": "auto",
        "distributions": {'u': ('triangular', 0, 1, 0.5)},
        "mc_func": lambda u: -(u**3) / 6,
        "mc_bounds": {'u': (0, 1)},
        "mc_generators": {'u': lambda n: np.random.triangular(0, 0.5, 1, n)}
    },

    # ---------------------------------------------------------
    # DISJOINT / PIECEWISE EXAMPLES
    # ---------------------------------------------------------
    "cav10_uni": {
        "name": "If/Else Disjoint (Uni)",
        "formula": "If[x^2 - x >= 0, x/10, x^2 + 2] < t && 0 < x < 10",
        "vars": ["t", "x"],
        "t_min": 0, "t_max": 5,
        "integrand": "auto",
        "mc_func": lambda x: np.where(x**2 - x >= 0, x/10, x**2 + 2),
        "mc_bounds": {'x': (0, 10)}
    },
    "cav10_tri": {
        "name": "If/Else Disjoint (Tri)",
        "formula": "If[x^2 - x >= 0, x/10, x^2 + 2] < t && 0 < x < 10",
        "vars": ["t", "x"],
        "t_min": 0, "t_max": 5,
        "integrand": "auto",
        "distributions": {'x': ('triangular', 0, 10, 5)},
        "mc_func": lambda x: np.where(x**2 - x >= 0, x/10, x**2 + 2),
        "mc_bounds": {'x': (0, 10)},
        "mc_generators": {'x': lambda n: np.random.triangular(0, 5, 10, n)}
    },
    # ---------------------------------------------------------
    # DOPPLER EFFECT & COMPLEX GEOMETRIES
    # ---------------------------------------------------------
    "doppler1": {
        "name": "Doppler Effect (Test 1)",
        "formula": "(-(3314/10 + (6/10)*T) * v) / ((3314/10 + (6/10)*T + u)^2) < t && -100 < u < 100 && 20 < v < 20000 && -30 < T < 50",
        "vars": ["t", "u", "v", "T"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "doppler_test_1",
        "mc_func": lambda u, v, T: (-(331.4 + 0.6 * T) * v) / ((331.4 + 0.6 * T + u) ** 2),
        "mc_bounds": {'u': (-100, 100), 'v': (20, 20000), 'T': (-30, 50)}
    },
    "doppler2": {
        "name": "Doppler Effect (Test 2)",
        "formula": "(-(3314/10 + (6/10)*T) * v) / ((3314/10 + (6/10)*T + u)^2) < t && -125 < u < 125 && 15 < v < 25000 && -40 < T < 60",
        "vars": ["t", "u", "v", "T"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "doppler_test_2",
        "mc_func": lambda u, v, T: (-(331.4 + 0.6 * T) * v) / ((331.4 + 0.6 * T + u) ** 2),
        "mc_bounds": {'u': (-125, 125), 'v': (15, 25000), 'T': (-40, 60)}
    },
    "doppler3": {
        "name": "Doppler Effect (Test 3)",
        "formula": "(-(3314/10 + (6/10)*T) * v) / ((3314/10 + (6/10)*T + u)^2) < t && -30 < u < 120 && 320 < v < 20300 && -50 < T < 30",
        "vars": ["t", "u", "v", "T"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "doppler_test_3",
        "mc_func": lambda u, v, T: (-(331.4 + 0.6 * T) * v) / ((331.4 + 0.6 * T + u) ** 2),
        "mc_bounds": {'u': (-30, 120), 'v': (320, 20300), 'T': (-50, 30)}
    },
    "doppler1_tri": {
        "name": "Doppler 1 (Triangular)",
        "formula": "(-(3314/10 + (6/10)*T) * v) / ((3314/10 + (6/10)*T + u)^2) < t && -100 < u < 100 && 20 < v < 20000 && -30 < T < 50",
        "vars": ["t", "u", "v", "T"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "doppler1_tri",
        "distributions": {
            'u': ('triangular', -100, 100, 0),
            'v': ('triangular', 20, 20000, 10010),
            'T': ('triangular', -30, 50, 10)
        },
        "mc_func": lambda u, v, T: (-(331.4 + 0.6 * T) * v) / ((331.4 + 0.6 * T + u) ** 2),
        "mc_bounds": {'u': (-100, 100), 'v': (20, 20000), 'T': (-30, 50)},
        "mc_generators": {
            'u': lambda n: np.random.triangular(-100, 0, 100, n),
            'v': lambda n: np.random.triangular(20, 10010, 20000, n),
            'T': lambda n: np.random.triangular(-30, 10, 50, n)
        }
    },
    "doppler2_tri": {
        "name": "Doppler 2 (Triangular)",
        "formula": "(-(3314/10 + (6/10)*T) * v) / ((3314/10 + (6/10)*T + u)^2) < t && -125 < u < 125 && 15 < v < 25000 && -40 < T < 60",
        "vars": ["t", "u", "v", "T"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "doppler2_tri",
        "distributions": {
            'u': ('triangular', -125, 125, 0),
            'v': ('triangular', 15, 25000, 12507),
            'T': ('triangular', -40, 60, 10)
        },
        "mc_func": lambda u, v, T: (-(331.4 + 0.6 * T) * v) / ((331.4 + 0.6 * T + u) ** 2),
        "mc_bounds": {'u': (-125, 125), 'v': (15, 25000), 'T': (-40, 60)},
        "mc_generators": {
            'u': lambda n: np.random.triangular(-125, 0, 125, n),
            'v': lambda n: np.random.triangular(15, 12507, 25000, n),
            'T': lambda n: np.random.triangular(-40, 10, 60, n)
        }
    },
    "doppler3_tri": {
        "name": "Doppler 3 (Triangular)",
        "formula": "(-(3314/10 + (6/10)*T) * v) / ((3314/10 + (6/10)*T + u)^2) < t && -30 < u < 120 && 320 < v < 20300 && -50 < T < 30",
        "vars": ["t", "u", "v", "T"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "doppler3_tri",
        "distributions": {
            'u': ('triangular', -30, 120, 45),
            'v': ('triangular', 320, 20300, 10310),
            'T': ('triangular', -50, 30, -10)
        },
        "mc_func": lambda u, v, T: (-(331.4 + 0.6 * T) * v) / ((331.4 + 0.6 * T + u) ** 2),
        "mc_bounds": {'u': (-30, 120), 'v': (320, 20300), 'T': (-50, 30)},
        "mc_generators": {
            'u': lambda n: np.random.triangular(-30, 45, 120, n),
            'v': lambda n: np.random.triangular(320, 10310, 20300, n),
            'T': lambda n: np.random.triangular(-50, -10, 30, n)
        }
    },
    "v_norm": {
        "name": "Vector Norm",
        "formula": "x/(rt(2, x^2+y^2+z^2)) < t && 0 < x < 1 && 0 < y < 1 && 0 < z < 1",
        "vars": ["t", "x", "y", "z"],
        "integrand": "auto",
        "t_min": None, "t_max": None,
        "filename": "v_norm",
        "mc_func": lambda x, y, z: x / np.sqrt(x ** 2 + y ** 2 + z ** 2),
        "mc_bounds": {'x': (0, 1), 'y': (0, 1), 'z': (0, 1)}
    },
    "rigidbody1": {
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
    "sum_beta": {
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
    "x_by_xy_beta": {
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
    "nonlin1_beta": {
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
    "bspline3_beta": {
        "name": "BSpline3 (Beta 2,2)",
        "formula": "-(u^3)/6 < t && 0 < u < 1",
        "vars": ["t", "u"],
        "integrand": "auto",
        "distributions": {
            'u': ('beta', 2, 2, 0, 1)
        },
        "mc_func": lambda u: -(u**3) / 6,
        "mc_bounds": {'u': (0, 1)},
        "mc_generators": {
            'u': lambda n: np.random.beta(2, 2, n)
        }
    },
    "cav10_beta": {
        "name": "If/Else Disjoint (Beta 2,2)",
        "formula": "If[x^2 - x >= 0, x/10, x^2 + 2] < t && 0 < x < 10",
        "vars": ["t", "x"],
        "t_min": 0, "t_max": 5,
        "integrand": "auto",
        "distributions": {
            'x': ('beta', 2, 2, 0, 10)
        },
        "mc_func": lambda x: np.where(x**2 - x >= 0, x/10, x**2 + 2),
        "mc_bounds": {'x': (0, 10)},
        "mc_generators": {
            'x': lambda n: 10 * np.random.beta(2, 2, n)
        }
    },
    "doppler1_beta": {
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
    "doppler2_beta": {
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
    "doppler3_beta": {
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
    "IMUFilter": {
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

}

def run_benchmarks(skip_imu: bool = False):
    os.makedirs("./outputs/benchmark_plots", exist_ok=True)
    csv_filename = f"benchmark_results_{int(time.time())}.csv"

    print(f"{'=' * 125}")
    print(f"{'Py-CAD Performance Benchmarking Suite (V4 - Native Orchestrator)':^125}")
    print(f"{'=' * 125}")
    print(
        f"{'Benchmark Name':<25} | {'CAD (s)':<10} | {'CDF (s)':<10} | {'PDF (s)':<10} | {'Plot (s)':<10} | {'Total (s)':<10} | {'Status'} | {'KS':<10}")
    print(f"{'-' * 125}")

    results = []

    for key, tc in BENCHMARK_SUITE.items():
        if skip_imu and key == "IMUFilter":
            print(f"{'IMU Complementary Filter':<25} | {'--':^10} | {'--':^10} | {'--':^10} | {'--':^10} | {'--':^10} | SKIPPED | {'--':^10}")
            continue

        # Inject the benchmark_plots path into the filename
        tc["filename"] = f"/benchmark_plots/{tc.get('filename', key)}"

        try:
            # 100% equivalent to running it locally
            pdf_expr, timings = run_test_orchestrator(tc, z3_flg=True, silent=True, plot=True)

            ks = timings.get("KS", -1.0)
            if ks < 0:
                status = "NO_KS"    # No Monte Carlo overlay — can't verify
            elif ks <= 0.02:
                status = "PASSED"   # Mathematically correct (within MC noise)
            elif ks <= 0.05:
                status = "MARGINAL" # Correct shape, minor numerical artifacts
            else:
                status = "FAILED"   # Visible discrepancy from Monte Carlo
        except Exception as e:
            status = "CRASHED"
            timings = {"cad": 0.0, "cdf": 0.0, "pdf": 0.0, "plot": 0.0, "total": 0.0, "KS": -1.0}

        # Print & Record
        print(f"{tc['name']:<25} | {timings['cad']:<10.4f} | {timings['cdf']:<10.4f} | "
              f"{timings['pdf']:<10.4f} | {timings['plot']:<10.4f} | {timings['total']:<10.4f} | {status} | {timings['KS']:<10.4f}")

        results.append({
            "Name": tc['name'],
            "CAD Time": round(timings['cad'], 4),
            "CDF Time": round(timings['cdf'], 4),
            "PDF Time": round(timings['pdf'], 4),
            "Plot Time": round(timings['plot'], 4),
            "Total Time": round(timings['total'], 4),
            "KS": round(timings['KS'], 4),
            "Status": status
        })

    print(f"{'=' * 125}")

    # Write Results to CSV
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Name", "CAD Time", "CDF Time", "PDF Time", "Plot Time", "Total Time",
                                                  "Status", "KS"])
        writer.writeheader()
        writer.writerows(results)

    # Calculate Averages for Passed Tests
    passed = [r for r in results if "PASSED" in r["Status"]]
    if passed:
        avg_cad = sum(r["CAD Time"] for r in passed) / len(passed)
        avg_cdf = sum(r["CDF Time"] for r in passed) / len(passed)
        avg_pdf = sum(r["PDF Time"] for r in passed) / len(passed)
        avg_plot = sum(r["Plot Time"] for r in passed) / len(passed)
        avg_tt = sum(r["Total Time"] for r in passed) / len(passed)

        print(f"Total Tests: {len(results)} | Fully Passed: {len(passed)}")
        print(f"Average CAD Time   : {avg_cad:.4f}s")
        print(f"Average CDF Time   : {avg_cdf:.4f}s")
        print(f"Average PDF Time   : {avg_pdf:.4f}s")
        print(f"Average Plot Time  : {avg_plot:.4f}s")
        print(f"Average Total Time : {avg_tt:.4f}s")
        print(f"\n[INFO] Benchmark report saved to: {csv_filename}")
        print(f"[INFO] Benchmark plots saved to directory: ./outputs/benchmark_plots/")
    print(f"{'=' * 125}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    skip_imu = "--skip-imu" in sys.argv
    run_benchmarks(skip_imu=skip_imu)