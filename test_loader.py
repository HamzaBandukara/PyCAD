"""
test_loader.py — Loads YAML test case files into the TEST_CASES dict format.

Converts a human-readable YAML file into the internal dict expected by
run_test_orchestrator(), including auto-generating mc_func and mc_generators
from the formula and distribution specifications.

Usage:
    from test_loader import load_test_file
    tc = load_test_file("examples/my_test.yaml")
    pdf, timings = run_test_orchestrator(tc)
"""
import yaml
import re
import numpy as np


def _parse_mc_formula(formula_str: str) -> str:
    """
    Attempts to extract the LHS of 'EXPR < t' from the CAD formula string
    and convert it into a Python/NumPy-evaluable expression.
    Returns a string like 'x / (x + y)'.
    """
    # Strip everything after the first '< t' or '<= t'
    # The LHS of the threshold comparison is the function of interest
    match = re.match(r'(.+?)\s*<[=]?\s*t\b', formula_str)
    if not match:
        return None

    expr = match.group(1).strip()

    # Convert Mathematica-style syntax to Python/NumPy
    expr = expr.replace('^', '**')
    expr = expr.replace('pi', 'np.pi')

    # Convert rt(n, expr) -> expr**(1/n)
    # Handle nested rt() by working from innermost out
    while 'rt(' in expr:
        expr = re.sub(
            r'rt\((\d+),\s*([^()]+(?:\([^()]*\))*[^()]*)\)',
            r'(\2)**(1/\1)',
            expr
        )

    # Convert If[cond, true, false] -> np.where(cond, true, false)
    expr = expr.replace('If[', 'np.where(').replace(']', ')')

    return expr


def _build_mc_func(mc_formula_str: str, var_names: list):
    """
    Builds a callable lambda from a NumPy expression string.
    The lambda accepts keyword arguments matching the variable names.
    """
    # Variables excluding 't'
    func_vars = [v for v in var_names if v != 't']

    # Build: lambda x, y, z: <expr>
    args_str = ', '.join(func_vars)
    func_code = f"lambda {args_str}: {mc_formula_str}"

    try:
        return eval(func_code, {"np": np, "math": __import__('math')})
    except Exception as e:
        print(f"[WARNING] Could not compile mc_func: {e}")
        print(f"          Expression was: {func_code}")
        return None


def _build_mc_generators(distributions: dict) -> dict:
    """
    Builds Monte Carlo sample generators from the distribution specifications.
    Returns a dict of {var_name: lambda n: np.random.<dist>(params, n)}.
    """
    generators = {}

    for var_name, dist_info in distributions.items():
        dist_type = dist_info[0].lower()

        if dist_type == 'triangular':
            a, b, mode = dist_info[1], dist_info[2], dist_info[3]
            generators[var_name] = (
                lambda n, _a=a, _mode=mode, _b=b: np.random.triangular(_a, _mode, _b, n)
            )
        elif dist_type == 'normal':
            mu, sigma = dist_info[1], dist_info[2]
            generators[var_name] = (
                lambda n, _mu=mu, _sigma=sigma: np.random.normal(_mu, _sigma, n)
            )
        elif dist_type == 'exponential':
            rate = dist_info[1]
            generators[var_name] = (
                lambda n, _scale=1.0/rate: np.random.exponential(_scale, n)
            )
        elif dist_type == 'beta':
            alpha, beta_p = dist_info[1], dist_info[2]
            a = dist_info[3] if len(dist_info) > 3 else 0
            b = dist_info[4] if len(dist_info) > 4 else 1
            generators[var_name] = (
                lambda n, _a=alpha, _b=beta_p, _lo=a, _hi=b:
                    np.random.beta(_a, _b, n) * (_hi - _lo) + _lo
            )
        # uniform: no generator needed, mc_bounds handles it

    return generators if generators else None


def _extract_mc_bounds_from_formula(formula_str: str, var_names: list) -> dict:
    """
    Extracts simple constant bounds (e.g., '1 < x < 4') from the formula.
    Returns {var_name: (lower, upper)}.
    """
    bounds = {}

    for var in var_names:
        if var == 't':
            continue

        # Match patterns like: NUM < var < NUM or NUM <= var <= NUM
        pattern = rf'([\d.eE+\-]+)\s*<[=]?\s*{re.escape(var)}\s*<[=]?\s*([\d.eE+\-]+)'
        match = re.search(pattern, formula_str)
        if match:
            bounds[var] = (float(match.group(1)), float(match.group(2)))
            continue

        # Also try: -NUM < var
        pattern2 = rf'(-[\d.eE+]+)\s*<[=]?\s*{re.escape(var)}\s*<[=]?\s*([\d.eE+\-]+)'
        match2 = re.search(pattern2, formula_str)
        if match2:
            bounds[var] = (float(match2.group(1)), float(match2.group(2)))

    return bounds


def _parse_distributions(dist_dict: dict) -> dict:
    """
    Converts YAML distribution specs into the tuple format expected by PyCAD.

    YAML format:
        x:
          type: triangular
          params: [1, 4, 2.5]

    Output format:
        {'x': ('triangular', 1, 4, 2.5)}
    """
    if not dist_dict:
        return None

    result = {}
    for var_name, spec in dist_dict.items():
        dist_type = spec['type'].lower()
        params = spec['params']
        result[var_name] = tuple([dist_type] + [float(p) for p in params])

    return result


def load_test_file(filepath: str) -> dict:
    """
    Loads a YAML test case file and returns a dict compatible with
    run_test_orchestrator().

    Automatically generates:
      - mc_func (from mc_formula or from the formula's LHS)
      - mc_bounds (from the formula or explicit specification)
      - mc_generators (from distribution specifications)
      - distributions (converted from YAML to tuple format)
    """
    with open(filepath, 'r') as f:
        raw = yaml.safe_load(f)

    # ── Required fields ──────────────────────────────────────────────────
    tc = {
        "name": raw.get("name", filepath),
        "formula": raw["formula"].strip(),
        "vars": raw["vars"],
        "integrand": raw.get("integrand", "auto"),
        "t_min": raw.get("t_min", None),
        "t_max": raw.get("t_max", None),
        "filename": raw.get("filename", "output"),
    }

    # ── Distributions ────────────────────────────────────────────────────
    if "distributions" in raw:
        tc["distributions"] = _parse_distributions(raw["distributions"])
    else:
        tc["distributions"] = None

    # ── Monte Carlo bounds ───────────────────────────────────────────────
    if "mc_bounds" in raw:
        tc["mc_bounds"] = {
            k: tuple(float(x) for x in v)
            for k, v in raw["mc_bounds"].items()
        }
    else:
        tc["mc_bounds"] = _extract_mc_bounds_from_formula(
            tc["formula"], tc["vars"]
        )

    # ── Monte Carlo function ─────────────────────────────────────────────
    mc_formula_str = raw.get("mc_formula", None)
    if mc_formula_str is None:
        mc_formula_str = _parse_mc_formula(tc["formula"])

    if mc_formula_str:
        mc_formula_str = mc_formula_str.strip()
        tc["mc_func"] = _build_mc_func(mc_formula_str, tc["vars"])
    else:
        tc["mc_func"] = None

    # ── Monte Carlo generators (for non-uniform distributions) ───────────
    if tc["distributions"]:
        tc["mc_generators"] = _build_mc_generators(tc["distributions"])
    else:
        tc["mc_generators"] = None

    return tc


def load_test_dir(dirpath: str) -> dict:
    """
    Loads all .yaml/.yml files from a directory into a numbered dict
    compatible with TEST_CASES.
    """
    import os
    import glob

    files = sorted(
        glob.glob(os.path.join(dirpath, "*.yaml")) +
        glob.glob(os.path.join(dirpath, "*.yml"))
    )

    cases = {}
    for i, fpath in enumerate(files, start=1000):
        try:
            cases[i] = load_test_file(fpath)
            print(f"[LOADER] Test {i}: {cases[i]['name']} ({os.path.basename(fpath)})")
        except Exception as e:
            print(f"[LOADER ERROR] Failed to load {fpath}: {e}")

    return cases


# ── CLI entrypoint for testing the loader ────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python3 test_loader.py <file.yaml>")
        sys.exit(1)

    tc = load_test_file(sys.argv[1])

    # Print the loaded test case (without lambdas for readability)
    printable = {k: v for k, v in tc.items() if not callable(v) and v is not None}
    print("\n=== Loaded Test Case ===")
    for k, v in printable.items():
        print(f"  {k}: {v}")

    if tc["mc_func"]:
        print(f"  mc_func: <compiled lambda>")
    if tc.get("mc_generators"):
        print(f"  mc_generators: {list(tc['mc_generators'].keys())}")
    print("========================")