import subprocess
import re
import functools
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from .preprocessing import preprocess_sugar, extract_base_support, ast_normalize_inequalities, optimize_variable_order, \
    apply_groebner_reduction
from py_cad_modules.utils import dbprint, apiprint


def format_for_qepcad(formula: str) -> str:
    formula = formula.replace("==", "=").replace("&&", "/\\ ").replace("||", "\\/ ")
    formula = formula.replace("**", "^").replace("*", " ")
    return f"[ {formula} ]."


# Added lru_cache to dramatically speed up identical QEPCAD queries
@functools.lru_cache(maxsize=128)
def extract_qepcad_tree(formula: str, vars_tuple: tuple):
    q_formula = format_for_qepcad(formula)
    var_str = "(" + ",".join(vars_tuple) + ")"
    script = f"[]\n{var_str}\n{len(vars_tuple)}\n{q_formula}\ngo\ngo\ngo\nd-proj-polynomials\nd-true-cells\nquit\n"

    # Passed script directly to input pipe to prevent parallel write collisions
    result = subprocess.run(["qepcad", "+N500000000"], input=script, capture_output=True, text=True)
    return result.stdout


def parse_polynomials(raw_output: str, formula: str, syms: list):
    local_syms = {str(s): s for s in syms}
    j_polys = {i: [] for i in range(1, len(syms) + 1)}

    current_lvl = None
    for line in raw_output.splitlines():
        line = line.strip()
        if line.startswith("J_"):
            current_lvl = int(line.split("_")[1].split(",")[0])
        elif line.startswith("=") and current_lvl is not None:
            eq_str = line[1:].strip()
            if "res(" not in eq_str and "dis(" not in eq_str:
                eq_str = eq_str.replace("^", "**")

                # Safely multiply QEPCAD spaces (i.e., turn spaces into *'s)
                eq_str = re.sub(r'([a-zA-Z0-9_])\s+([a-zA-Z0-9_])', r'\1*\2', eq_str)

                try:
                    expr = parse_expr(eq_str, local_dict=local_syms, transformations=standard_transformations)
                    j_polys[current_lvl].append(expr)
                except Exception:
                    pass

    parts = re.split(r'(\&\&|\|\||\[|\]|\~)', formula)
    for cond in parts:
        cond = cond.strip()
        match = re.search(r'(<=|>=|<|>|==)', cond)
        if match:
            op = match.group(1)
            lhs_str, rhs_str = cond.split(op)
            try:
                lhs_str = re.sub(r'([a-zA-Z0-9_])\s+([a-zA-Z0-9_])', r'\1*\2', lhs_str.replace("^", "**"))
                rhs_str = re.sub(r'([a-zA-Z0-9_])\s+([a-zA-Z0-9_])', r'\1*\2', rhs_str.replace("^", "**"))

                lhs = parse_expr(lhs_str, local_dict=local_syms, transformations=standard_transformations)
                rhs = parse_expr(rhs_str, local_dict=local_syms, transformations=standard_transformations)
                expr = sp.cancel(lhs - rhs)

                highest_lvl = 0
                for v in expr.free_symbols:
                    for i, s in enumerate(syms):
                        if str(s) == str(v): highest_lvl = max(highest_lvl, i + 1)

                if highest_lvl > 0:
                    if expr not in j_polys[highest_lvl] and -expr not in j_polys[highest_lvl]:
                        j_polys[highest_lvl].append(expr)
            except Exception:
                pass

    return j_polys


def get_cad(formula: str, vars_list: list) -> dict:
    """Takes a formula and bounds, returns the raw CAD geometry dictionary."""
    formula = preprocess_sugar(formula)
    formula, vars_list = apply_groebner_reduction(formula, vars_list)
    vars_list = optimize_variable_order(formula, vars_list)
    syms = tuple(sp.symbols(" ".join(vars_list), real=True))
    local_syms = {str(s): s for s in syms}
    base_bounds = extract_base_support(formula, syms, local_syms)
    formula, vars_list_updated = ast_normalize_inequalities(formula, vars_list.copy())
    dbprint("[DEBUG/GET_CAD] Formula/vars_list before send: ", vars_list_updated, formula)


    # Convert to tuple here so lru_cache can hash the arguments
    raw_output = extract_qepcad_tree(formula, tuple(vars_list_updated))

    blocks = raw_output.split("Before Solution >")
    if len(blocks) < 3:
        dbprint(f"[DEBUG/QEPCAD] {blocks}")
        return {"success": False, "error": "QEPCAD parsing failed."}
    updated_syms = tuple(sp.symbols(" ".join(vars_list_updated), real=True))

    level_polys = parse_polynomials(raw_output, formula, updated_syms)
    for level in level_polys:
        level_polys[level] = [p for p in level_polys[level] if not p.is_number]
    cells_raw = blocks[2].split("---------- Information about the cell")

    return {
        "success": True,
        "base_bounds": base_bounds,
        "level_polys": level_polys,
        "cells_raw": cells_raw[1:],
        "vars_list": vars_list_updated
    }


def auto_derive_uniform_integrand(syms, base_bounds):
    """
    Derives the exact uniform probability density function (1 / Total Volume)
    by calculating the Cartesian product of the base parameter bounds.
    """
    total_volume = 1

    # do not include threshold var in volume calculation
    integration_vars = syms[1:]

    for var in integration_vars:
        var_str = str(var)

        # Skip any internal slack variables used by CAD
        # (slack variables are now removed, keeping in-case)
        if var_str.startswith('w'):
            continue

        bounds = base_bounds.get(var_str)
        if not bounds:
            apiprint(f"[WARNING] No base bounds found for {var_str}. Defaulting dimension volume to 1.")
            continue

        try:
            # Extract the strict numerical limits from the global base bounds
            lower = float(bounds['lower'])
            upper = float(bounds['upper'])

            # Inf Shield
            if lower == -float('inf') or upper == float('inf'):
                apiprint(f"[WARNING] Infinite bounds for {var_str}. Defaulting dimension volume to 1.")
                continue

            # Multiply the length of this specific dimension
            dimension_length = upper - lower
            total_volume *= dimension_length

        except (ValueError, TypeError, KeyError):
            apiprint(f"[WARNING] Non-numeric base bounds for {var_str}. Defaulting dimension volume to 1.")
            continue

    if total_volume <= 0:
        apiprint("[ERROR] Total volume calculated as 0 or negative! Defaulting integrand to 1.")
        return sp.S.One

    # Exact fractions keep the algebraic engine efficinet.
    return sp.Rational(1, str(total_volume))


def auto_derive_uniform_integrand_global(mc_bounds: dict):
    """Calculates true global volume using the test's Monte Carlo bounds."""
    total_volume = 1
    for var, limits in mc_bounds.items():
        # limits are given in tuple form
        dimension_length = limits[1] - limits[0]
        total_volume *= dimension_length

    return sp.Rational(1, str(total_volume))


def derive_joint_pdf(vars_syms: tuple, distributions: dict) -> sp.Expr:
    """
    Constructs the joint PDF expression given a dictionary of distributions.
    Automatically handles standard tuples (e.g. ('uniform', 0, 1)) or
    pre-computed custom SymPy expressions (e.g., from Compositional Sub-CADs).
    """

    joint_pdf = sp.S.One

    for var in vars_syms:
        v_str = str(var)

        if v_str not in distributions:
            print(f"[WARNING] No distribution for {v_str}. Defaulting to Uniform(0, 1).")
            # If no bounds exist, we assume standard uniform [0,1]
            joint_pdf *= sp.Piecewise((1, sp.And(var >= 0, var <= 1)), (0, True))
            continue

        dist_info = distributions[v_str]

        # Handle Custom Compositional Expressions
        if isinstance(dist_info, sp.Expr):
            joint_pdf *= dist_info
            continue

        # Logic for standard distributions
        try:
            dist_type = dist_info[0].lower()
            if dist_type == 'uniform':
                lower, upper = sp.sympify(dist_info[1]), sp.sympify(dist_info[2])
                pdf_expr = sp.Piecewise((1 / (upper - lower), sp.And(var >= lower, var <= upper)), (0, True))
            elif dist_type == 'exponential':
                rate = sp.sympify(dist_info[1])
                pdf_expr = sp.Piecewise((rate * sp.exp(-rate * var), var >= 0), (0, True))
            elif dist_type == 'normal':
                mu, sigma = sp.sympify(dist_info[1]), sp.sympify(dist_info[2])
                pdf_expr = (1 / (sigma * sp.sqrt(2 * sp.pi))) * sp.exp(-0.5 * ((var - mu) / sigma) ** 2)

            # Triangular Distribution Parser
            elif dist_type == 'triangular':
                # Format: ('triangular', a, b, mode)
                a_val, b_val, c_val = dist_info[1], dist_info[2], dist_info[3]

                # Force exact rational math to prevent floating point drift in the CAD engine
                a = sp.Rational(str(a_val))
                b = sp.Rational(str(b_val))
                c = sp.Rational(str(c_val))

                # Left slope: 2*(x - a) / ((b - a)*(c - a))
                left_prob = (2 * (var - a)) / ((b - a) * (c - a))

                # Right slope: 2*(b - x) / ((b - a)*(b - c))
                right_prob = (2 * (b - var)) / ((b - a) * (b - c))

                pdf_expr = sp.Piecewise(
                    (left_prob, sp.And(var >= a, var < c)),
                    (right_prob, sp.And(var >= c, var <= b)),
                    (0, True)
                )

            # --- Beta Distribution ---
            elif dist_type == 'beta':
                # Format: ('beta', alpha, beta_param, lower, upper)
                alpha_val = dist_info[1]
                beta_val = dist_info[2]
                a_val = dist_info[3] if len(dist_info) > 3 else 0
                b_val = dist_info[4] if len(dist_info) > 4 else 1

                alpha = sp.Rational(str(alpha_val))
                beta_p = sp.Rational(str(beta_val))
                a = sp.Rational(str(a_val))
                b = sp.Rational(str(b_val))

                # Beta normalizing constant: B(α,β) = Γ(α)Γ(β)/Γ(α+β)
                B_const = sp.gamma(alpha) * sp.gamma(beta_p) / sp.gamma(alpha + beta_p)

                # Scaled variable: z = (x - a) / (b - a), maps [a,b] → [0,1]
                z = (var - a) / (b - a)

                # Beta PDF: f(x) = z^(α-1) * (1-z)^(β-1) / (B(α,β) * (b-a))
                beta_pdf = z**(alpha - 1) * (1 - z)**(beta_p - 1) / (B_const * (b - a))

                pdf_expr = sp.Piecewise(
                    (beta_pdf, sp.And(var >= a, var <= b)),
                    (0, True)
                )
            # -------------------------------------------

            else:
                raise ValueError(f"Unsupported distribution type: {dist_type}")

            joint_pdf *= pdf_expr
        except Exception as e:
            print(f"[ERROR] Failed to parse distribution for {v_str}: {e}")
            joint_pdf *= 1

    return joint_pdf


def evaluate_cell_boundary(poly_list: list, pt: dict, var: sp.Symbol, syms: tuple, base_bound: dict) -> tuple:
    """Evaluates standard roots and quintic RootOf fallbacks to find exact geometric bounds."""
    lower_cands, upper_cands = [], []

    for poly in poly_list:
        if var not in poly.free_symbols: continue

        roots = []
        try:
            roots = sp.solve(poly, var)
        except Exception:
            pass

        valid_roots_found = False
        for root in roots:
            try:
                exact_pt = {s: sp.Rational(str(pt[s])) for s in syms}
                val_complex = complex(root.subs(exact_pt).evalf()) if exact_pt else complex(root.evalf())
                if abs(val_complex.imag) > 1e-7: continue
                val_root = val_complex.real

                if root.has(sp.RootOf) or 'RootOf' in str(type(root)):
                    dummy_var = sp.Symbol(f"{var}_val", real=True)
                    poly_dummy = poly.subs(var, dummy_var)
                    clean_root = sp.Function('RootOf')(poly_dummy, dummy_var)
                else:
                    clean_root = root

                if val_root < pt[var]:
                    lower_cands.append((clean_root, val_root))
                elif val_root > pt[var]:
                    upper_cands.append((clean_root, val_root))
                valid_roots_found = True
            except Exception:
                pass

        if not valid_roots_found:
            try:
                exact_pt = {s: sp.Rational(str(pt[s])) for s in syms if s != var}
                poly_at_pt = poly.subs(exact_pt) if exact_pt else poly
                num_roots = sp.real_roots(sp.nsimplify(poly_at_pt, rational=True))
                for num_r in num_roots:
                    try:
                        c_val = complex(num_r.evalf())
                        if abs(c_val.imag) > 1e-7: continue
                        val_root = c_val.real

                        dummy_var = sp.Symbol(f"{var}_val", real=True)
                        poly_dummy = poly.subs(var, dummy_var)
                        implicit_expr = sp.Function('RootOf')(poly_dummy, dummy_var)

                        if val_root < pt[var]:
                            lower_cands.append((implicit_expr, val_root))
                        elif val_root > pt[var]:
                            upper_cands.append((implicit_expr, val_root))
                    except Exception:
                        pass
            except Exception:
                pass

    base_L = base_bound.get("lower", -sp.oo)
    base_U = base_bound.get("upper", sp.oo)
    if base_L != -sp.oo:
        try:
            val_l = complex(sp.sympify(base_L).subs(pt)).real
            lower_cands.append((sp.sympify(base_L), val_l))
        except Exception:
            pass
    if base_U != sp.oo:
        try:
            val_u = complex(sp.sympify(base_U).subs(pt)).real
            upper_cands.append((sp.sympify(base_U), val_u))
        except Exception:
            pass

    # USE SYMBOLIC MIN/MAX TO PRESERVE DYNAMIC BOUNDARY CROSSINGS
    if not lower_cands: clamped_lower = -sp.oo
    elif len(lower_cands) == 1: clamped_lower = lower_cands[0][0]
    else: clamped_lower = sp.Max(*[c[0] for c in lower_cands])

    if not upper_cands: clamped_upper = sp.oo
    elif len(upper_cands) == 1: clamped_upper = upper_cands[0][0]
    else: clamped_upper = sp.Min(*[c[0] for c in upper_cands])

    return clamped_lower, clamped_upper