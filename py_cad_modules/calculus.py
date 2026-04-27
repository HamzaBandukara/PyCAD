import sympy as sp
import re
import numpy as np
from .cad_core import evaluate_cell_boundary, auto_derive_uniform_integrand, auto_derive_uniform_integrand_global, \
    derive_joint_pdf
from sympy.core.relational import Relational
import pycad_cpp_engine  # compile the C++ module
from py_cad_modules.utils import call_maxima_integration, has_radical, extract_cell_point, apiprint, sympy_to_exprtk, dbprint
import subprocess
import scipy.integrate as scipy_int
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import warnings

import matplotlib.pyplot as plt

# --- GLOBAL INTEGRATION CACHE ---
INDEFINITE_INTEGRAL_CACHE = {}
PLOT_POINTS=400

def set_plot_points(plot_points):
    global PLOT_POINTS
    PLOT_POINTS = plot_points


def call_maxima_integrate_batch(valid_requests, timeout_sec=3, local_syms=None):
    """
    Batches multiple integrations into a single Maxima subprocess.
    Includes robust sanitization to prevent 'unexpected indent' errors from long strings.
    """

    timeout_sec = max(10.0, len(valid_requests) * timeout_sec)

    def sanitize_for_maxima(expr_obj):
        """Removes SymPy artifacts and translates Piecewise/Max/Min to Maxima syntax."""
        if isinstance(expr_obj, sp.Basic) and expr_obj.has(sp.Piecewise):
            expr_obj = expr_obj.rewrite(sp.Heaviside)

        s = str(expr_obj)
        s = s.replace('\\', '').replace('\n', '').replace('\r', '')
        s = ' '.join(s.split())
        s = s.replace('**', '^')
        s = s.replace('Heaviside', 'unit_step')
        # Safety: translate Max/Min in case they appear in bounds
        s = s.replace('Max(', 'max(')
        s = s.replace('Min(', 'min(')
        return s


    maxima_code = "display2d: false$\n"  # Flat 1D string output
    maxima_code += "keepfloat: true$\n"  # Prevent Maxima from converting floats to huge fractions
    maxima_code += "load(\"abs_integrate\")$\n"
    dbprint("[DEBUG/MAXIMA] Now printing requests...")

    for idx, req in enumerate(valid_requests):
        dbprint(f"[DEBUG/MAXIMA #{idx}] Completing: {req}")
        vol, req_var, req_lower, req_upper, cell_data = req

        # 1. Extract CAD boundaries
        assumptions = []

        if cell_data["domain"] != sp.S.true:
            for rel in cell_data["domain"].atoms(Relational):
                assumptions.append(sanitize_for_maxima(rel))

        for prev_var, prev_lower, prev_upper in cell_data.get("limits", []):
            if prev_lower != -sp.oo:
                assumptions.append(f"{prev_var} > {sanitize_for_maxima(prev_lower)}")
            if prev_upper != sp.oo:
                assumptions.append(f"{prev_var} < {sanitize_for_maxima(prev_upper)}")

        # 2. Write maxima instructions
        maxima_code += "forget(facts())$\n"
        for assume_str in assumptions:
            maxima_code += f"assume({assume_str})$\n"

        # Convert SymPy's unevaluated "Integral" into Maxima's "integrate"
        vol_str = sanitize_for_maxima(vol)
        dbprint(f"[DEBUG/MAXIMA] Sending to Maxima: {vol_str[:100]}...")
        dbprint(f"[DEBUG/MAXIMA] Above created from0: {vol}")
        vol_str = re.sub(r'Integral\((.*?), \((.*?), (.*?), (.*?)\)\)', r'integrate(\1, \2, \3, \4)', vol_str)

        lower_str = sanitize_for_maxima(req_lower)
        upper_str = sanitize_for_maxima(req_upper)

        maxima_code += f"res: integrate({vol_str}, {req_var}, {lower_str}, {upper_str})$\n"
        maxima_code += f'print("MAXIMA_START_{idx}")$\n'
        maxima_code += "print(res)$\n"
        maxima_code += f'print("MAXIMA_END_{idx}")$\n'
        dbprint("[DEBUG/MAXIMA] Complete!")

    maxima_code += "quit();\n"

    # 3. Execute maxima as a subprocess (batched, less call overhead)
    try:
        process = subprocess.Popen(
            ['maxima', '--very-quiet'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=maxima_code, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        process.kill()
        dbprint("[DEBUG/MAXIMA (WARNING)] Subprocess timed out. Falling back to unevaluated integrals.")
        return [sp.Integral(req[0], (req[1], req[2], req[3])) for req in valid_requests]

    # 4. Parse results
    results = []
    for idx, req in enumerate(valid_requests):
        vol, req_var, req_lower, req_upper, _ = req
        try:
            start_marker = f"MAXIMA_START_{idx}"
            end_marker = f"MAXIMA_END_{idx}"

            if start_marker in stdout and end_marker in stdout:
                raw_out = stdout.split(start_marker)[1].split(end_marker)[0].strip()

                if "integrate" in raw_out.lower():
                    results.append(sp.Integral(vol, (req_var, req_lower, req_upper)))
                    continue

                # Clean maxima output
                clean_out = raw_out.replace('\\', '').replace('\n', '').replace('\r', '')
                clean_out = clean_out.replace("%pi", "pi").replace("%e", "E").replace("%i", "I")
                clean_out = clean_out.replace("^", "**")

                from sympy.parsing.sympy_parser import parse_expr, standard_transformations, \
                    implicit_multiplication_application
                transformations = standard_transformations + (implicit_multiplication_application,)

                parsed_expr = parse_expr(clean_out, local_dict=local_syms, transformations=transformations)
                results.append(parsed_expr)
            else:
                results.append(sp.Integral(vol, (req_var, req_lower, req_upper)))

        except Exception as e:
            print(f"[MAXIMA PARSE ERROR] Cell {idx}: {e}")
            results.append(sp.Integral(vol, (req_var, req_lower, req_upper)))

    return results


def integrate_cad(cad_data: dict, num_params: int = 1,
                  integrand: str = "auto", global_bounds: dict = None, distributions: dict = None,
                  force_closed_form: bool = False, is_recursive: bool = False) -> dict:
    if not cad_data.get("success", False): return {"calculus": "Failed"}

    base_bounds = cad_data["base_bounds"]
    level_polys = cad_data["level_polys"]
    cells_raw = cad_data["cells_raw"]
    vars_list = cad_data["vars_list"]
    apiprint("\n[INTEGRATE CAD] Got Distributions:", distributions)
    syms = tuple(sp.symbols(" ".join(vars_list), real=True))
    local_syms = {str(s): s for s in syms}

    valid_cells = []
    apiprint("\n[API] Parsing Valid CAD Cells...")

    # =========================================================================
    # STEP 1: PRE-PROCESS CELLS
    # =========================================================================
    for cell in cells_raw:
        idx_match = re.search(r"\(([\d,]+)\)", cell)
        if not idx_match: continue
        indices = [int(x) for x in idx_match.group(1).split(",")]

        is_boundary = any(idx % 2 == 0 and not vars_list[k].startswith('w') for k, idx in enumerate(indices))
        if is_boundary: continue

        pt = extract_cell_point(cell, syms)

        normalized_pt = {}
        for k, v in pt.items():
            if isinstance(k, str):
                normalized_pt[sp.Symbol(k)] = v
            else:
                normalized_pt[k] = v

        if integrand == "auto" or integrand is None:
            target_var = str(syms[0])
            input_syms = tuple(s for s in syms if str(s) != target_var and not str(s).startswith('w'))

            if distributions:
                vol = derive_joint_pdf(input_syms, distributions)
            elif global_bounds:
                vol = auto_derive_uniform_integrand_global(global_bounds)
            else:
                vol = auto_derive_uniform_integrand(input_syms, base_bounds)
        else:
            vol = sp.sympify(integrand, locals=local_syms)

        t_var = syms[0]
        t_lower, t_upper = evaluate_cell_boundary(level_polys[1], normalized_pt, t_var, tuple([t_var]),
                                                  base_bounds.get(str(t_var), {}))

        if t_lower != -sp.oo and t_upper != sp.oo:
            domain = sp.And(t_var > t_lower, t_var < t_upper)
        elif t_lower != -sp.oo:
            domain = t_var > t_lower
        elif t_upper != sp.oo:
            domain = t_var < t_upper
        else:
            domain = sp.S.true

        valid_cells.append({
            "pt": normalized_pt,
            "vol": vol,
            "domain": domain,
            "limit_history": [],
            "limits": []
        })

    if not valid_cells:
        return {"calculus": "0", "cdf_expr": sp.S.Zero, "t_var": syms[0]}

    apiprint(f"\n[API] Running Batched Mathematica-Grade Disjoint Integrator on {len(valid_cells)} cells...\n")

    # =========================================================================
    # STEP 2: BREADTH-FIRST BATCH INTEGRATION
    # =========================================================================
    for lvl in range(len(syms), num_params, -1):
        var = syms[lvl - 1]
        if str(var).startswith('w'): continue
        batch_requests = []

        for i, cell_data in enumerate(valid_cells):
            vol = cell_data["vol"]

            c_lower, c_upper = evaluate_cell_boundary(level_polys[lvl], cell_data["pt"], var, syms,
                                                      base_bounds.get(str(var), {}))

            def resolve_bound(expr, pt_dict):
                if not isinstance(expr, sp.Basic) or not expr.has(sp.Max, sp.Min):
                    return expr
                val_map = {str(k).strip(): float(v) for k, v in pt_dict.items()}

                def evaluate_arg(arg, fallback_val):
                    try:
                        if arg.is_number: return float(arg)
                        sub_list = []
                        for s in arg.free_symbols:
                            name = str(s).strip()
                            if name in val_map:
                                sub_list.append((s, sp.Rational(str(val_map[name]))))
                        subbed_arg = arg.subs(sub_list)
                        root_func = sp.Function('RootOf')
                        if subbed_arg.has(root_func):
                            for node in subbed_arg.atoms(root_func):
                                try:
                                    r_roots = sp.real_roots(node.args[0])
                                    if r_roots:
                                        subbed_arg = subbed_arg.subs(node, r_roots[0].evalf())
                                except Exception:
                                    pass
                        return float(sp.re(subbed_arg.evalf()))
                    except Exception:
                        return fallback_val

                while expr.has(sp.Max):
                    for node in expr.find(sp.Max):
                        expr = expr.xreplace({node: max(node.args, key=lambda a: evaluate_arg(a, -float('inf')))})
                while expr.has(sp.Min):
                    for node in expr.find(sp.Min):
                        expr = expr.xreplace({node: min(node.args, key=lambda a: evaluate_arg(a, float('inf')))})
                return expr

            c_lower_res = resolve_bound(c_lower, cell_data["pt"])
            c_upper_res = resolve_bound(c_upper, cell_data["pt"])

            # Store RAW bounds in limit_history (for deduplication signature).
            # Using resolved bounds can merge cells that should be separate.
            cell_data["limit_history"].append(f"{var}:[{c_lower},{c_upper}]")
            cell_data["limits"].append((var, c_lower_res, c_upper_res))

            if vol == sp.S.Zero:
                batch_requests.append(None)
                continue

            # ==========================================================
            # PIECEWISE BRANCH EXTRACTOR (for Triangular distributions)
            # ==========================================================
            vol_for_symbolic = vol
            did_extract = False

            if vol.has(sp.Piecewise):
                val_map = {}
                for k, v in cell_data["pt"].items():
                    val_map[k] = sp.Rational(str(v))

                def extract_active_branch(pw_expr):
                    """Evaluate the sample point to find which Piecewise branch is active."""
                    for branch_expr, condition in pw_expr.args:
                        if condition == sp.S.true:
                            return branch_expr
                        try:
                            evaluated = condition.subs(val_map)
                            # Force numeric evaluation of any remaining relationals
                            if evaluated == sp.S.true:
                                return branch_expr
                            if isinstance(evaluated, sp.Basic):
                                if evaluated.equals(sp.S.true):
                                    return branch_expr
                                # Try numerical evaluation (fallback)
                                try:
                                    if bool(evaluated):
                                        return branch_expr
                                except (TypeError, ValueError):
                                    pass
                        except Exception:
                            pass
                    return sp.S.Zero

                extracted_vol = vol
                try:
                    max_iters = 20  # safety limit for nested Piecewise
                    iters = 0
                    while extracted_vol.has(sp.Piecewise) and iters < max_iters:
                        iters += 1
                        found_pw = False
                        for pw_node in extracted_vol.find(sp.Piecewise):
                            replacement = extract_active_branch(pw_node)
                            extracted_vol = extracted_vol.xreplace({pw_node: replacement})
                            found_pw = True
                            break
                        if not found_pw:
                            break

                    # Verify extraction succeeded
                    # Expected: No piecewise
                    if not extracted_vol.has(sp.Piecewise):
                        vol_for_symbolic = sp.expand(extracted_vol)
                        did_extract = True
                        dbprint(f"[BRANCH-EXTRACT] Cell {i}: extracted smooth branch: "
                                f"{str(vol_for_symbolic)[:80]}...")
                except Exception as e:
                    dbprint(f"[BRANCH-EXTRACT] Cell {i}: extraction failed ({e}), "
                            f"falling back to numerical")

            # ==========================================================
            # LEVEL 1: C++ O(1) Pattern Matching
            # ==========================================================
            integrand_sym = sp.sympify(vol)
            lower_sym = sp.sympify(c_lower_res)
            upper_sym = sp.sympify(c_upper_res)

            try:
                cpp_anti_deriv_str = pycad_cpp_engine.fast_integrate(str(integrand_sym), str(var))
                if cpp_anti_deriv_str != "NO_CPP_RULE_MATCHED" and not cpp_anti_deriv_str.startswith("ERROR"):
                    anti_deriv = sp.parse_expr(cpp_anti_deriv_str.replace('^', '**'))
                    vol = sp.simplify(anti_deriv.subs(var, upper_sym) - anti_deriv.subs(var, lower_sym))
                    valid_cells[i]["vol"] = vol
                    batch_requests.append(None)
                    continue
            except Exception:
                pass

            # ==========================================================
            # PATH S: Symbolic Maxima Integration
            # ==========================================================
            # Route to Maxima ONLY when bounds are clean (no Max/Min).
            integrand_is_clean = (
                (did_extract and not vol_for_symbolic.has(sp.Piecewise)) or
                (not vol.has(sp.Piecewise) and not vol.has(sp.Integral))
            )
            symbolic_vol = vol_for_symbolic if did_extract else vol

            if integrand_is_clean:
                can_do_symbolic = True

                # For sub-CAD (is_recursive), use RESOLVED bounds for the
                # Max/Min check.
                check_lower = c_lower_res if is_recursive else c_lower
                check_upper = c_upper_res if is_recursive else c_upper

                if (check_lower.has(sp.Max, sp.Min) or check_upper.has(sp.Max, sp.Min) or
                        has_radical(c_lower_res) or has_radical(c_upper_res) or
                        has_radical(symbolic_vol) or
                        symbolic_vol.has(sp.asin, sp.acos, sp.atan)):
                    can_do_symbolic = False

                if can_do_symbolic:
                    dbprint(f"[SYMBOLIC-ROUTE] Cell {i}: routing to Maxima: "
                            f"integrate({str(symbolic_vol)[:60]}, {var}, "
                            f"{c_lower_res}, {c_upper_res})")
                    batch_requests.append(
                        (symbolic_vol, var, c_lower_res, c_upper_res, cell_data))
                    continue

            # ==========================================================
            # PATH A: The High-Speed C++ Numerical Target (DEFAULT)
            # ==========================================================
            # Two flags:
            # 1. is_recursive: if called recursively, we must be in closed form
            # 2. Can force closed_form from user, flag not implemented yet as further testing required
            if not (force_closed_form or is_recursive):
                # Use the extracted smooth integrand if branch extraction succeeded,
                # otherwise fall back to the original integrand.
                num_vol = symbolic_vol if (did_extract and not symbolic_vol.has(sp.Piecewise)) else vol
                valid_cells[i]["vol"] = sp.Integral(num_vol, (var, c_lower, c_upper))
                batch_requests.append(None)
                continue

            # ==========================================================
            # PATH B: The Full Symbolic CAS Target
            # ==========================================================
            if (vol.has(sp.Integral) or c_lower_res.has(sp.Max, sp.Min) or c_upper_res.has(sp.Max, sp.Min) or
                    has_radical(vol) or has_radical(c_lower_res) or has_radical(c_upper_res) or
                    vol.has(sp.asin, sp.acos, sp.atan)):
                valid_cells[i]["vol"] = sp.Integral(vol, (var, c_lower, c_upper))
                batch_requests.append(None)
                continue

            batch_requests.append((vol, var, c_lower_res, c_upper_res, cell_data))

        valid_requests = []
        result_queue = []
        for req in batch_requests:
            if req is None:
                result_queue.append(None)
                continue

            vol, req_var, req_lower, req_upper, cell_data = req
            cache_key = (str(vol), str(req_var))

            if cache_key in INDEFINITE_INTEGRAL_CACHE:
                indef_F = INDEFINITE_INTEGRAL_CACHE[cache_key]
                if indef_F is not None and not indef_F.has(sp.Integral):
                    generic_var = sp.Symbol(str(req_var))
                    def_integral = indef_F.subs(generic_var, req_upper) - indef_F.subs(generic_var, req_lower)
                    result_queue.append(def_integral)
                    continue

            valid_requests.append(req)
            result_queue.append("PENDING")

        if valid_requests:
            valid_results = call_maxima_integrate_batch(
                valid_requests,
                local_syms=local_syms
            )

            for req, res in zip(valid_requests, valid_results):
                for idx, queued_item in enumerate(result_queue):
                    if queued_item == "PENDING" and batch_requests[idx] == req:
                        result_queue[idx] = res
                        break

        for i, res in enumerate(result_queue):
            if res is not None:
                if not res.has(sp.Integral) and res != sp.S.Zero:
                    req_info = batch_requests[i]
                    if req_info is not None:
                        _, _, req_lower, req_upper, _ = req_info
                        if not (req_lower.is_number and req_upper.is_number):
                            is_valid = req_upper >= req_lower
                            res = sp.Piecewise((res, is_valid), (0, True))

                valid_cells[i]["vol"] = res

    # =========================================================================
    # STEP 3: ASSEMBLE FINAL PIECEWISE CDF
    # =========================================================================
    unique_pieces = []
    seen_signatures = set()

    for c in valid_cells:
        vol = c["vol"]
        domain = c["domain"]
        limit_sig = "||".join(c["limit_history"])
        sig = f"{vol}|||{domain}|||{limit_sig}"
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique_pieces.append((vol, domain))

    total_cdf = sum(
        [vol if domain == sp.S.true else sp.Piecewise((vol, domain), (0, True)) for vol, domain in unique_pieces],
        sp.S.Zero
    )

    ops_count = total_cdf.count_ops()
    if not force_closed_form and (total_cdf.has(sp.Integral) or ops_count > 400 or len(unique_pieces) > 10):
        clean_cdf = total_cdf
        output_str = "--- Final Mathematica-Grade CDF ---\n      F(t) = [Massive Piecewise Sum Generated]\n"
    else:
        clean_cdf = sp.piecewise_fold(total_cdf)
        output_str = "--- Final Mathematica-Grade CDF ---\n"
        if isinstance(clean_cdf, sp.Piecewise):
            for expr, cond in clean_cdf.args: output_str += f"For {cond}:\n      F({syms[0]}) = {expr}\n\n"
        else:
            output_str += f"      F({syms[0]}) = {clean_cdf}\n\n"

    # =========================================================================
    # STEP 4: EXTRACT CLOSED-FORM FULL CAD DNF
    # =========================================================================
    cell_conditions = []
    for c in valid_cells:
        conds = [c["domain"]] if c["domain"] != sp.S.true else []
        for var, lower, upper in c.get("limits", []):
            if lower != -sp.oo: conds.append(var >= lower)
            if upper != sp.oo: conds.append(var <= upper)
        if conds:
            cell_conditions.append(sp.And(*conds))

    full_cad_dnf = sp.Or(*cell_conditions)

    return {"calculus": output_str.strip(), "cdf_expr": clean_cdf, "t_var": syms[0], "cad_dnf": full_cad_dnf}

def extract_optimized_cad_dnf(valid_cells, syms):
    """
    Extracts the exact physical boundaries of the CAD cells,
    converts them to Boolean logic, and mathematically merges contiguous regions.
    """

    cell_conditions = []

    for c in valid_cells:
        # 1. Start with the outer domain (t)
        conds = [c["domain"]] if c["domain"] != sp.S.true else []

        # 2. Add the inner spatial limits
        for var, lower, upper in c["limits"]:
            if lower != -sp.oo:
                conds.append(var >= lower)
            if upper != sp.oo:
                conds.append(var <= upper)

        # 3. A cell is the intersection (AND) of all its boundaries
        if conds:
            cell_conditions.append(sp.And(*conds))

    # 4. The total CAD space is the union (OR) of all valid cells
    raw_dnf = sp.Or(*cell_conditions)

    # 5. Topological merge
    apiprint("\n[CAD EXTRACTOR] Compressing CAD Disjuncts into Minimal DNF...")
    try:
        optimized_dnf = sp.simplify_logic(raw_dnf, form='dnf', deep=True)
    except Exception as e:
        apiprint(f"[CAD EXTRACTOR WARNING] Logic simplification timed out/failed: {e}")
        optimized_dnf = raw_dnf

    return raw_dnf, optimized_dnf


def extract_full_cad_dnf(valid_cells, syms):
    """
    Extracts the complete multi-dimensional boundaries of the CAD cells.
    """

    cell_conditions = []

    for c in valid_cells:
        # 1. Outer projection domain (t)
        conds = [c["domain"]] if c["domain"] != sp.S.true else []

        # 2. Inner spatial cylinders (x, u0)
        for var, lower, upper in c["limits"]:
            if lower != -sp.oo:
                conds.append(var > lower)
            if upper != sp.oo:
                conds.append(var < upper)

        if conds:
            cell_conditions.append(sp.And(*conds))

    raw_dnf = sp.Or(*cell_conditions)

    apiprint("\n[CAD EXTRACTOR] Compressing full spatial CAD...")
    try:
        # Attempt boolean compression
        # may not merge continuous float boundaries properly
        optimized_dnf = sp.simplify_logic(raw_dnf, form='dnf', deep=True)
    except Exception as e:
        optimized_dnf = raw_dnf

    return raw_dnf, optimized_dnf

def get_outermost_integrals(expr):
    """Recursively finds the outermost integrals to prevent unordered .atoms() traversal."""
    if isinstance(expr, sp.Integral):
        return [expr]
    if not hasattr(expr, 'args') or not expr.args:
        return []
    outermost = []
    for arg in expr.args:
        outermost.extend(get_outermost_integrals(arg))
    return outermost


def evaluate_numeric_integral(integrand, limits):

    if len(limits) == 0:
        try:
            return float(complex(integrand.evalf()).real)
        except Exception:
            return 0.0

    vars_list = [l[0] for l in limits]

    safe_math = {
        'Max': lambda *args: max(args),
        'Min': lambda *args: min(args),
        'Abs': abs,
        'sign': lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0),
        'Heaviside': lambda x: 1.0 if x >= 0 else 0.0,
        'sqrt': lambda x: math.sqrt(x) if x >= 0 else 0.0,
        'asin': lambda x: math.asin(1.0) if x > 1.0 else (math.asin(-1.0) if x < -1.0 else math.asin(x)),
        'acos': lambda x: math.acos(1.0) if x > 1.0 else (math.acos(-1.0) if x < -1.0 else math.acos(x))
    }

    try:
        f_integrand = sp.lambdify(vars_list, integrand, modules=[safe_math, 'math'])
    except Exception:
        f_integrand = None

    def safe_integrand(*args):
        if f_integrand is not None:
            try:
                val = float(f_integrand(*args))
                if math.isnan(val) or math.isinf(val): return 0.0
                return val
            except Exception:
                return 0.0
        return 0.0

    ranges = []
    opts = []

    for i, (v, a_expr, b_expr) in enumerate(limits):
        outer_vars = vars_list[i + 1:]

        a_func = sp.lambdify(outer_vars, a_expr, modules=[safe_math, 'math']) if outer_vars else sp.lambdify((), a_expr,
                                                                                                             modules=[
                                                                                                                 safe_math,
                                                                                                                 'math'])
        b_func = sp.lambdify(outer_vars, b_expr, modules=[safe_math, 'math']) if outer_vars else sp.lambdify((), b_expr,
                                                                                                             modules=[
                                                                                                                 safe_math,
                                                                                                                 'math'])

        def make_range(ac, bc):
            def bound(*args):
                try:
                    a_val = float(ac(*args))
                except Exception:
                    a_val = 0.0
                try:
                    b_val = float(bc(*args))
                except Exception:
                    b_val = 0.0

                if a_val >= b_val:
                    return [0.0, 0.0]

                return [a_val, b_val]

            return bound

        bound_func = make_range(a_func, b_func)
        ranges.append(bound_func)

        sym_points = []
        if integrand.has(sp.Piecewise):
            for pw in integrand.atoms(sp.Piecewise):
                for _, cond in pw.args:
                    for rel in cond.atoms(sp.core.relational.Relational):
                        if v in rel.free_symbols:
                            try:
                                if rel.lhs == v and rel.rhs.is_number:
                                    sym_points.append(rel.rhs)
                                elif rel.rhs == v and rel.lhs.is_number:
                                    sym_points.append(rel.lhs)
                                else:
                                    eq = rel.lhs - rel.rhs
                                    if not eq.has(sp.asin, sp.acos, sp.Abs, sp.Max, sp.Min, sp.sign, sp.sqrt):
                                        roots = sp.solve(eq, v)
                                        if roots: sym_points.append(roots[0])
                            except Exception:
                                pass

        sym_points = list(set(sym_points))
        pt_funcs = [sp.lambdify(outer_vars, pt, modules=[safe_math, 'math']) if outer_vars else sp.lambdify((), pt,
                                                                                                            modules=[
                                                                                                                safe_math,
                                                                                                                'math'])
                    for pt in sym_points]

        def make_opts(ptfs, bnd_func):
            def opts_callable(*args):
                limits_val = bnd_func(*args)
                a_val, b_val = min(limits_val), max(limits_val)
                pts = []
                for pf in ptfs:
                    try:
                        val = float(pf(*args))
                        if a_val + 1e-5 < val < b_val - 1e-5:
                            pts.append(val)
                    except Exception:
                        pass
                return {'limit': 50, 'epsabs': 1e-3, 'epsrel': 1e-3, 'points': list(set(pts))}

            return opts_callable

        opts.append(make_opts(pt_funcs, bound_func))

    try:
        res = scipy_int.nquad(safe_integrand, ranges, opts=opts)[0]
        return res
    except Exception:
        return 0.0


def resolve_numeric_roots(expr):
    """
    Evaluates purely numerical RootOf objects to physical floats.
    Catches BOTH SymPy native RootOf and CAD-generated custom Function('RootOf').
    """
    # SymPy AST nodes inherit from sp.Basic.
    # Ignore if its int/float
    if not isinstance(expr, sp.Basic):
        return expr

    replacements = {}

    root_nodes = []
    for n in sp.preorder_traversal(expr):
        func_name = getattr(getattr(n, 'func', None), '__name__', '')
        class_name = str(type(n).__name__)
        if isinstance(n, sp.RootOf) or func_name == 'RootOf' or class_name == 'RootOf':
            root_nodes.append(n)

    if not root_nodes:
        return expr

    for r in set(root_nodes):
        # 1. Identify if this is a CAD-generated unindexed RootOf
        is_unindexed = len(r.args) >= 2 and isinstance(r.args[1], sp.Symbol)
        target_var = r.args[1] if is_unindexed else None

        # An unindexed RootOf inherently contains its own dummy variable.
        allowed_syms = {target_var} if target_var else set()

        # Only evaluate if there are NO OTHER free symbols
        if r.free_symbols.issubset(allowed_syms):
            try:
                if is_unindexed:
                    poly = r.args[0]

                    # Clean the CAD Abs/Sign artifacts
                    clean_poly = poly.replace(sp.sign, lambda arg: 1).replace(sp.Abs, lambda arg: arg)
                    clean_poly = sp.expand(clean_poly)

                    # Force numerical root finding directly on the restored polynomial
                    n_roots = sp.nroots(clean_poly)
                    real_roots = [rt for rt in n_roots if abs(sp.im(rt)) < 1e-7]
                    replacements[r] = float(sp.re(real_roots[0])) if real_roots else sp.nan

                # Standard indexed SymPy CRootOf
                else:
                    c_val = complex(r.evalf(chop=True))
                    replacements[r] = c_val.real if abs(c_val.imag) < 1e-7 else sp.nan
            except Exception as e:
                pass

    return expr.xreplace(replacements)


def flatten_integrals(expr):
    """
    Applies linearity of integration to structurally flatten the AST.
    Deeply distributes Integrals into Additions and Piecewise branches.
    """
    if getattr(expr, "is_Atom", False) or not hasattr(expr, "args"):
        return expr

    # Process children bottom-up
    args = [flatten_integrals(arg) for arg in expr.args]
    expr = expr.func(*args)

    if isinstance(expr, sp.Integral):
        integrand = expr.args[0]
        limits = expr.args[1:]

        # 1. Push external multipliers INSIDE Piecewise branches
        # e.g., z * Piecewise((IntA, cond)) -> Piecewise((z * IntA, cond))
        if integrand.has(sp.Piecewise):
            integrand = sp.piecewise_fold(integrand)

        # 2. Deeply distribute multiplication over additions
        if integrand.has(sp.Integral) and isinstance(integrand, sp.Mul):
            integrand = sp.expand(integrand)

        # 3. Distribute Integral over Additions
        if isinstance(integrand, sp.Add):
            return sp.Add(*[flatten_integrals(sp.Integral(arg, *limits)) for arg in integrand.args])

        # 4. Pull inner integrals out of Multiplications
        if isinstance(integrand, sp.Mul):
            inner_integrals = [arg for arg in integrand.args if isinstance(arg, sp.Integral)]
            others = [arg for arg in integrand.args if not isinstance(arg, sp.Integral)]

            if inner_integrals:
                inner_int = inner_integrals[0]
                inner_integrand = inner_int.args[0]
                inner_limits = inner_int.args[1:]
                new_inner_integrand = inner_integrand * sp.Mul(*(others + inner_integrals[1:]))
                return flatten_integrals(sp.Integral(sp.Integral(new_inner_integrand, *inner_limits), *limits))

    return expr


def fast_eval(expr):
    """Evaluates SymPy expressions using the newly upgraded Breakpoint-Aware C++ Engine."""
    try:
        expr = resolve_numeric_roots(expr)
        expr = flatten_integrals(expr)
        outermost_integrals = get_outermost_integrals(expr)

        for idx, integ in enumerate(set(outermost_integrals)):
            core_integrand = integ.args[0]
            lims = list(integ.args[1:])

            curr = core_integrand
            while isinstance(curr, sp.Integral):
                lims = list(curr.args[1:]) + lims
                curr = curr.args[0]

            cpp_integrand = sympy_to_exprtk(curr)

            var_names = []
            lower_strs = []
            upper_strs = []
            breakpoint_strs = []

            for i, limit in enumerate(lims):
                var, a_expr, b_expr = limit
                var_names.append(str(var))

                lower_str = sympy_to_exprtk(a_expr)
                upper_str = sympy_to_exprtk(b_expr)

                lower_strs.append(lower_str)
                upper_strs.append(f"max({lower_str}, {upper_str})")

                # Scans the Piecewise conditions to find the exact modes/kinks
                dim_bps = []
                if curr.has(sp.Piecewise):
                    for pw in curr.atoms(sp.Piecewise):
                        for _, cond in pw.args:
                            for rel in cond.atoms(sp.core.relational.Relational):
                                if var in rel.free_symbols:
                                    try:
                                        # If condition is like (x < 2.5) or (x > mode)
                                        if rel.lhs == var and not rel.rhs.has(var):
                                            dim_bps.append(sympy_to_exprtk(rel.rhs))
                                        elif rel.rhs == var and not rel.lhs.has(var):
                                            dim_bps.append(sympy_to_exprtk(rel.lhs))
                                    except Exception:
                                        pass

                # Append unique breakpoints for this specific dimension
                breakpoint_strs.append(list(set(dim_bps)))

            # Calling C++ module for symbolic integration
            dbprint(f"[DEBUG/CPP-ENGINE] Calling integration with\n\t->{cpp_integrand}\n\t->{var_names}\n\t->{lower_strs}\n\t->{upper_strs}\n\t->{breakpoint_strs}")
            val = pycad_cpp_engine.fast_nd_quadrature(
                cpp_integrand,
                var_names,
                lower_strs,
                upper_strs,
                1e-6,
                1e-6,
                8,
                breakpoint_strs
            )
            expr = expr.subs(integ, val)

        c_val = complex(expr.evalf())
        return c_val.real if abs(c_val.imag) < 1e-5 else np.nan

    except Exception as e:
        return np.nan


def _evaluate_pdf_chunk(t_chunk, cdf_expr, t_var_str):

    t_var = sp.Symbol(t_var_str, real=True)
    c_vals, p_vals = [], []

    for i, val in enumerate(t_chunk):
        exact_val = sp.Rational(str(val))

        dbprint(f"\n============================================================")
        dbprint(f"[DEBUG/CDF-EVAL] Evaluating t = {float(exact_val):.4f} (Point {i + 1}/{len(t_chunk)})")

        # 1. Substitute 't' into the CDF Piecewise
        try:
            math_cdf = cdf_expr.subs(t_var, exact_val)
        except (ValueError, TypeError):
            try:
                math_cdf = cdf_expr.subs(t_var, float(val))
            except Exception:
                c_vals.append(np.nan)
                p_vals.append(0.0)
                continue

        c_float = np.nan
        try:
            # 2. Evaluate using C++ engine
            c_val = fast_eval(math_cdf)
            if not np.isnan(c_val):
                c_float = float(c_val)
            else:
                c_float = float(math_cdf.evalf(chop=True))

            dbprint(f"\n[DEBUG/CDF-EVAL] TOTAL C++ CDF VOLUME at t={float(exact_val):.4f} = {c_float}")
        except Exception as e:
            # Retry with float substitution
            try:
                math_cdf_float = cdf_expr.subs(t_var, float(val))
                c_val = fast_eval(math_cdf_float)
                if not np.isnan(c_val):
                    c_float = float(c_val)
                else:
                    c_float = float(math_cdf_float.evalf(chop=True))
                dbprint(f"\n[DEBUG/CDF-EVAL] TOTAL C++ CDF VOLUME (float fallback) at t={float(exact_val):.4f} = {c_float}")
            except Exception as e2:
                # Fallback: pure numerical evaluation via evalf(subs=...)
                try:
                    raw_val = cdf_expr.evalf(subs={t_var: float(val)}, chop=True)
                    c_val_raw = complex(raw_val)
                    c_float = c_val_raw.real if abs(c_val_raw.imag) < 1e-5 else np.nan
                    dbprint(f"\n[DEBUG/CDF] TOTAL EVALF CDF VOLUME (numerical fallback) at t={float(exact_val):.4f}] = {c_float}")
                except Exception as e3:
                    dbprint(f"[CDF EVAL ERROR]: {e} -> {e2} -> {e3}")

        c_vals.append(c_float)

        # 3. Fallback Finite Difference
        p_float = np.nan
        if np.isnan(p_float) and not np.isnan(c_float):
            try:
                h = 5e-2
                val_h = val + h
                try:
                    math_cdf_h = cdf_expr.subs(t_var, sp.Rational(str(val_h)))
                except (ValueError, TypeError):
                    math_cdf_h = cdf_expr.subs(t_var, float(val_h))

                try:
                    c_val_h = fast_eval(math_cdf_h)
                    if np.isnan(c_val_h):
                        c_val_h = float(math_cdf_h.evalf(chop=True))
                except Exception:
                    # Retry with float substitution
                    math_cdf_h = cdf_expr.subs(t_var, float(val_h))
                    try:
                        c_val_h = fast_eval(math_cdf_h)
                    except Exception:
                        c_val_h = np.nan

                if not np.isnan(c_val_h):
                    p_float = (c_val_h - c_float) / h
            except Exception:
                pass

        p_vals.append(p_float)
    return c_vals, p_vals

def add_monte_carlo_overlay(python_lambda_func, var_bounds, mc_generators=None, num_samples=500000):
    samples = {}
    for var, (low, high) in var_bounds.items():
        # Check if a custom generator (e.g.,  Triangular) was passed for this variable
        if mc_generators and var in mc_generators:
            samples[var] = mc_generators[var](num_samples)
        else:
            # Fallback to Uniform only if nothing was provided
            samples[var] = np.random.uniform(low, high, num_samples)

    # Evaluate the numpy lambda function across all half-million samples instantly
    t_simulated = python_lambda_func(**samples)

    # Filter out imaginary/NaN/Inf values that fall outside physical bounds
    t_simulated = t_simulated[~np.isnan(t_simulated) & ~np.isinf(t_simulated)]
    return t_simulated



def calculate_pdf(cad_integration_result: dict, force_closed_form: bool = False, is_recursive: bool = False):
    """Symbolically differentiates the CDF to find the exact PDF."""
    if "cdf_expr" not in cad_integration_result:
        apiprint("Error: No valid CDF provided to differentiate.")
        return None

    cdf_expr = cad_integration_result["cdf_expr"]
    t_var = cad_integration_result["t_var"]

    apiprint("\n[API] Calculating Probability Density Function (PDF)...")


    needs_symbolic = is_recursive or force_closed_form
    root_func = sp.Function('RootOf')
    if not needs_symbolic and (cdf_expr.has(sp.Integral) or cdf_expr.has(sp.RootOf) or cdf_expr.has(root_func)):
        apiprint("[HEURISTIC ENGINE] Unevaluated integrals or roots detected.")
        apiprint("[HEURISTIC ENGINE] Engaging SciPy Fast Numerical Calculus Fallback...")
        pdf_expr = None  # Safely flag for numerical fallback
        output_str = "--- Final Mathematica-Grade PDF ---\n      f(t) = [Computed Numerically via SciPy Finite Differences]\n"
    else:
        def fast_piecewise_diff(expr, var):
            """Differentiates massive piecewise functions while strictly blocking SymPy's auto-simplifier."""
            if not expr.has(sp.Piecewise):
                # Evaluate the raw derivative, forbidding any deep evaluation/simplification
                return sp.diff(expr, var, evaluate=False).doit(deep=False)

            # If sum of cells, recursively differentiate each cell
            if isinstance(expr, sp.Add):
                return sp.Add(*[fast_piecewise_diff(arg, var) for arg in expr.args])

            # If Piecewise block, differentiate the branches safely
            if isinstance(expr, sp.Piecewise):
                new_branches = []
                for branch_expr, condition in expr.args:
                    if branch_expr == sp.S.Zero:
                        new_branches.append((sp.S.Zero, condition))
                    else:
                        raw_diff = sp.diff(branch_expr, var)
                        new_branches.append((raw_diff, condition))
                return sp.Piecewise(*new_branches)

            # Fallback
            return sp.diff(expr, var)

        # Safely extract the derivative
        raw_pdf = fast_piecewise_diff(cdf_expr, t_var)

        if not needs_symbolic and raw_pdf.has(sp.Derivative):
            apiprint("[HEURISTIC ENGINE] Unresolved symbolic derivatives detected.")
            apiprint("[HEURISTIC ENGINE] Engaging SciPy Fast Numerical Calculus Fallback...")
            pdf_expr = None
            output_str = "--- Final Mathematica-Grade PDF ---\n      f(t) = [Computed Numerically via SciPy Finite Differences]\n"
        else:
            pdf_expr = raw_pdf

            output_str = f"--- Final Mathematica-Grade PDF ---\n"
            if isinstance(pdf_expr, sp.Piecewise):
                for expr, cond in pdf_expr.args:
                    output_str += f"For {cond}:\n      f({t_var}) = {expr}\n\n"
            else:
                output_str += f"      f({t_var}) = {pdf_expr}\n\n"

    dbprint(output_str.strip())
    return pdf_expr


def plot_distributions(cad_integration_result: dict, pdf_expr, t_min: float = None, t_max: float = None,
                       filename: str = "distribution", mc_func=None, mc_bounds=None, mc_generators=None):
    """Numerically evaluates the CDF/PDF and renders the matplotlib graphs."""

    if "cdf_expr" not in cad_integration_result:
        return

    cdf_expr = cad_integration_result["cdf_expr"]
    t_var = cad_integration_result["t_var"]

    def _sanitize_for_numpy(expr):
        if not isinstance(expr, sp.Expr): return expr

        root_map = {}
        dummy_counter = 0

        def hide_roots(e):
            nonlocal dummy_counter
            if isinstance(e, sp.RootOf) or str(type(e).__name__) == 'RootOf':
                d = sp.Symbol(f"DUMMY_ROOT_{dummy_counter}")
                root_map[d] = e
                dummy_counter += 1
                return d
            if not getattr(e, 'args', ()): return e
            return e.func(*[hide_roots(a) for a in e.args])

        safe_expr = hide_roots(expr)

        # Clamp roots and fix NaN Fractional Powers
        for node in safe_expr.find(sp.Pow):
            if node.exp == sp.S.Half or node.exp == 0.5:
                safe_expr = safe_expr.xreplace({node: sp.Pow(sp.Max(0.0, node.base), node.exp)})
            elif node.exp == -sp.S.Half or node.exp == -0.5:
                safe_expr = safe_expr.xreplace({node: sp.Pow(sp.Max(1e-12, node.base), node.exp)})
            elif node.exp.is_Rational and node.exp.q % 2 != 0:
                p = node.exp.p
                safe_base = node.base
                safe_expr = safe_expr.xreplace({
                    node: (sp.sign(safe_base) ** p) * sp.Pow(sp.Abs(safe_base), node.exp)
                })

        # Put roots back
        return safe_expr.xreplace(root_map)

    numerical_cdf = _sanitize_for_numpy(cdf_expr)
    numerical_pdf = _sanitize_for_numpy(pdf_expr) if pdf_expr is not None else None
    if cdf_expr.has(sp.Integral):
        numerical_pdf = None
        pdf_expr = None
    if t_min is None or t_max is None:
        extracted_bounds = []
        for rel in cdf_expr.atoms(Relational):
            is_lhs_target = str(rel.lhs) == str(t_var) or (str(t_var) == 't' and str(rel.lhs).startswith('u0'))
            is_rhs_target = str(rel.rhs) == str(t_var) or (str(t_var) == 't' and str(rel.rhs).startswith('u0'))

            if is_lhs_target and rel.rhs.is_number:
                extracted_bounds.append(float(rel.rhs))
            elif is_rhs_target and rel.lhs.is_number:
                extracted_bounds.append(float(rel.lhs))

        if extracted_bounds:
            b_min, b_max = min(extracted_bounds), max(extracted_bounds)
            padding = max(0.1, (b_max - b_min) * 0.1)
            t_min = t_min if t_min is not None else b_min - padding
            t_max = t_max if t_max is not None else b_max + padding
        else:
            t_min, t_max = -1, 5  # Fallback

    apiprint(f"\n[API] Generating graphs bounds [{t_min:.2f}, {t_max:.2f}]... Saving to {filename}.png")
    global PLOT_POINTS
    apiprint("Plot Points:", PLOT_POINTS)
    t_vals = np.linspace(t_min + 1e-3, t_max - 1e-3, PLOT_POINTS) + (np.sqrt(2) * 1e-8)
    cdf_vals, pdf_vals = [], []

    root_func = sp.Function('RootOf')
    has_roots = cdf_expr.has(root_func) or cdf_expr.has(sp.RootOf)
    if pdf_expr is not None:
        has_roots = has_roots or pdf_expr.has(root_func) or pdf_expr.has(sp.RootOf)

    is_closed_form = pdf_expr is not None and not cdf_expr.has(sp.Integral) and not has_roots

    if is_closed_form:
        apiprint("[API] Closed form detected! Compiling via SymPy Lambdify for instant evaluation...")
        try:
            cdf_fast = sp.lambdify(t_var, numerical_cdf, modules=['numpy', 'math'])
            pdf_fast = sp.lambdify(t_var, numerical_pdf, modules=['numpy', 'math'])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cdf_vals = [float(cdf_fast(v)) for v in t_vals]
                pdf_vals = [float(pdf_fast(v)) for v in t_vals]
        except Exception as e:
            apiprint(f"[API] Lambdify fallback triggered. Switching to robust evaluation...")
            is_closed_form = False

    if not is_closed_form:
        apiprint(f"[API] Firing Parallel Numerical Evaluation across CPU cores...")
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        t_chunks = np.array_split(t_vals, num_cores)

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(_evaluate_pdf_chunk, chunk, numerical_cdf, str(t_var)) for chunk in t_chunks]
            for future in futures:
                c_res, p_res = future.result()
                cdf_vals.extend(c_res)
                pdf_vals.extend(p_res)

    cdf_raw = np.array(cdf_vals, dtype=float)
    for idx_nan in range(len(cdf_raw)):
        if np.isnan(cdf_raw[idx_nan]):
            cdf_raw[idx_nan] = cdf_raw[idx_nan - 1] if idx_nan > 0 else 0.0
    cdf_raw = np.maximum(cdf_raw, 0.0)

    cdf_stitched = cdf_raw.copy()
    cumulative_offset = 0.0
    stitch_indices = []
    for idx_st in range(1, len(cdf_stitched)):
        raw_jump = cdf_raw[idx_st] - cdf_raw[idx_st - 1]
        if raw_jump < -0.005:
            corrected_prev = cdf_raw[idx_st - 1] + cumulative_offset
            cumulative_offset = corrected_prev - cdf_raw[idx_st]
            stitch_indices.append(idx_st)
        cdf_stitched[idx_st] = cdf_raw[idx_st] + cumulative_offset

    # Cosine blending at stitch points
    blend_radius = max(5, len(cdf_stitched) // 40)
    for si in stitch_indices:
        lo_b = max(0, si - blend_radius)
        hi_b = min(len(cdf_stitched) - 1, si + blend_radius)
        if hi_b > lo_b:
            v_s = cdf_stitched[lo_b]
            v_e = cdf_stitched[hi_b]
            for j_b in range(lo_b, hi_b + 1):
                alpha = (j_b - lo_b) / (hi_b - lo_b)
                w_b = 0.5 * (1.0 - np.cos(np.pi * alpha))
                cdf_stitched[j_b] = v_s * (1.0 - w_b) + v_e * w_b

    # Normalize
    tail_start_idx = max(1, int(0.9 * len(cdf_stitched)))
    plateau_val = np.mean(cdf_stitched[tail_start_idx:])
    if plateau_val > 0.01:
        cdf_stitched = cdf_stitched / plateau_val
    cdf_stitched = np.clip(cdf_stitched, 0.0, 1.0)

    # Isotonic regression safety net
    def _isotonic_regression(y):
        n = len(y)
        blocks = [[y[idx_iso], 1] for idx_iso in range(n)]
        idx_iso = 0
        while idx_iso < len(blocks) - 1:
            if blocks[idx_iso][0] / blocks[idx_iso][1] > blocks[idx_iso + 1][0] / blocks[idx_iso + 1][1]:
                blocks[idx_iso][0] += blocks[idx_iso + 1][0]
                blocks[idx_iso][1] += blocks[idx_iso + 1][1]
                blocks.pop(idx_iso + 1)
                if idx_iso > 0: idx_iso -= 1
            else:
                idx_iso += 1
        out = np.empty(n)
        pos = 0
        for s_val, c_val_iso in blocks:
            out[pos:pos + c_val_iso] = s_val / c_val_iso
            pos += c_val_iso
        return out

    cdf_final = _isotonic_regression(cdf_stitched)
    cdf_vals = cdf_final.tolist()

    # Re-derive PDF from corrected CDF
    pdf_corrected = np.zeros(len(t_vals))
    for idx_pdf in range(1, len(t_vals) - 1):
        pdf_corrected[idx_pdf] = (cdf_final[idx_pdf + 1] - cdf_final[idx_pdf - 1]) / (t_vals[idx_pdf + 1] - t_vals[idx_pdf - 1])
    pdf_corrected[0] = (cdf_final[1] - cdf_final[0]) / (t_vals[1] - t_vals[0])
    pdf_corrected[-1] = (cdf_final[-1] - cdf_final[-2]) / (t_vals[-1] - t_vals[-2])
    pdf_corrected = np.maximum(pdf_corrected, 0.0)
    pdf_vals = pdf_corrected.tolist()

    if stitch_indices:
        apiprint(f"[API] Band-stitching: {len(stitch_indices)} repairs, "
                 f"offset: {cumulative_offset:.4f}")


    # Pre-generate Monte Carlo samples (shared by both subplots)
    ks_dist = None
    t_sim_array = None
    t_sim_sorted = None
    ecdf_vals = None

    if mc_func and mc_bounds:
        apiprint("[API] Overlaying Monte Carlo Simulation (500k samples)...")
        t_sim = add_monte_carlo_overlay(mc_func, mc_bounds, mc_generators=mc_generators)
        t_sim_array = np.array(t_sim)

        # Sort to create the Empirical CDF
        t_sim_sorted = np.sort(t_sim_array)

        # Calculate cumulative probability of the MC samples at evaluation points
        ecdf_vals = np.searchsorted(t_sim_sorted, t_vals, side='right') / len(t_sim_sorted)

        # K-S Distance: max difference between Math CDF and Empirical CDF
        cdf_array = np.array(cdf_vals)
        ks_dist = float(np.max(np.abs(cdf_array - ecdf_vals)))
        apiprint(f"[API] Computed K-S Distance: {ks_dist:.6f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if t_sim_array is not None:
        # Monte Carlo ECDF histogram: cumulative, normalized to [0,1]
        plt.hist(t_sim_array, bins=200, density=False, cumulative=True,
                 weights=np.ones(len(t_sim_array)) / len(t_sim_array),
                 alpha=0.3, color='green', label='empirical')
    plt.plot(t_vals, cdf_vals, label="semi-evaluated", color='blue', linewidth=2)
    # plt.title("Cumulative Distribution Function")
    plt.ylim(bottom=-0.05, top=1.05)
    # plt.grid(True, linestyle='--', alpha=0.7)
    plt.grid(False)
    plt.legend()

    # --- PDF SUBPLOT ---
    plt.subplot(1, 2, 2)
    if t_sim_array is not None:
        plt.hist(t_sim_array, bins=200, density=True, alpha=0.3, color='green', label='histogram')
    plt.plot(t_vals, pdf_vals, label="semi-evaluated", color='red', linewidth=2)
    # plt.title("Probability Density Function")
    # plt.grid(True, linestyle='--', alpha=0.7)
    plt.grid(False)
    plt.legend()

    valid_pdfs = [p for p in pdf_vals if not np.isnan(p) and not np.isinf(p)]
    if valid_pdfs:
        max_pdf = max(valid_pdfs)
        y_pad = max_pdf * 0.05 if max_pdf > 0 else 0.1
        plt.ylim(bottom=-y_pad, top=max_pdf * 1.1)

    plt.tight_layout()
    plt.savefig(f"./outputs/{filename}.png", dpi=300)
    plt.close()
    apiprint(f"[API] Graph saved successfully!")
    return ks_dist
