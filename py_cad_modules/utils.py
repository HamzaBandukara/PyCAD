import re
import sympy as sp
import subprocess
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.relational import Relational

# Print functions

DEBUG = False
APIPrint = False

def update_Debug():
    global DEBUG
    DEBUG = True

def update_APIPrint():
    global APIPrint
    APIPrint = True

def get_debug():
    global DEBUG
    return DEBUG

def get_apiprint():
    global APIPrint
    return APIPrint

def dbprint(*args, **kwargs):
    """Prints debug information if DEBUG is True."""
    if get_debug():
        print(*args, **kwargs)

def apiprint(*args, **kwargs):
    """Prints API point information if APIPrint is True."""
    if get_apiprint():
        print(*args, **kwargs)

# --------------------------------------------------------

def format_clean_dnf(dnf_expr):
    """
    Formats a SymPy boolean DNF into a clean, readable text block,
    avoiding massive 2D Unicode art.
    """
    if dnf_expr in (sp.S.true, True): return "True (All Space)"
    if dnf_expr in (sp.S.false, False): return "False (Empty Space)"

    # If it's a massive OR statement, break each cell onto a new line
    if isinstance(dnf_expr, sp.Or):
        cell_strings = []
        for cell in dnf_expr.args:
            # Format the inner AND conditions cleanly
            if isinstance(cell, sp.And):
                conds = [str(arg).replace('**', '^') for arg in cell.args]
                cell_strings.append(" AND ".join(conds))
            else:
                cell_strings.append(str(cell).replace('**', '^'))

        # Join the cells with a clear visual separator
        return "\n  OR  [ ".join(["[ " + cell_strings[0]] + cell_strings[1:]) + " ]"

    # Fallback for single conditions
    return str(dnf_expr).replace('**', '^')

def has_radical(expr) -> bool:
    """Checks if a SymPy expression contains fractional exponents."""
    if not hasattr(expr, 'atoms'): return False
    for p in expr.atoms(sp.Pow):
        if not p.exp.is_integer: return True
    return False

def extract_cell_point(cell_str: str, syms: tuple) -> dict:
    """Parses a QEPCAD cell string and extracts the floating-point coordinate sample."""
    pt = {}
    for i in range(len(syms)):
        coord_marker = f"Coordinate {i + 1} ="
        if coord_marker not in cell_str: return None

        if i < len(syms) - 1:
            next_marker = f"Coordinate {i + 2} ="
            chunk = cell_str.split(coord_marker)[1].split(next_marker)[0]
        else:
            chunk = cell_str.split(coord_marker)[1]

        dec_match = re.search(r"=\s*([+\-]?\d+\.\d+[+\-]?)", chunk)
        if dec_match:
            clean_val = dec_match.group(1).strip()
            if clean_val.endswith('+') or clean_val.endswith('-'):
                clean_val = clean_val[:-1]
            try:
                pt[syms[i]] = float(clean_val)
            except Exception:
                return None
        else:
            return None
    return pt




def call_maxima_integration(integrand_str, var_str, lower_str, upper_str, assumptions_list):
    """
    Passes the integral to Maxima, injecting the CAD cell boundaries as strict assumptions
    to prevent domain blindness and imaginary number crashes.
    """
    # 1. Build the assumption database
    maxima_code = "display2d: false$\n"  # Force flat string output

    for assume_str in assumptions_list:
        # e.g., assume(t > 0)$ assume(t < 1/2)$
        maxima_code += f"assume({assume_str})$\n"

    # 2. Ask Maxima to integrate
    maxima_code += f"result: integrate({integrand_str}, {var_str}, {lower_str}, {upper_str})$\n"
    maxima_code += "print(result)$\n"

    try:
        # 3. Fire the Subprocess
        process = subprocess.Popen(
            ['maxima', '--very-quiet'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=maxima_code, timeout=10)

        raw_output = stdout.strip()

        # Maxima sometimes returns noun forms (unevaluated) if it fails.
        if "integrate(" in raw_output:
            return None  # Maxima failed to find a closed form

        # Clean Maxima syntax for SymPy parsing
        clean_out = raw_output.replace("%pi", "pi").replace("%e", "E")

        return parse_expr(clean_out)

    except Exception as e:
        print(f"[MAXIMA ERROR] {e}")
        return None


def sympy_to_exprtk(expr):
    """
    Recursively compiles a SymPy AST into a pure C++ ExprTk execution string.
    Includes real-domain protection and dynamic Boolean resolution.
    """
    # 1. Base Cases & Safe NaN Injection
    if expr == sp.I or expr.has(sp.I):
        return "(0.0 / 0.0)"  # Safely yields NaN in C++
    elif isinstance(expr, sp.Symbol):
        return str(expr)
    elif isinstance(expr, sp.Rational):
        return f"({expr.p} / {expr.q}.0)"
    elif isinstance(expr, sp.Number):
        return str(float(expr))
    elif expr == sp.pi:
        return "pi"
    elif expr == sp.E:
        return "e"

    # 2. Standard Algebra
    elif isinstance(expr, sp.Add):
        return "(" + " + ".join(sympy_to_exprtk(arg) for arg in expr.args) + ")"
    elif isinstance(expr, sp.Mul):
        return "(" + " * ".join(sympy_to_exprtk(arg) for arg in expr.args) + ")"
    elif isinstance(expr, sp.Pow):
        base = sympy_to_exprtk(expr.base)
        exp = sympy_to_exprtk(expr.exp)
        return f"({base} ^ {exp})"

    # 3. Relational Logic
    elif isinstance(expr, Relational):
        op_map = {
            sp.StrictLessThan: "<", sp.LessThan: "<=",
            sp.StrictGreaterThan: ">", sp.GreaterThan: ">=",
            sp.Equality: "==", sp.Unequality: "!="
        }
        return f"({sympy_to_exprtk(expr.lhs)} {op_map.get(type(expr), '==')} {sympy_to_exprtk(expr.rhs)})"

    # 4. Boolean Logic Gates (ExprTk Native)
    elif isinstance(expr, sp.And):
        return "(" + " and ".join(sympy_to_exprtk(arg) for arg in expr.args) + ")"
    elif isinstance(expr, sp.Or):
        return "(" + " or ".join(sympy_to_exprtk(arg) for arg in expr.args) + ")"
    elif isinstance(expr, sp.Not):
        # Numerical inversion perfectly mirrors boolean NOT
        return f"(1.0 - ({sympy_to_exprtk(expr.args[0])}))"

    # 5. Multi-Argument Min/Max Nesting
    elif isinstance(expr, sp.Max):
        if len(expr.args) == 1: return sympy_to_exprtk(expr.args[0])
        res = sympy_to_exprtk(expr.args[0])
        for arg in expr.args[1:]:
            res = f"max({res}, {sympy_to_exprtk(arg)})"
        return res

    elif isinstance(expr, sp.Min):
        if len(expr.args) == 1: return sympy_to_exprtk(expr.args[0])
        res = sympy_to_exprtk(expr.args[0])
        for arg in expr.args[1:]:
            res = f"min({res}, {sympy_to_exprtk(arg)})"
        return res

    # 6. Piecewise Branch Resolution
    elif isinstance(expr, sp.Piecewise):
        terms = []
        accumulated_nots = []

        for branch_expr, cond in expr.args:
            branch_str = sympy_to_exprtk(branch_expr)
            if cond == sp.S.true or cond == True:
                if accumulated_nots:
                    terms.append(f"(({' and '.join(accumulated_nots)}) * ({branch_str}))")
                else:
                    terms.append(f"({branch_str})")
                break
            else:
                cond_str = sympy_to_exprtk(cond)
                if accumulated_nots:
                    # FIX: Using single quotes inside the f-string expression
                    terms.append(f"(({' and '.join(accumulated_nots)} and ({cond_str})) * ({branch_str}))")
                else:
                    terms.append(f"(({cond_str}) * ({branch_str}))")
                accumulated_nots.append(f"(1.0 - ({cond_str}))")

        return "(" + " + ".join(terms) + ")"

    # 7. Safe Function Mapping
    else:
        func_name = str(expr.func.__name__)

        # Protect against imaginary numbers leaking into the adaptive grid
        if func_name == "sqrt":
            arg_str = sympy_to_exprtk(expr.args[0])
            return f"sqrt(max(0.0, {arg_str}))"

        name_map = {"sign": "sgn", "Abs": "abs", "log": "log"}
        mapped_name = name_map.get(func_name, func_name)

        args = ", ".join(sympy_to_exprtk(arg) for arg in expr.args)
        return f"{mapped_name}({args})"