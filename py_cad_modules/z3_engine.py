import z3
import sympy as sp
import re
from .preprocessing import preprocess_sugar, ast_normalize_inequalities
from .utils import apiprint


def z3_to_infix(expr, fmt="mathematica"):
    """Recursively formats Z3 AST into Mathematica-style strings for QEPCAD."""
    if z3.is_and(expr):
        return "[ " + " && ".join(z3_to_infix(arg, fmt) for arg in expr.children()) + " ]"
    elif z3.is_or(expr):
        return "[ " + " || ".join(z3_to_infix(arg, fmt) for arg in expr.children()) + " ]"
    elif z3.is_not(expr):
        child = expr.children()[0]
        if z3.is_le(child): return f"[ {z3_to_infix(child.children()[0], fmt)} > {z3_to_infix(child.children()[1], fmt)} ]"
        if z3.is_ge(child): return f"[ {z3_to_infix(child.children()[0], fmt)} < {z3_to_infix(child.children()[1], fmt)} ]"
        if z3.is_lt(child): return f"[ {z3_to_infix(child.children()[0], fmt)} >= {z3_to_infix(child.children()[1], fmt)} ]"
        if z3.is_gt(child): return f"[ {z3_to_infix(child.children()[0], fmt)} <= {z3_to_infix(child.children()[1], fmt)} ]"
        if z3.is_eq(child):
            # Translate Negated Equality to QEPCAD-safe bounds
            a = z3_to_infix(child.children()[0], fmt)
            b = z3_to_infix(child.children()[1], fmt)
            return f"[ [ {a} < {b} ] || [ {a} > {b} ] ]"
        return "[ 0 < 0 ]" # Failsafe False
    elif z3.is_le(expr):
        return f"[ {z3_to_infix(expr.children()[0], fmt)} <= {z3_to_infix(expr.children()[1], fmt)} ]"
    elif z3.is_ge(expr):
        return f"[ {z3_to_infix(expr.children()[0], fmt)} >= {z3_to_infix(expr.children()[1], fmt)} ]"
    elif z3.is_lt(expr):
        return f"[ {z3_to_infix(expr.children()[0], fmt)} < {z3_to_infix(expr.children()[1], fmt)} ]"
    elif z3.is_gt(expr):
        return f"[ {z3_to_infix(expr.children()[0], fmt)} > {z3_to_infix(expr.children()[1], fmt)} ]"
    elif z3.is_eq(expr):
        if z3.is_bool(expr.children()[0]):
            a = z3_to_infix(expr.children()[0], fmt)
            b = z3_to_infix(expr.children()[1], fmt)
            return f"[ [ {a} && {b} ] || [ [ !{a} ] && [ !{b} ] ] ]"
        return f"[ {z3_to_infix(expr.children()[0], fmt)} == {z3_to_infix(expr.children()[1], fmt)} ]"
    elif z3.is_distinct(expr):
        # Translate Distinct (XOR / !=) to QEPCAD-safe bounds
        if z3.is_bool(expr.children()[0]):
            a = z3_to_infix(expr.children()[0], fmt)
            b = z3_to_infix(expr.children()[1], fmt)
            return f"[ [ {a} && [ !{b} ] ] || [ [ !{a} ] && {b} ] ]"
        a = z3_to_infix(expr.children()[0], fmt)
        b = z3_to_infix(expr.children()[1], fmt)
        return f"[ [ {a} < {b} ] || [ {a} > {b} ] ]"
    else:
        clean_str = str(expr).replace(" ", "").replace("\n", "")
        if fmt == "mathematica":
            clean_str = clean_str.replace("**", "^").replace("+-", "-").replace("-+", "-")
        return clean_str


def sympy_to_z3(expr, z3_vars):
    """Recursively converts a SymPy AST into a Z3 SMT AST."""
    if expr.is_Symbol: return z3_vars[str(expr)]
    if expr.is_Number: return z3.RealVal(str(expr))
    if expr.is_Add: return z3.Sum([sympy_to_z3(arg, z3_vars) for arg in expr.args])
    if expr.is_Mul:
        res = 1
        for arg in expr.args: res *= sympy_to_z3(arg, z3_vars)
        return res
    if expr.is_Pow: return sympy_to_z3(expr.args[0], z3_vars) ** sympy_to_z3(expr.args[1], z3_vars)
    raise ValueError(f"Unsupported SymPy expression: {type(expr)}")


def z3_minimize_for_qepcad(formula: str, vars_list: list) -> str:
    formula = preprocess_sugar(formula)
    try:
        norm_formula, _ = ast_normalize_inequalities(formula, vars_list.copy())
    except Exception:
        norm_formula = formula

    z3_vars = {v: z3.Real(v) for v in vars_list}
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    transformations = standard_transformations + (implicit_multiplication_application,)
    local_syms = {v: sp.Symbol(v, real=True) for v in vars_list}

    cond_pattern = re.compile(r'([^\[\]\&\&\|\|\~]+)\s*(<=|>=|<|>|==)\s*([^\[\]\&\&\|\|\~]+)')
    z3_conds_dict = {}
    counter = 0

    def replacer(match):
        nonlocal counter
        lhs_str = match.group(1).strip()
        op = match.group(2).strip()
        rhs_str = match.group(3).strip()

        # Safely trap known boolean strings before SymPy evaluates them
        if "True" in lhs_str or "False" in lhs_str or "0" == lhs_str:
            if lhs_str == "0" and rhs_str == "0":
                if op == "==" or op in ["<=", ">="]: return "Z3_TRUE"
                if op in ["<", ">"]: return "Z3_FALSE"
            if lhs_str == "0 < 0": return "Z3_FALSE"
            if lhs_str == "0 == 0": return "Z3_TRUE"

        try:
            lhs_sp = parse_expr(lhs_str.replace("^", "**"), local_dict=local_syms, transformations=transformations)
            rhs_sp = parse_expr(rhs_str.replace("^", "**"), local_dict=local_syms, transformations=transformations)

            lhs_z3 = sympy_to_z3(lhs_sp, z3_vars)
            rhs_z3 = sympy_to_z3(rhs_sp, z3_vars)

            z3_c = None
            if op == '<':
                z3_c = (lhs_z3 < rhs_z3)
            elif op == '<=':
                z3_c = (lhs_z3 <= rhs_z3)
            elif op == '>':
                z3_c = (lhs_z3 > rhs_z3)
            elif op == '>=':
                z3_c = (lhs_z3 >= rhs_z3)
            elif op == '==':
                z3_c = (lhs_z3 == rhs_z3)

            var_name = f"C_{counter}"
            z3_conds_dict[var_name] = z3_c
            counter += 1
            return var_name
        except Exception:
            return match.group(0)

    logic_str = cond_pattern.sub(replacer, norm_formula)
    logic_str = logic_str.replace('[', '(').replace(']', ')')
    logic_str = logic_str.replace('&&', '&').replace('||', '|')
    logic_str = logic_str.replace('~', '~')

    # Map Z3 Bools securely
    z3_conds_dict['Z3_TRUE'] = z3.BoolVal(True)
    z3_conds_dict['Z3_FALSE'] = z3.BoolVal(False)
    z3_conds_dict['True'] = z3.BoolVal(True)
    z3_conds_dict['False'] = z3.BoolVal(False)

    try:
        z3_ast = eval(logic_str, {"__builtins__": None}, z3_conds_dict)
    except Exception as e:
        apiprint(f"[Z3 WARNING] Eval Failed: {e}. Falling back.")
        return norm_formula

    goal = z3.Goal()
    goal.add(z3_ast)

    # 1. Simplify logic
    # 2. Push all NOTs to the leaves (removes ~ from the string)
    # 3. Context solver simplification
    tactic = z3.Then(
        z3.Tactic('simplify'),
        z3.Tactic('nnf'),
        z3.TryFor(z3.Tactic('ctx-solver-simplify'), 2000)
    )

    apiprint("\n[Z3 ENGINE] Firing Context-Solver Simplification...")
    import time
    start = time.time()
    result = tactic(goal)
    end = time.time()

    minimized_ast = result[0].as_expr()

    if z3.is_false(minimized_ast): return "[ 0 < 0 ]"
    if z3.is_true(minimized_ast): return "[ 0 == 0 ]"

    minimized_string = z3_to_infix(minimized_ast, fmt="mathematica")

    # Final sweep to ensure no ! or ~ remain to anger QEPCAD
    minimized_string = minimized_string.replace('![', '[ 0 < 0 ] || [')

    apiprint(f"[Z3 ENGINE] Minimized in {(end - start):.4f} seconds!")
    if "!" in minimized_string:
        apiprint("[Z3 WARNING] Skolem variable injected. Rejecting compression to preserve CAD integrity.")
        return formula  # Return the original, unpolluted formula

    return minimized_string