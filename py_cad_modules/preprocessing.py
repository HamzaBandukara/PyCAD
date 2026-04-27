import re
import sympy as sp
from sympy.core.relational import Relational
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

from py_cad_modules.utils import dbprint
from py_cad_modules.cpp_oracle import call_cpp_oracle
import itertools

def custom_rt(degree, inner):
    """Wraps SymPy's root function to match the rt(degree, inner) string format."""
    return sp.root(inner, degree)


def preprocess_sugar(formula: str) -> str:
    """Converts chained inequalities a < b < c into a < b && b < c."""
    # Split by logical AND / OR to isolate individual constraint blocks
    parts = re.split(r'(\&\&|\|\|)', formula)
    processed_parts = []

    for part in parts:
        if part in ['&&', '||']:
            processed_parts.append(part)
            continue

        # Isolate relational operators
        sub_parts = re.split(r'(<=|>=|<|>|==)', part)

        # If there's more than one operator (length > 3), it's a chained inequality
        # E.g., ['0 ', '<', ' ax ', '<', ' 1']
        # Parses the syntactic sugar such that it is only 1 operator
        if len(sub_parts) > 3:
            new_sub = []
            for i in range(0, len(sub_parts) - 2, 2):
                expr1 = sub_parts[i]
                op1 = sub_parts[i + 1]
                expr2 = sub_parts[i + 2]
                new_sub.append(f"{expr1}{op1}{expr2}")
            # Rejoin the broken chain with &&
            processed_parts.append(" && ".join(new_sub))
        else:
            processed_parts.append(part)

    return "".join(processed_parts)


def expand_ifs_to_list(formula: str) -> list:
    """Recursively splits If[...] statements into a flat list of independent logical branches."""
    import re
    if "If[" not in formula:
        return [formula]

    start_idx = formula.find("If[")

    # 1. Find the end of the If[...] block
    depth = 0
    if_inner_start = start_idx + 3
    if_inner_end = -1

    for i in range(if_inner_start, len(formula)):
        if formula[i] == '[':
            depth += 1
        elif formula[i] == ']':
            if depth == 0:
                if_inner_end = i
                break
            depth -= 1

    if if_inner_end == -1: return [formula]

    inner_str = formula[if_inner_start:if_inner_end]

    # 2. Split arguments respecting nesting
    args = []
    curr = ""
    b_depth = p_depth = c_depth = 0

    for char in inner_str:
        if char == '[':
            b_depth += 1
        elif char == ']':
            b_depth -= 1
        elif char == '(':
            p_depth += 1
        elif char == ')':
            p_depth -= 1
        elif char == '{':
            c_depth += 1
        elif char == '}':
            c_depth -= 1
        elif char == ',' and b_depth == 0 and p_depth == 0 and c_depth == 0:
            args.append(curr.strip())
            curr = ""
            continue
        curr += char
    args.append(curr.strip())

    if len(args) != 3: return [formula]

    c, e1, e2 = args

    # 3. Extract the Operator and RHS
    remainder = formula[if_inner_end + 1:].lstrip()
    op_match = re.match(r'^(<=|>=|<|>|==)', remainder)
    if not op_match: return [formula]

    op = op_match.group(1)
    remainder = remainder[len(op):].lstrip()

    # Find where the RHS ends (usually at the next && or ||)
    rhs_match = re.search(r'(\&\&|\|\||\])', remainder)
    if rhs_match:
        rhs_end = rhs_match.start()
        rhs = remainder[:rhs_end].strip()
        suffix = remainder[rhs_end:]
    else:
        rhs = remainder.strip()
        suffix = ""

    prefix = formula[:start_idx]

    # 4. Construct the two independent strings
    branch_true = f"{prefix} ( {c} ) && ( {e1} {op} {rhs} ) {suffix}"

    # Safely invert the condition for the False branch
    neg_c = f"~( {c} )"
    if ">=" in c:
        neg_c = c.replace(">=", "<")
    elif "<=" in c:
        neg_c = c.replace("<=", ">")
    elif ">" in c:
        neg_c = c.replace(">", "<=")
    elif "<" in c:
        neg_c = c.replace("<", ">=")

    branch_false = f"{prefix} ( {neg_c} ) && ( {e2} {op} {rhs} ) {suffix}"

    # 5. Recurse to catch any deeply nested Ifs!
    return expand_ifs_to_list(branch_true) + expand_ifs_to_list(branch_false)


def extract_base_support(formula_sugar_free: str, syms: tuple, local_syms: dict) -> dict:
    """Extracts constant boundary support limits (e.g., 0 < x < 1)."""
    import re
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

    base_bounds = {str(s): {"lower": -sp.oo, "upper": sp.oo} for s in syms}

    if 'custom_rt' in globals():
        local_syms['rt'] = custom_rt
    local_syms['pi'] = sp.pi

    # Strip Master Z3 Brackets
    s = formula_sugar_free.strip()
    while s.startswith('[') and s.endswith(']'):
        depth, is_master = 0, True
        for i, char in enumerate(s):
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
            if depth == 0 and i < len(s) - 1:
                is_master = False
                break
        if is_master:
            s = s[1:-1].strip()
        else:
            break

    # Split safely by top-level &&
    top_level_conds = []
    depth = 0
    current_chunk = ""
    i = 0
    while i < len(s):
        char = s[i]
        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1

        if depth == 0 and s[i:i + 2] == '&&':
            top_level_conds.append(current_chunk.strip())
            current_chunk = ""
            i += 2
            continue

        current_chunk += char
        i += 1
    if current_chunk: top_level_conds.append(current_chunk.strip())

    # Evaluate each top-level condition
    for cond in top_level_conds:
        cond = cond.strip()
        if cond.startswith('[') and cond.endswith(']'):
            cond = cond[1:-1].strip()

        if not cond or "||" in cond or "&&" in cond:
            continue

        match = re.search(r'(<=|>=|<|>|==)', cond)
        if not match: continue
        op = match.group(1)

        try:
            lhs_str, rhs_str = cond.split(op, 1)

            transformations = standard_transformations + (implicit_multiplication_application,)
            lhs = parse_expr(lhs_str.replace("^", "**"), local_dict=local_syms, transformations=transformations)
            rhs = parse_expr(rhs_str.replace("^", "**"), local_dict=local_syms, transformations=transformations)

            if lhs.has(sp.pi): lhs = lhs.subs(sp.pi, sp.Rational(314159265, 100000000))
            if rhs.has(sp.pi): rhs = rhs.subs(sp.pi, sp.Rational(314159265, 100000000))

            for f in lhs.atoms(sp.Float): lhs = lhs.subs(f, sp.Rational(str(f)))
            for f in rhs.atoms(sp.Float): rhs = rhs.subs(f, sp.Rational(str(f)))

            if lhs.is_Symbol and rhs.is_number:
                var = str(lhs)
                if op in ['<', '<=']: base_bounds[var]["upper"] = sp.Min(base_bounds[var]["upper"], rhs)
                if op in ['>', '>=']: base_bounds[var]["lower"] = sp.Max(base_bounds[var]["lower"], rhs)
            elif rhs.is_Symbol and lhs.is_number:
                var = str(rhs)
                if op in ['<', '<=']: base_bounds[var]["lower"] = sp.Max(base_bounds[var]["lower"], lhs)
                if op in ['>', '>=']: base_bounds[var]["upper"] = sp.Min(base_bounds[var]["upper"], lhs)
        except Exception:
            continue

    return base_bounds

def prune_positive_factors(expr):
    """Algorithms to strip positive weights and SCALAR MULTIPLIERS to save QEPCAD memory."""
    if expr.is_number: return expr
    try:
        # 1. Strip all integer scalar multipliers (e.g., 16x - 32 -> x - 2)
        _, expr = expr.as_content_primitive()

        factored = sp.sqrfree(expr)
        if factored.is_Mul:
            clean_expr = sp.S.One
            sign = 1
            for arg in factored.args:
                if arg.is_Pow and arg.exp % 2 == 0:
                    continue
                elif arg.is_number:
                    if arg.is_real and arg < 0: sign *= -1
                else:
                    clean_expr *= arg
            expr = sp.expand(clean_expr * sign)
        _, expr = expr.as_content_primitive()
        return sp.expand(expr)
    except:
        pass
    return sp.expand(expr)


def auto_split_distribution_modes(branches: list, distributions: dict) -> list:
    """
    Automatically shatters formulas into mutually exclusive branches at the mode
    of any triangular distribution, guaranteeing smooth piecewise integration.
    """
    if not distributions:
        return branches

    split_conditions = []
    for var, dist_info in distributions.items():
        if dist_info[0].lower() == 'triangular':
            mode = dist_info[3]
            # Convert float (1.5) to exact fraction string ("3/2") for CAD safety
            mode_rat = sp.Rational(str(mode))
            mode_str = f"{mode_rat.p}/{mode_rat.q}" if mode_rat.q != 1 else str(mode_rat.p)

            # Create the two mutually exclusive realities for this variable
            split_conditions.append([f"{var} < {mode_str}", f"{var} > {mode_str}"])

    if not split_conditions:
        return branches

    # Generate the combinatorial matrix (3 variables = 8 unique realities)
    combinations = list(itertools.product(*split_conditions))

    shattered_branches = []
    for branch in branches:
        for combo in combinations:
            combo_str = " && ".join(combo)
            shattered_branches.append(f"{branch} && {combo_str}")

    return shattered_branches

def ast_normalize_inequalities(formula_sugar_free: str, vars_list: list) -> tuple:
    syms = list(sp.symbols(" ".join(vars_list), real=True))
    local_syms = {str(s): s for s in syms}
    if 'custom_rt' in globals(): local_syms['rt'] = custom_rt
    local_syms['pi'] = sp.pi

    # Inherently wrap all operations in QEPCAD-safe protective brackets.
    def AND(a, b):
        a, b = a.strip(), b.strip()
        if a == "[ 0 < 0 ]" or b == "[ 0 < 0 ]": return "[ 0 < 0 ]"
        if a == "[ 0 == 0 ]": return b
        if b == "[ 0 == 0 ]": return a
        return f"[ {a} && {b} ]"

    def OR(a, b):
        a, b = a.strip(), b.strip()
        if a == "[ 0 == 0 ]" or b == "[ 0 == 0 ]": return "[ 0 == 0 ]"
        if a == "[ 0 < 0 ]": return b
        if b == "[ 0 < 0 ]": return a
        return f"[ {a} || {b} ]"

    def cond(v_str, cop):
        """Evaluates constants instantly to collapse the tree, or formats securely."""
        try:
            v = float(sp.sympify(v_str.replace('^', '**')))
            if cop == '<': return "[ 0 == 0 ]" if v < 0 else "[ 0 < 0 ]"
            if cop == '<=': return "[ 0 == 0 ]" if v <= 0 else "[ 0 < 0 ]"
            if cop == '>': return "[ 0 == 0 ]" if v > 0 else "[ 0 < 0 ]"
            if cop == '>=': return "[ 0 == 0 ]" if v >= 0 else "[ 0 < 0 ]"
            if cop == '==': return "[ 0 == 0 ]" if v == 0 else "[ 0 < 0 ]"
        except:
            pass
        return f"[ {v_str} {cop} 0 ]"

    while True:
        parts = re.split(r'(\&\&|\|\||\[|\]|\~)', formula_sugar_free)
        changed = False
        new_parts = []

        for part in parts:
            part_stripped = part.strip()
            if part_stripped in ['&&', '||', '[', ']', '~', '']:
                new_parts.append(part)
                continue

            match = re.search(r'(<=|>=|<|>|==)', part_stripped)
            if match:
                op = match.group(1)
                lhs_str, rhs_str = part_stripped.split(op, 1)

                transformations = standard_transformations + (implicit_multiplication_application,)
                lhs = parse_expr(lhs_str.replace("^", "**"), local_dict=local_syms, transformations=transformations)
                rhs = parse_expr(rhs_str.replace("^", "**"), local_dict=local_syms, transformations=transformations)

                raw_expr = lhs - rhs

                if raw_expr.has(sp.pi): raw_expr = raw_expr.subs(sp.pi, sp.Rational(314159265, 100000000))
                for f in raw_expr.atoms(sp.Float): raw_expr = raw_expr.subs(f, sp.Rational(str(f)))

                for p in raw_expr.atoms(sp.Pow):
                    if p.exp.is_real and p.exp < 0:
                        raw_expr = raw_expr.subs(p, 1 / (p.base ** (-p.exp)))

                expr_together = sp.together(raw_expr)
                numer, denom = expr_together.as_numer_denom()

                if denom != 1 and denom != -1:
                    rads_in_denom = [p for p in denom.atoms(sp.Pow) if not p.exp.is_integer]
                    if rads_in_denom or denom.is_nonnegative:
                        numer = prune_positive_factors(numer)
                        numer_str = str(numer).replace('**', '^')
                        new_parts.append(f" {cond(numer_str, op)} ")
                    else:
                        numer = prune_positive_factors(numer)
                        denom = prune_positive_factors(denom)
                        n_str = str(numer).replace('**', '^')
                        d_str = str(denom).replace('**', '^')
                        op_flip = {'<': '>', '<=': '>=', '>': '<', '>=': '<=', '==': '=='}[op]

                        # Securely wrapped Denominator Split
                        left_branch = AND(cond(n_str, op), cond(d_str, '>'))
                        right_branch = AND(cond(n_str, op_flip), cond(d_str, '<'))
                        new_parts.append(f" {OR(left_branch, right_branch)} ")
                    changed = True
                    continue

                numer_str = str(numer).replace('**', '^')
                oracle_result = call_cpp_oracle(numer_str)

                if oracle_result.get("TYPE") == "ROOT":
                    b_expr = parse_expr(oracle_result.get("B", "0").replace("^", "**"), local_dict=local_syms)
                    c_expr = parse_expr(oracle_result.get("C", "0").replace("^", "**"), local_dict=local_syms)
                    d_expr = parse_expr(oracle_result.get("D", "0").replace("^", "**"), local_dict=local_syms)
                    bc2_expr = parse_expr(oracle_result.get("BC2", "0").replace("^", "**"), local_dict=local_syms)
                    d2_expr = parse_expr(oracle_result.get("D2", "0").replace("^", "**"), local_dict=local_syms)

                    surface_expr = prune_positive_factors(bc2_expr - d2_expr)
                    c_expr = prune_positive_factors(c_expr)
                    d_expr = prune_positive_factors(d_expr)
                    b_expr = prune_positive_factors(b_expr)

                    b_str = str(b_expr).replace('**', '^')
                    c_str = str(c_expr).replace('**', '^')
                    d_str = str(d_expr).replace('**', '^')
                    s_str = str(surface_expr).replace('**', '^')

                    # Isolate components to evaluate constants immediately
                    d_lt, d_le = cond(d_str, '<'), cond(d_str, '<=')
                    d_gt, d_ge, d_eq = cond(d_str, '>'), cond(d_str, '>='), cond(d_str, '==')
                    s_lt, s_le = cond(s_str, '<'), cond(s_str, '<=')
                    s_gt, s_ge, s_eq = cond(s_str, '>'), cond(s_str, '>='), cond(s_str, '==')

                    # Build securely wrapped collapsed sub-trees
                    if op == '<':
                        pos_logic = AND(d_lt, s_lt)
                        neg_logic = OR(d_lt, AND(d_ge, s_gt))
                        zero_logic = d_lt
                    elif op == '<=':
                        pos_logic = AND(d_le, s_le)
                        neg_logic = OR(d_le, AND(d_gt, s_ge))
                        zero_logic = d_le
                    elif op == '>':
                        pos_logic = OR(d_gt, AND(d_le, s_gt))
                        neg_logic = AND(d_gt, s_lt)
                        zero_logic = d_gt
                    elif op == '>=':
                        pos_logic = OR(d_ge, AND(d_lt, s_ge))
                        neg_logic = AND(d_ge, s_le)
                        zero_logic = d_ge
                    elif op == '==':
                        pos_logic = AND(d_le, s_eq)
                        neg_logic = AND(d_ge, s_eq)
                        zero_logic = d_eq

                    if c_expr.is_number:
                        c_val = float(c_expr)
                        if c_val > 0:
                            logic_str = pos_logic
                        elif c_val < 0:
                            logic_str = neg_logic
                        else:
                            logic_str = zero_logic
                    else:
                        t1 = AND(cond(c_str, '>'), pos_logic)
                        t2 = AND(cond(c_str, '<'), neg_logic)
                        t3 = AND(cond(c_str, '=='), zero_logic)
                        logic_str = OR(t1, OR(t2, t3))

                    final_logic = AND(cond(b_str, '>='), logic_str)
                    new_parts.append(f" {final_logic} ")
                    changed = True

                elif oracle_result.get("TYPE") == "POLY":
                    expr_str = oracle_result.get("EXPANDED", "0")
                    if expr_str in ['t', '-t']:
                        # Securely wrap the fallback string
                        new_parts.append(f" [ {lhs_str} {op} {rhs_str} ] ")
                    else:
                        expr_obj = parse_expr(expr_str.replace("^", "**"), local_dict=local_syms)
                        expr_obj = prune_positive_factors(expr_obj)
                        new_parts.append(f" {cond(str(expr_obj).replace('**', '^'), op)} ")
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)

        formula_sugar_free = "".join(new_parts)

        # Bracket flattener
        while True:
            old_formula = formula_sugar_free
            # Matches [[X]] ONLY if X contains no other brackets, safely avoiding logic jumps
            formula_sugar_free = re.sub(r'\[\s*\[([^\[\]]+)\]\s*\]', r'[\1]', formula_sugar_free)
            if old_formula == formula_sugar_free: break

        if not changed: break

    return formula_sugar_free, vars_list


# For better variable ordering (using Brown's heurstic)
def optimize_variable_order(formula: str, vars_list: list) -> list:
    """
    Implements Brown's Heuristic to dynamically sort variables for QEPCAD.
    Returns a reordered vars_list where the most complex variables are kept
    at the front (outermost), and the simplest at the back (eliminated first).
    The first variable (usually 't') is locked at index 0.
    """
    if len(vars_list) <= 2:
        return vars_list  # Nothing to optimize if it's just 2D

    target_var = vars_list[0]
    search_vars = [sp.Symbol(v, real=True) for v in vars_list[1:]]

    local_syms = {v: sp.Symbol(v, real=True) for v in vars_list}
    local_syms['rt'] = sp.root
    local_syms['pi'] = sp.pi
    local_syms['rt'] = custom_rt

    # 1. Extract all raw polynomials from the formula
    exprs = []
    parts = re.split(r'(\&\&|\|\||\[|\]|\~)', formula)
    for part in parts:
        part = part.strip()
        match = re.search(r'(<=|>=|<|>|==)', part)
        if match:
            op = match.group(1)
            lhs_str, rhs_str = part.split(op)
            try:
                transformations = standard_transformations + (implicit_multiplication_application,)
                lhs = parse_expr(lhs_str.replace("^", "**"), local_dict=local_syms, transformations=transformations)
                rhs = parse_expr(rhs_str.replace("^", "**"), local_dict=local_syms, transformations=transformations)
                expr = sp.cancel(lhs - rhs)

                # Clear denominators to expose the true polynomial
                numer, denom = expr.as_numer_denom()
                if denom != 1 and denom != -1:
                    expr = numer * denom
                exprs.append(sp.expand(expr))
            except Exception:
                pass

    # 2. Score each variable using Brown's Criteria
    # Structure: { var_name: [max_deg, max_total_deg, term_count] }
    scores = {str(v): [0, 0, 0] for v in search_vars}

    for expr in exprs:
        if not hasattr(expr, 'args'): continue
        # Break polynomial into individual terms
        terms = expr.args if expr.func == sp.Add else [expr]

        for v in search_vars:
            v_str = str(v)
            if not expr.has(v): continue

            # Criterion 1: Max degree of v in the polynomial
            try:
                deg = sp.degree(expr, v)
                scores[v_str][0] = max(scores[v_str][0], int(deg))
            except Exception:
                pass

            # Criterion 2 & 3: Total Degree & Term Count
            for term in terms:
                if term.has(v):
                    scores[v_str][2] += 1
                    try:
                        # Sum of degrees of all variables in this specific term
                        tot_deg = sum(int(sp.degree(term, sv)) for sv in search_vars if term.has(sv))
                        scores[v_str][1] = max(scores[v_str][1], tot_deg)
                    except Exception:
                        pass

    # 3. Sort variables based on scores (Descending: Highest score goes first)
    # Most complex vars go to the front (eliminated LAST).
    # Simplest vars go to the back (eliminated FIRST).
    sorted_free_vars = sorted(
        [str(v) for v in search_vars],
        key=lambda v: (scores[v][0], scores[v][1], scores[v][2]),
        reverse=True
    )

    optimized_list = [target_var] + sorted_free_vars

    dbprint(f"\n[HEURISTIC ENGINE] Brown's Criteria applied. Optimal Order: {optimized_list}")
    for v in sorted_free_vars:
        dbprint(f"      {v} -> Max Degree: {scores[v][0]} | Total Degree: {scores[v][1]} | Terms: {scores[v][2]}")

    return optimized_list


# Pre-Reductions
def apply_groebner_reduction(formula: str, vars_list: list) -> tuple:
    """
    Pillar 2: Gröbner Basis Dimensionality Reduction.
    Finds equational constraints (==), computes their Gröbner basis,
    and substitutes linear variables out of the system entirely to drop dimensions.
    """
    if "==" not in formula:
        return formula, vars_list


    local_syms = {v: sp.Symbol(v, real=True) for v in vars_list}
    local_syms['rt'] = sp.root
    local_syms['pi'] = sp.pi
    transformations = standard_transformations + (implicit_multiplication_application,)

    # 1. Extract equalities
    parts = re.split(r'(\&\&|\|\||\[|\]|\~)', formula)
    eq_exprs = []

    for part in parts:
        if "==" in part:
            try:
                lhs_str, rhs_str = part.split("==")
                lhs = parse_expr(lhs_str.replace("^", "**"), local_dict=local_syms, transformations=transformations)
                rhs = parse_expr(rhs_str.replace("^", "**"), local_dict=local_syms, transformations=transformations)
                eq_exprs.append(lhs - rhs)
            except Exception:
                pass

    if not eq_exprs:
        return formula, vars_list

    # 2. Compute Gröbner basis (excluding the target variable 't')
    free_syms = [sp.Symbol(v, real=True) for v in vars_list if v != vars_list[0]]
    try:
        basis = sp.groebner(eq_exprs, *free_syms).polys
    except Exception:
        return formula, vars_list

    subs_dict = {}
    eliminated_vars = []

    # 3. Find linear relationships to drop dimensions
    for b in basis:
        expr = b.as_expr()
        for v in free_syms:
            if v not in eliminated_vars and sp.degree(expr, v) == 1:
                sol = sp.solve(expr, v)
                if sol:
                    subs_dict[v] = sol[0]
                    eliminated_vars.append(v)
                    break

    if not subs_dict:
        return formula, vars_list

    dbprint(f"\n[HEURISTIC ENGINE] Gröbner Basis Dimensionality Reduction:")
    for v, eq in subs_dict.items():
        dbprint(f"      Eliminating {v}: Substituted with ({eq})")

    # 4. String substitution using word boundaries to protect substrings
    new_formula = formula
    for v, sub_expr in subs_dict.items():
        sub_str = str(sub_expr).replace("**", "^")
        new_formula = re.sub(rf'\b{str(v)}\b', f"({sub_str})", new_formula)

    # 5. Drop the eliminated dimensions from the CAD payload entirely!
    new_vars_list = [v for v in vars_list if sp.Symbol(v, real=True) not in eliminated_vars]

    return new_formula, new_vars_list


# Experimental function, not used. Too "hacky"
def apply_cylindrical_override(formula_str: str, vars_list: list) -> tuple:
    """
    Detects rotational symmetry in the XY plane.
    If mathematically sound, collapses 4D (ax, ay) geometry into 3D (r) geometry.
    """
    if 'ax' not in vars_list or 'ay' not in vars_list:
        return formula_str, vars_list



    syms = {v: sp.Symbol(v, real=True) for v in vars_list}
    ax, ay = syms['ax'], syms['ay']
    r = sp.Symbol('r', nonnegative=True, real=True)
    transformations = standard_transformations + (implicit_multiplication_application,)

    parts = re.split(r'(\&\&|\|\||\[|\]|\~)', formula_str)
    new_parts = []
    is_symmetric = True
    actually_collapsed = False

    for part in parts:
        part_stripped = part.strip()
        if part_stripped in ['&&', '||', '[', ']', '~', 'TRUE', 'FALSE', '']:
            new_parts.append(part)
            continue

        match = re.search(r'(<=|>=|<|>|==)', part_stripped)
        if match:
            op = match.group(1)
            lhs_str, rhs_str = part_stripped.split(op, 1)

            # Skip linear bounding boxes for the symmetry check
            if lhs_str.strip() in ['ax', 'ay', '-ax', '-ay'] or rhs_str.strip() in ['ax', 'ay', '-ax', '-ay']:
                new_parts.append("TRUE")  # Temp placeholder, stripped later
                continue

            try:
                lhs = parse_expr(lhs_str.replace("^", "**"), local_dict=syms, transformations=transformations)
                rhs = parse_expr(rhs_str.replace("^", "**"), local_dict=syms, transformations=transformations)
                expr = sp.expand(lhs - rhs)

                if expr.has(ax) or expr.has(ay):
                    # If it has odd powers of ax or ay, it cannot be symmetric
                    if not expr.is_polynomial(ax) or not expr.is_polynomial(ay):
                        is_symmetric = False
                        break

                    # Substitute ax^2 -> r^2 - ay^2 and expand!
                    test_expr = sp.expand(expr.subs(ax ** 2, r ** 2 - ay ** 2))

                    if test_expr.has(ay) or test_expr.has(ax):
                        is_symmetric = False
                        break

                    new_parts.append(f" {str(test_expr).replace('**', '^')} {op} 0 ")
                    actually_collapsed = True
                else:
                    new_parts.append(part)
            except Exception:
                new_parts.append(part)
        else:
            new_parts.append(part)

    if is_symmetric and actually_collapsed:
        dbprint("\n[PHYSICS OVERRIDE] Cylindrical Symmetry detected! Collapsing 4D space to 3D...")

        # Build the new 3D formula
        new_formula = "".join(new_parts)
        # Strip the temporary TRUE placeholders
        new_formula = re.sub(r'\[\s*TRUE\s*\]', '', new_formula)
        new_formula = new_formula.replace("&&  &&", "&&").replace("||  ||", "||")

        # Inject the new radial bounds (0 <= r <= 1)
        new_formula += " && [ r >= 0 ] && [ 1 - r >= 0 ] "

        # Update vars_list: Drop ax/ay, insert r
        new_vars = [v for v in vars_list if v not in ['ax', 'ay']]
        new_vars.append('r')

        return new_formula, new_vars

    return formula_str, vars_list


def parse_mathematica_string(formula_str: str, local_dict: dict):
    """
    Front-Door: Converts Mathematica '&&' and '0 < x < 1' into a pristine SymPy AST.
    Includes failsafes for dangling parentheses and Mathematica caret exponents.
    """
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    # 1. Expand chained inequalities via your preprocessor
    expanded_str = preprocess_sugar(formula_str)

    # 2. Split by the Mathematica '&&' operator
    conditions = [c.strip() for c in expanded_str.split('&&') if c.strip()]
    parsed_conds = []

    for cond in conditions:
        # Safely normalize Mathematica exponents
        cond = cond.replace('^', '**')

        while cond.count('(') > cond.count(')') and cond.startswith('('):
            cond = cond[1:].strip()

        while cond.count(')') > cond.count('(') and cond.endswith(')'):
            cond = cond[:-1].strip()

        # Safely parse the isolated, balanced chunk
        parsed_conds.append(parse_expr(cond, local_dict=local_dict, evaluate=False))

    # Wrap them all in a SymPy And() gate
    return sp.And(*parsed_conds)


def ast_to_cad_string(expr):
    """Back-Door: Converts a SymPy AST into the exact '[ Eq < 0 ] &&' string the CAD engine expects."""
    import sympy as sp
    if isinstance(expr, sp.And):
        return " && ".join(ast_to_cad_string(arg) for arg in expr.args)
    elif isinstance(expr, sp.Or):
        return " || ".join(ast_to_cad_string(arg) for arg in expr.args)
    elif isinstance(expr, Relational):
        # 1. Isolate the mathematical expression
        raw_expr = expr.lhs - expr.rhs

        # QEPCAD crashes if it sees 'z - 3/2 < 0'. It requires '2*z - 3 < 0'.
        # sp.cancel combines it into (2*z - 3)/2.
        frac_expr = sp.cancel(raw_expr)
        numer, denom = frac_expr.as_numer_denom()

        # If the denominator is a negative number, we must flip the sign of the numerator.
        if denom.is_number and denom < 0:
            numer = -numer
            denom = -denom

        # If the denominator is a NON-CONSTANT polynomial (e.g., x+y),
        # we must create a denominator split to preserve the sign information.
        # This generates: (numer OP 0 AND denom > 0) OR (numer FLIP 0 AND denom < 0)
        if not denom.is_number and denom != sp.S.One:
            n_str = str(numer).replace("**", "^")
            d_str = str(denom).replace("**", "^")
            op = expr.rel_op
            op_flip = {'<': '>', '<=': '>=', '>': '<', '>=': '<=', '==': '=='}[op]
            left = f"[ [ {n_str} {op} 0 ] && [ {d_str} > 0 ] ]"
            right = f"[ [ {n_str} {op_flip} 0 ] && [ {d_str} < 0 ] ]"
            return f"[ {left} || {right} ]"

        math_str = str(numer).replace("**", "^")


        return f"[ {math_str} {expr.rel_op} 0 ]"
    else:
        return f"[ {str(expr).replace('**', '^')} < 0 ]"