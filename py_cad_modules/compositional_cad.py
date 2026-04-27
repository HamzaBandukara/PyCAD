import sympy as sp
from typing import List, Dict, Any

from sympy.core.relational import Relational


def find_disjoint_subtrees(expr: sp.Expr, system_vars: List[sp.Symbol]) -> Dict[sp.Expr, Dict[str, Any]]:
    """
    Analyzes an expression to find additive/multiplicative subtrees.
    Recursively extracts pure math, incinerating all Boolean/Piecewise wrappers
    to prevent SymPy core crashes during substitution.
    """
    if len(system_vars) <= 3:
        # If the expression contains division or powers, skip Leibniz.
        # C++ 3D native engine can do it.
        if expr.has(sp.Pow) or any(
                isinstance(n, sp.Mul) and any(sp.simplify(arg).is_negative for arg in n.args if isinstance(arg, sp.Pow))
                for n in expr.atoms(sp.Mul)):
            return {}
    spatial_vars = [v for v in system_vars if str(v) != 't']
    polys_to_check = []

    # 1. Extracts pure algebra out of any logic container
    def extract_math(node):
        if isinstance(node, Relational):
            # Only subtract if both sides are pure math expressions
            if isinstance(node.lhs, sp.Expr) and isinstance(node.rhs, sp.Expr):
                polys_to_check.append(node.lhs - node.rhs)
        elif isinstance(node, sp.Piecewise):
            for branch, cond in node.args:
                extract_math(branch)
        elif isinstance(node, (sp.And, sp.Or, sp.Not)):
            for arg in node.args:
                extract_math(arg)
        elif isinstance(node, sp.Expr):
            # If it's an expression with no hidden logic gates
            if not node.has(sp.Piecewise, Relational, sp.And, sp.Or, sp.Not):
                polys_to_check.append(node)
            else:
                # If it has logic hidden inside (e.g., a function), dig deeper
                for arg in node.args:
                    extract_math(arg)
    # Execute extraction
    extract_math(expr)

    # Clean up zeros and duplicates
    polys_to_check = list(set([p for p in polys_to_check if p != sp.S.Zero]))

    # 2. Base expressions (raw and factored pure algebra)
    base_exprs = []
    for p in polys_to_check:
        base_exprs.append(p)
        try:
            base_exprs.append(sp.factor(p))
        except Exception:
            pass

    # 3. Add variable collections
    exprs_to_test = list(base_exprs)
    for p in base_exprs:
        for v in system_vars + [sp.Symbol('t', real=True)]:
            try:
                exprs_to_test.append(sp.collect(sp.expand(p), v))
            except Exception:
                pass

    disjoint_subtrees = {}
    system_vars_set = set(system_vars)
    dummy = sp.Symbol('DUMMY_VAR')

    # 4. Analyze the pure math for subtrees
    for test_expr in exprs_to_test:

        def traverse(node):
            if not isinstance(node, sp.Basic): return

            # 1. Check the node itself
            if node != test_expr:
                sub_vars = node.free_symbols.intersection(system_vars_set)

                # A random variable composite cannot depend on the integration threshold 't'
                if len(sub_vars) > 1 and not any(str(s) == 't' for s in node.free_symbols):
                    rest_of_expr = test_expr.xreplace({node: dummy})
                    rest_vars = rest_of_expr.free_symbols.intersection(system_vars_set)
                    physical_rest_vars = {v for v in rest_vars if str(v) != 't'}

                    if sub_vars.isdisjoint(rest_vars):
                        if not physical_rest_vars:
                            return
                        disjoint_subtrees[node] = {
                            "vars": list(sub_vars),
                            "rest": rest_of_expr
                        }
                        return  # Stop at first find, too complex to track + decide. Greedy-style.


            if isinstance(node, (sp.Add, sp.Mul)):
                from itertools import combinations
                args = node.args

                for r in range(2, len(args)):
                    for combo in combinations(args, r):
                        sub_expr = node.func(*combo)
                        sub_vars = sub_expr.free_symbols.intersection(system_vars_set)

                        if not sub_vars: continue

                        if any(str(s) == 't' for s in sub_expr.free_symbols):
                            continue

                        if isinstance(node, sp.Add):
                            modified_node = dummy + sp.expand(node - sub_expr)
                        else:
                            modified_node = dummy * sp.cancel(node / sub_expr)

                        rest_of_expr = test_expr.xreplace({node: modified_node})
                        rest_vars = rest_of_expr.free_symbols.intersection(system_vars_set)
                        physical_rest_vars = {v for v in rest_vars if str(v) != 't'}
                        if sub_vars.isdisjoint(rest_vars) and len(sub_vars) > 1:
                            if not physical_rest_vars:
                                continue
                            disjoint_subtrees[sub_expr] = {
                                "vars": list(sub_vars),
                                "rest": rest_of_expr
                            }
                            return  # Stop at first find

            for arg in node.args:
                traverse(arg)
                if disjoint_subtrees: return

        traverse(test_expr)

        if disjoint_subtrees:
            break
    return disjoint_subtrees



if __name__ == "__main__":
    x, y, z, t = sp.symbols('x y z t')

    # Test Case
    formula = x / (x ** 2 + y ** 2 + z ** 2)
    print(f"Analyzing: {formula}")

    clusters = find_disjoint_subtrees(formula, [x, y, z])
    for subtree, vars_in_tree in clusters.items():
        print(f"--> Found Isolated Subtree: [{subtree}] containing variables {vars_in_tree}")