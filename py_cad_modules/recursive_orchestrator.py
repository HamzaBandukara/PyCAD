import sympy as sp
from sympy.core.relational import Relational
from sympy.parsing.sympy_parser import parse_expr

from py_cad_modules.compositional_cad import find_disjoint_subtrees
from py_cad_modules.utils import apiprint, dbprint
from py_cad_modules.cad_core import get_cad
from py_cad_modules.calculus import integrate_cad, calculate_pdf
from scipy.optimize import minimize



class RecursiveOrchestrator:
    def __init__(self, base_bounds: dict, distributions: dict=None, force_closed_form: bool = False):
        """
        Manages the bottom-up evaluation of compositional CAD formulas.
        :param base_bounds: The global bounding box dict (e.g. {'x': (0, 1), 'y': (0, 1)})
        """
        self.base_bounds = base_bounds
        self.distributions = distributions or {}
        self.subtree_cache = {}  # Caches PDFs: { "y**2 + z**2": pdf_expr }
        self.dummy_var_counter = 0
        self.force_closed_form = force_closed_form

    def can_extract_subtree(self, subtree_vars):
        """
        Prevents the Orchestrator from grouping variables that do not
        have reproductive properties (like Triangular).
        Uniform distributions ARE allowed — their convolutions produce
        clean Piecewise polynomials after doit() resolves the integrals.
        """
        for var in subtree_vars:
            dist_info = self.distributions.get(str(var))
            if dist_info:
                dist_type = dist_info[0].lower()
                # Block Triangular, Piecewise, and Beta — sub-CAD mode kinks
                # and fractional powers cause issues in symbolic integration
                if dist_type in ['triangular', 'piecewise', 'beta']:
                    dbprint(f"[RECURSION SHIELD] Blocked! {var} is {dist_type} (Non-Reproductive).")
                    return False
        return True

    def _generate_dummy_var(self) -> sp.Symbol:
        """Generates a unique 'un' variable for subtrees."""
        var_name = f"u{self.dummy_var_counter}"
        self.dummy_var_counter += 1
        # Enforce positivity so SymPy cancels roots.
        return sp.Symbol(var_name, real=True, positive=True)

    def _infer_subtree_bounds(self, subtree_expr: sp.Expr) -> tuple:
        """
        Calculates the absolute minimum and maximum values a subtree can take,
        given the global boundary conditions of its variables.
        """

        sub_vars = list(subtree_expr.free_symbols)

        # Map the subtree variables to their global boundaries
        scipy_bounds = []
        for v in sub_vars:
            v_str = str(v)
            if v_str in self.base_bounds:
                l_bound, u_bound = self.base_bounds[v_str]
                # Fallback to +/- 1000 for infinite bounds to allow numerical optimization
                l_val = float(l_bound) if l_bound != -sp.oo else -1000.0
                u_val = float(u_bound) if u_bound != sp.oo else 1000.0
                scipy_bounds.append((l_val, u_val))
            else:
                scipy_bounds.append((-1000.0, 1000.0))

        # Compile SymPy expression into a C/NumPy function
        fast_func = sp.lambdify(sub_vars, subtree_expr, "numpy")

        def min_objective(x):
            return fast_func(*x)

        def max_objective(x):
            return -fast_func(*x)

        # 3. Start search from the mathematical center of the bounded space
        x0 = [(b[0] + b[1]) / 2.0 for b in scipy_bounds]

        # 4. Fire the Optimization Engine
        res_min = minimize(min_objective, x0, bounds=scipy_bounds)
        res_max = minimize(max_objective, x0, bounds=scipy_bounds)

        # Clean up floating-point noise (e.g., -1.12e-16 -> 0.0)
        w_min = round(float(res_min.fun), 5)
        w_max = round(float(-res_max.fun), 5)

        return w_min, w_max

    def _evaluate_subtree_pdf(self, subtree_expr: sp.Expr, sub_vars: list, w_var: sp.Symbol, w_bounds: tuple = None) -> sp.Expr:
        """
        Fires an isolated CAD pipeline to determine the exact analytical PDF of the subtree.
        """
        w_str = str(w_var)
        w_min, w_max = w_bounds
        bounds_strs = []

        # Extract the base bounds for the isolated variables
        for v in sub_vars:
            v_str = str(v)
            if v_str in self.base_bounds:
                lower, upper = self.base_bounds[v_str]
                if lower is not None and lower != -sp.oo:
                    bounds_strs.append(f"{lower} < {v_str}")
                if upper is not None and upper != sp.oo:
                    bounds_strs.append(f"{v_str} < {upper}")

        # Ensure that the sub-CAD respects the peaks (modes) of the distributions it is evaluating.
        for v in sub_vars:
            v_str = str(v)
            if self.distributions and v_str in self.distributions:
                dist_info = self.distributions[v_str]
                if isinstance(dist_info, tuple) and dist_info[0].lower() == 'triangular':
                    mode = dist_info[3]
                    mode_rat = sp.Rational(str(mode))

                    # Safely build a QEPCAD fractional polynomial (e.g., mode 1.5 -> 2*x - 3)
                    poly = sp.Symbol(v_str, real=True) * mode_rat.q - mode_rat.p
                    poly_str = str(poly).replace('**', '^')

                    # Create a hard cell wall at the peak
                    bounds_strs.append(f"[ [ {poly_str} < 0 ] || [ {poly_str} > 0 ] ]")

        # Append the subtree constraint (e.g., "y^2 + z^2 < w_0")
        subtree_str = str(subtree_expr).replace("**", "^")
        bounds_strs.append(f"{subtree_str} < {w_str}")

        full_formula = " && ".join(bounds_strs)

        cad_vars = [w_str] + [str(v) for v in sub_vars]

        apiprint(f"\n   -> [SUB-CAD] Firing isolated CAD for subtree: {full_formula}")

        # Generate CAD
        cad_result = get_cad(full_formula, cad_vars)
        if not cad_result or not cad_result.get("success"):
            raise ValueError(f"Subtree CAD Generation failed for {full_formula}")

        # Integrate CAD
        integration_result = integrate_cad(
            cad_data=cad_result,
            num_params=1,
            force_closed_form=self.force_closed_form,
            is_recursive=True,
            integrand="auto",
            distributions=self.distributions
        )

        if not integration_result or "cdf_expr" not in integration_result:
            raise ValueError("Subtree Integration failed.")

        # Calculate PDF
        dbprint(f"   -> [SUB-CAD] Deriving Exact Analytical PDF for {w_str}...")
        pdf_expr = calculate_pdf(integration_result, force_closed_form=self.force_closed_form, is_recursive=True)

        # FORCE-EVALUATE UNEVALUATED INTEGRALS
        if pdf_expr.has(sp.Integral):
            dbprint(f"   -> [SUB-CAD] Resolving unevaluated integrals...")
            try:
                pdf_expr = pdf_expr.doit()
                pdf_expr = sp.piecewise_fold(pdf_expr)
                dbprint(f"   -> [SUB-CAD] Resolved to: {str(pdf_expr)[:200]}...")
            except Exception as e:
                dbprint(f"   -> [SUB-CAD] WARNING: Could not resolve integrals: {e}")

        # PDF VALIDATION
        # Check that the PDF has no contradictory Piecewise conditions
        # and integrates to approximately 1.0. If validation fails,
        # the sub-CAD produced a broken distribution.
        pdf_valid = True

        # double check for unevaluated integrals
        if pdf_expr.has(sp.Integral):
            dbprint(f"   -> [SUB-CAD] WARNING: PDF still has unevaluated integrals. Marking invalid.")
            pdf_valid = False

        # evaluate PDF at the midpoint of the bounds
        if pdf_valid:
            try:
                mid_val = (float(w_min) + float(w_max)) / 2.0
                test_val = float(pdf_expr.subs(w_var, mid_val).evalf(chop=True))
                if test_val == 0.0 or abs(test_val) < 1e-15:
                    dbprint(f"   -> [SUB-CAD] WARNING: PDF evaluates to 0 at midpoint {mid_val}. Marking invalid.")
                    pdf_valid = False
                elif test_val < 0:
                    dbprint(f"   -> [SUB-CAD] WARNING: PDF evaluates to negative {test_val} at midpoint. Marking invalid.")
                    pdf_valid = False
            except Exception as e:
                dbprint(f"   -> [SUB-CAD] WARNING: PDF evaluation failed at midpoint: {e}. Marking invalid.")
                pdf_valid = False

        if not pdf_valid:
            raise ValueError(f"Sub-CAD PDF validation failed for {w_str}. Falling back to direct CAD.")

        # Purify the algebra
        if not pdf_expr.has(sp.Integral):
            ops_count = pdf_expr.count_ops()
            dbprint(f"   -> [SUB-CAD] Passing raw AST ({ops_count} ops) to Outer CAD without cosmetic simplification.")

        return pdf_expr

    def evaluate_formula(self, formula_expr: sp.Expr, system_vars: list) -> tuple:
        dbprint(f"\n[RECURSION] Analyzing AST for compositional subtrees...")
        dbprint(f"\n[RECURSION-F] Evaluating {formula_expr}...")
        clusters = find_disjoint_subtrees(formula_expr, system_vars)

        # Filter out non-reproductive distributions (Triangles, Piecewise)
        valid_clusters = {}
        for subtree, data in clusters.items():
            if self.can_extract_subtree(data["vars"]):
                valid_clusters[subtree] = data
            else:
                dbprint(f"[RECURSION SHIELD] Subtree {subtree} blocked! Falling back to Native CAD.")

        clusters = valid_clusters
        if not clusters:
            dbprint("[RECURSION] No independent subtrees found. Proceeding with standard CAD.")
            return formula_expr, system_vars, {}, {}, {}

        active_vars = list(system_vars)
        custom_distributions = {}
        custom_bounds = {}
        sub_map = {}

        # EXPLICIT AST REBUILDER
        def safe_boolean_subs(node, old_val, new_val):
            # 1. Pure Logic Gates
            if isinstance(node, (sp.And, sp.Or, sp.Not)):
                return node.func(*[safe_boolean_subs(arg, old_val, new_val) for arg in node.args])

            # 2. Relational Inequalities (<, >, <=, >=, ==)
            elif isinstance(node, Relational):
                return node.func(
                    safe_boolean_subs(node.lhs, old_val, new_val),
                    safe_boolean_subs(node.rhs, old_val, new_val)
                )

            # 3. Piecewise Functions
            elif isinstance(node, sp.Piecewise):
                new_args = []
                for branch_expr, branch_cond in node.args:
                    new_expr = safe_boolean_subs(branch_expr, old_val, new_val)
                    new_cond = safe_boolean_subs(branch_cond, old_val, new_val)
                    new_args.append((new_expr, new_cond))
                return sp.Piecewise(*new_args)

            # 4. Pure Math Expressions
            elif isinstance(node, sp.Expr):
                return node.xreplace({old_val: new_val})

            # Fallback for unhandled types
            return node

        processed_formula_str = str(formula_expr)
        skipped_subtrees = set()

        for subtree, data in clusters.items():
            sub_vars = data["vars"]

            apiprint(f"\n[RECURSION] Isolating Subtree: {subtree}")
            subtree_str = str(subtree)

            if subtree_str in self.subtree_cache:
                w_var, pdf_expr, bnds = self.subtree_cache[subtree_str]
            else:
                try:
                    w_var = self._generate_dummy_var()
                    w_min, w_max = self._infer_subtree_bounds(subtree)
                    dbprint(f"   -> [INTERVAL] Inferred bounds for {w_var}: [{w_min}, {w_max}]")
                    pdf_expr = self._evaluate_subtree_pdf(subtree, sub_vars, w_var, (w_min, w_max))
                    bnds = (w_min, w_max)
                    self.subtree_cache[subtree_str] = (w_var, pdf_expr, bnds)
                except (ValueError, Exception) as e:
                    apiprint(f"   -> [RECURSION FALLBACK] Subtree {subtree} failed: {e}")
                    apiprint(f"   -> [RECURSION FALLBACK] Keeping subtree in formula for direct CAD.")
                    skipped_subtrees.add(subtree_str)
                    continue

            # 1. Strip whitespace to ensure a perfect string match
            clean_subtree_str = str(subtree).replace(" ", "")
            clean_master_str = processed_formula_str.replace(" ", "")

            # 2. Perform the raw text swap (e.g., y**2+z**2 -> u0)
            replaced_str = clean_master_str.replace(clean_subtree_str, str(w_var))

            # 3. Update the master string for the next loop
            processed_formula_str = replaced_str

            for v in sub_vars:
                if v in active_vars: active_vars.remove(v)
            active_vars.append(w_var)

            custom_distributions[str(w_var)] = pdf_expr
            custom_bounds[str(w_var)] = bnds
            sub_map[w_var] = subtree
        local_dict = {str(s): s for s in system_vars + [sp.Symbol('t', real=True)]}
        for w_var in sub_map.keys():
            local_dict[str(w_var)] = w_var

        # Re-parse the clean string
        local_dict = {str(s): s for s in system_vars + [sp.Symbol('t', real=True)]}
        for w_var in sub_map.keys():
            local_dict[str(w_var)] = w_var

        processed_formula = parse_expr(processed_formula_str, local_dict=local_dict)

        # GHOST VARIABLE PURGE
        # delete anything that references eliminated sub-variables
        active_sym_names = set([str(v) for v in active_vars])

        if isinstance(processed_formula, sp.And):
            kept_args = []
            for arg in processed_formula.args:
                arg_sym_names = set([str(s) for s in arg.free_symbols])

                # Only keep the condition if ALL its variables are in the active list
                if arg_sym_names.issubset(active_sym_names):
                    kept_args.append(arg)

            # If everything was purged, fallback to True
            # Edge case, rare
            processed_formula = sp.And(*kept_args) if kept_args else sp.S.true


        return processed_formula, active_vars, custom_distributions, custom_bounds, sub_map

if __name__ == "__main__":
    x, y, z, t = sp.symbols('x y z t', real=True)
    formula = x / (x ** 2 + y ** 2 + z ** 2) - t

    orchestrator = RecursiveOrchestrator(base_bounds={'x': (0, 1), 'y': (0, 1), 'z': (0, 1)})

    new_form, new_vars, dists = orchestrator.evaluate_formula(formula, [x, y, z])

    print("\n=======================================================")
    print("FINAL RECURSIVE OUTPUT:")
    print(f"Outer Formula: {new_form} < 0")
    print(f"Outer Vars:    {new_vars}")
    for var, pdf in dists.items():
        print(f"Computed Custom Weighting PDF for {var}: \n{pdf}")
    print("=======================================================")