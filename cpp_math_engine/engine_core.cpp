#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Magic header: Converts C++ std::map to Python dict!
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <stdexcept>
#include <cmath>
#include <functional>
#include <algorithm>
#include <symengine/basic.h>
#include <symengine/parser.h>
#include <symengine/subs.h>
#include <symengine/symbol.h>
#include <symengine/pow.h>
#include <symengine/rational.h>
#include <symengine/real_double.h>
#include <symengine/integer.h>
#include <symengine/add.h>
#include <symengine/mul.h>

#include "exprtk/exprtk.hpp"


// This file contains a lot of mathematical helper functions written in C++. This also includes integration.
// Contains duplicate code as <ROOT>/simplifier, however, seems to run better for that as a subprocess.

namespace py = pybind11;
using namespace SymEngine;
using namespace std;


bool contains_pow_with_base(const RCP<const Basic> &expr, const RCP<const Basic> &target_base) {
    if (is_a<Pow>(*expr)) {
        RCP<const Pow> p = rcp_static_cast<const Pow>(expr);
        if (eq(*p->get_base(), *target_base)) {
            bool is_rad = false;
            if (is_a<Rational>(*p->get_exp()) && !rcp_static_cast<const Rational>(p->get_exp())->is_int()) is_rad = true;
            else if (is_a<RealDouble>(*p->get_exp())) is_rad = true;
            if (is_rad) return true;
        }
    }
    for (const auto &arg : expr->get_args()) {
        if (contains_pow_with_base(arg, target_base)) return true;
    }
    return false;
}

void collect_rads(const RCP<const Basic> &expr, vector<RCP<const Basic>> &rads) {
    if (is_a<Pow>(*expr)) {
        RCP<const Pow> p = rcp_static_cast<const Pow>(expr);
        bool is_radical = false;
        if (is_a<Rational>(*p->get_exp()) && !rcp_static_cast<const Rational>(p->get_exp())->is_int()) is_radical = true;
        else if (is_a<RealDouble>(*p->get_exp())) is_radical = true;

        if (is_radical) {
            bool exists = false;
            for (const auto &r : rads) {
                if (eq(*r, *expr)) { exists = true; break; }
            }
            if (!exists) rads.push_back(expr);
        }
        collect_rads(p->get_base(), rads);
        collect_rads(p->get_exp(), rads);
    } else {
        for (const auto &arg : expr->get_args()) collect_rads(arg, rads);
    }
}

// --- THE PYTHON API ENDPOINT ---
// Replaces main(). Returns a dict to Python instead of printing to cout.
map<string, string> simplify_radicals(const string& expr_str) {
    map<string, string> result;
    try {
        RCP<const Basic> expr = parse(expr_str);
        vector<RCP<const Basic>> rads;
        collect_rads(expr, rads);

        RCP<const Basic> outermost = RCP<const Basic>();

        for (const auto &r : rads) {
            bool inside_other = false;
            RCP<const Pow> r_pow = rcp_static_cast<const Pow>(r);
            RCP<const Basic> r_base = r_pow->get_base();

            for (const auto &other : rads) {
                if (eq(*r, *other)) continue;
                RCP<const Pow> other_pow = rcp_static_cast<const Pow>(other);
                RCP<const Basic> other_base = other_pow->get_base();
                if (contains_pow_with_base(other_base, r_base)) {
                    inside_other = true;
                    break;
                }
            }
            if (!inside_other) {
                outermost = r;
                break;
            }
        }

        if (outermost != RCP<const Basic>()) {
            RCP<const Pow> root = rcp_static_cast<const Pow>(outermost);
            RCP<const Basic> B = root->get_base();
            RCP<const Symbol> dummy = symbol("DUMMY_RAD");

            map<RCP<const Basic>, RCP<const Basic>, RCPBasicKeyLess> sub_map;
            sub_map[outermost] = dummy;
            RCP<const Basic> replaced = expand(expr->subs(sub_map));

            map<RCP<const Basic>, RCP<const Basic>, RCPBasicKeyLess> zero_map;
            zero_map[dummy] = integer(0);
            RCP<const Basic> D = expand(replaced->subs(zero_map));

            RCP<const Basic> C = expand(mul(sub(replaced, D), pow(dummy, integer(-1))));
            C = expand(C->subs(zero_map));

            RCP<const Basic> BC2 = expand(mul(B, pow(C, integer(2))));
            RCP<const Basic> D2 = expand(pow(D, integer(2)));

            string b_str = B->__str__();
            string c_str = C->__str__();
            string d_str = D->__str__();
            string bc2_str = BC2->__str__();
            string d2_str = D2->__str__();

            if (b_str.find("zoo") != string::npos || c_str.find("zoo") != string::npos ||
                d_str.find("zoo") != string::npos || bc2_str.find("zoo") != string::npos ||
                d2_str.find("zoo") != string::npos || b_str.find("nan") != string::npos) {
                throw runtime_error("Engine generated Complex Infinity");
            }

            // Populate the dictionary
            result["TYPE"] = "ROOT";
            result["B"] = b_str;
            result["C"] = c_str;
            result["D"] = d_str;
            result["BC2"] = bc2_str;
            result["D2"] = d2_str;
        } else {
            result["TYPE"] = "POLY";
            result["EXPANDED"] = expand(expr)->__str__();
        }
    } catch (const exception &e) {
        result["TYPE"] = "ERROR";
        result["MSG"] = e.what();
    }
    return result;
}


// --- THE C++ INTEGRATION ENGINE ---
string fast_integrate(const string& integrand_str, const string& var_str) {
    try {
        // 1. Pattern matching engine
        bool has_asin_1 = integrand_str.find("asin(1/sqrt(") != string::npos;
        bool has_asin_2 = integrand_str.find("asin(sqrt(") != string::npos;
        bool has_var = integrand_str.find(var_str) != string::npos;

        // The 2D Compositional PDF (e.g., u0 = y^2 + z^2 for u0 in [1, 2])
        if (has_asin_1 && has_asin_2 && has_var) {
            string u = var_str;
            string anti_deriv = u + "*asin(1/sqrt(" + u + "))/2 - " + u + "*asin(sqrt(" + u + " - 1)/sqrt(" + u + "))/2 + sqrt(" + u + " - 1)";

            return anti_deriv;
        }

        // RULE 2: The inner Uniform PDF (Strict Match)
        if (integrand_str == "pi/4") {
             return "pi/4 * " + var_str;
        }

        // NO RULE MATCHED: Fallback to the Python CAS
        return "NO_CPP_RULE_MATCHED";

    } catch (const exception& e) {
        return string("ERROR: ") + e.what();
    }
}

// ============================================================================
// BOOLEAN/PIECEWISE MASK EVALUATOR
// ============================================================================
py::array_t<double> fast_vectorized_eval(
    const std::string& expression,
    const std::vector<std::string>& var_names,
    py::array_t<double> points_array)
{
    // 1. Request buffer information from the NumPy array
    py::buffer_info buf = points_array.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Points array must be 2-dimensional (N points x M variables)");
    }

    size_t num_points = buf.shape[0];
    size_t num_vars = buf.shape[1];

    if (num_vars != var_names.size()) {
        throw std::runtime_error("Number of columns in array must match number of variable names");
    }

    // 2. Set up the ExprTk compiler and symbol table
    exprtk::symbol_table<double> symbol_table;
    std::vector<double> current_point(num_vars, 0.0);

    // Register variables with the symbol table (ExprTk binds them by reference!)
    for (size_t i = 0; i < num_vars; ++i) {
        symbol_table.add_variable(var_names[i], current_point[i]);
    }

    exprtk::expression<double> expr;
    expr.register_symbol_table(symbol_table);

    exprtk::parser<double> parser;

    // 3. Compile the string expression into C++ execution nodes
    if (!parser.compile(expression, expr)) {
        throw std::runtime_error("ExprTk Compilation Error: " + parser.error());
    }

    // 4. Prepare the output NumPy array
    auto result_array = py::array_t<double>(num_points);
    py::buffer_info res_buf = result_array.request();
    double* ptr_in = static_cast<double*>(buf.ptr);
    double* ptr_out = static_cast<double*>(res_buf.ptr);

    // 5. The big loop
    for (size_t i = 0; i < num_points; ++i) {
        // Update the bound variables
        for (size_t j = 0; j < num_vars; ++j) {
            current_point[j] = ptr_in[i * num_vars + j];
        }
        // Evaluate the expression and store the result
        ptr_out[i] = expr.value();
    }

    return result_array;
}

// ============================================================================
// BOOLEAN SCALAR EVALUATOR (For Piecewise Branch Collapsing)
// ============================================================================
bool fast_evaluate_condition(const std::string& expression, const py::dict& pt_dict) {
    exprtk::symbol_table<double> symbol_table;
    std::vector<std::string> keys;
    std::vector<double> values;

// Unpack Python dictionary into C++ vectors using PyBind11's native casting
    for (auto item : pt_dict) {
        // Extract the handle directly to a native C++ string
        keys.push_back(item.first.cast<std::string>());

        // Extract the handle directly to a native C++ double
        values.push_back(item.second.cast<double>());
    }

    // Bind memory to ExprTk
    for (size_t i = 0; i < keys.size(); ++i) {
        symbol_table.add_variable(keys[i], values[i]);
    }

    exprtk::expression<double> expr;
    expr.register_symbol_table(symbol_table);
    exprtk::parser<double> parser;

    // Clean up SymPy logic syntax for ExprTk
    std::string clean_expr = expression;

    if (!parser.compile(clean_expr, expr)) {
        throw std::runtime_error("ExprTk Compilation Error");
    }

    // ExprTk evaluates true logic as 1.0, false as 0.0
    return expr.value() != 0.0;
}

// ============================================================================
// C++ 1D INTEGRATOR (Riemann).
// ============================================================================
double fast_1d_quadrature(
    const std::string& integrand_str,
    double t_val,
    double a, double b,
    std::string var_name)
{
    double t = t_val, x = 0.0;
    exprtk::symbol_table<double> symbol_table;
    symbol_table.add_variable("t", t);
    symbol_table.add_variable(var_name, x);

    exprtk::expression<double> integrand_expr;
    integrand_expr.register_symbol_table(symbol_table);

    exprtk::parser<double> parser;
    if (!parser.compile(integrand_str, integrand_expr)) return 0.0;

    int samples = 5000;
    double dx = (b - a) / samples;
    double total_volume = 0.0;

    for(int i = 0; i < samples; ++i) {
        x = a + (i + 0.5) * dx; // Midpoint evaluation
        total_volume += integrand_expr.value();
    }

    return total_volume * dx;
}

// ============================================================================
// C++ 2D INTEGRATOR
// ============================================================================
double fast_2d_quadrature(
    const std::string& integrand_str,
    const std::string& lower_str, const std::string& upper_str,
    double t_val, double outer_min, double outer_max,
    std::string outer_var, std::string inner_var)
{
    double t = t_val, out_val = 0.0, in_val = 0.0;
    exprtk::symbol_table<double> st;
    st.add_variable("t", t); st.add_variable(outer_var, out_val); st.add_variable(inner_var, in_val);
    exprtk::expression<double> expr, l_expr, u_expr;
    expr.register_symbol_table(st); l_expr.register_symbol_table(st); u_expr.register_symbol_table(st);
    exprtk::parser<double> p;
    if(!p.compile(integrand_str, expr) || !p.compile(lower_str, l_expr) || !p.compile(upper_str, u_expr)) return 0.0;

    int samples = 10000;
    static std::vector<double> h2, h3;
    static bool init = false;
    if(!init) {
        h2.resize(samples); h3.resize(samples);
        for(int i=1; i<=samples; ++i) {
            double f=0.5, h=0; int tmp=i; while(tmp>0){h+=f*(tmp&1); tmp>>=1; f*=0.5;} h2[i-1]=h;
            f=1.0/3.0; h=0; tmp=i; while(tmp>0){h+=f*(tmp%3); tmp/=3; f/=3.0;} h3[i-1]=h;
        }
        init = true;
    }

    double total = 0.0;
    for(int i=0; i<samples; ++i) {
        out_val = outer_min + h2[i]*(outer_max - outer_min);
        double l = l_expr.value(); double u = u_expr.value();
        if(l < u) {
            in_val = l + h3[i]*(u - l);
            total += expr.value() * (u - l);
        }
    }
    return (total / samples) * (outer_max - outer_min);
}

// ============================================================================
// C++ 3D INTEGRATOR
// ============================================================================
double fast_3d_quadrature(
    const std::string& integrand_str,
    const std::string& v_lower_str, const std::string& v_upper_str,
    const std::string& u_lower_str, const std::string& u_upper_str,
    double t_val, double T_min, double T_max,
    std::string var_T, std::string var_u, std::string var_v)
{
    double t = t_val, T = 0.0, u = 0.0, v = 0.0;
    exprtk::symbol_table<double> st;
    st.add_variable("t", t); st.add_variable(var_T, T); st.add_variable(var_u, u); st.add_variable(var_v, v);
    exprtk::expression<double> expr, vl_expr, vu_expr, ul_expr, uu_expr;
    expr.register_symbol_table(st); vl_expr.register_symbol_table(st); vu_expr.register_symbol_table(st);
    ul_expr.register_symbol_table(st); uu_expr.register_symbol_table(st);
    exprtk::parser<double> p;
    if(!p.compile(integrand_str, expr) || !p.compile(v_lower_str, vl_expr) || !p.compile(v_upper_str, vu_expr) ||
       !p.compile(u_lower_str, ul_expr) || !p.compile(u_upper_str, uu_expr)) return 0.0;

    int samples = 15000;
    static std::vector<double> h2, h3, h5;
    static bool init = false;
    if(!init) {
        h2.resize(samples); h3.resize(samples); h5.resize(samples);
        for(int i=1; i<=samples; ++i) {
            double f=0.5, h=0; int tmp=i; while(tmp>0){h+=f*(tmp&1); tmp>>=1; f*=0.5;} h2[i-1]=h;
            f=1.0/3.0; h=0; tmp=i; while(tmp>0){h+=f*(tmp%3); tmp/=3; f/=3.0;} h3[i-1]=h;
            f=1.0/5.0; h=0; tmp=i; while(tmp>0){h+=f*(tmp%5); tmp/=5; f/=5.0;} h5[i-1]=h;
        }
        init = true;
    }

    double total = 0.0;
    for(int i=0; i<samples; ++i) {
        T = T_min + h2[i]*(T_max - T_min);
        double l_u = ul_expr.value(); double u_u = uu_expr.value();
        if(l_u < u_u) {
            u = l_u + h3[i]*(u_u - l_u);
            double l_v = vl_expr.value(); double u_v = vu_expr.value();
            if(l_v < u_v) {
                v = l_v + h5[i]*(u_v - l_v);
                // Jacobian transform to scale the unit volume back to physical geometry
                total += expr.value() * (u_u - l_u) * (u_v - l_v);
            }
        }
    }
    return (total / samples) * (T_max - T_min);
}


// ============================================================================
// THE N-DIMENSIONAL ADAPTIVE C++ INTEGRATOR
// ============================================================================
// Replaces the SciPy recursive quad fallback with pure C++ Gauss-Kronrod.
//
// Interface from Python:
//   fast_nd_quadrature(
//       integrand_str,                          # "1/(2*sqrt(u0 - y^2))"
//       var_names,                              # ["u0", "z", "y"]  (inner → outer)
//       lower_bound_strs,                       # ["997/500", "max(1,-sqrt(...))", "1"]
//       upper_bound_strs,                       # ["4003/500", "min(2,sqrt(...))", "2"]
//   )
//
// All bound expressions are ExprTk strings that may reference outer variables.
// Integration order: var_names[last] is outermost, var_names[0] is innermost.
// ============================================================================

// --- 7-point Gauss and 15-point Kronrod nodes/weights on [-1, 1] ---
// Pre-computed to full double precision for G7-K15 adaptive quadrature.
static const int GK_N = 15;

static const double gk_nodes[15] = {
    -0.991455371120813, -0.949107912342759, -0.864864423359769,
    -0.741531185599394, -0.586087235467691, -0.405845151377397,
    -0.207784955007898,  0.0,
     0.207784955007898,  0.405845151377397,  0.586087235467691,
     0.741531185599394,  0.864864423359769,  0.949107912342759,
     0.991455371120813
};

// Kronrod weights (15-point)
static const double gk_wK[15] = {
    0.022935322010529, 0.063092092629979, 0.104790010322250,
    0.140653259715525, 0.169004726639268, 0.190350578064785,
    0.204432940075298, 0.209482141084728,
    0.204432940075298, 0.190350578064785, 0.169004726639268,
    0.140653259715525, 0.104790010322250, 0.063092092629979,
    0.022935322010529
};

// Gauss weights (7-point - only at odd indices of the 15-point rule)
static const double gk_wG[15] = {
    0.0,              0.129484966168870, 0.0,
    0.279705391489277, 0.0,              0.381830050505119,
    0.0,              0.417959183673469,
    0.0,              0.381830050505119, 0.0,
    0.279705391489277, 0.0,              0.129484966168870,
    0.0
};

// Forward declaration
static double nd_integrate_recursive(
    exprtk::expression<double>& integrand_expr,
    std::vector<exprtk::expression<double>>& lo_exprs,
    std::vector<exprtk::expression<double>>& hi_exprs,
    std::vector<std::vector<exprtk::expression<double>>>& bp_exprs,
    std::vector<double*>& var_ptrs,
    int dim,           // current dimension index (0 = innermost)
    int max_dim,       // total number of dimensions
    double abs_tol,
    double rel_tol,
    int max_depth
);

// Core adaptive G7-K15 quadrature on a single smooth subinterval (no breakpoints).
static double adaptive_gk15_smooth(
    std::function<double(double)> f,
    double a, double b,
    double abs_tol, double rel_tol, int depth_remaining)
{
    double mid = 0.5 * (a + b);
    double half = 0.5 * (b - a);

    double result_K = 0.0;  // 15-point Kronrod estimate
    double result_G = 0.0;  // 7-point Gauss estimate

    double fvals[GK_N];

    for (int i = 0; i < GK_N; ++i) {
        double x = mid + half * gk_nodes[i];
        fvals[i] = f(x);
        result_K += gk_wK[i] * fvals[i];
        result_G += gk_wG[i] * fvals[i];
    }

    result_K *= half;
    result_G *= half;

    double err = std::abs(result_K - result_G);

    // Accept if error is within tolerance or we've exhausted recursion depth
    if (depth_remaining <= 0 || err <= std::max(abs_tol, rel_tol * std::abs(result_K))) {
        return result_K;
    }

    // Subdivide
    return adaptive_gk15_smooth(f, a, mid, abs_tol * 0.5, rel_tol, depth_remaining - 1)
         + adaptive_gk15_smooth(f, mid, b, abs_tol * 0.5, rel_tol, depth_remaining - 1);
}

// Breakpoint-aware adaptive G7-K15 quadrature for a single dimension.
// Splits the interval [a, b] at any breakpoints that fall strictly inside it,
// then calls the smooth core on each sub-interval.
static double adaptive_gk15(
    std::function<double(double)> f,
    double a, double b,
    double abs_tol, double rel_tol, int depth_remaining,
    const std::vector<double>& breakpoints = {})
{
    // Collect breakpoints that fall strictly inside (a, b)
    std::vector<double> splits;
    for (double bp : breakpoints) {
        if (bp > a && bp < b) {
            splits.push_back(bp);
        }
    }

    if (splits.empty()) {
        // No kinks in this interval - use the smooth core directly
        return adaptive_gk15_smooth(f, a, b, abs_tol, rel_tol, depth_remaining);
    }

    // Sort and deduplicate
    std::sort(splits.begin(), splits.end());
    splits.erase(std::unique(splits.begin(), splits.end()), splits.end());

    // Build the sub-interval edges
    std::vector<double> edges;
    edges.reserve(splits.size() + 2);
    edges.push_back(a);
    for (double s : splits) edges.push_back(s);
    edges.push_back(b);

    int n_panels = (int)edges.size() - 1;
    double panel_tol = abs_tol / n_panels;  // distribute tolerance budget

    double total = 0.0;
    for (int i = 0; i < n_panels; ++i) {
        total += adaptive_gk15_smooth(f, edges[i], edges[i + 1],
                                       panel_tol, rel_tol, depth_remaining);
    }
    return total;
}

// The recursive N-dimensional integration engine.
// At each dimension level, it evaluates the bounds (which may depend on outer variables
// already set), then adaptively integrates over that dimension, with the integrand
// being the recursive call to the next inner dimension.
// Breakpoint expressions for each dimension are evaluated at integration time so they
// can depend on outer variables (e.g., the mode of a triangular distribution that is
// parameterised by an outer integration variable).
// NOTE: Breakpoints do not work for tri-distributions in Py-CAD. Leave them empty!
static double nd_integrate_recursive(
    exprtk::expression<double>& integrand_expr,
    std::vector<exprtk::expression<double>>& lo_exprs,
    std::vector<exprtk::expression<double>>& hi_exprs,
    std::vector<std::vector<exprtk::expression<double>>>& bp_exprs,
    std::vector<double*>& var_ptrs,
    int dim,
    int max_dim,
    double abs_tol,
    double rel_tol,
    int max_depth)
{
    // Evaluate bounds for this dimension (~dependency on outer vars)
    double a = lo_exprs[dim].value();
    double b = hi_exprs[dim].value();

    if (a >= b || std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b)) {
        return 0.0;
    }

    // Evaluate breakpoints for this dimension (~dependency on outer vars)
    std::vector<double> bps;
    for (auto& bp_expr : bp_exprs[dim]) {
        double bp_val = bp_expr.value();
        if (std::isfinite(bp_val)) {
            bps.push_back(bp_val);
        }
    }

    if (dim == 0) {
        // Innermost dimension: integrate the actual integrand expression
        auto f_inner = [&](double x) -> double {
            *(var_ptrs[0]) = x;
            double val = integrand_expr.value();
            return std::isfinite(val) ? val : 0.0;
        };
        return adaptive_gk15(f_inner, a, b, abs_tol, rel_tol, max_depth, bps);
    } else {
        // Outer dimension: integrate the inner integral as a function of this variable
        auto f_outer = [&](double x) -> double {
            *(var_ptrs[dim]) = x;
            return nd_integrate_recursive(
                integrand_expr, lo_exprs, hi_exprs, bp_exprs, var_ptrs,
                dim - 1, max_dim, abs_tol, rel_tol, max_depth
            );
        };
        // Use fewer subdivisions for outer dimensions to control total cost
        int outer_depth = std::max(3, max_depth - dim);
        return adaptive_gk15(f_outer, a, b, abs_tol, rel_tol, outer_depth, bps);
    }
}

// Python-facing function.
// breakpoint_strs is optional: a vector of vectors of ExprTk expression strings,
// one vector per dimension (inner-to-outer, matching var_names).
// --------------------------------------------------------------------------------
// Each expression is evaluated at integration time and may reference outer variables.
// For a triangular distribution, pass the mode expression as a breakpoint on the
// dimension being integrated so the kink is always a panel boundary.
// NOTE: this doesn't actually work great for tri-dists. Kept as legacy, use WITHOUT BREAKPOINTS
double fast_nd_quadrature(
    const std::string& integrand_str,
    const std::vector<std::string>& var_names,       // inner-to-outer order
    const std::vector<std::string>& lower_bound_strs,
    const std::vector<std::string>& upper_bound_strs,
    double abs_tol,
    double rel_tol,
    int max_depth,
    const std::vector<std::vector<std::string>>& breakpoint_strs = {})
{
    int ndim = (int)var_names.size();
    if (ndim == 0) return 0.0;
    if ((int)lower_bound_strs.size() != ndim || (int)upper_bound_strs.size() != ndim) {
        throw std::runtime_error("fast_nd_quadrature: var_names, lower_bounds, upper_bounds must have same length");
    }

    // 1. Create a shared symbol table with all variable slots
    exprtk::symbol_table<double> st;
    std::vector<double> var_storage(ndim, 0.0);
    std::vector<double*> var_ptrs(ndim);

    for (int i = 0; i < ndim; ++i) {
        st.add_variable(var_names[i], var_storage[i]);
        var_ptrs[i] = &var_storage[i];
    }

    double t_dummy = 0.0;
    bool has_t = false;
    for (int i = 0; i < ndim; ++i) {
        if (var_names[i] == "t") { has_t = true; break; }
    }
    if (!has_t) {
        st.add_variable("t", t_dummy);
    }

    // 2. Compile the integrand
    exprtk::expression<double> integrand_expr;
    integrand_expr.register_symbol_table(st);
    exprtk::parser<double> parser;
    if (!parser.compile(integrand_str, integrand_expr)) {
        throw std::runtime_error("fast_nd_quadrature: Failed to compile integrand: "
                                 + integrand_str + " | Error: " + parser.error());
    }

    // 3. Compile all bound expressions
    std::vector<exprtk::expression<double>> lo_exprs(ndim), hi_exprs(ndim);
    for (int i = 0; i < ndim; ++i) {
        lo_exprs[i].register_symbol_table(st);
        hi_exprs[i].register_symbol_table(st);
        if (!parser.compile(lower_bound_strs[i], lo_exprs[i])) {
            throw std::runtime_error("fast_nd_quadrature: Failed to compile lower bound[" +
                                     std::to_string(i) + "]: " + lower_bound_strs[i] +
                                     " | Error: " + parser.error());
        }
        if (!parser.compile(upper_bound_strs[i], hi_exprs[i])) {
            throw std::runtime_error("fast_nd_quadrature: Failed to compile upper bound[" +
                                     std::to_string(i) + "]: " + upper_bound_strs[i] +
                                     " | Error: " + parser.error());
        }
    }

    // 4. Compile breakpoint expressions for each dimension.
    //    If breakpoint_strs is empty or shorter than ndim, missing dimensions
    //    simply have no breakpoints.
    std::vector<std::vector<exprtk::expression<double>>> bp_exprs(ndim);
    for (int i = 0; i < ndim && i < (int)breakpoint_strs.size(); ++i) {
        for (const auto& bp_str : breakpoint_strs[i]) {
            if (bp_str.empty()) continue;
            exprtk::expression<double> bp_expr;
            bp_expr.register_symbol_table(st);
            if (!parser.compile(bp_str, bp_expr)) {
                throw std::runtime_error("fast_nd_quadrature: Failed to compile breakpoint[" +
                                         std::to_string(i) + "]: " + bp_str +
                                         " | Error: " + parser.error());
            }
            bp_exprs[i].push_back(std::move(bp_expr));
        }
    }

    // 5. Call the recursive adaptive integrator (outermost dimension = ndim-1)
    return nd_integrate_recursive(
        integrand_expr, lo_exprs, hi_exprs, bp_exprs, var_ptrs,
        ndim - 1, ndim, abs_tol, rel_tol, max_depth
    );
}


// ============================================================================
// PYBIND11 MODULE DEFINITION
// ============================================================================
PYBIND11_MODULE(pycad_cpp_engine, m) {
    m.doc() = "C++ Engine for Fast CAS Operations";

    m.def("simplify_radicals", &simplify_radicals, "Decomposes radical formulas into B, C, D components.");
    m.def("fast_integrate", &fast_integrate, "Analytically integrate using C++ custom rules");
    m.def("fast_evaluate_condition", &fast_evaluate_condition, "Evaluates boolean scalars quickly; useful for piecewise branch collapsing");

    // 1D Interceptor Binding
    m.def("fast_1d_quadrature", &fast_1d_quadrature, "Fast 1D Integrator",
          py::arg("integrand_str"), py::arg("t_val"),
          py::arg("a"), py::arg("b"), py::arg("var_name"));

    // 2D Interceptor Binding
    m.def("fast_2d_quadrature", &fast_2d_quadrature, "Fast 2D Quasi-Monte Carlo",
          py::arg("integrand_str"), py::arg("lower_bound_str"), py::arg("upper_bound_str"),
          py::arg("t_val"), py::arg("outer_min"), py::arg("outer_max"),
          py::arg("outer_var"), py::arg("inner_var"));

    // 3D Interceptor Binding
    m.def("fast_3d_quadrature", &fast_3d_quadrature, "Performs superfast nested 3D integration",
          py::arg("integrand_str"),
          py::arg("v_lower_str"), py::arg("v_upper_str"),
          py::arg("u_lower_str"), py::arg("u_upper_str"),
          py::arg("t_val"), py::arg("T_min"), py::arg("T_max"),
          py::arg("var_T"), py::arg("var_u"), py::arg("var_v"));

    m.def("fast_vectorized_eval", &fast_vectorized_eval,
          "Evaluates a math/logic string over a 2D numpy array of points",
          py::arg("expression"), py::arg("var_names"), py::arg("points_array"));

    // N-Dimensional Adaptive Integrator (with optional breakpoints for kink-aware integration)
    m.def("fast_nd_quadrature", &fast_nd_quadrature,
          "Adaptive Gauss-Kronrod N-dimensional nested integration with variable bounds "
          "and optional per-dimension breakpoints for non-smooth integrands (e.g. triangular PDF modes)",
          py::arg("integrand_str"),
          py::arg("var_names"),
          py::arg("lower_bound_strs"),
          py::arg("upper_bound_strs"),
          py::arg("abs_tol") = 1e-6,
          py::arg("rel_tol") = 1e-6,
          py::arg("max_depth") = 8,
          py::arg("breakpoint_strs") = std::vector<std::vector<std::string>>{});
}