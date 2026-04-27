#include <iostream>
#include <vector>
#include <map>
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

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;

    try {
        RCP<const Basic> expr = parse(argv[1]);
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

            // Block Complex Infinity from entering Python
            if (b_str.find("zoo") != string::npos || c_str.find("zoo") != string::npos ||
                d_str.find("zoo") != string::npos || bc2_str.find("zoo") != string::npos ||
                d2_str.find("zoo") != string::npos || b_str.find("nan") != string::npos) {
                throw runtime_error("Engine generated Complex Infinity");
            }

            cout << "TYPE|ROOT" << endl;
            cout << "B|" << b_str << endl;
            cout << "C|" << c_str << endl;
            cout << "D|" << d_str << endl;
            cout << "BC2|" << bc2_str << endl;
            cout << "D2|" << d2_str << endl;
        } else {
            cout << "TYPE|POLY" << endl;
            cout << "EXPANDED|" << expand(expr)->__str__() << endl;
        }
    } catch (const exception &e) {
        cout << "TYPE|ERROR" << endl;
        cout << "MSG|" << e.what() << endl;
        return 1;
    }
    return 0;
}