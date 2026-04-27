import subprocess
import os


def call_cpp_oracle(expression_str):
    """Passes a string to the C++ SymEngine Oracle and parses the algebraic components."""
    binary_path = os.path.join(os.getcwd(), "simplifier", "build", "simplifier")

    # SymEngine C++ requires ** for powers, not ^
    safe_expr = expression_str.replace('^', '**')

    try:
        output = subprocess.check_output([binary_path, safe_expr], text=True).strip()
    except subprocess.CalledProcessError:
        return {"TYPE": "ERROR"}

    result = {}
    for line in output.split('\n'):
        if '|' in line:
            key, val = line.split('|', 1)
            # Convert SymEngine's ** back to ^ for QEPCAD
            result[key] = val.replace('**', '^')

    return result