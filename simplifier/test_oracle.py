import subprocess
import os


def run_tests():
    # Target the newly compiled C++ binary
    binary_path = os.path.join(os.getcwd(), "simplifier", "build", "simplifier")

    tests = [
        ("1. Simple Polynomial", "ax^2 + ay^2 - 1"),
        ("2. Standard Root", "az * (ax^2 + ay^2)^0.5 + 1"),
        ("3. Division by Root (Negative Power)", "az * (ax^2 + ay^2)^(-0.5)"),
        ("4. The 'Zoo' Killer (Nested Roots)", "(-ay - 2*t * (0.5 - 0.5*az*(ax**2 + ay**2 + az**2)**(-0.5))**0.5)")
    ]

    print("====================================")
    print("--- C++ SYMENGINE ORACLE TESTS ---")
    print("====================================\n")

    for name, expr in tests:
        print(f"--- TEST: {name} ---")
        print(f"INPUT:  {expr}")
        try:
            # Oracle requires Python's ** for parsing
            safe_expr = expr.replace('^', '**')
            output = subprocess.check_output([binary_path, safe_expr], text=True).strip()

            for line in output.split('\n'):
                print(f"  > {line}")
        except Exception as e:
            print(f"  > FAILED: {e}")
        print("")


if __name__ == "__main__":
    run_tests()