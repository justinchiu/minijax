"""
Tests for MiniJax-Py.

Tests cover:
1. Basic evaluation (foo)
2. JVP (forward-mode AD)
3. Higher-order derivatives (0th through 4th)
4. Perturbation confusion (f/g test)
5. Staging (Jaxpr building and evaluation)
"""

from minijax import (
    foo, add, mul,
    jvp, derivative, nth_order_derivative,
    build_jaxpr, eval_jaxpr,
)


def test(name: str, condition: bool):
    """Simple test helper."""
    status = "\u2713" if condition else "\u2717"
    print(f"  {status} {name}")
    return condition


def approx_eq(a: float, b: float, eps: float = 1e-9) -> bool:
    """Check if two floats are approximately equal."""
    return abs(a - b) < eps


def main():
    passed = 0
    failed = 0

    def check(name: str, condition: bool):
        nonlocal passed, failed
        if test(name, condition):
            passed += 1
        else:
            failed += 1

    # =========================================================================
    # Eval Interpreter Tests
    # =========================================================================
    print("\nEval Interpreter:")
    check("add floats", add(2.0, 3.0) == 5.0)
    check("mul floats", mul(2.0, 3.0) == 6.0)
    check("foo(2) = 10", foo(2.0) == 10.0)

    # =========================================================================
    # JVP (Forward-Mode AD) Tests
    # =========================================================================
    print("\nJVP (Forward-Mode AD):")

    # JVP of add: d(x + 3)/dx = 1
    p, t = jvp(lambda x: add(x, 3.0), 2.0, 1.0)
    check("jvp of add: primal=5", approx_eq(p, 5.0))
    check("jvp of add: tangent=1", approx_eq(t, 1.0))

    # JVP of mul: d(x * 3)/dx = 3
    p, t = jvp(lambda x: mul(x, 3.0), 2.0, 1.0)
    check("jvp of mul: primal=6", approx_eq(p, 6.0))
    check("jvp of mul: tangent=3", approx_eq(t, 3.0))

    # JVP of foo at x=2 with tangent=1: (primal=10, tangent=7)
    primal, tangent = jvp(foo, 2.0, 1.0)
    check("jvp of foo at x=2: primal=10", approx_eq(primal, 10.0))
    check("jvp of foo at x=2: tangent=7", approx_eq(tangent, 7.0))

    # derivative is just jvp with tangent=1, extracting the tangent
    d = derivative(foo, 2.0)
    check("derivative of foo at x=2 is 7", approx_eq(d, 7.0))

    # =========================================================================
    # Higher-Order Derivatives Tests
    # =========================================================================
    print("\nHigher-Order Derivatives:")

    # foo(x) = x^2 + 3x
    # foo(2) = 10
    # foo'(x) = 2x + 3, foo'(2) = 7
    # foo''(x) = 2
    # foo'''(x) = 0 (foo is quadratic)
    # foo''''(x) = 0

    d0 = nth_order_derivative(0, foo, 2.0)
    check("0th derivative (just evaluation): 10", approx_eq(d0, 10.0))

    d1 = nth_order_derivative(1, foo, 2.0)
    check("1st derivative: 2x + 3 = 7 at x=2", approx_eq(d1, 7.0))

    d2 = nth_order_derivative(2, foo, 2.0)
    check("2nd derivative: 2", approx_eq(d2, 2.0))

    d3 = nth_order_derivative(3, foo, 2.0)
    check("3rd derivative: 0 (foo is quadratic)", approx_eq(d3, 0.0))

    d4 = nth_order_derivative(4, foo, 2.0)
    check("4th derivative: 0", approx_eq(d4, 0.0))

    # =========================================================================
    # Perturbation Confusion Tests
    # =========================================================================
    print("\nPerturbation Confusion:")

    # This is the classic perturbation confusion test:
    # f(x) = x * derivative(g, 0) where g(y) = x
    #
    # g is constant in y (it ignores y and just returns x), so g'(y) = 0.
    # Therefore f(x) = x * 0 = 0, and f'(x) = 0.
    #
    # Without proper tagging, a naive implementation would confuse the
    # dual number for x (from the outer derivative) with the dual number
    # for y (from the inner derivative), incorrectly computing g'(y) = 1.

    def f(x):
        def g(y):
            return x  # g is constant in y, ignores y completely
        should_be_zero = derivative(g, 0.0)
        return mul(x, should_be_zero)

    result = derivative(f, 1.0)
    check("should handle nested differentiation correctly (f/g test)", approx_eq(result, 0.0))

    # =========================================================================
    # Staging Tests
    # =========================================================================
    print("\nStaging:")

    # Build jaxpr from foo
    jaxpr = build_jaxpr(foo, 1)

    # Check structure
    check("jaxpr has 1 parameter", len(jaxpr.parameters) == 1)
    check("jaxpr has 2 equations", len(jaxpr.equations) == 2)
    check("jaxpr return is v_3", jaxpr.return_val == "v_3")

    # Pretty print
    print(f"\n    Jaxpr for foo:")
    for line in str(jaxpr).split('\n'):
        print(f"      {line}")
    print()

    # Evaluate jaxpr
    result = eval_jaxpr(jaxpr, (2.0,))
    check("eval_jaxpr gives same result as direct evaluation", approx_eq(result, 10.0))

    # Can differentiate through eval_jaxpr!
    primal, tangent = jvp(lambda x: eval_jaxpr(jaxpr, (x,)), 2.0, 1.0)
    check("can differentiate through eval_jaxpr: primal=10", approx_eq(primal, 10.0))
    check("can differentiate through eval_jaxpr: tangent=7", approx_eq(tangent, 7.0))

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
