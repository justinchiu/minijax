/**
 * Tests for MiniJax-TS
 */

import {
  Op,
  VFloat,
  add,
  mul,
  foo,
  jvp,
  derivative,
  nthDerivative,
  buildJaxpr,
  evalJaxpr,
  jaxprToString,
  valueToNumber,
} from "./minijax.ts";

import type { Value } from "./minijax.ts";

// Simple test helper
let passed = 0;
let failed = 0;

function test(name: string, fn: () => void): void {
  try {
    fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (e) {
    console.log(`  ✗ ${name}`);
    console.log(`    ${e}`);
    failed++;
  }
}

function assertEqual<T>(actual: T, expected: T, msg?: string): void {
  if (actual !== expected) {
    throw new Error(
      `${msg || "Assertion failed"}: expected ${expected}, got ${actual}`
    );
  }
}

function assertClose(actual: number, expected: number, eps = 1e-9): void {
  if (Math.abs(actual - expected) > eps) {
    throw new Error(`Expected ${expected}, got ${actual}`);
  }
}

// =============================================================================
// Tests
// =============================================================================

console.log("\nEval Interpreter:");

test("add floats", () => {
  const result = add(VFloat(2), VFloat(3));
  assertEqual(valueToNumber(result), 5);
});

test("mul floats", () => {
  const result = mul(VFloat(2), VFloat(3));
  assertEqual(valueToNumber(result), 6);
});

test("foo(2) = 10", () => {
  const result = foo(VFloat(2));
  assertEqual(valueToNumber(result), 10);
});

console.log("\nJVP (Forward-Mode AD):");

test("jvp of foo at x=2", () => {
  const [p, t] = jvp(foo, VFloat(2), VFloat(1));
  assertEqual(valueToNumber(p), 10, "primal");
  assertEqual(valueToNumber(t), 7, "tangent");
});

test("derivative of foo at x=2", () => {
  const result = derivative(foo, 2);
  assertEqual(valueToNumber(result), 7);
});

console.log("\nHigher-Order Derivatives:");

test("0th derivative (just evaluation)", () => {
  const result = nthDerivative(0, foo, 2);
  assertEqual(valueToNumber(result), 10);
});

test("1st derivative: 2x + 3 = 7 at x=2", () => {
  const result = nthDerivative(1, foo, 2);
  assertEqual(valueToNumber(result), 7);
});

test("2nd derivative: 2", () => {
  const result = nthDerivative(2, foo, 2);
  assertEqual(valueToNumber(result), 2);
});

test("3rd derivative: 0 (foo is quadratic)", () => {
  const result = nthDerivative(3, foo, 2);
  assertEqual(valueToNumber(result), 0);
});

test("4th derivative: 0", () => {
  const result = nthDerivative(4, foo, 2);
  assertEqual(valueToNumber(result), 0);
});

console.log("\nPerturbation Confusion:");

test("should handle nested differentiation correctly", () => {
  // f(x) = x * derivative(g, 0) where g(y) = x
  // g is constant in y, so derivative(g, 0) = 0
  // Therefore f(x) = x * 0 = 0, and f'(x) = 0
  function f(x: Value): Value {
    const g = (_y: Value): Value => x;
    const shouldBeZero = derivative(g, 0);
    return mul(x, shouldBeZero);
  }

  const result = derivative(f, 1);
  assertEqual(valueToNumber(result), 0, "Should avoid perturbation confusion");
});

console.log("\nStaging:");

test("build jaxpr from foo", () => {
  const jaxpr = buildJaxpr((args) => foo(args[0]), 1);

  assertEqual(jaxpr.params.length, 1);
  assertEqual(jaxpr.equations.length, 2);

  // First equation: v_2 = Add(v_1, 3.0)
  assertEqual(jaxpr.equations[0].op, Op.Add);

  // Second equation: v_3 = Mul(v_1, v_2)
  assertEqual(jaxpr.equations[1].op, Op.Mul);
});

test("eval jaxpr gives same result as direct evaluation", () => {
  const jaxpr = buildJaxpr((args) => foo(args[0]), 1);
  const result = evalJaxpr(jaxpr, [VFloat(2)]);
  assertEqual(valueToNumber(result), 10);
});

test("can differentiate through eval_jaxpr", () => {
  const jaxpr = buildJaxpr((args) => foo(args[0]), 1);
  const [p, t] = jvp((x) => evalJaxpr(jaxpr, [x]), VFloat(2), VFloat(1));
  assertEqual(valueToNumber(p), 10, "primal");
  assertEqual(valueToNumber(t), 7, "tangent");
});

console.log("\nJaxpr Pretty Print:");

test("jaxpr string format", () => {
  const jaxpr = buildJaxpr((args) => foo(args[0]), 1);
  const str = jaxprToString(jaxpr);
  console.log("    " + str.split("\n").join("\n    "));
  // Just verify it doesn't throw
});

// =============================================================================
// Summary
// =============================================================================

console.log(`\n${passed} passed, ${failed} failed\n`);

if (failed > 0) {
  process.exit(1);
}
