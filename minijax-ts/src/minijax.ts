/**
 * MiniJax-TS: a minimal JAX-like interpreter in TypeScript.
 *
 * Design: Global Interpreter with Tagged Duals
 * =============================================
 * This implementation follows the Python version in autodidax2.md closely.
 * Key design choices:
 *
 * 1. GLOBAL CURRENT INTERPRETER: A module-level variable `currentInterpreter`
 *    determines how operations behave. User-facing functions `add` and `mul`
 *    dispatch to it:
 *
 *      let currentInterpreter: Interpreter = evalInterpreter;
 *      const add = (x: Value, y: Value) => currentInterpreter.interpretOp(Op.Add, [x, y]);
 *
 * 2. TAGGED DUAL NUMBERS: Each Dual carries a reference to its creating
 *    interpreter to avoid perturbation confusion in higher-order AD:
 *
 *      interface Dual {
 *        interpreter: Interpreter;
 *        primal: Value;
 *        tangent: Value;
 *      }
 *
 * 3. NESTING VIA VALUE UNION: The Value type can be Float, Dual, or Atom.
 *    Since Dual contains Values, we get natural nesting for higher-order AD.
 *
 * Perturbation Confusion
 * ----------------------
 * When lifting a value to a Dual, we check if it's already a Dual from the
 * CURRENT interpreter. If not, it's treated as a constant (tangent = 0).
 * This prevents mixing tangents from different differentiation contexts.
 */

// =============================================================================
// Primitive Operations
// =============================================================================

/** The primitive operations in our language. */
export const Op = {
  Add: "Add",
  Mul: "Mul",
} as const;

export type Op = (typeof Op)[keyof typeof Op];

// =============================================================================
// Intermediate Representation (Jaxpr)
// =============================================================================

/** Variable names in our IR. */
export type Var = string;

/** Atoms are arguments to operations: either variables or literals. */
export type Atom = { kind: "Var"; name: Var } | { kind: "Lit"; value: number };

/** Helper constructors for Atom. */
export const VarAtom = (name: Var): Atom => ({ kind: "Var", name });
export const LitAtom = (value: number): Atom => ({ kind: "Lit", value });

/** A single equation/instruction in our IR. */
export interface Equation {
  variable: Var;
  op: Op;
  args: Atom[];
}

/**
 * Jaxpr (JAX expression) - our intermediate representation.
 *
 * A Jaxpr represents a function in ANF (A-normal form):
 * - params: the input variables
 * - equations: a sequence of let-bindings
 * - returnVal: the output atom
 *
 * Example for `foo(x) = x * (x + 3)`:
 * ```
 * v_1 ->
 *   v_2 = Add(v_1, 3.0)
 *   v_3 = Mul(v_1, v_2)
 * v_3
 * ```
 */
export interface Jaxpr {
  params: Var[];
  equations: Equation[];
  returnVal: Atom;
}

// =============================================================================
// Values
// =============================================================================

/** Forward declaration for Interpreter (needed for Dual). */
export interface Interpreter {
  interpretOp(op: Op, args: Value[]): Value;
}

/**
 * A dual number for forward-mode automatic differentiation.
 *
 * Mathematically, a dual number is `primal + tangent * ε` where ε² = 0.
 * Propagating these through arithmetic gives us derivatives via the chain rule.
 *
 * The `interpreter` field identifies which JVP context created this dual.
 * This prevents perturbation confusion in higher-order differentiation.
 */
export interface Dual {
  interpreter: Interpreter;
  primal: Value;
  tangent: Value;
}

/**
 * Values that flow through interpreters.
 *
 * The key to our design is that Value can represent different things:
 * - Float: concrete numeric values (used by EvalInterpreter)
 * - Atom: symbolic variables (used by StageInterpreter)
 * - Dual: primal-tangent pairs (used by JvpInterpreter)
 *
 * Crucially, Dual contains nested Values, enabling higher-order AD.
 */
export type Value =
  | { kind: "Float"; value: number }
  | { kind: "Atom"; atom: Atom }
  | { kind: "Dual"; dual: Dual };

/** Helper constructors for Value. */
export const VFloat = (value: number): Value => ({ kind: "Float", value });
export const VAtom = (atom: Atom): Value => ({ kind: "Atom", atom });
export const VDual = (dual: Dual): Value => ({ kind: "Dual", dual });

/** Create a zero value with the same structure as the input. */
function zeroLike(value: Value): Value {
  switch (value.kind) {
    case "Float":
      return VFloat(0);
    case "Atom":
      return VAtom(LitAtom(0));
    case "Dual":
      return VDual({
        interpreter: value.dual.interpreter,
        primal: zeroLike(value.dual.primal),
        tangent: zeroLike(value.dual.tangent),
      });
  }
}

/** Extract a number from a (possibly nested) Value. */
export function valueToNumber(value: Value): number {
  switch (value.kind) {
    case "Float":
      return value.value;
    case "Dual":
      return valueToNumber(value.dual.primal);
    case "Atom":
      throw new Error("Cannot convert Atom to number");
  }
}

// =============================================================================
// Interpreter Trait and Global State
// =============================================================================

/**
 * The evaluation interpreter: performs ordinary concrete arithmetic.
 */
const evalInterpreter: Interpreter = {
  interpretOp(op: Op, args: Value[]): Value {
    const [x, y] = args;
    if (x.kind !== "Float" || y.kind !== "Float") {
      throw new Error("EvalInterpreter expects Float arguments");
    }
    switch (op) {
      case Op.Add:
        return VFloat(x.value + y.value);
      case Op.Mul:
        return VFloat(x.value * y.value);
    }
  },
};

/**
 * The current interpreter. Initially set to evalInterpreter.
 * Operations dispatch to this interpreter.
 */
let currentInterpreter: Interpreter = evalInterpreter;

/**
 * Temporarily set the interpreter and run a function.
 * Restores the previous interpreter afterward (even if an error occurs).
 */
export function withInterpreter<T>(interp: Interpreter, fn: () => T): T {
  const prev = currentInterpreter;
  currentInterpreter = interp;
  try {
    return fn();
  } finally {
    currentInterpreter = prev;
  }
}

// =============================================================================
// User-Facing Operations
// =============================================================================

/** Add two values using the current interpreter. */
export function add(x: Value, y: Value): Value {
  return currentInterpreter.interpretOp(Op.Add, [x, y]);
}

/** Multiply two values using the current interpreter. */
export function mul(x: Value, y: Value): Value {
  return currentInterpreter.interpretOp(Op.Mul, [x, y]);
}

// =============================================================================
// JVP (Forward-Mode AD) Interpreter
// =============================================================================

/**
 * Create a JVP interpreter.
 *
 * The JVP interpreter implements forward-mode automatic differentiation.
 * It interprets values as dual numbers and propagates them using:
 * - Addition: d(x + y) = dx + dy
 * - Multiplication: d(x * y) = x*dy + dx*y (product rule)
 *
 * @param prev The interpreter to use for primal/tangent operations
 */
function makeJvpInterpreter(prev: Interpreter): Interpreter {
  const jvpInterpreter: Interpreter = {
    interpretOp(op: Op, args: Value[]): Value {
      const [x, y] = args;

      // Lift both arguments to Dual numbers
      const dx = lift(x);
      const dy = lift(y);

      // Switch to previous interpreter for computing on primals/tangents
      return withInterpreter(prev, () => {
        switch (op) {
          case Op.Add: {
            // d(x + y) = dx + dy
            const p = add(dx.primal, dy.primal);
            const t = add(dx.tangent, dy.tangent);
            return VDual({ interpreter: jvpInterpreter, primal: p, tangent: t });
          }
          case Op.Mul: {
            // d(x * y) = x*dy + dx*y (product rule)
            const p = mul(dx.primal, dy.primal);
            const t1 = mul(dx.primal, dy.tangent);
            const t2 = mul(dx.tangent, dy.primal);
            const t = add(t1, t2);
            return VDual({ interpreter: jvpInterpreter, primal: p, tangent: t });
          }
        }
      });
    },
  };

  /**
   * Lift a value to a Dual number.
   *
   * If the value is already a Dual from THIS interpreter, return it as-is.
   * Otherwise, treat it as a constant with tangent = 0.
   *
   * This is key to avoiding perturbation confusion.
   */
  function lift(value: Value): Dual {
    if (value.kind === "Dual" && value.dual.interpreter === jvpInterpreter) {
      return value.dual;
    }
    return {
      interpreter: jvpInterpreter,
      primal: value,
      tangent: zeroLike(value),
    };
  }

  return jvpInterpreter;
}

/**
 * Compute the Jacobian-vector product (JVP) of a function.
 *
 * Given a function `f`, a primal input, and a tangent vector,
 * returns [f(primal), df/dx * tangent].
 */
export function jvp(
  f: (x: Value) => Value,
  primal: Value,
  tangent: Value
): [Value, Value] {
  const jvpInterp = makeJvpInterpreter(currentInterpreter);
  const dualIn: Value = VDual({
    interpreter: jvpInterp,
    primal,
    tangent,
  });

  const result = withInterpreter(jvpInterp, () => f(dualIn));

  // Extract primal and tangent from result
  if (result.kind === "Dual" && result.dual.interpreter === jvpInterp) {
    return [result.dual.primal, result.dual.tangent];
  }
  return [result, zeroLike(result)];
}

/**
 * Compute the derivative of f at x.
 * This is a convenience wrapper around jvp with tangent = 1.
 */
export function derivative(f: (x: Value) => Value, x: number): Value {
  return derivativeValue(f, VFloat(x));
}

/**
 * Compute derivative where x can be any Value (for higher-order AD).
 */
export function derivativeValue(f: (x: Value) => Value, x: Value): Value {
  const [_, t] = jvp(f, x, VFloat(1));
  return t;
}

/**
 * Compute the nth-order derivative of f at x.
 *
 * For n=0, returns f(x).
 * For n>0, returns d^n f / dx^n evaluated at x.
 */
export function nthDerivative(
  n: number,
  f: (x: Value) => Value,
  x: number
): Value {
  return nthDerivativeValue(n, f, VFloat(x));
}

function nthDerivativeValue(
  n: number,
  f: (x: Value) => Value,
  x: Value
): Value {
  if (n === 0) {
    return f(x);
  }
  // Differentiate "the function that computes the (n-1)th derivative"
  return derivativeValue((v) => nthDerivativeValue(n - 1, f, v), x);
}

// =============================================================================
// Staging Interpreter
// =============================================================================

interface StageState {
  equations: Equation[];
  nameCounter: number;
}

/**
 * Create a staging interpreter.
 *
 * Instead of computing results, it:
 * 1. Generates fresh variable names for results
 * 2. Records each operation as an Equation
 * 3. Returns symbolic Atom values
 */
function makeStageInterpreter(state: StageState): Interpreter {
  function freshVar(): Var {
    state.nameCounter++;
    return `v_${state.nameCounter}`;
  }

  function valueToAtom(value: Value): Atom {
    switch (value.kind) {
      case "Atom":
        return value.atom;
      case "Float":
        return LitAtom(value.value);
      case "Dual":
        throw new Error("Cannot stage a dual number");
    }
  }

  return {
    interpretOp(op: Op, args: Value[]): Value {
      const variable = freshVar();
      const atoms = args.map(valueToAtom);
      state.equations.push({ variable, op, args: atoms });
      return VAtom(VarAtom(variable));
    },
  };
}

/**
 * Build a Jaxpr from a function by tracing it with symbolic inputs.
 */
export function buildJaxpr(
  f: (args: Value[]) => Value,
  numArgs: number
): Jaxpr {
  const state: StageState = { equations: [], nameCounter: 0 };
  const stageInterp = makeStageInterpreter(state);

  // Create symbolic input variables
  const params: Var[] = [];
  const args: Value[] = [];
  for (let i = 0; i < numArgs; i++) {
    state.nameCounter++;
    const v = `v_${state.nameCounter}`;
    params.push(v);
    args.push(VAtom(VarAtom(v)));
  }

  // Run the function under the staging interpreter
  const result = withInterpreter(stageInterp, () => f(args));

  // Convert result to Atom
  let returnVal: Atom;
  switch (result.kind) {
    case "Atom":
      returnVal = result.atom;
      break;
    case "Float":
      returnVal = LitAtom(result.value);
      break;
    case "Dual":
      throw new Error("Cannot stage a dual number");
  }

  return { params, equations: state.equations, returnVal };
}

/**
 * Evaluate a Jaxpr with concrete arguments.
 *
 * Uses the current interpreter, so we can differentiate through eval_jaxpr!
 */
export function evalJaxpr(jaxpr: Jaxpr, args: Value[]): Value {
  const env = new Map<Var, Value>();

  // Initialize environment with parameters
  jaxpr.params.forEach((param, i) => {
    env.set(param, args[i]);
  });

  function evalAtom(atom: Atom): Value {
    switch (atom.kind) {
      case "Var":
        const val = env.get(atom.name);
        if (val === undefined) throw new Error(`Unknown variable: ${atom.name}`);
        return val;
      case "Lit":
        return VFloat(atom.value);
    }
  }

  // Evaluate each equation using the current interpreter
  for (const eqn of jaxpr.equations) {
    const argVals = eqn.args.map(evalAtom);
    const result = currentInterpreter.interpretOp(eqn.op, argVals);
    env.set(eqn.variable, result);
  }

  return evalAtom(jaxpr.returnVal);
}

/**
 * Pretty-print a Jaxpr.
 */
export function jaxprToString(jaxpr: Jaxpr): string {
  const lines: string[] = [];
  lines.push(`${jaxpr.params.join(", ")} ->`);
  for (const eqn of jaxpr.equations) {
    const argsStr = eqn.args
      .map((a) => (a.kind === "Var" ? a.name : String(a.value)))
      .join(", ");
    lines.push(`  ${eqn.variable} = ${eqn.op}(${argsStr})`);
  }
  const retStr =
    jaxpr.returnVal.kind === "Var"
      ? jaxpr.returnVal.name
      : String(jaxpr.returnVal.value);
  lines.push(retStr);
  return lines.join("\n");
}

// =============================================================================
// Example Function
// =============================================================================

/**
 * Example function: foo(x) = x * (x + 3)
 *
 * This is equivalent to x² + 3x, which has:
 * - foo(2) = 10
 * - foo'(x) = 2x + 3, so foo'(2) = 7
 * - foo''(x) = 2
 * - foo'''(x) = 0
 */
export function foo(x: Value): Value {
  return mul(x, add(x, VFloat(3)));
}
