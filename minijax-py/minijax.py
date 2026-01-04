"""
MiniJax-Py: a minimal JAX-like interpreter in Python.

Design: Global Interpreter with Tagged Duals
============================================
This implementation follows autodidax2.md closely. Key design choices:

1. GLOBAL CURRENT INTERPRETER: A module-level variable `current_interpreter`
   determines how operations behave. User-facing functions `add` and `mul`
   dispatch to it:

     current_interpreter = EvalInterpreter()
     def add(x, y): return current_interpreter.interpret_op(Op.add, (x, y))

2. TAGGED DUAL NUMBERS: Each TaggedDualNumber carries a reference to its
   creating interpreter to avoid perturbation confusion in higher-order AD:

     @dataclass
     class TaggedDualNumber:
       interpreter: Interpreter
       primal: Any
       tangent: Any

3. NESTING VIA VALUE TYPES: The primal/tangent fields can themselves be
   TaggedDualNumbers from outer interpreters, enabling higher-order AD.

Perturbation Confusion
----------------------
When lifting a value to a dual, we check if it's already a TaggedDualNumber
from the CURRENT interpreter. If not, it's treated as a constant (tangent = 0).
This prevents mixing tangents from different differentiation contexts.

Example of perturbation confusion (handled correctly):
    def f(x):
        def g(y): return x  # g is constant in y
        return x * derivative(g, 0)  # should be x * 0 = 0

    derivative(f, 1.0)  # Should be 0, not 1
"""

from enum import Enum, auto
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Tuple, List, Union


# =============================================================================
# Primitive Operations
# =============================================================================

class Op(Enum):
    """The primitive operations in our language.

    In a full JAX implementation, this would include the entire NumPy API
    (sin, cos, matmul, etc.). We keep it minimal with just add and mul.
    """
    add = auto()  # addition on floats
    mul = auto()  # multiplication on floats


# =============================================================================
# Interpreter Base and Global State
# =============================================================================

class Interpreter:
    """Base class for interpreters.

    Different interpreters implement different semantics:
    - EvalInterpreter: performs actual arithmetic on floats
    - JVPInterpreter: propagates dual numbers for forward-mode AD
    - StagingInterpreter: records operations to build a Jaxpr IR
    """
    def interpret_op(self, op: Op, args: Tuple[Any, ...]) -> Any:
        raise NotImplementedError("subclass should implement this")


class EvalInterpreter(Interpreter):
    """The evaluation interpreter: performs ordinary concrete arithmetic.

    This is the default interpreter - it just does normal math on floats.
    """
    def interpret_op(self, op: Op, args: Tuple[Any, ...]) -> float:
        assert all(isinstance(arg, (int, float)) for arg in args), \
            f"EvalInterpreter expects numeric arguments, got {args}"
        match op:
            case Op.add:
                x, y = args
                return x + y
            case Op.mul:
                x, y = args
                return x * y
            case _:
                raise ValueError(f"Unrecognized primitive op: {op}")


# The current interpreter. Initially set to EvalInterpreter.
# All operations dispatch to this interpreter.
current_interpreter: Interpreter = EvalInterpreter()


@contextmanager
def set_interpreter(new_interpreter: Interpreter):
    """Temporarily set the interpreter and restore it afterward.

    This is crucial for:
    1. Running functions under different interpreters (JVP, staging)
    2. Ensuring cleanup even if exceptions occur
    3. Enabling nested interpreter contexts for higher-order AD
    """
    global current_interpreter
    prev_interpreter = current_interpreter
    try:
        current_interpreter = new_interpreter
        yield
    finally:
        current_interpreter = prev_interpreter


# =============================================================================
# User-Facing Operations
# =============================================================================

def add(x: Any, y: Any) -> Any:
    """Add two values using the current interpreter.

    The actual behavior depends on which interpreter is active:
    - EvalInterpreter: returns x + y (float)
    - JVPInterpreter: returns dual number with propagated tangents
    - StagingInterpreter: returns a variable name, records equation
    """
    return current_interpreter.interpret_op(Op.add, (x, y))


def mul(x: Any, y: Any) -> Any:
    """Multiply two values using the current interpreter."""
    return current_interpreter.interpret_op(Op.mul, (x, y))


# =============================================================================
# JVP (Forward-Mode AD) Interpreter
# =============================================================================

@dataclass
class TaggedDualNumber:
    """A dual number for forward-mode automatic differentiation.

    Mathematically, a dual number is `primal + tangent * epsilon` where
    epsilon^2 = 0. Propagating these through arithmetic gives us derivatives
    via the chain rule.

    The `interpreter` field is crucial for avoiding perturbation confusion:
    it identifies which JVP context created this dual number. When lifting
    a value, we check if it's a dual from the *current* interpreter - if not,
    it's treated as a constant (tangent = 0).

    Attributes:
        interpreter: The JVPInterpreter that created this dual number
        primal: The original value (can be float or nested TaggedDualNumber)
        tangent: The derivative value (can be float or nested TaggedDualNumber)
    """
    interpreter: 'JVPInterpreter'
    primal: Any
    tangent: Any


class JVPInterpreter(Interpreter):
    """The JVP interpreter implements forward-mode automatic differentiation.

    It interprets values as dual numbers (primal + tangent*epsilon) and
    propagates them through operations using differentiation rules:
    - Addition: d(x + y) = dx + dy
    - Multiplication: d(x * y) = x*dy + dx*y (product rule)

    The `prev_interpreter` stores the interpreter that was active when this
    JVP interpreter was created. We use it to evaluate primal/tangent
    operations, which enables higher-order AD.
    """

    def __init__(self, prev_interpreter: Interpreter):
        # Keep a pointer to the interpreter that was current when this
        # interpreter was first invoked. That's the context in which our
        # differentiation rules should run.
        self.prev_interpreter = prev_interpreter

    def interpret_op(self, op: Op, args: Tuple[Any, ...]) -> TaggedDualNumber:
        # Lift all arguments to dual numbers (constants get tangent = 0)
        args = tuple(self.lift(arg) for arg in args)

        # Switch to the previous interpreter for computing on primals/tangents.
        # This prevents infinite recursion and enables nesting for higher-order AD.
        with set_interpreter(self.prev_interpreter):
            match op:
                case Op.add:
                    # d(x + y) = dx + dy
                    x, y = args
                    return self.dual_number(
                        add(x.primal, y.primal),
                        add(x.tangent, y.tangent)
                    )
                case Op.mul:
                    # d(x * y) = x*dy + dx*y (product rule)
                    x, y = args
                    return self.dual_number(
                        mul(x.primal, y.primal),
                        add(mul(x.primal, y.tangent), mul(x.tangent, y.primal))
                    )

    def dual_number(self, primal: Any, tangent: Any) -> TaggedDualNumber:
        """Create a dual number tagged with this interpreter."""
        return TaggedDualNumber(self, primal, tangent)

    def lift(self, x: Any) -> TaggedDualNumber:
        """Lift a value to a TaggedDualNumber.

        If the value is already a TaggedDualNumber from THIS interpreter,
        return it as-is. Otherwise, treat it as a constant with tangent = 0.

        This is the key to avoiding perturbation confusion: a dual number
        from a *different* JVP interpreter (e.g., an outer differentiation
        context) is constant with respect to THIS differentiation.
        """
        if isinstance(x, TaggedDualNumber) and x.interpreter is self:
            return x
        else:
            # Constant with respect to this differentiation
            return self.dual_number(x, zero_like(x))


def zero_like(x: Any) -> Any:
    """Create a zero value with the same structure as the input.

    For nested duals (higher-order AD), this preserves the nesting structure.
    """
    if isinstance(x, TaggedDualNumber):
        return TaggedDualNumber(
            x.interpreter,
            zero_like(x.primal),
            zero_like(x.tangent)
        )
    else:
        return 0.0


def jvp(f: Callable, primal: Any, tangent: Any) -> Tuple[Any, Any]:
    """Compute the Jacobian-vector product (JVP) of a function.

    Given a function `f`, a primal input, and a tangent vector,
    returns (f(primal), df/dx * tangent).

    For scalar functions, this is equivalent to (f(x), f'(x) * tangent).

    Args:
        f: The function to differentiate
        primal: The point at which to evaluate f
        tangent: The tangent vector (typically 1.0 for scalar derivatives)

    Returns:
        A tuple (primal_out, tangent_out) where:
        - primal_out = f(primal)
        - tangent_out = df/dx * tangent
    """
    jvp_interpreter = JVPInterpreter(current_interpreter)
    dual_number_in = jvp_interpreter.dual_number(primal, tangent)
    with set_interpreter(jvp_interpreter):
        result = f(dual_number_in)
    dual_number_out = jvp_interpreter.lift(result)
    return dual_number_out.primal, dual_number_out.tangent


def derivative(f: Callable, x: float) -> Any:
    """Compute the derivative of f at x.

    This is a convenience wrapper around jvp with tangent = 1.0.
    """
    _, tangent = jvp(f, x, 1.0)
    return tangent


def nth_order_derivative(n: int, f: Callable, x: float) -> Any:
    """Compute the nth-order derivative of f at x.

    For n=0, returns f(x).
    For n>0, returns d^n f / dx^n evaluated at x.

    This works by recursively applying `derivative`, which creates
    nested JVP interpreters. Each level of nesting corresponds to
    one order of differentiation.

    Example for foo(x) = x * (x + 3) = x^2 + 3x:
        nth_order_derivative(0, foo, 2.0) = 10.0  (just evaluation)
        nth_order_derivative(1, foo, 2.0) = 7.0   (2x + 3 at x=2)
        nth_order_derivative(2, foo, 2.0) = 2.0   (constant)
        nth_order_derivative(3, foo, 2.0) = 0.0   (foo is quadratic)
    """
    if n == 0:
        return f(x)
    else:
        # Differentiate "the function that computes the (n-1)th derivative"
        return derivative(lambda y: nth_order_derivative(n - 1, f, y), x)


# =============================================================================
# Intermediate Representation (Jaxpr)
# =============================================================================

# Variables are just strings in this untyped IR
Var = str

# Atoms (arguments to operations) can be variables or float literals
Atom = Union[Var, float]


@dataclass
class Equation:
    """A single equation/instruction in our IR.

    Example: `v_3 = mul(v_1, v_2)` binds the result of multiplying
    v_1 and v_2 to the variable v_3.
    """
    var: Var           # The variable name of the result
    op: Op             # The primitive operation we're applying
    args: Tuple[Atom, ...]  # The arguments to the operation


@dataclass
class Jaxpr:
    """Jaxpr (JAX expression) - our intermediate representation.

    A Jaxpr represents a function in ANF (A-normal form):
    - parameters: the input variables
    - equations: a sequence of let-bindings
    - return_val: the output atom

    Example for `foo(x) = x * (x + 3)`:
        v_1 ->
          v_2 = Op.add(v_1, 3.0)
          v_3 = Op.mul(v_1, v_2)
        v_3
    """
    parameters: List[Var]
    equations: List[Equation]
    return_val: Atom

    def __str__(self) -> str:
        lines = []
        lines.append(', '.join(b for b in self.parameters) + ' ->')
        for eqn in self.equations:
            args_str = ', '.join(str(arg) for arg in eqn.args)
            lines.append(f'  {eqn.var} = {eqn.op.name}({args_str})')
        lines.append(str(self.return_val))
        return '\n'.join(lines)


# =============================================================================
# Staging Interpreter
# =============================================================================

class StagingInterpreter(Interpreter):
    """The staging interpreter builds a Jaxpr IR by recording operations.

    Instead of computing results, it:
    1. Generates fresh variable names for results
    2. Records each operation as an Equation
    3. Returns variable names that refer to the results

    This is essential for transformations that need the whole program,
    like dead-code elimination or reverse-mode AD.
    """

    def __init__(self):
        self.equations: List[Equation] = []  # All operations seen so far
        self.name_counter: int = 0           # For generating unique names

    def fresh_var(self) -> Var:
        """Generate a fresh variable name."""
        self.name_counter += 1
        return f"v_{self.name_counter}"

    def interpret_op(self, op: Op, args: Tuple[Atom, ...]) -> Var:
        """Record an operation and return a fresh variable for its result."""
        binder = self.fresh_var()
        self.equations.append(Equation(binder, op, args))
        return binder


def build_jaxpr(f: Callable, num_args: int) -> Jaxpr:
    """Build a Jaxpr from a function by tracing it with symbolic inputs.

    This runs the function under the staging interpreter, recording
    all operations. The result is a Jaxpr that can be:
    - Pretty-printed
    - Evaluated with `eval_jaxpr`
    - Differentiated (via eval_jaxpr under JVP interpreter)
    """
    interpreter = StagingInterpreter()
    parameters = tuple(interpreter.fresh_var() for _ in range(num_args))
    with set_interpreter(interpreter):
        result = f(*parameters)
    return Jaxpr(list(parameters), interpreter.equations, result)


def eval_jaxpr(jaxpr: Jaxpr, args: Tuple[Any, ...]) -> Any:
    """Evaluate a Jaxpr with concrete arguments.

    This interprets the Jaxpr by walking through equations and evaluating
    each one using the current interpreter. Because we use `current_interpreter`,
    we can differentiate through `eval_jaxpr` by running it under a JVP
    interpreter!
    """
    # Build environment mapping variables to values
    env = dict(zip(jaxpr.parameters, args))

    def eval_atom(x: Atom) -> Any:
        """Evaluate an atom: look up variables, pass through literals."""
        return env[x] if isinstance(x, str) else x

    # Evaluate each equation using the current interpreter
    for eqn in jaxpr.equations:
        arg_vals = tuple(eval_atom(x) for x in eqn.args)
        env[eqn.var] = current_interpreter.interpret_op(eqn.op, arg_vals)

    return eval_atom(jaxpr.return_val)


# =============================================================================
# Example Function
# =============================================================================

def foo(x: Any) -> Any:
    """Example function: foo(x) = x * (x + 3)

    This is equivalent to x^2 + 3x, which has:
    - foo(2) = 10
    - foo'(x) = 2x + 3, so foo'(2) = 7
    - foo''(x) = 2
    - foo'''(x) = 0
    """
    return mul(x, add(x, 3.0))
