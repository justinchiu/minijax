//! MiniJax-RS: a minimal JAX-like interpreter in Rust.
//!
//! Mirrors autodidax2.md: eval, forward-mode JVP, and staging to a tiny IR.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Add,
    Mul,
}

pub type Var = String;

#[derive(Clone, Debug, PartialEq)]
pub enum Atom {
    Var(Var),
    Lit(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Equation {
    pub var: Var,
    pub op: Op,
    pub args: Vec<Atom>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Jaxpr {
    pub params: Vec<Var>,
    pub equations: Vec<Equation>,
    pub return_val: Atom,
}

impl fmt::Display for Jaxpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} ->", self.params.join(", "))?;
        for eqn in &self.equations {
            let args_str: Vec<String> = eqn.args.iter().map(|a| match a {
                Atom::Var(v) => v.clone(),
                Atom::Lit(x) => format!("{}", x),
            }).collect();
            writeln!(f, "  {} = {:?}({})", eqn.var, eqn.op, args_str.join(", "))?;
        }
        match &self.return_val {
            Atom::Var(v) => write!(f, "{}", v),
            Atom::Lit(x) => write!(f, "{}", x),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Float(f64),
    Atom(Atom),
    Dual(Box<Dual>),
}

#[derive(Clone)]
pub struct Dual {
    pub interpreter: Rc<dyn Interpreter>,
    pub primal: Value,
    pub tangent: Value,
}

impl fmt::Debug for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dual")
            .field("interpreter", &"<interpreter>")
            .field("primal", &self.primal)
            .field("tangent", &self.tangent)
            .finish()
    }
}

impl PartialEq for Dual {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.interpreter, &other.interpreter)
            && self.primal == other.primal
            && self.tangent == other.tangent
    }
}

fn zero_like(value: &Value) -> Value {
    match value {
        Value::Float(_) => Value::Float(0.0),
        Value::Atom(_) => Value::Atom(Atom::Lit(0.0)),
        Value::Dual(d) => Value::Dual(Box::new(Dual {
            interpreter: d.interpreter.clone(),
            primal: zero_like(&d.primal),
            tangent: zero_like(&d.tangent),
        })),
    }
}

pub trait Interpreter {
    fn interpret_op(&self, op: Op, args: &[Value]) -> Value;
}

struct EvalInterpreter;

impl Interpreter for EvalInterpreter {
    fn interpret_op(&self, op: Op, args: &[Value]) -> Value {
        match args {
            [Value::Float(x), Value::Float(y)] => match op {
                Op::Add => Value::Float(x + y),
                Op::Mul => Value::Float(x * y),
            },
            _ => panic!("eval expects two Float args"),
        }
    }
}

thread_local! {
    static CURRENT: RefCell<Rc<dyn Interpreter>> = RefCell::new(Rc::new(EvalInterpreter));
}

pub struct InterpreterGuard {
    prev: Rc<dyn Interpreter>,
}

impl Drop for InterpreterGuard {
    fn drop(&mut self) {
        let prev = self.prev.clone();
        CURRENT.with(|cell| {
            *cell.borrow_mut() = prev;
        });
    }
}

fn current_interpreter() -> Rc<dyn Interpreter> {
    CURRENT.with(|cell| cell.borrow().clone())
}

pub fn set_interpreter(interp: Rc<dyn Interpreter>) -> InterpreterGuard {
    let prev = CURRENT.with(|cell| {
        let prev = cell.borrow().clone();
        *cell.borrow_mut() = interp;
        prev
    });
    InterpreterGuard { prev }
}

pub fn add(x: Value, y: Value) -> Value {
    current_interpreter().interpret_op(Op::Add, &[x, y])
}

pub fn mul(x: Value, y: Value) -> Value {
    current_interpreter().interpret_op(Op::Mul, &[x, y])
}

struct JvpInterpreter {
    prev: Rc<dyn Interpreter>,
}

impl JvpInterpreter {
    fn new(prev: Rc<dyn Interpreter>) -> Self {
        Self { prev }
    }

    fn dual_value(interpreter: Rc<dyn Interpreter>, primal: Value, tangent: Value) -> Value {
        Value::Dual(Box::new(Dual {
            interpreter,
            primal,
            tangent,
        }))
    }

    fn lift(&self, value: &Value) -> Dual {
        let self_interp = current_interpreter();
        match value {
            Value::Dual(d) if Rc::ptr_eq(&d.interpreter, &self_interp) => (**d).clone(),
            _ => Dual {
                interpreter: self_interp,
                primal: value.clone(),
                tangent: zero_like(value),
            },
        }
    }
}

impl Interpreter for JvpInterpreter {
    fn interpret_op(&self, op: Op, args: &[Value]) -> Value {
        match args {
            [x, y] => {
                let self_interp = current_interpreter();
                let dx = self.lift(x);
                let dy = self.lift(y);
                let _guard = set_interpreter(self.prev.clone());
                match op {
                    Op::Add => {
                        let Dual { primal: px, tangent: tx, .. } = dx;
                        let Dual { primal: py, tangent: ty, .. } = dy;
                        let p = add(px, py);
                        let t = add(tx, ty);
                        Self::dual_value(self_interp, p, t)
                    }
                    Op::Mul => {
                        let Dual { primal: px, tangent: tx, .. } = dx;
                        let Dual { primal: py, tangent: ty, .. } = dy;
                        let p = mul(px.clone(), py.clone());
                        let t1 = mul(px, ty);
                        let t2 = mul(tx, py);
                        let t = add(t1, t2);
                        Self::dual_value(self_interp, p, t)
                    }
                }
            }
            _ => panic!("jvp expects two args"),
        }
    }
}

pub fn jvp<F>(f: F, primal: Value, tangent: Value) -> (Value, Value)
where
    F: Fn(Value) -> Value,
{
    let jvp_interpreter = Rc::new(JvpInterpreter::new(current_interpreter()));
    let dual_in = JvpInterpreter::dual_value(jvp_interpreter.clone(), primal, tangent);
    let _guard = set_interpreter(jvp_interpreter.clone());
    let result = f(dual_in);
    let Dual { primal, tangent, .. } = jvp_interpreter.lift(&result);
    (primal, tangent)
}

pub fn derivative<F>(f: F, x: f64) -> Value
where
    F: Fn(Value) -> Value,
{
    derivative_value(f, Value::Float(x))
}

/// Compute derivative where x can be any Value (including nested duals for higher-order AD).
pub fn derivative_value<F>(f: F, x: Value) -> Value
where
    F: Fn(Value) -> Value,
{
    let (_p, t) = jvp(f, x, Value::Float(1.0));
    t
}

/// Compute the nth-order derivative of f at x.
/// For n=0, returns f(x). For n>0, returns d^n f / dx^n evaluated at x.
pub fn nth_order_derivative<F>(n: usize, f: F, x: f64) -> Value
where
    F: Fn(Value) -> Value + Clone + 'static,
{
    nth_order_derivative_value(n, f, Value::Float(x))
}

/// Internal helper for nth_order_derivative that works with Values.
fn nth_order_derivative_value<F>(n: usize, f: F, x: Value) -> Value
where
    F: Fn(Value) -> Value + Clone + 'static,
{
    if n == 0 {
        f(x)
    } else {
        let f_clone = f.clone();
        derivative_value(
            move |v| nth_order_derivative_value(n - 1, f_clone.clone(), v),
            x,
        )
    }
}

#[derive(Default)]
struct StageState {
    equations: Vec<Equation>,
    name_counter: usize,
}

struct StageInterpreter {
    state: RefCell<StageState>,
}

impl StageInterpreter {
    fn new() -> Self {
        Self {
            state: RefCell::new(StageState::default()),
        }
    }

    fn fresh_var(&self) -> Var {
        let mut st = self.state.borrow_mut();
        st.name_counter += 1;
        format!("v_{}", st.name_counter)
    }

    fn value_to_atom(value: &Value) -> Atom {
        match value {
            Value::Atom(atom) => atom.clone(),
            Value::Float(x) => Atom::Lit(*x),
            Value::Dual(_) => panic!("cannot stage a dual number"),
        }
    }
}

impl Interpreter for StageInterpreter {
    fn interpret_op(&self, op: Op, args: &[Value]) -> Value {
        let var = self.fresh_var();
        let atoms = args.iter().map(Self::value_to_atom).collect::<Vec<_>>();
        let mut st = self.state.borrow_mut();
        st.equations.push(Equation {
            var: var.clone(),
            op,
            args: atoms,
        });
        Value::Atom(Atom::Var(var))
    }
}

pub fn build_jaxpr<F>(f: F, num_args: usize) -> Jaxpr
where
    F: Fn(Vec<Value>) -> Value,
{
    let stage = Rc::new(StageInterpreter::new());
    let params = (0..num_args).map(|_| stage.fresh_var()).collect::<Vec<_>>();
    let args = params
        .iter()
        .map(|v| Value::Atom(Atom::Var(v.clone())))
        .collect::<Vec<_>>();
    let _guard = set_interpreter(stage.clone());
    let result = f(args);
    let return_val = StageInterpreter::value_to_atom(&result);
    let equations = stage.state.borrow().equations.clone();
    Jaxpr {
        params,
        equations,
        return_val,
    }
}

pub fn eval_jaxpr(jaxpr: &Jaxpr, args: Vec<Value>) -> Value {
    let mut env: HashMap<Var, Value> = HashMap::new();
    for (v, a) in jaxpr.params.iter().cloned().zip(args.into_iter()) {
        env.insert(v, a);
    }
    let eval_atom = |atom: &Atom, env: &HashMap<Var, Value>| match atom {
        Atom::Var(v) => env.get(v).cloned().expect("missing var"),
        Atom::Lit(x) => Value::Float(*x),
    };
    for eqn in &jaxpr.equations {
        let arg_vals = eqn
            .args
            .iter()
            .map(|a| eval_atom(a, &env))
            .collect::<Vec<_>>();
        let result = current_interpreter().interpret_op(eqn.op, &arg_vals);
        env.insert(eqn.var.clone(), result);
    }
    eval_atom(&jaxpr.return_val, &env)
}

pub fn foo(x: Value) -> Value {
    mul(x.clone(), add(x, Value::Float(3.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn eval_basic_ops() {
        assert_eq!(add(Value::Float(2.0), Value::Float(3.0)), Value::Float(5.0));
        assert_eq!(mul(Value::Float(2.0), Value::Float(3.0)), Value::Float(6.0));
    }

    #[test]
    fn jvp_foo() {
        let (p, t) = jvp(foo, Value::Float(2.0), Value::Float(1.0));
        assert_eq!(p, Value::Float(10.0));
        assert_eq!(t, Value::Float(7.0));
    }

    #[test]
    fn stage_and_eval_jaxpr() {
        let jaxpr = build_jaxpr(
            |mut args| {
                let x = args.remove(0);
                foo(x)
            },
            1,
        );
        let expected = Jaxpr {
            params: vec!["v_1".to_string()],
            equations: vec![
                Equation {
                    var: "v_2".to_string(),
                    op: Op::Add,
                    args: vec![Atom::Var("v_1".to_string()), Atom::Lit(3.0)],
                },
                Equation {
                    var: "v_3".to_string(),
                    op: Op::Mul,
                    args: vec![Atom::Var("v_1".to_string()), Atom::Var("v_2".to_string())],
                },
            ],
            return_val: Atom::Var("v_3".to_string()),
        };
        assert_eq!(jaxpr, expected);

        let result = eval_jaxpr(&jaxpr, vec![Value::Float(2.0)]);
        match result {
            Value::Float(x) => assert!(float_eq(x, 10.0)),
            _ => panic!("expected float"),
        }
    }

    #[test]
    fn perturbation_confusion_example() {
        fn f(x: Value) -> Value {
            let x_for_g = x.clone();
            let g = move |_y: Value| x_for_g.clone();
            let should_be_zero = derivative(g, 0.0);
            mul(x, should_be_zero)
        }

        let result = derivative(f, 0.0);
        assert_eq!(result, Value::Float(0.0));
    }

    #[test]
    fn higher_order_derivatives() {
        // foo(x) = x * (x + 3) = x^2 + 3x
        // foo'(x) = 2x + 3
        // foo''(x) = 2
        // foo'''(x) = 0
        // foo''''(x) = 0

        // 0th derivative (just evaluation)
        let d0 = nth_order_derivative(0, foo, 2.0);
        assert_eq!(d0, Value::Float(10.0));

        // 1st derivative at x=2: 2*2 + 3 = 7
        let d1 = nth_order_derivative(1, foo, 2.0);
        assert_eq!(d1, Value::Float(7.0));

        // 2nd derivative: 2
        let d2 = nth_order_derivative(2, foo, 2.0);
        assert_eq!(d2, Value::Float(2.0));

        // 3rd derivative: 0 (foo is only quadratic)
        let d3 = nth_order_derivative(3, foo, 2.0);
        assert_eq!(d3, Value::Float(0.0));

        // 4th derivative: 0
        let d4 = nth_order_derivative(4, foo, 2.0);
        assert_eq!(d4, Value::Float(0.0));
    }

    #[test]
    fn jvp_roundtrip_through_jaxpr() {
        // Build jaxpr from foo, then differentiate the evaluated jaxpr
        let jaxpr = build_jaxpr(
            |mut args| {
                let x = args.remove(0);
                foo(x)
            },
            1,
        );

        // JVP through eval_jaxpr should give same result as direct JVP
        let (p, t) = jvp(|x| eval_jaxpr(&jaxpr, vec![x]), Value::Float(2.0), Value::Float(1.0));
        assert_eq!(p, Value::Float(10.0));
        assert_eq!(t, Value::Float(7.0));
    }
}
