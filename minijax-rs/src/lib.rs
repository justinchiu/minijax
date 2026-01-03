//! MiniJax-RS: a minimal JAX-like interpreter in Rust.
//!
//! Mirrors autodidax2.md: eval, forward-mode JVP, and staging to a tiny IR.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

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

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Float(f64),
    Atom(Atom),
    Dual(Box<Dual>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dual {
    pub tag: u64,
    pub primal: Value,
    pub tangent: Value,
}

fn zero_like(value: &Value) -> Value {
    match value {
        Value::Float(_) => Value::Float(0.0),
        Value::Atom(_) => Value::Atom(Atom::Lit(0.0)),
        Value::Dual(d) => Value::Dual(Box::new(Dual {
            tag: d.tag,
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

static NEXT_TAG: AtomicU64 = AtomicU64::new(1);

fn fresh_tag() -> u64 {
    NEXT_TAG.fetch_add(1, Ordering::Relaxed)
}

struct JvpInterpreter {
    tag: u64,
    prev: Rc<dyn Interpreter>,
}

impl JvpInterpreter {
    fn new(prev: Rc<dyn Interpreter>) -> Self {
        Self { tag: fresh_tag(), prev }
    }

    fn dual_value(&self, primal: Value, tangent: Value) -> Value {
        Value::Dual(Box::new(Dual { tag: self.tag, primal, tangent }))
    }

    fn lift(&self, value: &Value) -> Dual {
        match value {
            Value::Dual(d) if d.tag == self.tag => (**d).clone(),
            _ => Dual {
                tag: self.tag,
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
                let dx = self.lift(x);
                let dy = self.lift(y);
                let _guard = set_interpreter(self.prev.clone());
                match op {
                    Op::Add => {
                        let p = add(dx.primal.clone(), dy.primal.clone());
                        let t = add(dx.tangent.clone(), dy.tangent.clone());
                        self.dual_value(p, t)
                    }
                    Op::Mul => {
                        let p = mul(dx.primal.clone(), dy.primal.clone());
                        let t1 = mul(dx.primal.clone(), dy.tangent.clone());
                        let t2 = mul(dx.tangent.clone(), dy.primal.clone());
                        let t = add(t1, t2);
                        self.dual_value(p, t)
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
    let prev = current_interpreter();
    let jvp_interpreter = Rc::new(JvpInterpreter::new(prev));
    let dual_in = jvp_interpreter.dual_value(primal, tangent);
    let _guard = set_interpreter(jvp_interpreter.clone());
    let result = f(dual_in);
    let dual_out = jvp_interpreter.lift(&result);
    (dual_out.primal.clone(), dual_out.tangent.clone())
}

pub fn derivative<F>(f: F, x: f64) -> Value
where
    F: Fn(Value) -> Value,
{
    let (_p, t) = jvp(f, Value::Float(x), Value::Float(1.0));
    t
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
        Self { state: RefCell::new(StageState::default()) }
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
        st.equations.push(Equation { var: var.clone(), op, args: atoms });
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
    Jaxpr { params, equations, return_val }
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
        let arg_vals = eqn.args.iter().map(|a| eval_atom(a, &env)).collect::<Vec<_>>();
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
}
