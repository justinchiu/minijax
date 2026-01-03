# minijax-rs

A minimal JAX-like interpreter in Rust, following `autodidax2.md`:

- Primitive ops: `add`, `mul`
- Interpreters: eval, forward-mode JVP, staging to a tiny Jaxpr IR

## Quick sketch

```rust
use minijax::{add, mul, Value, jvp, build_jaxpr, foo};

let v = add(Value::Float(2.0), Value::Float(3.0));

let (p, t) = jvp(foo, Value::Float(2.0), Value::Float(1.0));

let jaxpr = build_jaxpr(|args| {
    let x = args.into_iter().next().expect("arg");
    foo(x)
}, 1);
```

Notes:
- The JVP interpreter uses runtime tags to avoid perturbation confusion.
- Higher-order AD is not fully type-safe in this minimal dynamic-value model.
