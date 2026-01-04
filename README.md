# minijax

MiniJax is a tiny, multi-language reimplementation of the core ideas in
the jax-from-scratch tutorial `autodidax2.md`: a minimal JAX-style system built around context-sensitive
interpretation. The focus is on just two primitives (`add`, `mul`) and a small
set of interpreters:

- Eval (concrete execution)
- Forward-mode AD (JVP)
- Staging to a tiny Jaxpr-like IR

## Implementations

| Language | Directory | Run Tests |
|----------|-----------|-----------|
| Haskell | `minijax-hs/` | `stack test` |
| OCaml | `minijax-ml/` | `dune test` |
| Rust | `minijax-rs/` | `cargo test` |
| TypeScript | `minijax-ts/` | `npm test` |
| Python | `minijax-py/` | `python3 test_minijax.py` |

---

## Feature Coverage by Implementation

### Haskell (`minijax-hs/`)

Haskell has two main approaches: **Direct** (monadic, like Python/OCaml) and **Tagless Final** (type-class based). The Tagless approach has three JVP variants with different perturbation confusion handling.

| Feature | Direct | Tagless Dynamic | Tagless TaggedStatic | Tagless TaggedDynamic |
|---------|--------|-----------------|----------------------|-----------------------|
| Eval (add, mul) | ✓ | ✓ | ✓ | ✓ |
| foo(2) = 10 | ✓ | ✓ | ✓ | ✓ |
| JVP of add | ✓ | ✓ | ✓ | ✓ |
| JVP of mul | ✓ | ✓ | ✓ | ✓ |
| JVP of foo | ✓ | ✓ | ✓ | ✓ |
| Higher-order derivatives (0-4th) | ✓ | - | - | ✓ |
| Perturbation confusion (f/g) | ✓ | ✗ (known bug) | ✓ | ✓ |
| Staging (build jaxpr) | ✓ | ✓ | ✓ | ✓ |
| Eval jaxpr | ✓ | ✓ | ✓ | ✓ |
| JVP through eval_jaxpr | ✓ | - | - | - |

**Notes:**
- **Tagless Dynamic** intentionally fails perturbation confusion to demonstrate the bug
- **Direct** style is closest to the Python/TypeScript implementations
- **TaggedDynamic** is recommended for higher-order AD with correct perturbation handling

### OCaml (`minijax-ml/`)

OCaml has five different implementation styles exploring various design patterns.

| Feature | GADT | Global | Reader Monad | Reader Style | Tagged Functor |
|---------|------|--------|--------------|--------------|----------------|
| Eval (add, mul) | ✓ | ✓ | ✓ | ✓ | ✓ |
| foo(2) = 10 | ✓ | ✓ | ✓ | ✓ | ✓ |
| Finite difference approx | - | ✓ | ✓ | ✓ | - |
| JVP of add | - | ✓ | ✓ | ✓ | - |
| JVP of mul | - | ✓ | ✓ | ✓ | - |
| JVP of foo | ✓ | ✓ | ✓ | ✓ | ✓ |
| derivative helper | ✓ | - | - | - | ✓ |
| Higher-order derivatives (0-4th) | - | ✓ | ✓ (0-3) | ✓ (0-3) | - |
| Perturbation confusion (f/g) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Staging (build jaxpr) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Eval jaxpr | ✓ | ✓ | ✓ | ✓ | ✓ |
| jvp2 (2nd order in one pass) | - | - | - | - | ✓ |
| jvp_n (n derivatives at once) | - | - | - | - | ✓ |

**Notes:**
- **GADT** style uses explicit expression AST
- **Global** style uses mutable global interpreter (closest to Python)
- **Reader Monad** uses `ReaderT` for interpreter threading
- **Reader Style** passes interpreter explicitly
- **Tagged Functor** uses OCaml modules/functors for type-safe tagging

### Rust (`minijax-rs/`)

Single implementation using thread-local state and `Rc<dyn Interpreter>`.

| Feature | Status |
|---------|--------|
| Eval (add, mul) | ✓ |
| foo(2) = 10 | (via JVP) |
| JVP of foo | ✓ |
| Higher-order derivatives (0-4th) | ✓ |
| Perturbation confusion (f/g) | ✓ |
| Staging (build jaxpr) | ✓ |
| Eval jaxpr | ✓ |
| JVP through eval_jaxpr | ✓ |

### TypeScript (`minijax-ts/`)

Single implementation using global interpreter with tagged duals.

| Feature | Status |
|---------|--------|
| Eval (add, mul) | ✓ |
| foo(2) = 10 | ✓ |
| JVP of foo | ✓ |
| derivative helper | ✓ |
| Higher-order derivatives (0-4th) | ✓ |
| Perturbation confusion (f/g) | ✓ |
| Staging (build jaxpr) | ✓ |
| Eval jaxpr | ✓ |
| JVP through eval_jaxpr | ✓ |
| Jaxpr pretty print | ✓ |

### Python (`minijax-py/`)

Single implementation following `autodidax2.md` closely with tagged duals.

| Feature | Status |
|---------|--------|
| Eval (add, mul) | ✓ |
| foo(2) = 10 | ✓ |
| JVP of add | ✓ |
| JVP of mul | ✓ |
| JVP of foo | ✓ |
| derivative helper | ✓ |
| Higher-order derivatives (0-4th) | ✓ |
| Perturbation confusion (f/g) | ✓ |
| Staging (build jaxpr) | ✓ |
| Eval jaxpr | ✓ |
| JVP through eval_jaxpr | ✓ |

---

## Quick Start

### Haskell

```sh
cd minijax-hs
stack test
```

Imports:
```haskell
import MiniJax.Direct           -- Direct style (recommended)
import MiniJax.Tagless          -- Tagless final core (JaxSym, JaxAD)
import MiniJax.Tagless.Eval     -- Eval interpreter
import MiniJax.Tagless.JVP.Dynamic        -- Untagged JVP (has perturbation confusion bug)
import MiniJax.Tagless.JVP.TaggedDynamic  -- Runtime-tagged JVP
import MiniJax.Tagless.JVP.TaggedStatic   -- Type-tagged JVP
import MiniJax.Tagless.Stage    -- Staging to Jaxpr
import MiniJax.Common           -- Shared types (Op, Jaxpr, etc.)
```

### OCaml

```sh
cd minijax-ml
dune test
```

### Rust

```sh
cd minijax-rs
cargo test
```

### TypeScript

```sh
cd minijax-ts
npm install
npm test
```

### Python

```sh
cd minijax-py
python3 test_minijax.py
```

---

## Design Notes

The key insight from `autodidax2.md` is **context-sensitive interpretation**: operations like `add` and `mul` dispatch to the "current interpreter" rather than having fixed semantics. This allows the same user code to be:

1. **Evaluated** on concrete floats
2. **Differentiated** by interpreting as dual numbers
3. **Staged** to an IR by recording operations symbolically

### Perturbation Confusion

In higher-order AD, we must distinguish dual numbers from different differentiation contexts. The classic bug:

```python
def f(x):
    def g(y): return x  # g is constant in y
    return x * derivative(g, 0)  # should be x * 0 = 0
```

Without proper tagging, a naive implementation confuses the dual number for `x` (outer derivative) with `y` (inner derivative), incorrectly computing `g'(y) = 1` instead of `0`.

Solutions:
- **Tagged duals**: Each dual carries a reference to its creating interpreter
- **Type-level tags**: Use phantom types to distinguish differentiation contexts (Haskell TaggedStatic)
- **Module functors**: Use OCaml's module system for type-safe tagging
