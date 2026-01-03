# minijax

MiniJax is a tiny, multi-language reimplementation of the core ideas in
`autodidax2.md`: a minimal JAX-style system built around context-sensitive
interpretation. The focus is on just two primitives (`add`, `mul`) and a small
set of interpreters:

- Eval (concrete execution)
- Forward-mode AD (JVP)
- Staging to a tiny Jaxpr-like IR

## Tutorial

`autodidax2.md` is the narrative walkthrough and reference for the design. Each
implementation mirrors that structure and examples.

## Implementations

### Haskell (`minijax-hs/`)

Tagless-final encoding with multiple interpreters, plus an AST for simple
interpretation tests.

Interpreter-style imports:

```hs
import MiniJax.Tagless -- core language (JaxSym, JaxAD)
import MiniJax.Tagless.Eval -- Eval interpreter
import MiniJax.Tagless.JVP.Dynamic -- untagged JVP interpreter (shows perturbation confusion)
import MiniJax.Tagless.JVP.TaggedDynamic -- runtime-tagged JVP
import MiniJax.Tagless.JVP.TaggedStatic -- type-tagged JVP
import MiniJax.Tagless.Stage -- staging to Jaxpr
```

Shared core types:

```hs
import MiniJax.Common
```

Convenience entry point (core types only):

```hs
import MiniJax
```

Run tests:

```sh
cd minijax-hs
stack test
```

### OCaml (`minijax-ml/`)

A minimal dynamic-value interpreter with eval, tagged JVP, and staging to a
tiny IR. See `minijax-ml/README.md` for a quick sketch.

### Rust (`minijax-rs/`)

A minimal dynamic-value interpreter with eval, tagged JVP, and staging to a
tiny IR. See `minijax-rs/README.md` for a quick sketch.

Run tests:

```sh
cd minijax-rs
cargo test
```
