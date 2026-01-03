# minijax-ml

A minimal JAX-like interpreter in OCaml, following `autodidax2.md`:

- Primitive ops: `add`, `mul`
- Interpreters: eval, forward-mode JVP, staging to a tiny Jaxpr IR
- Three front-ends:
  - `minijax_global.ml` (global interpreter)
  - `minijax.ml` (reader-style argument passing)
  - `minijax_reader.ml` (Reader monad with `let*`)
- Type-safe variants:
  - `minijax_gadt.ml` (GADT-typed expression AST)
  - `minijax_tagged.ml` (tagless-final with type-indexed JVP tags)

## Quick sketch

```ocaml
#use "minijax.ml";;

let foo interp x = mul interp x (add interp x (VFloat 3.0));;

(* Evaluate *)
let v = foo eval_interpreter (VFloat 2.0);;

(* JVP (primal, tangent) at x=2 with tangent=1 *)
let p, t = jvp ~base_interpreter:eval_interpreter foo (VFloat 2.0) (VFloat 1.0);;

(* Stage to IR *)
let jaxpr =
  build_jaxpr (fun interp args ->
    match args with
    | [x] -> foo interp x
    | _ -> failwith "expected one arg") 1;;
```

Notes:
- The JVP interpreter is tagged to avoid perturbation confusion.
- Higher-order AD is not fully type-safe in this minimal dynamic-value model.
