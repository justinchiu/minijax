# minijax-ml

A minimal JAX-like interpreter in OCaml, following `autodidax2.md`:

- Primitive ops: `add`, `mul`
- Interpreters: eval, forward-mode JVP, staging to a tiny Jaxpr IR

## Quick sketch

```ocaml
#use "minijax.ml";;

let foo x = mul x (add x (VFloat 3.0));;

(* Evaluate *)
let v = set_interpreter eval_interpreter (fun () -> foo (VFloat 2.0));;

(* JVP (primal, tangent) at x=2 with tangent=1 *)
let p, t = jvp foo (VFloat 2.0) (VFloat 1.0);;

(* Stage to IR *)
let jaxpr = build_jaxpr (fun args ->
  match args with
  | [x] -> foo x
  | _ -> failwith "expected one arg") 1;;
```

Notes:
- The JVP interpreter is tagged to avoid perturbation confusion.
- Higher-order AD is not fully type-safe in this minimal dynamic-value model.
