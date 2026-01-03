(* MiniJax-ML: a minimal JAX-like interpreter in OCaml.

   Design: GADT Expression AST
   ===========================
   This implementation uses a GADT (Generalized Algebraic Data Type) to represent
   expressions as an explicit AST:

     type _ expr =
       | Lit : float -> float expr
       | Var : float expr
       | Add : float expr * float expr -> float expr
       | Mul : float expr * float expr -> float expr

   Programs are data structures that can be interpreted in different ways:

     let foo_expr = Mul (Var, Add (Var, Lit 3.0))

   The interpreter record provides different semantics:

     type interpreter =
       { add : value -> value -> value
       ; mul : value -> value -> value
       ; lit : float -> value
       }

   To evaluate: `eval foo_expr ~x:2.0` traverses the AST, dispatching operations
   to the current interpreter.

   Comparison to Other Approaches:
   - Unlike tagless final (minijax_tagged.ml), the AST is explicit data
   - Unlike dynamic dispatch (minijax_global.ml), programs are static structures
   - Staging is trivial: just walk the AST and generate IR

   Advantages:
   - AST can be inspected, optimized, serialized
   - Easy to implement pattern-matching transformations
   - Type parameter ensures well-typed expressions

   Disadvantages:
   - Fixed set of operations (closed world)
   - Adding new interpreters is easy; adding new operations requires AST changes *)

type op = Add | Mul

(* IR types *)
type var = string

type atom =
  | VarAtom of var
  | LitAtom of float

type equation =
  { var : var
  ; op : op
  ; args : atom list
  }

type jaxpr =
  { params : var list
  ; equations : equation list
  ; return_val : atom
  }

type _ expr =
  | Lit : float -> float expr
  | Var : float expr
  | Add : float expr * float expr -> float expr
  | Mul : float expr * float expr -> float expr

(* Values that flow through an interpreter. *)
type value =
  | VFloat of float
  | VDual of dual

and dual =
  { interpreter : interpreter
  ; primal : value
  ; tangent : value
  }

and interpreter =
  { add : value -> value -> value
  ; mul : value -> value -> value
  ; lit : float -> value
  }

let float_of_value = function
  | VFloat x -> x
  | _ -> invalid_arg "expected VFloat"

let rec zero_like = function
  | VFloat _ -> VFloat 0.0
  | VDual d ->
      let zp = zero_like d.primal in
      let zt = zero_like d.tangent in
      VDual { interpreter = d.interpreter; primal = zp; tangent = zt }

let add interp x y = interp.add x y
let mul interp x y = interp.mul x y

(* Eval interpreter. *)
let eval_interpreter =
  let add_value x y =
    match x, y with
    | VFloat a, VFloat b -> VFloat (a +. b)
    | _ -> invalid_arg "eval expects two VFloat args"
  in
  let mul_value x y =
    match x, y with
    | VFloat a, VFloat b -> VFloat (a *. b)
    | _ -> invalid_arg "eval expects two VFloat args"
  in
  { add = add_value; mul = mul_value; lit = (fun x -> VFloat x) }

let rec eval_with interp expr x =
  match expr with
  | Lit v -> interp.lit v
  | Var -> x
  | Add (a, b) -> add interp (eval_with interp a x) (eval_with interp b x)
  | Mul (a, b) -> mul interp (eval_with interp a x) (eval_with interp b x)

let eval expr ~x =
  match eval_with eval_interpreter expr (VFloat x) with
  | VFloat v -> v
  | _ -> invalid_arg "eval produced non-float value"

(* JVP interpreter (tagged, dynamic). *)
let make_jvp_interpreter prev_interpreter =
  let rec dual_number primal tangent =
    VDual { interpreter = jvp_interpreter; primal; tangent }
  and lift v =
    match v with
    | VDual d when d.interpreter == jvp_interpreter -> d
    | _ -> { interpreter = jvp_interpreter; primal = v; tangent = zero_like v }
  and add_value x y =
    let dx = lift x in
    let dy = lift y in
    let p = add prev_interpreter dx.primal dy.primal in
    let t = add prev_interpreter dx.tangent dy.tangent in
    dual_number p t
  and mul_value x y =
    let dx = lift x in
    let dy = lift y in
    let p = mul prev_interpreter dx.primal dy.primal in
    let t1 = mul prev_interpreter dx.primal dy.tangent in
    let t2 = mul prev_interpreter dx.tangent dy.primal in
    let t = add prev_interpreter t1 t2 in
    dual_number p t
  and jvp_interpreter = { add = add_value; mul = mul_value; lit = prev_interpreter.lit } in
  (jvp_interpreter, dual_number, lift)

let jvp ~base_interpreter expr primal tangent =
  let jvp_interpreter, dual_number, lift = make_jvp_interpreter base_interpreter in
  let dual_in = dual_number primal tangent in
  let result = eval_with jvp_interpreter expr dual_in in
  let dual_out = lift result in
  (dual_out.primal, dual_out.tangent)

let derivative ~base_interpreter expr x =
  let _p, t = jvp ~base_interpreter expr (VFloat x) (VFloat 1.0) in
  t

let jvp_fun ~base_interpreter f primal tangent =
  let jvp_interpreter, dual_number, lift = make_jvp_interpreter base_interpreter in
  let dual_in = dual_number primal tangent in
  let result = f jvp_interpreter dual_in in
  let dual_out = lift result in
  (dual_out.primal, dual_out.tangent)

let derivative_fun ~base_interpreter f x =
  let _p, t =
    jvp_fun ~base_interpreter f (VFloat x) (VFloat 1.0)
  in
  t

(* Staging interpreter. *)
type stage_state =
  { mutable equations : equation list
  ; mutable name_counter : int
  }

let fresh_var st =
  st.name_counter <- st.name_counter + 1;
  "v_" ^ string_of_int st.name_counter

let build_jaxpr expr =
  let st = { equations = []; name_counter = 1 } in
  let param = "v_1" in
  let rec stage = function
    | Lit x -> LitAtom x
    | Var -> VarAtom param
    | Add (a, b) ->
        let lhs = stage a in
        let rhs = stage b in
        let var = fresh_var st in
        st.equations <- st.equations @ [{ var; op = Add; args = [lhs; rhs] }];
        VarAtom var
    | Mul (a, b) ->
        let lhs = stage a in
        let rhs = stage b in
        let var = fresh_var st in
        st.equations <- st.equations @ [{ var; op = Mul; args = [lhs; rhs] }];
        VarAtom var
  in
  let return_val = stage expr in
  { params = [param]; equations = st.equations; return_val }

let eval_jaxpr interp jaxpr args =
  let env = Hashtbl.create 16 in
  List.iter2 (fun v a -> Hashtbl.add env v a) jaxpr.params args;
  let eval_atom = function
    | VarAtom v -> Hashtbl.find env v
    | LitAtom x -> interp.lit x
  in
  List.iter
    (fun eqn ->
      let arg_vals = List.map eval_atom eqn.args in
      let result =
        match eqn.op, arg_vals with
        | Add, [x; y] -> add interp x y
        | Mul, [x; y] -> mul interp x y
        | _ -> invalid_arg "jaxpr expects two args"
      in
      Hashtbl.replace env eqn.var result)
    jaxpr.equations;
  eval_atom jaxpr.return_val

let foo_expr =
  Mul (Var, Add (Var, Lit 3.0))
