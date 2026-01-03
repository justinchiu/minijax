(* MiniJax-ML (reader monad with let-star): a minimal JAX-like interpreter in OCaml.
   Mirrors autodidax2.md, passing interpreter via a Reader monad. *)

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

(* Values that flow through an interpreter. *)
type value =
  | VFloat of float
  | VAtom of atom
  | VDual of dual

and dual =
  { interpreter : interpreter
  ; primal : value
  ; tangent : value
  }

and interpreter =
  { interpret_op : op -> value list -> value
  }

let float_of_value = function
  | VFloat x -> x
  | _ -> invalid_arg "expected VFloat"

let primal d = d.primal
let tangent d = d.tangent

let atom_of_value = function
  | VAtom a -> a
  | VFloat x -> LitAtom x
  | VDual _ -> invalid_arg "cannot stage a dual number"

let rec zero_like = function
  | VFloat _ -> VFloat 0.0
  | VAtom _ -> VAtom (LitAtom 0.0)
  | VDual d ->
      let zp = zero_like d.primal in
      let zt = zero_like d.tangent in
      VDual { interpreter = d.interpreter; primal = zp; tangent = zt }

(* Reader monad. *)
type 'a reader = interpreter -> 'a

let return x : 'a reader = fun _ -> x
let pure v : value reader = fun _ -> v

let bind (m : 'a reader) (f : 'a -> 'b reader) : 'b reader =
  fun interp -> f (m interp) interp

let ( let* ) = bind

let run interp m = m interp

(* Eval interpreter. *)
let eval_interpreter =
  let eval_binop op args =
    match args with
    | [VFloat x; VFloat y] ->
        begin
          match op with
          | Add -> VFloat (x +. y)
          | Mul -> VFloat (x *. y)
        end
    | _ -> invalid_arg "eval expects two VFloat args"
  in
  { interpret_op = eval_binop }

let add x y : value reader =
  fun interp -> interp.interpret_op Add [x; y]

let mul x y : value reader =
  fun interp -> interp.interpret_op Mul [x; y]

(* JVP interpreter (tagged, dynamic). *)
let make_jvp_interpreter prev_interpreter =
  let rec dual_number primal tangent =
    VDual { interpreter = jvp_interpreter; primal; tangent }
  and lift v =
    match v with
    | VDual d when d.interpreter == jvp_interpreter -> d
    | _ -> { interpreter = jvp_interpreter; primal = v; tangent = zero_like v }
  and interpret_op op args =
    match args with
    | [x; y] ->
        let dx = lift x in
        let dy = lift y in
        begin
          match op with
          | Add ->
              let p = prev_interpreter.interpret_op Add [dx.primal; dy.primal] in
              let t = prev_interpreter.interpret_op Add [dx.tangent; dy.tangent] in
              dual_number p t
          | Mul ->
              let p = prev_interpreter.interpret_op Mul [dx.primal; dy.primal] in
              let t1 = prev_interpreter.interpret_op Mul [dx.primal; dy.tangent] in
              let t2 = prev_interpreter.interpret_op Mul [dx.tangent; dy.primal] in
              let t = prev_interpreter.interpret_op Add [t1; t2] in
              dual_number p t
        end
    | _ -> invalid_arg "jvp expects two args"
  and jvp_interpreter = { interpret_op } in
  (jvp_interpreter, dual_number, lift)

let jvp f primal tangent =
  let jvp_interpreter, dual_number, lift = make_jvp_interpreter eval_interpreter in
  let dual_in = dual_number primal tangent in
  let result = run jvp_interpreter (f dual_in) in
  let dual_out = lift result in
  (dual_out.primal, dual_out.tangent)

let derivative f x =
  let p, t = jvp f (VFloat x) (VFloat 1.0) in
  ignore p;
  t

(* Staging interpreter. *)
type stage_state =
  { mutable equations : equation list
  ; mutable name_counter : int
  }

let fresh_var st =
  st.name_counter <- st.name_counter + 1;
  "v_" ^ string_of_int st.name_counter

let make_stage_interpreter st =
  let interpret_op op args =
    let var = fresh_var st in
    let atoms = List.map atom_of_value args in
    st.equations <- st.equations @ [{ var; op; args = atoms }];
    VAtom (VarAtom var)
  in
  { interpret_op }

let build_jaxpr f num_args =
  let st = { equations = []; name_counter = 0 } in
  let params =
    List.init num_args (fun _ ->
      st.name_counter <- st.name_counter + 1;
      "v_" ^ string_of_int st.name_counter)
  in
  let args = List.map (fun v -> VAtom (VarAtom v)) params in
  let stage_interpreter = make_stage_interpreter st in
  let result = run stage_interpreter (f args) in
  { params; equations = st.equations; return_val = atom_of_value result }

let eval_jaxpr interp jaxpr args =
  let env = Hashtbl.create 16 in
  List.iter2 (fun v a -> Hashtbl.add env v a) jaxpr.params args;
  let eval_atom = function
    | VarAtom v -> Hashtbl.find env v
    | LitAtom x -> VFloat x
  in
  List.iter
    (fun eqn ->
      let arg_vals = List.map eval_atom eqn.args in
      let result = interp.interpret_op eqn.op arg_vals in
      Hashtbl.replace env eqn.var result)
    jaxpr.equations;
  eval_atom jaxpr.return_val

(* Example function: foo(x) = x * (x + 3). *)
let foo x : value reader =
  let* y = add x (VFloat 3.0) in
  mul x y
