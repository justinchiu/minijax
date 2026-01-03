(* MiniJax-ML (global interpreter): a minimal JAX-like interpreter in OCaml.
   Mirrors autodidax2.md with a global current interpreter. *)

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

(* Values that flow through the current interpreter. *)
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

type 'a reader = interpreter -> 'a

let float_of_value = function
  | VFloat x -> x
  | _ -> invalid_arg "expected VFloat"

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

let current_interpreter : interpreter ref = ref eval_interpreter

let set_interpreter new_interpreter f =
  let prev = !current_interpreter in
  current_interpreter := new_interpreter;
  match f () with
  | result ->
      current_interpreter := prev;
      result
  | exception exn ->
      current_interpreter := prev;
      raise exn

let add_raw x y = (!current_interpreter).interpret_op Add [x; y]
let mul_raw x y = (!current_interpreter).interpret_op Mul [x; y]

let run interp m = set_interpreter interp (fun () -> m interp)
let pure v : value reader = fun _ -> v
let add x y : value reader = fun _ -> add_raw x y
let mul x y : value reader = fun _ -> mul_raw x y

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
        let compute () =
          match op with
          | Add ->
              let p = add_raw dx.primal dy.primal in
              let t = add_raw dx.tangent dy.tangent in
              dual_number p t
          | Mul ->
              let p = mul_raw dx.primal dy.primal in
              let t1 = mul_raw dx.primal dy.tangent in
              let t2 = mul_raw dx.tangent dy.primal in
              let t = add_raw t1 t2 in
              dual_number p t
        in
        set_interpreter prev_interpreter compute
    | _ -> invalid_arg "jvp expects two args"
  and jvp_interpreter = { interpret_op } in
  (jvp_interpreter, dual_number, lift)

let jvp f primal tangent =
  let jvp_interpreter, dual_number, lift = make_jvp_interpreter eval_interpreter in
  let dual_in = dual_number primal tangent in
  let result = set_interpreter jvp_interpreter (fun () -> run jvp_interpreter (f dual_in)) in
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
  let result =
    set_interpreter stage_interpreter (fun () -> run stage_interpreter (f args))
  in
  { params; equations = st.equations; return_val = atom_of_value result }

let eval_jaxpr interp jaxpr args =
  set_interpreter interp (fun () ->
    let env = Hashtbl.create 16 in
    List.iter2 (fun v a -> Hashtbl.add env v a) jaxpr.params args;
    let eval_atom = function
      | VarAtom v -> Hashtbl.find env v
      | LitAtom x -> VFloat x
    in
    List.iter
      (fun eqn ->
        let arg_vals = List.map eval_atom eqn.args in
        let result = (!current_interpreter).interpret_op eqn.op arg_vals in
        Hashtbl.replace env eqn.var result)
      jaxpr.equations;
    eval_atom jaxpr.return_val)

(* Example function: foo(x) = x * (x + 3). *)
let foo x : value reader =
  fun _ -> mul_raw x (add_raw x (VFloat 3.0))
