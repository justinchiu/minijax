(* Minimal tests for minijax-ml. *)

let float_eq a b =
  let eps = 1e-9 in
  abs_float (a -. b) < eps

module type TESTABLE = sig
  type op
  type atom
  type equation
  type jaxpr
  type value
  type program

  val name : string
  val vfloat : float -> value
  val float_of_value : value -> float option
  val pure : value -> program
  val add : value -> value -> program
  val mul : value -> value -> program
  val foo : value -> program
  val run_eval : program -> value
  val jvp : (value -> program) -> value -> value -> value * value
  val build_jaxpr : (value list -> program) -> int -> jaxpr
  val eval_jaxpr : jaxpr -> value list -> value
  val op_add : op
  val op_mul : op
  val atom_var : string -> atom
  val atom_lit : float -> atom
  val equation : string -> op -> atom list -> equation
  val jaxpr : string list -> equation list -> atom -> jaxpr
end

module Global : TESTABLE = struct
  type op = Minijax_global.op
  type atom = Minijax_global.atom
  type equation = Minijax_global.equation
  type jaxpr = Minijax_global.jaxpr
  type value = Minijax_global.value
  type program = Minijax_global.interpreter -> value

  let name = "global"
  let vfloat x = Minijax_global.VFloat x

  let float_of_value = function
    | Minijax_global.VFloat x -> Some x
    | _ -> None

  let pure v = fun _ -> v
  let add x y = fun _ -> Minijax_global.add x y
  let mul x y = fun _ -> Minijax_global.mul x y
  let foo x = fun _ -> Minijax_global.foo x

  let run_eval prog =
    Minijax_global.set_interpreter Minijax_global.eval_interpreter (fun () ->
      prog Minijax_global.eval_interpreter)

  let jvp f primal tangent =
    Minijax_global.jvp
      (fun v ->
        let prog = f v in
        prog Minijax_global.eval_interpreter)
      primal
      tangent

  let build_jaxpr f num_args =
    Minijax_global.build_jaxpr
      (fun args ->
        let prog = f args in
        prog Minijax_global.eval_interpreter)
      num_args

  let eval_jaxpr = Minijax_global.eval_jaxpr
  let op_add = Minijax_global.Add
  let op_mul = Minijax_global.Mul
  let atom_var v = Minijax_global.VarAtom v
  let atom_lit x = Minijax_global.LitAtom x
  let equation var op args = { Minijax_global.var = var; op; args }
  let jaxpr params equations return_val =
    { Minijax_global.params = params; equations; return_val }
end

module ReaderStyle : TESTABLE = struct
  type op = Minijax.op
  type atom = Minijax.atom
  type equation = Minijax.equation
  type jaxpr = Minijax.jaxpr
  type value = Minijax.value
  type program = Minijax.interpreter -> value

  let name = "reader-style"
  let vfloat x = Minijax.VFloat x

  let float_of_value = function
    | Minijax.VFloat x -> Some x
    | _ -> None

  let pure v = fun _ -> v
  let add x y = fun interp -> Minijax.add interp x y
  let mul x y = fun interp -> Minijax.mul interp x y
  let foo x = fun interp -> Minijax.foo interp x
  let run_eval prog = prog Minijax.eval_interpreter

  let jvp f primal tangent =
    Minijax.jvp
      ~base_interpreter:Minijax.eval_interpreter
      (fun interp v -> (f v) interp)
      primal
      tangent

  let build_jaxpr f num_args =
    Minijax.build_jaxpr
      (fun interp args -> (f args) interp)
      num_args

  let eval_jaxpr jaxpr args =
    Minijax.eval_jaxpr Minijax.eval_interpreter jaxpr args

  let op_add = Minijax.Add
  let op_mul = Minijax.Mul
  let atom_var v = Minijax.VarAtom v
  let atom_lit x = Minijax.LitAtom x
  let equation var op args = { Minijax.var = var; op; args }
  let jaxpr params equations return_val =
    { Minijax.params = params; equations; return_val }
end

module ReaderMonad : TESTABLE = struct
  type op = Minijax_reader.op
  type atom = Minijax_reader.atom
  type equation = Minijax_reader.equation
  type jaxpr = Minijax_reader.jaxpr
  type value = Minijax_reader.value
  type program = value Minijax_reader.reader

  let name = "reader-monad"
  let vfloat x = Minijax_reader.VFloat x

  let float_of_value = function
    | Minijax_reader.VFloat x -> Some x
    | _ -> None

  let pure v = Minijax_reader.return v
  let add = Minijax_reader.add
  let mul = Minijax_reader.mul
  let foo = Minijax_reader.foo
  let run_eval prog = Minijax_reader.run Minijax_reader.eval_interpreter prog

  let jvp f primal tangent =
    Minijax_reader.jvp ~base_interpreter:Minijax_reader.eval_interpreter f primal tangent

  let build_jaxpr = Minijax_reader.build_jaxpr
  let eval_jaxpr jaxpr args =
    Minijax_reader.eval_jaxpr Minijax_reader.eval_interpreter jaxpr args

  let op_add = Minijax_reader.Add
  let op_mul = Minijax_reader.Mul
  let atom_var v = Minijax_reader.VarAtom v
  let atom_lit x = Minijax_reader.LitAtom x
  let equation var op args = { Minijax_reader.var = var; op; args }
  let jaxpr params equations return_val =
    { Minijax_reader.params = params; equations; return_val }
end

module Tests (M : TESTABLE) = struct
  let float_exn = function
    | Some x -> x
    | None -> assert false

  let run () =
    (* Eval interpreter *)
    let v_add = M.run_eval (M.add (M.vfloat 2.0) (M.vfloat 3.0)) in
    let v_mul = M.run_eval (M.mul (M.vfloat 2.0) (M.vfloat 3.0)) in
    assert (v_add = M.vfloat 5.0);
    assert (v_mul = M.vfloat 6.0);

    (* Eval foo example *)
    let v_foo = M.run_eval (M.foo (M.vfloat 2.0)) in
    assert (v_foo = M.vfloat 10.0);

    (* Finite difference for foo (approx) *)
    let x = 2.0 in
    let eps = 1.0e-5 in
    let f_eval a =
      M.run_eval (M.foo (M.vfloat a)) |> M.float_of_value |> float_exn
    in
    let diff = (f_eval (x +. eps) -. f_eval x) /. eps in
    assert (abs_float (diff -. 7.000009999913458) < 1.0e-6);

    (* Primal-tangent packing example (approx) *)
    let v_pack = f_eval 2.00001 in
    assert (abs_float (v_pack -. 10.0000700001) < 1.0e-9);

    (* JVP for foo at x=2, tangent=1 -> (10, 7) *)
    let p, t = M.jvp (fun v -> M.foo v) (M.vfloat 2.0) (M.vfloat 1.0) in
    assert (p = M.vfloat 10.0);
    assert (t = M.vfloat 7.0);

    (* JVP for add and mul *)
    let p_add, t_add =
      M.jvp
        (fun v -> M.add v (M.vfloat 3.0))
        (M.vfloat 2.0)
        (M.vfloat 1.0)
    in
    assert (p_add = M.vfloat 5.0);
    assert (t_add = M.vfloat 1.0);

    let p_mul, t_mul =
      M.jvp
        (fun v -> M.mul v (M.vfloat 3.0))
        (M.vfloat 2.0)
        (M.vfloat 1.0)
    in
    assert (p_mul = M.vfloat 6.0);
    assert (t_mul = M.vfloat 3.0);

    (* Staging foo into a jaxpr *)
    let jaxpr =
      M.build_jaxpr
        (fun args ->
          match args with
          | [x] -> M.foo x
          | _ -> failwith "expected one arg")
        1
    in
    let expected =
      M.jaxpr
        ["v_1"]
        [ M.equation "v_2" M.op_add [M.atom_var "v_1"; M.atom_lit 3.0]
        ; M.equation "v_3" M.op_mul [M.atom_var "v_1"; M.atom_var "v_2"]
        ]
        (M.atom_var "v_3")
    in
    assert (jaxpr = expected);

    (* Eval jaxpr *)
    let result = M.eval_jaxpr jaxpr [M.vfloat 2.0] in
    assert (match M.float_of_value result with Some _ -> true | None -> false);
    let result_value = result |> M.float_of_value |> float_exn in
    assert (float_eq result_value 10.0);

    (* Perturbation confusion should pass with tagged dynamic JVP *)
    let f x =
      let g _ = M.pure x in
      let _p, should_be_zero = M.jvp g (M.vfloat 0.0) (M.vfloat 1.0) in
      M.mul x should_be_zero
    in
    let _p_conf, t_conf = M.jvp f (M.vfloat 0.0) (M.vfloat 1.0) in
    assert (t_conf = M.vfloat 0.0)
end

let () =
  let modules : (module TESTABLE) list =
    [ (module Global); (module ReaderStyle); (module ReaderMonad) ]
  in
  List.iter
    (fun (module M : TESTABLE) ->
      ignore M.name;
      let module T = Tests(M) in
      T.run ())
    modules
