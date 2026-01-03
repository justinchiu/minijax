let float_eq a b =
  let eps = 1e-9 in
  abs_float (a -. b) < eps

module Tests (M : Minijax_api.S) = struct
  let float_exn = function
    | M.VFloat x -> x
    | _ -> assert false

  let run () =
    (* Eval interpreter *)
    let v_add = M.run M.eval_interpreter (M.add (M.VFloat 2.0) (M.VFloat 3.0)) in
    let v_mul = M.run M.eval_interpreter (M.mul (M.VFloat 2.0) (M.VFloat 3.0)) in
    assert (v_add = M.VFloat 5.0);
    assert (v_mul = M.VFloat 6.0);

    (* Eval foo example *)
    let v_foo = M.run M.eval_interpreter (M.foo (M.VFloat 2.0)) in
    assert (v_foo = M.VFloat 10.0);

    (* Finite difference for foo (approx) *)
    let x = 2.0 in
    let eps = 1.0e-5 in
    let f_eval a =
      M.run M.eval_interpreter (M.foo (M.VFloat a)) |> float_exn
    in
    let diff = (f_eval (x +. eps) -. f_eval x) /. eps in
    assert (abs_float (diff -. 7.000009999913458) < 1.0e-6);

    (* Primal-tangent packing example (approx) *)
    let v_pack = f_eval 2.00001 in
    assert (abs_float (v_pack -. 10.0000700001) < 1.0e-9);

    (* JVP for foo at x=2, tangent=1 -> (10, 7) *)
    let p, t = M.jvp (fun v -> M.foo v) (M.VFloat 2.0) (M.VFloat 1.0) in
    assert (p = M.VFloat 10.0);
    assert (t = M.VFloat 7.0);

    (* JVP for add and mul *)
    let p_add, t_add =
      M.jvp
        (fun v -> M.add v (M.VFloat 3.0))
        (M.VFloat 2.0)
        (M.VFloat 1.0)
    in
    assert (p_add = M.VFloat 5.0);
    assert (t_add = M.VFloat 1.0);

    let p_mul, t_mul =
      M.jvp
        (fun v -> M.mul v (M.VFloat 3.0))
        (M.VFloat 2.0)
        (M.VFloat 1.0)
    in
    assert (p_mul = M.VFloat 6.0);
    assert (t_mul = M.VFloat 3.0);

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
      let open M in
      { params = ["v_1"]
      ; equations =
          [ { var = "v_2"; op = Add; args = [VarAtom "v_1"; LitAtom 3.0] }
          ; { var = "v_3"; op = Mul; args = [VarAtom "v_1"; VarAtom "v_2"] }
          ]
      ; return_val = VarAtom "v_3"
      }
    in
    assert (jaxpr = expected);

    (* Eval jaxpr *)
    let result = M.eval_jaxpr M.eval_interpreter jaxpr [M.VFloat 2.0] in
    let result_value = float_exn result in
    assert (float_eq result_value 10.0);

    (* Perturbation confusion should pass with tagged dynamic JVP *)
    let f x =
      let g _ = M.pure x in
      let _p, should_be_zero = M.jvp g (M.VFloat 0.0) (M.VFloat 1.0) in
      M.mul x should_be_zero
    in
    let _p_conf, t_conf = M.jvp f (M.VFloat 0.0) (M.VFloat 1.0) in
    assert (t_conf = M.VFloat 0.0)
end

let () =
  let modules : (module Minijax_api.S) list =
    [ (module Minijax_global)
    ; (module Minijax)
    ; (module Minijax_reader)
    ]
  in
  List.iter
    (fun (module M : Minijax_api.S) ->
      let module T = Tests(M) in
      T.run ())
    modules
