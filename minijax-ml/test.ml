(* Minimal tests for minijax-ml. *)

#use "minijax.ml";;

let float_eq a b =
  let eps = 1e-9 in
  abs_float (a -. b) < eps

let () =
  (* Eval interpreter *)
  let v_add = set_interpreter eval_interpreter (fun () -> add (VFloat 2.0) (VFloat 3.0)) in
  let v_mul = set_interpreter eval_interpreter (fun () -> mul (VFloat 2.0) (VFloat 3.0)) in
  assert (v_add = VFloat 5.0);
  assert (v_mul = VFloat 6.0);

  (* JVP for foo at x=2, tangent=1 -> (10, 7) *)
  let p, t = jvp foo (VFloat 2.0) (VFloat 1.0) in
  assert (p = VFloat 10.0);
  assert (t = VFloat 7.0);

  (* Staging foo into a jaxpr *)
  let jaxpr =
    build_jaxpr
      (fun args ->
        match args with
        | [x] -> foo x
        | _ -> failwith "expected one arg")
      1
  in
  let expected =
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
  let result = eval_jaxpr jaxpr [VFloat 2.0] in
  match result with
  | VFloat x -> assert (float_eq x 10.0)
  | _ -> assert false
