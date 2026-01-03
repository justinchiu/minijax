open Minijax_gadt

let float_eq a b =
  let eps = 1e-9 in
  abs_float (a -. b) < eps

let foo_expr =
  Mul (Var, Add (Var, Lit 3.0))

let foo x = eval foo_expr ~x

(* Test functions. *)
let g x _ = x

let f interp x =
  let g_x = g x in
  let _p, should_be_zero =
    jvp_fun
      ~base_interpreter:eval_interpreter
      (fun _ v -> g_x v)
      (VFloat 0.0)
      (VFloat 1.0)
  in
  mul interp x should_be_zero

let () =
  let v = foo 2.0 in
  assert (float_eq v 10.0);

  let p, t =
    jvp ~base_interpreter:eval_interpreter foo_expr (VFloat 2.0) (VFloat 1.0)
  in
  assert (float_eq (float_of_value p) 10.0);
  assert (float_eq (float_of_value t) 7.0);

  let d1 = derivative ~base_interpreter:eval_interpreter foo_expr 2.0 in
  assert (float_eq (float_of_value d1) 7.0);

  let eps = 1.0e-5 in
  let d2 =
    derivative ~base_interpreter:eval_interpreter foo_expr (2.0 +. eps)
  in
  let d2_approx = (float_of_value d2 -. float_of_value d1) /. eps in
  assert (abs_float (d2_approx -. 2.0) < 1.0e-4);

  let jaxpr = build_jaxpr foo_expr in
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

  let result = eval_jaxpr eval_interpreter jaxpr [VFloat 2.0] in
  assert (float_eq (float_of_value result) 10.0);

  let _p_conf, t_conf =
    jvp_fun ~base_interpreter:eval_interpreter f (VFloat 0.0) (VFloat 1.0)
  in
  assert (float_eq (float_of_value t_conf) 0.0)
