open Minijax_global

let float_eq a b =
  let eps = 1e-9 in
  abs_float (a -. b) < eps

(* Test functions. *)
let foo x =
  mul x (add x (VFloat 3.0))

let g x _ = x

let f x =
  let should_be_zero =
    set_interpreter eval_interpreter (fun () -> derivative (g x) 0.0)
  in
  mul x should_be_zero

let () =
  (* Eval interpreter *)
  let v_add = set_interpreter eval_interpreter (fun () -> add (VFloat 2.0) (VFloat 3.0)) in
  let v_mul = set_interpreter eval_interpreter (fun () -> mul (VFloat 2.0) (VFloat 3.0)) in
  assert (v_add = VFloat 5.0);
  assert (v_mul = VFloat 6.0);

  (* Eval foo example *)
  let v_foo = set_interpreter eval_interpreter (fun () -> foo (VFloat 2.0)) in
  assert (v_foo = VFloat 10.0);

  (* Finite difference for foo (approx) *)
  let x = 2.0 in
  let eps = 1.0e-5 in
  let f_eval a =
    match set_interpreter eval_interpreter (fun () -> foo (VFloat a)) with
    | VFloat v -> v
    | _ -> assert false
  in
  let diff = (f_eval (x +. eps) -. f_eval x) /. eps in
  assert (abs_float (diff -. 7.000009999913458) < 1.0e-6);

  (* Primal-tangent packing example (approx) *)
  let v_pack = f_eval 2.00001 in
  assert (abs_float (v_pack -. 10.0000700001) < 1.0e-9);

  (* JVP for foo at x=2, tangent=1 -> (10, 7) *)
  let p, t =
    set_interpreter eval_interpreter (fun () ->
      jvp foo (VFloat 2.0) (VFloat 1.0))
  in
  assert (p = VFloat 10.0);
  assert (t = VFloat 7.0);

  (* JVP for add and mul *)
  let p_add, t_add =
    set_interpreter eval_interpreter (fun () ->
      jvp (fun x -> add x (VFloat 3.0)) (VFloat 2.0) (VFloat 1.0))
  in
  assert (p_add = VFloat 5.0);
  assert (t_add = VFloat 1.0);

  let p_mul, t_mul =
    set_interpreter eval_interpreter (fun () ->
      jvp (fun x -> mul x (VFloat 3.0)) (VFloat 2.0) (VFloat 1.0))
  in
  assert (p_mul = VFloat 6.0);
  assert (t_mul = VFloat 3.0);

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
  let result = set_interpreter eval_interpreter (fun () -> eval_jaxpr jaxpr [VFloat 2.0]) in
  let result_value =
    match result with
    | VFloat v -> v
    | _ -> 0.0
  in
  assert (match result with VFloat _ -> true | _ -> false);
  assert (float_eq result_value 10.0);

  (* Perturbation confusion should pass with tagged dynamic JVP *)
  let _p_conf, t_conf =
    set_interpreter eval_interpreter (fun () ->
      jvp f (VFloat 0.0) (VFloat 1.0))
  in
  assert (t_conf = VFloat 0.0);

  (* Higher-order derivatives of foo(x) = x*(x+3) = x^2 + 3x at x=2
     foo(2) = 10
     foo'(x) = 2x + 3, foo'(2) = 7
     foo''(x) = 2
     foo'''(x) = 0
     foo''''(x) = 0 *)

  (* 0th derivative (just evaluation) *)
  let d0 = nth_derivative 0 foo 2.0 in
  assert (float_eq d0 10.0);

  (* 1st derivative *)
  let d1 = nth_derivative 1 foo 2.0 in
  assert (float_eq d1 7.0);

  (* 2nd derivative *)
  let d2 = nth_derivative 2 foo 2.0 in
  assert (float_eq d2 2.0);

  (* 3rd derivative (foo is quadratic, so 0) *)
  let d3 = nth_derivative 3 foo 2.0 in
  assert (float_eq d3 0.0);

  (* 4th derivative *)
  let d4 = nth_derivative 4 foo 2.0 in
  assert (float_eq d4 0.0);

  Printf.printf "All tests passed!\n"
