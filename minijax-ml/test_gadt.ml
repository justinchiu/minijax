open Minijax_gadt

let float_eq a b =
  let eps = 1e-9 in
  abs_float (a -. b) < eps

let () =
  let v = eval foo_expr ~x:2.0 in
  assert (float_eq v 10.0);

  let d = jvp foo_expr ~primal:2.0 ~tangent:1.0 in
  assert (float_eq d.primal 10.0);
  assert (float_eq d.tangent 7.0);

  let d1 = derivative foo_expr 2.0 in
  assert (float_eq d1 7.0);

  let eps = 1.0e-5 in
  let d2_approx = (derivative foo_expr (2.0 +. eps) -. d1) /. eps in
  assert (abs_float (d2_approx -. 2.0) < 1.0e-4)
