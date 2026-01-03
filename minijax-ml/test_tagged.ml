open Minijax_tagged

let float_eq a b =
  let eps = 1e-9 in
  abs_float (a -. b) < eps

module Foo (S : SYM) = struct
  let run x = S.mul x (S.add x (S.lit 3.0))
end

let () =
  let v = run_eval (module Foo) 2.0 in
  assert (float_eq v 10.0);

  let p, t = jvp (module Foo) 2.0 1.0 in
  assert (float_eq p 10.0);
  assert (float_eq t 7.0);

  let d = derivative (module Foo) 2.0 in
  assert (float_eq d 7.0);

  let ds = jvp_n (module Foo) 2 2.0 in
  assert (ds = [10.0; 7.0; 2.0]);

  let p2, t1, t2 = jvp2 (module Foo) 2.0 in
  assert (float_eq p2 10.0);
  assert (float_eq t1 7.0);
  assert (float_eq t2 2.0)
