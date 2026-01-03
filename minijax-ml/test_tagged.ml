open Minijax_tagged

let float_eq a b =
  let eps = 1e-9 in
  abs_float (a -. b) < eps

let () =
  let module Foo (S : SYM) = struct
    let run x = S.mul x (S.add x (S.lit 3.0))
  end in

  let module F (S : SYM) = struct
    let run x =
      let g _ = x in
      let _p, should_be_zero =
        jvp_fun (module S) g (S.lit 0.0) (S.lit 1.0)
      in
      S.mul x should_be_zero
  end in

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
  assert (float_eq t2 2.0);

  let jaxpr = build_jaxpr (module Foo) in
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

  let result = eval_jaxpr (module Eval) jaxpr [2.0] in
  assert (float_eq result 10.0);

  let _p_conf, t_conf = jvp (module F) 0.0 1.0 in
  assert (float_eq t_conf 0.0)
