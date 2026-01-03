(* MiniJax-ML (type-indexed tags): tagless-final with static JVP tags. *)

module type SYM = sig
  type t
  val add : t -> t -> t
  val mul : t -> t -> t
  val lit : float -> t
end

module Eval : SYM with type t = float = struct
  type t = float
  let add = ( +. )
  let mul = ( *. )
  let lit x = x
end

type ('tag, 'a) dual =
  { primal : 'a
  ; tangent : 'a
  }

module Jvp (Base : SYM) (Tag : sig type t end) : sig
  include SYM with type t = (Tag.t, Base.t) dual
  val dual : Base.t -> Base.t -> t
  val primal : t -> Base.t
  val tangent : t -> Base.t
end = struct
  type t = (Tag.t, Base.t) dual

  let dual primal tangent = { primal; tangent }
  let primal d = d.primal
  let tangent d = d.tangent

  let add x y =
    { primal = Base.add x.primal y.primal
    ; tangent = Base.add x.tangent y.tangent
    }

  let mul x y =
    { primal = Base.mul x.primal y.primal
    ; tangent =
        Base.add
          (Base.mul x.primal y.tangent)
          (Base.mul x.tangent y.primal)
    }

  let lit x = { primal = Base.lit x; tangent = Base.lit 0.0 }
end

module type PROG = functor (S : SYM) -> sig
  val run : S.t -> S.t
end

module type PACKED = sig
  module S : SYM
  val lift : float -> S.t
  val seed : S.t
  val input : float -> S.t
  val unpack : S.t -> float list
end

let last_exn xs =
  match List.rev xs with
  | [] -> invalid_arg "expected non-empty list"
  | x :: _ -> x

let rec build_packed n : (module PACKED) =
  if n = 0 then
    (module struct
      module S = Eval
      let lift x = x
      let seed = 1.0
      let input x = x
      let unpack x = [x]
    end)
  else
    let module B = (val build_packed (n - 1) : PACKED) in
    let module Tag = struct type t end in
    let module J = Jvp(B.S)(Tag) in
    (module struct
      module S = J
      let lift x = J.dual (B.lift x) (B.lift 0.0)
      let seed = J.dual B.seed (B.lift 0.0)
      let input x = J.dual (B.input x) B.seed
      let unpack v =
        let p_list = B.unpack (J.primal v) in
        let t_list = B.unpack (J.tangent v) in
        p_list @ [last_exn t_list]
    end)

let run_eval (module P : PROG) x =
  let module PE = P(Eval) in
  PE.run x

let jvp (module P : PROG) primal tangent =
  let module Tag = struct type t end in
  let module J = Jvp(Eval)(Tag) in
  let module PJ = P(J) in
  let out = PJ.run (J.dual primal tangent) in
  (J.primal out, J.tangent out)

let derivative p x =
  let _, t = jvp p x 1.0 in
  t

let jvp_n (module P : PROG) n x =
  let module Pack = (val build_packed n : PACKED) in
  let module PJ = P(Pack.S) in
  let out = PJ.run (Pack.input x) in
  Pack.unpack out

let jvp2 (module P : PROG) x =
  let module Tag1 = struct type t end in
  let module J1 = Jvp(Eval)(Tag1) in
  let module Tag2 = struct type t end in
  let module J2 = Jvp(J1)(Tag2) in
  let module PJ = P(J2) in
  let x1 = J1.dual x 1.0 in
  let x2 = J2.dual x1 (J1.dual 1.0 0.0) in
  let out = PJ.run x2 in
  let primal1 = J2.primal out in
  let tangent1 = J2.tangent out in
  let p = J1.primal primal1 in
  let t = J1.tangent primal1 in
  let t2 = J1.tangent tangent1 in
  (p, t, t2)
