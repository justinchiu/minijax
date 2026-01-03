(* MiniJax-ML: a minimal JAX-like interpreter in OCaml.

   Design: Tagless Final with Type-Level Tags
   ==========================================
   This implementation uses OCaml's module system (tagless final style) combined
   with type-level tags to avoid perturbation confusion in higher-order AD.

   Key ideas:

   1. TAGLESS FINAL: Operations are defined by a module signature (SYM):

        module type SYM = sig
          type t
          val add : t -> t -> t
          val mul : t -> t -> t
          val lit : float -> t
        end

      Different interpreters implement this signature with different `t`:
      - Eval: t = float
      - JVP: t = (Tag.t, Base.t) dual

   2. TYPE-LEVEL TAGS: Each JVP layer introduces a fresh phantom type Tag.t:

        module Jvp (Base : SYM) (Tag : sig type t end) : sig
          include SYM with type t = (Tag.t, Base.t) dual
          ...
        end

      The phantom type tag ensures dual numbers from different differentiation
      contexts cannot be mixed, preventing perturbation confusion at compile time.

   3. NESTED DIFFERENTIATION: Higher-order AD works by stacking JVP modules:

        module J1 = Jvp(Eval)(Tag1)      (* first derivative *)
        module J2 = Jvp(J1)(Tag2)        (* second derivative *)

   Advantages:
   - Type-safe: perturbation confusion is a type error
   - No runtime overhead for tag checking
   - Elegant use of OCaml's module system

   Disadvantages:
   - More complex types
   - Programs must be polymorphic over the SYM functor *)

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

(* Staging interpreter. *)
type stage_state =
  { mutable equations : equation list
  ; mutable name_counter : int
  }

let fresh_var st =
  st.name_counter <- st.name_counter + 1;
  "v_" ^ string_of_int st.name_counter

let make_stage_sym st : (module SYM with type t = atom) =
  (module struct
    type t = atom
    let lit x = LitAtom x

    let add x y =
      let var = fresh_var st in
      st.equations <- st.equations @ [{ var; op = Add; args = [x; y] }];
      VarAtom var

    let mul x y =
      let var = fresh_var st in
      st.equations <- st.equations @ [{ var; op = Mul; args = [x; y] }];
      VarAtom var
  end)

let build_jaxpr (module P : PROG) =
  let st = { equations = []; name_counter = 0 } in
  let param = fresh_var st in
  let module S = (val make_stage_sym st : SYM with type t = atom) in
  let module PS = P(S) in
  let result = PS.run (VarAtom param) in
  { params = [param]; equations = st.equations; return_val = result }

let eval_jaxpr (type a) (module S : SYM with type t = a) jaxpr args =
  let env = Hashtbl.create 16 in
  List.iter2 (fun v a -> Hashtbl.add env v a) jaxpr.params args;
  let eval_atom = function
    | VarAtom v -> Hashtbl.find env v
    | LitAtom x -> S.lit x
  in
  List.iter
    (fun eqn ->
      let arg_vals = List.map eval_atom eqn.args in
      let result =
        match eqn.op, arg_vals with
        | Add, [x; y] -> S.add x y
        | Mul, [x; y] -> S.mul x y
        | _ -> invalid_arg "jaxpr expects two args"
      in
      Hashtbl.replace env eqn.var result)
    jaxpr.equations;
  eval_atom jaxpr.return_val
