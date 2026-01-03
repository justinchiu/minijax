(* MiniJax-ML (GADT AST): a typed expression language with eval and JVP. *)

type _ expr =
  | Lit : float -> float expr
  | Var : float expr
  | Add : float expr * float expr -> float expr
  | Mul : float expr * float expr -> float expr

let rec eval expr ~x =
  match expr with
  | Lit v -> v
  | Var -> x
  | Add (a, b) -> eval a ~x +. eval b ~x
  | Mul (a, b) -> eval a ~x *. eval b ~x

type dual =
  { primal : float
  ; tangent : float
  }

let rec jvp expr ~primal ~tangent =
  match expr with
  | Lit v -> { primal = v; tangent = 0.0 }
  | Var -> { primal; tangent }
  | Add (a, b) ->
      let da = jvp a ~primal ~tangent in
      let db = jvp b ~primal ~tangent in
      { primal = da.primal +. db.primal; tangent = da.tangent +. db.tangent }
  | Mul (a, b) ->
      let da = jvp a ~primal ~tangent in
      let db = jvp b ~primal ~tangent in
      { primal = da.primal *. db.primal
      ; tangent = da.primal *. db.tangent +. da.tangent *. db.primal
      }

let derivative expr x =
  let d = jvp expr ~primal:x ~tangent:1.0 in
  d.tangent

let foo_expr =
  Mul (Var, Add (Var, Lit 3.0))
