module type S = sig
  type op = Add | Mul
  type var = string
  type atom = VarAtom of var | LitAtom of float
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
  type value = VFloat of float | VAtom of atom | VDual of dual
  and dual =
    { interpreter : interpreter
    ; primal : value
    ; tangent : value
    }
  and interpreter =
    { interpret_op : op -> value list -> value
    }
  type 'a reader = interpreter -> 'a

  val eval_interpreter : interpreter
  val run : interpreter -> 'a reader -> 'a
  val pure : value -> value reader
  val add : value -> value -> value reader
  val mul : value -> value -> value reader
  val jvp : (value -> value reader) -> value -> value -> value * value
  val build_jaxpr : (value list -> value reader) -> int -> jaxpr
  val eval_jaxpr : interpreter -> jaxpr -> value list -> value
  val foo : value -> value reader
end
