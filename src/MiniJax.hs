module MiniJax where

data Op = Add | Mul
  deriving (Show, Eq)

type Var = String

data Atom = VarAtom Var | LitAtom Float
  deriving (Show, Eq)
