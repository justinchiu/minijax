module MiniJax where

data Op = Add | Mul
  deriving (Show, Eq)

type Var = String

data Atom = VarAtom Var | LitAtom Float
  deriving (Show, Eq)

data Equation = Equation
  { getVar :: Var
  , getOp :: Op
  , getArgs :: [Atom]
  } deriving Show

data Jaxpr = Jaxpr
  { getParams :: [Var]
  , getEquations :: [Equation]
  , getReturn :: Atom
  } deriving Show

data Intepreter
  = EvalInterpreter
  | MockInterpreter
  | JVPInterpreter


add :: Float -> Float -> Float
add = (+)
mul :: Float -> Float -> Float
mul = (*)



