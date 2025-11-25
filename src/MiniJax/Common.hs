module MiniJax.Common where

-- | Primitive operations
data Op = Add | Mul
  deriving (Show, Eq)

-- | Variables are strings
type Var = String

-- | Atoms are either variables or literals
data Atom = VarAtom Var | LitAtom Float
  deriving (Show, Eq)

-- | Dual numbers for automatic differentiation
data Dual = Dual
  { primal :: Float
  , tangent :: Float
  } deriving (Show, Eq)

-- | Equation in the IR
data Equation = Equation
  { getVar :: Var
  , getOp :: Op
  , getArgs :: [Atom]
  } deriving (Show, Eq)

-- | Jaxpr - the IR representation
data Jaxpr = Jaxpr
  { getParams :: [Var]
  , getEquations :: [Equation]
  , getReturn :: Atom
  } deriving (Show, Eq)