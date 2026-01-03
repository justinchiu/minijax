-- | Common types shared across MiniJax interpreters.
--
-- This module defines the core data structures used throughout the library:
-- primitive operations, dual numbers for AD, the Jaxpr IR representation,
-- and the expression AST.
module MiniJax.Common
  ( Op(..)
  , Var
  , Atom(..)
  , Dual(..)
  , Equation(..)
  , Jaxpr(..)
  , Expr(..)
  ) where

-- | Primitive operations
data Op = Add | Mul
  deriving (Show, Eq)

-- | Variables are strings
type Var = String

-- | Atoms are either variables or literals
data Atom = VarAtom Var | LitAtom Double
  deriving (Show, Eq)

-- | Dual numbers for automatic differentiation
data Dual = Dual
  { primal :: Double
  , tangent :: Double
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

-- | Expression AST for the REPL
data Expr
  = Lit Double       -- ^ Literal value
  | EAdd Expr Expr   -- ^ Addition
  | EMul Expr Expr   -- ^ Multiplication
  deriving (Show, Eq)
