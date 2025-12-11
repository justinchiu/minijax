{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module MiniJax.Tagless where

import MiniJax.Common
import Control.Monad.State
import Control.Monad.Identity

-- | The tagless final encoding of our operations
class Monad m => JaxSym m where
  add :: Float -> Float -> m Float
  mul :: Float -> Float -> m Float

-- | The example function from autodidax2
foo :: JaxSym m => Float -> m Float
foo x = do
  y <- add x 3.0
  mul x y

-- | Evaluation interpreter - just performs arithmetic
newtype Eval a = Eval { unEval :: Identity a }
  deriving (Functor, Applicative, Monad)

runEval :: Eval a -> a
runEval = runIdentity . unEval

instance JaxSym Eval where
  add x y = return (x + y)
  mul x y = return (x * y)

-- | Interpret an AST using tagless final
interpret :: JaxSym m => Expr -> m Float
interpret (Lit x) = return x
interpret (EAdd e1 e2) = do
  x <- interpret e1
  y <- interpret e2
  add x y
interpret (EMul e1 e2) = do
  x <- interpret e1
  y <- interpret e2
  mul x y

-- | Placeholder for JVP interpreter
newtype JVP a = JVP (Identity a)
  deriving (Functor, Applicative, Monad)

runJVP :: JVP Dual -> (Float, Float)
runJVP = error "JVP not implemented yet"

instance JaxSym JVP where
  add _ _ = error "JVP add not implemented yet"
  mul _ _ = error "JVP mul not implemented yet"

-- | Placeholder for Stage interpreter
newtype Stage a = Stage (State StageState a)
  deriving (Functor, Applicative, Monad)

data StageState = StageState
  { equations :: [Equation]
  , nameCounter :: Int
  }

runStage :: (Var -> Stage Var) -> Jaxpr
runStage = error "Stage not implemented yet"

instance JaxSym Stage where
  add _ _ = error "Stage add not implemented yet"
  mul _ _ = error "Stage mul not implemented yet"