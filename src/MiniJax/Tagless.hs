{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeFamilies #-}

module MiniJax.Tagless where

import MiniJax.Common
import Control.Monad.State
import Control.Monad.Identity

-- | The tagless final encoding of our operations
class Monad m => JaxSym m where
  type JaxVal m
  add :: JaxVal m -> JaxVal m -> m (JaxVal m)
  mul :: JaxVal m -> JaxVal m -> m (JaxVal m)
  lit :: Float -> m (JaxVal m)

-- | The example function from autodidax2
foo :: JaxSym m => JaxVal m -> m (JaxVal m)
foo x = do
  c <- lit 3.0
  y <- add x c
  mul x y

-- | Evaluation interpreter - just performs arithmetic
newtype Eval a = Eval { unEval :: Identity a }
  deriving (Functor, Applicative, Monad)

runEval :: Eval a -> a
runEval = runIdentity . unEval

instance JaxSym Eval where
  type JaxVal Eval = Float
  add x y = return (x + y)
  mul x y = return (x * y)
  lit x = return x

-- | Interpret an AST using tagless final
interpret :: JaxSym m => Expr -> m (JaxVal m)
interpret (Lit x) = lit x
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

runJVP :: JVP a -> a
runJVP (JVP x) = runIdentity x

instance JaxSym JVP where
  type JaxVal JVP = Dual
  add x y = return (Dual (primal x + primal y) (tangent x + tangent y))
  mul x y = return (Dual (primal x * primal y) (primal x * tangent y + tangent x * primal y))
  lit x = return (Dual x 0.0)

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
  type JaxVal Stage = Atom
  add _ _ = error "Stage add not implemented yet"
  mul _ _ = error "Stage mul not implemented yet"
  lit x = return (LitAtom x)
