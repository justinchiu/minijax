{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeFamilies #-}

-- | Staging interpreter: converts programs to Jaxpr IR.
--
-- This interpreter stages programs into the 'Jaxpr' IR representation, enabling
-- program transformations that require examining the entire program structure
-- (e.g., dead-code elimination, reverse-mode AD via transposition).
--
-- /Note: This module is incomplete. The interpreter structure exists but/
-- /operations are not yet implemented./
module MiniJax.Tagless.Stage
  ( Stage(..)
  , runStage
  ) where

import Control.Monad.State
import MiniJax.Common
import MiniJax.Tagless

-- | Placeholder for Stage interpreter.
newtype Stage a = Stage (State StageState a)
  deriving (Functor, Applicative, Monad)

data StageState = StageState
  { equations :: [Equation]
  , nameCounter :: Int
  }

runStage :: (Var -> Stage Var) -> Jaxpr
runStage f =
  let inputVar = "v_1"
      initial = StageState { equations = [], nameCounter = 1 }
      Stage m = f inputVar
      (outVar, finalState) = runState m initial
  in Jaxpr
      { getParams = [inputVar]
      , getEquations = equations finalState
      , getReturn = VarAtom outVar
      }

freshVar :: Stage Var
freshVar = Stage $ do
  st <- get
  let next = nameCounter st + 1
      v = "v_" ++ show next
  put st { nameCounter = next }
  return v

emit :: Op -> Atom -> Atom -> Stage Atom
emit op x y = do
  v <- freshVar
  Stage $ modify (\st -> st { equations = equations st ++ [Equation v op [x, y]] })
  return (VarAtom v)

instance JaxSym Stage where
  type JaxVal Stage = Atom
  add x y = emit Add x y
  mul x y = emit Mul x y
  lit x = return (LitAtom x)
