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
module MiniJax.Tagless.Stage where

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
runStage = error "Stage not implemented yet"

instance JaxSym Stage where
  type JaxVal Stage = Atom
  add _ _ = error "Stage add not implemented yet"
  mul _ _ = error "Stage mul not implemented yet"
  lit x = return (LitAtom x)
