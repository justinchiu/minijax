{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeFamilies #-}

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
