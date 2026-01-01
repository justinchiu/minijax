{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeFamilies #-}

-- | Evaluation interpreter: performs ordinary arithmetic on concrete values.
--
-- This is the simplest interpreter, evaluating operations directly on 'Float'
-- values. Use this for normal computation without any transformations.
module MiniJax.Tagless.Eval where

import Control.Monad.Identity
import MiniJax.Tagless

-- | Evaluation interpreter - just performs arithmetic.
-- We wrap Identity so GND can derive Functor/Applicative/Monad.
newtype Eval a = Eval { unEval :: Identity a }
  deriving (Functor, Applicative, Monad)

runEval :: Eval a -> a
runEval = runIdentity . unEval

instance JaxSym Eval where
  type JaxVal Eval = Float
  add x y = return (x + y)
  mul x y = return (x * y)
  lit x = return x
