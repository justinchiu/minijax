{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeFamilies #-}

module MiniJax.Tagless.JVP.Dynamic where

import Control.Monad.Identity
import MiniJax.Common
import MiniJax.Tagless

-- | JVP (forward-mode AD) interpreter using untagged dual numbers.
newtype JVP a = JVP (Identity a)
  deriving (Functor, Applicative, Monad)

runJVP :: JVP a -> a
runJVP (JVP x) = runIdentity x

instance JaxSym JVP where
  type JaxVal JVP = Dual
  add x y = return (Dual (primal x + primal y) (tangent x + tangent y))
  mul x y = return (Dual (primal x * primal y) (primal x * tangent y + tangent x * primal y))
  lit x = return (Dual x 0.0)
