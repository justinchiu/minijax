{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeFamilies #-}

-- | Forward-mode automatic differentiation using untagged dual numbers.
--
-- This interpreter implements JVP (Jacobian-vector product) for forward-mode AD.
-- Values are interpreted as 'Dual' numbers carrying both primal and tangent
-- components. The derivative of a function @f@ at point @x@ with tangent @v@
-- can be computed by evaluating @f@ with input @Dual x v@ and reading the
-- tangent component of the result.
--
-- Note: This uses /untagged/ dual numbers, which can suffer from "perturbation
-- confusion" in higher-order differentiation scenarios. See the test suite
-- for an example of this limitation.
module MiniJax.Tagless.JVP.Dynamic
  ( JVP
  , runJVP
  , runJVPDual
  , runJVPTangent
  , dual
  , primal
  , tangent
  ) where

import Control.Monad.Identity
import MiniJax.Common
import MiniJax.Tagless

-- | JVP (forward-mode AD) interpreter using untagged dual numbers.
newtype JVP a = JVP (Identity a)
  deriving (Functor, Applicative, Monad)

runJVP :: JVP a -> a
runJVP (JVP x) = runIdentity x

-- | Run a JVP computation returning a dual number.
runJVPDual :: JVP Dual -> Dual
runJVPDual = runJVP

-- | Run a JVP computation and return the tangent component.
runJVPTangent :: JVP Dual -> Float
runJVPTangent m = tangent (runJVPDual m)

-- | Construct an untagged dual number.
dual :: Float -> Float -> Dual
dual = Dual

instance JaxSym JVP where
  type JaxVal JVP = Dual
  add x y = return (Dual (primal x + primal y) (tangent x + tangent y))
  mul x y = return (Dual (primal x * primal y) (primal x * tangent y + tangent x * primal y))
  lit x = return (Dual x 0.0)
