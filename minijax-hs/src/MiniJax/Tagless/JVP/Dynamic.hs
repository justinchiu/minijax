{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeFamilies #-}

-- | Forward-mode automatic differentiation using untagged dual numbers.
--
-- = Design: Simple Untagged JVP
--
-- This is the simplest forward-mode AD interpreter. Values are 'Dual' numbers
-- with primal and tangent components:
--
-- @
-- data Dual = Dual { primal :: Double, tangent :: Double }
-- @
--
-- The JVP rules are:
--
-- * @add (Dual px tx) (Dual py ty) = Dual (px + py) (tx + ty)@
-- * @mul (Dual px tx) (Dual py ty) = Dual (px * py) (px * ty + tx * py)@
--
-- = Perturbation Confusion
--
-- __Warning__: This interpreter suffers from \"perturbation confusion\" in
-- higher-order differentiation. Consider:
--
-- @
-- f x = let g y = x in derivative g 0.0
-- derivative f 0.0  -- Should be 0, but may give wrong answer!
-- @
--
-- The inner @g@ is constant in @y@, so its derivative should be 0. But with
-- untagged duals, the @x@ (which is a Dual from the outer differentiation)
-- gets confused with the inner differentiation variable.
--
-- For correct higher-order AD, use 'MiniJax.Tagless.JVP.TaggedDynamic' or
-- 'MiniJax.Tagless.JVP.TaggedStatic'.
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
runJVPTangent :: JVP Dual -> Double
runJVPTangent m = tangent (runJVPDual m)

-- | Construct an untagged dual number.
dual :: Double -> Double -> Dual
dual = Dual

instance JaxSym JVP where
  type JaxVal JVP = Dual
  add x y = return (Dual (primal x + primal y) (tangent x + tangent y))
  mul x y = return (Dual (primal x * primal y) (primal x * tangent y + tangent x * primal y))
  lit x = return (Dual x 0.0)

instance JaxAD JVP where
  derivative f x = do
    let t = tangent (runJVP (f (Dual x 1.0)))
    lit t
