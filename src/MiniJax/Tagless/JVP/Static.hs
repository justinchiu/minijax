{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

-- | Static (type-level tagged) JVP interpreter.
--
-- This interpreter avoids perturbation confusion by threading a phantom type
-- parameter through dual numbers. Each run of 'runStatic' chooses a fresh type
-- parameter, so duals from different runs cannot be mixed.
module MiniJax.Tagless.JVP.Static
  ( Static
  , SDual
  , runStatic
  , runStaticDual
  , runStaticTangent
  , staticDual
  , liftStatic
  , sPrimal
  , sTangent
  ) where

import Control.Monad.Identity
import MiniJax.Common
import MiniJax.Tagless

-- | Static JVP interpreter with a phantom tag.
newtype Static s a = Static { unStatic :: Identity a }
  deriving (Functor, Applicative, Monad)

-- | Dual numbers tagged at the type level.
newtype SDual s = SDual { unSDual :: Dual }
  deriving (Show, Eq)

-- | Run a static computation with a fresh type-level tag.
runStatic :: forall a. (forall s. Static s a) -> a
runStatic m = runIdentity (unStatic (m :: Static () a))

-- | Run a static computation that returns a tagged dual and erase the tag.
runStaticDual :: (forall s. Static s (SDual s)) -> Dual
runStaticDual m = unSDual (runIdentity (unStatic (m :: Static () (SDual ()))))

-- | Run a static computation and return the tangent component.
runStaticTangent :: (forall s. Static s (SDual s)) -> Float
runStaticTangent m = tangent (runStaticDual m)

-- | Construct a tagged dual number.
staticDual :: Float -> Float -> Static s (SDual s)
staticDual p t = return (SDual (Dual p t))

-- | Lift a dual from another tag as a constant in the current tag.
liftStatic :: SDual s -> Static t (SDual t)
liftStatic d = staticDual (sPrimal d) 0.0

-- | Accessors for tagged dual numbers.
sPrimal :: SDual s -> Float
sPrimal = primal . unSDual

sTangent :: SDual s -> Float
sTangent = tangent . unSDual

instance JaxSym (Static s) where
  type JaxVal (Static s) = SDual s
  add x y = return (SDual (Dual (sPrimal x + sPrimal y) (sTangent x + sTangent y)))
  mul x y =
    let p = sPrimal x * sPrimal y
        t = sPrimal x * sTangent y + sTangent x * sPrimal y
    in return (SDual (Dual p t))
  lit x = staticDual x 0.0
