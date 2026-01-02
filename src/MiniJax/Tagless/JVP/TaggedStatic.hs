{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

-- | Static (type-level tagged) JVP interpreter.
--
-- This interpreter avoids perturbation confusion by threading a phantom type
-- parameter through dual numbers. Each run of 'runStatic' chooses a fresh type
-- parameter, so duals from different runs cannot be mixed.
module MiniJax.Tagless.JVP.TaggedStatic
  ( TaggedStatic
  , TaggedDual
  , runTaggedStatic
  , runTaggedStaticDual
  , runTaggedStaticTangent
  , taggedDual
  , liftTagged
  , primal
  , tangent
  ) where

import Control.Monad.Identity
import Unsafe.Coerce (unsafeCoerce)
import qualified MiniJax.Common as Common
import MiniJax.Tagless

-- | Static JVP interpreter with a phantom tag.
newtype TaggedStatic s a = TaggedStatic { unTaggedStatic :: Identity a }
  deriving (Functor, Applicative, Monad)

-- | Dual numbers tagged at the type level.
newtype TaggedDual s = TaggedDual { unTaggedDual :: Common.Dual }
  deriving (Show, Eq)

-- | Run a static computation with a fresh type-level tag.
runTaggedStatic :: forall a. (forall s. TaggedStatic s a) -> a
runTaggedStatic m = runIdentity (unTaggedStatic (m :: TaggedStatic () a))

-- | Run a static computation that returns a tagged dual and erase the tag.
runTaggedStaticDual :: (forall s. TaggedStatic s (TaggedDual s)) -> Common.Dual
runTaggedStaticDual m = unTaggedDual (runIdentity (unTaggedStatic (m :: TaggedStatic () (TaggedDual ()))))

-- | Run a static computation and return the tangent component.
runTaggedStaticTangent :: (forall s. TaggedStatic s (TaggedDual s)) -> Float
runTaggedStaticTangent m = Common.tangent (runTaggedStaticDual m)

-- | Construct a tagged dual number.
taggedDual :: Float -> Float -> TaggedStatic s (TaggedDual s)
taggedDual p t = return (TaggedDual (Common.Dual p t))

-- | Lift a dual from another tag as a constant in the current tag.
liftTagged :: TaggedDual s -> TaggedStatic t (TaggedDual t)
liftTagged d = taggedDual (primal d) 0.0

-- | Accessors for tagged dual numbers.
primal :: TaggedDual s -> Float
primal = Common.primal . unTaggedDual

tangent :: TaggedDual s -> Float
tangent = Common.tangent . unTaggedDual

instance JaxSym (TaggedStatic s) where
  type JaxVal (TaggedStatic s) = TaggedDual s
  add x y = return (TaggedDual (Common.Dual (primal x + primal y) (tangent x + tangent y)))
  mul x y =
    let p = primal x * primal y
        t = primal x * tangent y + tangent x * primal y
    in return (TaggedDual (Common.Dual p t))
  lit x = taggedDual x 0.0

instance JaxAD (TaggedStatic s) where
  derivative f x = do
    -- Run the inner derivative in a fresh tag; coerce f across tags so a
    -- simple `g y = return x` API is possible at call sites.
    let f' :: forall t. TaggedDual t -> TaggedStatic t (TaggedDual t)
        f' = unsafeCoerce f
        t = runTaggedStaticTangent $ do
          input <- taggedDual x 1.0
          result <- f' input
          liftTagged result
    lit t
