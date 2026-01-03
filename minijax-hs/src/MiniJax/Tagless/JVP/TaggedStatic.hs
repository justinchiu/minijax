{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

-- | Static (type-level tagged) JVP interpreter.
--
-- = Design: Type-Level Tags (Phantom Types)
--
-- This interpreter avoids perturbation confusion by threading a phantom type
-- parameter through dual numbers. Each differentiation context has a unique
-- type tag @s@:
--
-- @
-- newtype TaggedStatic s a = TaggedStatic (Identity a)
-- newtype TaggedDual s = TaggedDual Dual
-- @
--
-- The phantom @s@ ensures duals from different contexts have incompatible types:
--
-- @
-- runTaggedStatic :: (forall s. TaggedStatic s a) -> a
-- @
--
-- The @forall s@ ensures the computation cannot depend on a specific tag,
-- which would allow mixing duals from different contexts.
--
-- = Type Safety vs Flexibility
--
-- This approach catches perturbation confusion at /compile time/, which is
-- the most type-safe option. However, it has limitations:
--
-- * Requires rank-2 types
-- * Uses @unsafeCoerce@ internally for nested differentiation
-- * Higher-order AD is more complex than with runtime tags
--
-- For simpler higher-order AD, see 'MiniJax.Tagless.JVP.TaggedDynamic'.
--
-- = Comparison to OCaml's Module System
--
-- This is analogous to the OCaml tagless-final approach with fresh type tags
-- via functor application (see @minijax_tagged.ml@), but uses Haskell's
-- rank-2 polymorphism instead of generative functors.
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
runTaggedStaticTangent :: (forall s. TaggedStatic s (TaggedDual s)) -> Double
runTaggedStaticTangent m = Common.tangent (runTaggedStaticDual m)

-- | Construct a tagged dual number.
taggedDual :: Double -> Double -> TaggedStatic s (TaggedDual s)
taggedDual p t = return (TaggedDual (Common.Dual p t))

-- | Lift a dual from another tag as a constant in the current tag.
liftTagged :: TaggedDual s -> TaggedStatic t (TaggedDual t)
liftTagged d = taggedDual (primal d) 0.0

-- | Accessors for tagged dual numbers.
primal :: TaggedDual s -> Double
primal = Common.primal . unTaggedDual

tangent :: TaggedDual s -> Double
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
