{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}

-- | Dynamically tagged JVP interpreter.
--
-- This module implements forward-mode AD using runtime tags to avoid perturbation
-- confusion. Each JVP interpreter instance has a unique tag, and dual numbers
-- are tagged with the interpreter that created them. When lifting values, if a
-- dual number has a different tag, it's treated as a constant (tangent = 0).
--
-- This is similar to the Python version in autodidax2.md.
module MiniJax.Tagless.JVP.TaggedDynamic
  ( TaggedDynamic
  , TaggedDual(..)
  , TaggedValue(..)
  , runTaggedDynamic
  , runTaggedDual
  , runTaggedTangent
  , taggedDual
  , taggedDualValue
  , liftTagged
  , primal
  , tangent
  , valueToFloat
  , nthDerivativeTagged
  ) where

import Control.Monad.Reader
import Data.Unique (Unique, newUnique)
import System.IO.Unsafe (unsafePerformIO)
import MiniJax.Tagless

-- | Values that flow through the TaggedDynamic interpreter.
-- Can be a plain float or a tagged dual number (for nesting).
data TaggedValue
  = TVFloat Float
  | TVDual TaggedDual
  deriving (Eq, Show)

-- | Tagged dual number with an interpreter tag to avoid perturbation confusion.
-- Primal and tangent are TaggedValue to support nested dual numbers for higher-order AD.
data TaggedDual = TaggedDual
  { tag :: Unique   -- ^ Unique identifier for the interpreter that created this
  , primalV :: TaggedValue
  , tangentV :: TaggedValue
  } deriving (Eq)

-- | Extract primal as Float (only works for non-nested duals)
primal :: TaggedDual -> Float
primal d = valueToFloat (primalV d)

-- | Extract tangent as Float (only works for non-nested duals)
tangent :: TaggedDual -> Float
tangent d = valueToFloat (tangentV d)

-- | Convert TaggedValue to Float
valueToFloat :: TaggedValue -> Float
valueToFloat (TVFloat x) = x
valueToFloat (TVDual d) = valueToFloat (primalV d)

-- | Create zero value with same structure
zeroLike :: TaggedValue -> TaggedValue
zeroLike (TVFloat _) = TVFloat 0.0
zeroLike (TVDual d) = TVDual $ TaggedDual (tag d) (zeroLike (primalV d)) (zeroLike (tangentV d))

instance Show TaggedDual where
  show (TaggedDual _ p t) =
    "TaggedDual {tag = <unique>, primal = " ++ show p ++ ", tangent = " ++ show t ++ "}"

-- | TaggedDynamic interpreter state
data TaggedDynamicState = TaggedDynamicState
  { interpreterTag :: Unique           -- ^ Tag for this interpreter instance
  , previousInterpreter :: ()   -- ^ Placeholder for previous interpreter context
  }

-- | TaggedDynamic JVP interpreter using runtime tags.
-- Uses ReaderT to track the current interpreter tag.
newtype TaggedDynamic a = TaggedDynamic (Reader TaggedDynamicState a)
  deriving (Functor, Applicative, Monad)

-- | Run a TaggedDynamic computation with a fresh interpreter tag.
--
-- Uses 'unsafePerformIO' because we need to read/modify a global 'IORef'
-- to generate unique tags, but we want a pure API. The 'do' notation here
-- is inside the 'IO' monad, and 'unsafePerformIO' converts 'IO a' to 'a'.
runTaggedDynamic :: TaggedDynamic a -> a
runTaggedDynamic (TaggedDynamic m) = unsafePerformIO $ do
  newTag <- newUnique
  return (runReader m (TaggedDynamicState { interpreterTag = newTag, previousInterpreter = () }))

-- | Run a TaggedDynamic computation returning a tagged dual.
runTaggedDual :: TaggedDynamic TaggedDual -> TaggedDual
runTaggedDual = runTaggedDynamic

-- | Run a TaggedDynamic computation and return the tangent component.
runTaggedTangent :: TaggedDynamic TaggedDual -> Float
runTaggedTangent m = tangent (runTaggedDual m)

-- | Create a tagged dual number with Float primal and tangent
taggedDual :: Float -> Float -> TaggedDynamic TaggedDual
taggedDual p t = taggedDualValue (TVFloat p) (TVFloat t)

-- | Create a tagged dual number with TaggedValue primal and tangent (for nesting)
taggedDualValue :: TaggedValue -> TaggedValue -> TaggedDynamic TaggedDual
taggedDualValue p t = TaggedDynamic $ do
  TaggedDynamicState { interpreterTag = currentTag } <- ask
  return (TaggedDual currentTag p t)

-- | Lift a value to a TaggedDual. If it's already a TaggedDual with the same
-- tag, return it. Otherwise, treat it as a constant (tangent = 0).
liftTagged :: TaggedDual -> TaggedDynamic TaggedDual
liftTagged td = TaggedDynamic $ do
  TaggedDynamicState { interpreterTag = currentTag } <- ask
  if tag td == currentTag
    then return td
    else return (TaggedDual currentTag (TVDual td) (zeroLike (TVDual td)))

-- | Add two TaggedValues (for nested AD)
addValue :: TaggedValue -> TaggedValue -> TaggedValue
addValue (TVFloat x) (TVFloat y) = TVFloat (x + y)
addValue (TVDual dx) (TVDual dy) =
  -- When adding nested duals, we add component-wise
  -- This only works if they have the same tag structure
  TVDual $ TaggedDual (tag dx) (addValue (primalV dx) (primalV dy)) (addValue (tangentV dx) (tangentV dy))
addValue _ _ = error "addValue: mismatched value types"

-- | Multiply two TaggedValues (for nested AD)
mulValue :: TaggedValue -> TaggedValue -> TaggedValue
mulValue (TVFloat x) (TVFloat y) = TVFloat (x * y)
mulValue (TVDual dx) (TVDual dy) =
  -- Product rule on nested duals
  let px = primalV dx
      py = primalV dy
      tx = tangentV dx
      ty = tangentV dy
      p = mulValue px py
      t = addValue (mulValue px ty) (mulValue tx py)
  in TVDual $ TaggedDual (tag dx) p t
mulValue _ _ = error "mulValue: mismatched value types"

instance JaxSym TaggedDynamic where
  type JaxVal TaggedDynamic = TaggedDual

  add x y = do
    x' <- liftTagged x
    y' <- liftTagged y
    let p = addValue (primalV x') (primalV y')
        t = addValue (tangentV x') (tangentV y')
    taggedDualValue p t

  mul x y = do
    x' <- liftTagged x
    y' <- liftTagged y
    -- Product rule: (fg)' = f'g + fg'
    let px = primalV x'
        py = primalV y'
        tx = tangentV x'
        ty = tangentV y'
        p = mulValue px py
        t = addValue (mulValue px ty) (mulValue tx py)
    taggedDualValue p t

  lit x = taggedDual x 0.0

instance JaxAD TaggedDynamic where
  derivative f x = do
    let t = runTaggedTangent $ do
          input <- taggedDual x 1.0
          result <- f input
          liftTagged result
    lit t

  -- Higher-order derivatives via nthDerivativeTagged
  nthDerivative n f x = lit $ nthDerivativeTagged n f x

-- | Compute JVP with TaggedValue inputs (supports nested duals for higher-order AD)
jvpTaggedValue :: (TaggedDual -> TaggedDynamic TaggedDual) -> TaggedValue -> TaggedValue -> (TaggedValue, TaggedValue)
jvpTaggedValue f primalIn tangentIn = unsafePerformIO $ do
  newTag <- newUnique
  let state = TaggedDynamicState { interpreterTag = newTag, previousInterpreter = () }
      TaggedDynamic m = do
        let dualIn = TaggedDual newTag primalIn tangentIn
        result <- f dualIn
        lifted <- liftTagged result
        return (primalV lifted, tangentV lifted)
  return (runReader m state)

-- | Compute derivative with TaggedValue input (for higher-order AD)
derivativeTaggedValue :: (TaggedDual -> TaggedDynamic TaggedDual) -> TaggedValue -> TaggedValue
derivativeTaggedValue f v =
  let (_, t) = jvpTaggedValue f v (TVFloat 1.0)
  in t

-- | Compute nth derivative using nested TaggedValues.
-- The key insight: for higher-order AD, we pass TaggedValues through recursion,
-- allowing the dual number structure to nest properly.
nthDerivativeTaggedValue :: Int -> (TaggedDual -> TaggedDynamic TaggedDual) -> TaggedValue -> TaggedValue
nthDerivativeTaggedValue n f v
  | n == 0    =
      -- Just evaluate f at v
      let result = unsafePerformIO $ do
            newTag <- newUnique
            let state = TaggedDynamicState { interpreterTag = newTag, previousInterpreter = () }
                TaggedDynamic m = do
                  let dualIn = TaggedDual newTag v (zeroLike v)
                  r <- f dualIn
                  liftTagged r
            return (runReader m state)
      in primalV result
  | otherwise =
      -- Differentiate the function that computes (n-1)th derivative
      derivativeTaggedValue (\u -> do
        let innerResult = nthDerivativeTaggedValue (n - 1) f (TVDual u)
        taggedDualValue innerResult (TVFloat 0.0)
      ) v

-- | Standalone function for higher-order derivatives (exported for testing)
nthDerivativeTagged :: Int -> (TaggedDual -> TaggedDynamic TaggedDual) -> Float -> Float
nthDerivativeTagged n f x = valueToFloat $ nthDerivativeTaggedValue n f (TVFloat x)
