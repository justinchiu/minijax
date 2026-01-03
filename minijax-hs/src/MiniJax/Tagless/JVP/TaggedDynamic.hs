{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}

-- | Dynamically tagged JVP interpreter.
--
-- = Design: Runtime Tags for Perturbation Safety
--
-- This module implements forward-mode AD using runtime tags to avoid
-- perturbation confusion. It closely follows the Python version in autodidax2.md.
--
-- Each JVP interpreter instance has a unique tag ('Data.Unique.Unique'), and
-- dual numbers carry the tag of their creating interpreter:
--
-- @
-- data TaggedDual = TaggedDual
--   { tag     :: Unique
--   , primalV :: TaggedValue
--   , tangentV :: TaggedValue
--   }
-- @
--
-- = Avoiding Perturbation Confusion
--
-- When lifting a value to a dual, we check if it's already a dual from the
-- /current/ interpreter. If not, it's treated as a constant:
--
-- @
-- lift v = case v of
--   TVDual d | tag d == currentTag -> d
--   _ -> TaggedDual currentTag v (zeroLike v)  -- constant!
-- @
--
-- This ensures that duals from outer differentiation contexts are correctly
-- treated as constants in inner differentiations.
--
-- = Higher-Order AD
--
-- Values are 'TaggedValue', which can be either @TVFloat Double@ or
-- @TVDual TaggedDual@. This nesting enables higher-order AD: when we
-- differentiate a derivative, the inner dual's components are themselves
-- duals from the outer differentiation.
--
-- See 'nthDerivativeTagged' for computing nth-order derivatives.
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
  = TVFloat Double
  | TVDual TaggedDual
  deriving (Eq, Show)

-- | Tagged dual number with an interpreter tag to avoid perturbation confusion.
-- Primal and tangent are TaggedValue to support nested dual numbers for higher-order AD.
data TaggedDual = TaggedDual
  { tag :: Unique   -- ^ Unique identifier for the interpreter that created this
  , primalV :: TaggedValue
  , tangentV :: TaggedValue
  } deriving (Eq)

-- | Extract primal as Double (only works for non-nested duals)
primal :: TaggedDual -> Double
primal d = valueToFloat (primalV d)

-- | Extract tangent as Double (only works for non-nested duals)
tangent :: TaggedDual -> Double
tangent d = valueToFloat (tangentV d)

-- | Convert TaggedValue to Double
valueToFloat :: TaggedValue -> Double
valueToFloat (TVFloat x) = x
valueToFloat (TVDual d) = valueToFloat (primalV d)

-- | Create zero value with same structure
zeroLike :: TaggedValue -> TaggedValue
zeroLike (TVFloat _) = TVFloat 0.0
zeroLike (TVDual d) = TVDual $ TaggedDual (tag d) (zeroLike (primalV d)) (zeroLike (tangentV d))

oneLike :: TaggedValue -> TaggedValue
oneLike (TVFloat _) = TVFloat 1.0
oneLike (TVDual d) =
  let p = oneLike (primalV d)
      t = zeroLike (tangentV d)
  in TVDual $ TaggedDual (tag d) p t

promoteScalarValue :: Double -> TaggedValue -> TaggedValue
promoteScalarValue x (TVFloat _) = TVFloat x
promoteScalarValue x (TVDual d) =
  let p = promoteScalarValue x (primalV d)
      t = zeroLike (tangentV d)
  in TVDual $ TaggedDual (tag d) p t

promoteScalar :: Double -> TaggedDual -> TaggedValue
promoteScalar x d =
  let p = promoteScalarValue x (primalV d)
      t = zeroLike (tangentV d)
  in TVDual $ TaggedDual (tag d) p t

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
runTaggedTangent :: TaggedDynamic TaggedDual -> Double
runTaggedTangent m = tangent (runTaggedDual m)

-- | Create a tagged dual number with Double primal and tangent
taggedDual :: Double -> Double -> TaggedDynamic TaggedDual
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
addValue (TVFloat x) (TVDual dy) = addValue (promoteScalar x dy) (TVDual dy)
addValue (TVDual dx) (TVFloat y) = addValue (TVDual dx) (promoteScalar y dx)

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
mulValue (TVFloat x) (TVDual dy) = mulValue (promoteScalar x dy) (TVDual dy)
mulValue (TVDual dx) (TVFloat y) = mulValue (TVDual dx) (promoteScalar y dx)

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
  let (_, t) = jvpTaggedValue f v (oneLike v)
  in t

buildNestedValue :: Int -> Double -> TaggedValue
buildNestedValue 0 x = TVFloat x
buildNestedValue n x =
  let inner = buildNestedValue (n - 1) x
      innerTag = unsafePerformIO newUnique
  in TVDual (TaggedDual innerTag inner (oneLike inner))

extractNthTangent :: Int -> TaggedDual -> TaggedValue
extractNthTangent 1 d = tangentV d
extractNthTangent n d =
  case tangentV d of
    TVDual inner -> extractNthTangent (n - 1) inner
    TVFloat _ -> error "extractNthTangent: unexpected scalar tangent"

-- | Standalone function for higher-order derivatives (exported for testing)
nthDerivativeTagged :: Int -> (TaggedDual -> TaggedDynamic TaggedDual) -> Double -> Double
nthDerivativeTagged n f x
  | n < 0 = error "nthDerivativeTagged: n must be non-negative"
  | n == 0 =
      let result = runTaggedDual $ do
            input <- taggedDual x 0.0
            f input
      in valueToFloat (primalV result)
  | otherwise =
      let inner = buildNestedValue (n - 1) x
          result = runTaggedDual $ do
            input <- taggedDualValue inner (oneLike inner)
            f input
      in valueToFloat (extractNthTangent n result)
