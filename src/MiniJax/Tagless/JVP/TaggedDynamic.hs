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
  , TaggedDual
  , runTaggedDynamic
  , runTaggedDual
  , runTaggedTangent
  , taggedDual
  , liftTagged
  , primal
  , tangent
  ) where

import Control.Monad.Reader
import Data.IORef
import System.IO.Unsafe (unsafePerformIO)
import MiniJax.Tagless

-- | Global counter for generating unique interpreter tags.
-- We need global mutable state to ensure each call to 'runTaggedDynamic' gets
-- a unique tag. This requires 'IORef' (which is in 'IO'), but we want a pure
-- API, so we use 'unsafePerformIO' to escape from 'IO'.
tagCounterRef :: IORef Int
tagCounterRef = unsafePerformIO (newIORef 0)
{-# NOINLINE tagCounterRef #-}

-- | Tagged dual number with an interpreter tag to avoid perturbation confusion.
data TaggedDual = TaggedDual
  { tag :: Int      -- ^ Unique identifier for the interpreter that created this
  , primal :: Float
  , tangent :: Float
  } deriving (Show, Eq)

-- | TaggedDynamic interpreter state
data TaggedDynamicState = TaggedDynamicState
  { interpreterTag :: Int              -- ^ Tag for this interpreter instance
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
-- is inside the 'IO' monad (needed for 'IORef' operations), and
-- 'unsafePerformIO' converts 'IO a' to 'a'.
runTaggedDynamic :: TaggedDynamic a -> a
runTaggedDynamic (TaggedDynamic m) = unsafePerformIO $ do
  newTag <- readIORef tagCounterRef
  modifyIORef' tagCounterRef (+1)
  return (runReader m (TaggedDynamicState { interpreterTag = newTag, previousInterpreter = () }))

-- | Run a TaggedDynamic computation returning a tagged dual.
runTaggedDual :: TaggedDynamic TaggedDual -> TaggedDual
runTaggedDual = runTaggedDynamic

-- | Run a TaggedDynamic computation and return the tangent component.
runTaggedTangent :: TaggedDynamic TaggedDual -> Float
runTaggedTangent m = tangent (runTaggedDual m)

-- | Create a tagged dual number with the current interpreter's tag
taggedDual :: Float -> Float -> TaggedDynamic TaggedDual
taggedDual p t = TaggedDynamic $ do
  TaggedDynamicState { interpreterTag = currentTag } <- ask
  return (TaggedDual currentTag p t)

-- | Lift a value to a TaggedDual. If it's already a TaggedDual with the same
-- tag, return it. Otherwise, treat it as a constant (tangent = 0).
liftTagged :: TaggedDual -> TaggedDynamic TaggedDual
liftTagged td = TaggedDynamic $ do
  TaggedDynamicState { interpreterTag = currentTag } <- ask
  if tag td == currentTag
    then return td
    else return (TaggedDual currentTag (primal td) 0.0)

instance JaxSym TaggedDynamic where
  type JaxVal TaggedDynamic = TaggedDual
  
  add x y = do
    x' <- liftTagged x
    y' <- liftTagged y
    -- For primal/tangent operations, use regular Float arithmetic
    -- (not interpreter dispatch, since we're computing on Float values)
    taggedDual (primal x' + primal y') (tangent x' + tangent y')
  
  mul x y = do
    x' <- liftTagged x
    y' <- liftTagged y
    -- Product rule: (fg)' = f'g + fg'
    let p = primal x' * primal y'
        t = primal x' * tangent y' + tangent x' * primal y'
    taggedDual p t
  
  lit x = taggedDual x 0.0
