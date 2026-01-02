{-# LANGUAGE TypeFamilies #-}

-- | Tagless final encoding of MiniJax operations.
--
-- This module defines the core language interface through the 'JaxSym' type class.
-- Different interpreters implement 'JaxSym' to provide different semantics:
--
-- * Evaluation: @MiniJax.Tagless.Eval@ interprets values as 'Float'
-- * Forward-mode AD: @MiniJax.Tagless.JVP.Dynamic@ interprets values as 'Dual'
-- * Staging: @MiniJax.Tagless.Stage@ interprets values as 'Atom' (for IR construction)
--
-- The same program can be interpreted in multiple ways without
-- modification, enabling evaluation, differentiation, compilation, etc.
module MiniJax.Tagless
  ( JaxSym(..)
  , JaxAD(..)
  ) where

-- | The tagless final encoding of our operations.
-- JaxSym m uses 'JaxVal m' as the concrete value type interpreted by monad m.
class Monad m => JaxSym m where
  type JaxVal m
  add :: JaxVal m -> JaxVal m -> m (JaxVal m)
  mul :: JaxVal m -> JaxVal m -> m (JaxVal m)
  lit :: Float -> m (JaxVal m)

-- Library users can define their own programs using 'JaxSym'.

-- | Automatic differentiation interface for interpreters that support it.
class JaxSym m => JaxAD m where
  derivative :: (JaxVal m -> m (JaxVal m)) -> Float -> m (JaxVal m)
