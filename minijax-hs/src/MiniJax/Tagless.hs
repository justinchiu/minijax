{-# LANGUAGE TypeFamilies #-}

-- | Tagless final encoding of MiniJax operations.
--
-- = Design: Tagless Final with Type Families
--
-- This module defines the core language interface through the 'JaxSym' type class.
-- The key idea is that programs are /polymorphic/ over the interpretation:
--
-- @
-- foo :: JaxSym m => JaxVal m -> m (JaxVal m)
-- foo x = do
--   y <- add x x
--   z <- lit 3.0
--   add y z
-- @
--
-- The same @foo@ can be:
--
-- * __Evaluated__: with @m = Eval@, @JaxVal m = Double@
-- * __Differentiated__: with @m = JVP@, @JaxVal m = Dual@
-- * __Staged__: with @m = Stage@, @JaxVal m = Atom@
--
-- = Type Family for Values
--
-- The associated type @JaxVal m@ maps each monad to its value type:
--
-- * @JaxVal Eval = Double@
-- * @JaxVal JVP = Dual@
-- * @JaxVal Stage = Atom@
--
-- This is more type-safe than the direct style ('MiniJax.Direct') which uses
-- a single @Value@ sum type, but makes higher-order AD more complex.
--
-- = Available Interpreters
--
-- * @MiniJax.Tagless.Eval@: concrete evaluation
-- * @MiniJax.Tagless.JVP.Dynamic@: untagged forward-mode AD (simple but limited)
-- * @MiniJax.Tagless.JVP.TaggedStatic@: type-level tagged AD (no perturbation confusion)
-- * @MiniJax.Tagless.JVP.TaggedDynamic@: runtime tagged AD (supports higher-order)
-- * @MiniJax.Tagless.Stage@: staging to Jaxpr IR
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
  lit :: Double -> m (JaxVal m)

-- Library users can define their own programs using 'JaxSym'.

-- | Automatic differentiation interface for interpreters that support it.
class JaxSym m => JaxAD m where
  derivative :: (JaxVal m -> m (JaxVal m)) -> Double -> m (JaxVal m)

  -- | Compute the nth-order derivative of a function at a point.
  -- Instances that support higher-order AD should override this.
  nthDerivative :: Int -> (JaxVal m -> m (JaxVal m)) -> Double -> m (JaxVal m)
  nthDerivative n _ _ = error $ "nthDerivative not implemented for n=" ++ show n
