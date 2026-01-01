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
-- The same program (e.g., 'foo') can be interpreted in multiple ways without
-- modification, enabling evaluation, differentiation, compilation, etc.
module MiniJax.Tagless where

import MiniJax.Common

-- | The tagless final encoding of our operations.
-- JaxSym m uses 'JaxVal m' as the concrete value type interpreted by monad m.
class Monad m => JaxSym m where
  type JaxVal m
  add :: JaxVal m -> JaxVal m -> m (JaxVal m)
  mul :: JaxVal m -> JaxVal m -> m (JaxVal m)
  lit :: Float -> m (JaxVal m)

-- | The example function from autodidax2.
foo :: JaxSym m => JaxVal m -> m (JaxVal m)
foo x = do
  c <- lit 3.0
  y <- add x c
  mul x y

-- | Interpret an AST using tagless final.
interpret :: JaxSym m => Expr -> m (JaxVal m)
interpret (Lit x) = lit x
interpret (EAdd e1 e2) = do
  x <- interpret e1
  y <- interpret e2
  add x y
interpret (EMul e1 e2) = do
  x <- interpret e1
  y <- interpret e2
  mul x y
