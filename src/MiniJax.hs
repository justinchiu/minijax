-- | MiniJax: A minimal JAX-like library using tagless final encoding.
--
-- This module provides a convenient entry point that re-exports the core
-- language and common types. For more control, import specific modules:
--
-- * @MiniJax.Tagless@ - Core tagless final language (@JaxSym@)
-- * @MiniJax.Tagless.Eval@ - Evaluation interpreter
-- * @MiniJax.Tagless.JVP.Dynamic@ - Forward-mode AD interpreter
-- * @MiniJax.Common@ - Shared types (@Dual@, @Jaxpr@, @Expr@, etc.)
module MiniJax
  ( module MiniJax.Common
  , module MiniJax.Tagless
  ) where

import MiniJax.Common
import MiniJax.Tagless
