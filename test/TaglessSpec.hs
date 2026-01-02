{-# LANGUAGE RankNTypes #-}

module TaglessSpec (spec) where
import Test.Hspec
import MiniJax.Common
import MiniJax.Tagless
import MiniJax.Tagless.Eval
import MiniJax.Tagless.JVP.Dynamic
import qualified MiniJax.Tagless.JVP.TaggedDynamic as TD
import qualified MiniJax.Tagless.JVP.TaggedStatic as TS

-- Test-only example function from autodidax2.
-- Python:
-- def foo(x):
--   return mul(x, add(x, 3.0))
foo :: JaxSym m => JaxVal m -> m (JaxVal m)
foo x = do
  c <- lit 3.0
  y <- add x c
  mul x y

-- Test-only helper for perturbation confusion examples.
-- Python:
-- def f(x):
--   def g(y):
--     return x
--   should_be_zero = derivative(g, 0.0)
--   return mul(x, should_be_zero)
--
-- print(derivative(f, 0.0))
f :: JaxAD m => JaxVal m -> m (JaxVal m)
f x = do
  let g _ = return x
  shouldBeZero <- derivative g 0.0
  mul x shouldBeZero

-- Interpret an AST using tagless final.
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

spec :: Spec
spec = do
  describe "Tagless final evaluation" $ do
    it "should evaluate simple operations" $ do
      runEval (add 2.0 3.0) `shouldBe` 5.0
      runEval (mul 2.0 3.0) `shouldBe` 6.0

    it "should evaluate foo example" $ do
      runEval (foo 2.0) `shouldBe` 10.0

  describe "AST interpretation" $ do
    it "should interpret literals" $ do
      runEval (interpret (Lit 3.0)) `shouldBe` 3.0

    it "should interpret add" $ do
      runEval (interpret (EAdd (Lit 2.0) (Lit 3.0))) `shouldBe` 5.0

    it "should interpret mul" $ do
      runEval (interpret (EMul (Lit 2.0) (Lit 3.0))) `shouldBe` 6.0

    it "should interpret foo as AST" $ do
      -- foo(x) = x * (x + 3), with x = 2.0
      let fooAST = EMul (Lit 2.0) (EAdd (Lit 2.0) (Lit 3.0))
      runEval (interpret fooAST) `shouldBe` 10.0

  describe "JVP (forward-mode AD)" $ do
    it "should differentiate add" $ do
      -- d/dx (x + 3) at x=2 with tangent=1 => (5.0, 1.0)
      let result = runJVP (add (Dual 2.0 1.0) (Dual 3.0 0.0))
      primal result `shouldBe` 5.0
      tangent result `shouldBe` 1.0

    it "should differentiate mul" $ do
      -- d/dx (x * 3) at x=2 with tangent=1 => (6.0, 3.0)
      let result = runJVP (mul (Dual 2.0 1.0) (Dual 3.0 0.0))
      primal result `shouldBe` 6.0
      tangent result `shouldBe` 3.0

    it "should differentiate foo" $ do
      -- foo(x) = x * (x + 3), foo'(x) = 2x + 3
      -- At x=2: foo(2) = 10, foo'(2) = 7
      let result = runJVP (foo (Dual 2.0 1.0))
      primal result `shouldBe` 10.0
      tangent result `shouldBe` 7.0

  describe "JVP perturbation confusion" $ do
    -- The test function f(x) = x * derivative(g, 0) where g(y) = x.
    -- Since g is constant in y, its derivative should be 0, so f'(0) should be 0.
    -- This test exposes perturbation confusion: naive implementations will incorrectly
    -- propagate the outer derivative into the inner derivative computation.

    it "Dynamic (naive) should fail perturbation confusion [EXPECTED FAILURE]" $ do
      -- Dynamic uses untagged dual numbers, so it will incorrectly compute
      -- the derivative due to perturbation confusion. The correct answer is 0.0,
      -- but Dynamic gives 1.0 instead. This test documents the known bug by
      -- expecting the wrong answer. If Dynamic is ever fixed to give 0.0, this
      -- test will fail, alerting us that the bug has been fixed.
      let result = primal (runJVP (derivative f 0.0 :: JVP Dual))
      -- Expect the wrong answer (1.0) due to perturbation confusion bug
      -- Correct answer would be 0.0, but Dynamic gives 1.0
      result `shouldBe` 1.0

    it "TaggedStatic should pass perturbation confusion" $ do
      -- TaggedStatic uses type-level tags to avoid perturbation confusion
      let result = primal (TS.runTaggedStaticDual (derivative f 0.0))
      result `shouldBe` 0.0

    it "TaggedDynamic should pass perturbation confusion" $ do
      -- TaggedDynamic uses runtime tags to avoid perturbation confusion
      -- The derivative function creates a new TaggedDynamic context for each differentiation
      let result = TD.primal (TD.runTaggedDual (derivative f 0.0 :: TD.TaggedDynamic TD.TaggedDual))
      result `shouldBe` 0.0
