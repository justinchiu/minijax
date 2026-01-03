{-# LANGUAGE RankNTypes #-}

module TaglessSpec (spec) where
import Test.Hspec
import MiniJax.Common
import MiniJax.Tagless
import MiniJax.Tagless.Eval
import MiniJax.Tagless.JVP.Dynamic
import MiniJax.Tagless.Stage
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

atomToVar :: Atom -> Var
atomToVar (VarAtom v) = v
atomToVar _ = error "expected VarAtom"

stageFoo :: Var -> Stage Var
stageFoo x = do
  y <- add (VarAtom x) (LitAtom 3.0)
  z <- mul (VarAtom x) y
  return (atomToVar z)

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

    it "should match finite difference for foo (approx)" $ do
      let x = 2.0
          eps = 1.0e-5
          f a = runEval (foo a)
          diff = (f (x + eps) - f x) / eps
      diff `shouldSatisfy` (\d -> abs (d - 7.000009999913458) < 1.0e-6)

    it "should show primal-tangent packing example (approx)" $ do
      runEval (foo 2.00001) `shouldSatisfy` (\v -> abs (v - 10.0000700001) < 1.0e-9)

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

  describe "Higher-order AD (TaggedDynamic)" $ do
    -- foo(x) = x * (x + 3) = x^2 + 3x
    -- foo'(x) = 2x + 3
    -- foo''(x) = 2
    -- foo'''(x) = 0
    -- foo''''(x) = 0

    it "should compute 0th-order derivative of foo at x=2" $ do
      -- 0th derivative = f(2) = 2*(2+3) = 10
      TD.nthDerivativeTagged 0 foo 2.0 `shouldBe` 10.0

    it "should compute 1st-order derivative of foo at x=2" $ do
      -- 1st derivative = 2*2 + 3 = 7
      TD.nthDerivativeTagged 1 foo 2.0 `shouldBe` 7.0

    it "should compute 2nd-order derivative of foo at x=2" $ do
      -- 2nd derivative = 2
      TD.nthDerivativeTagged 2 foo 2.0 `shouldBe` 2.0

    it "should compute 3rd-order derivative of foo at x=2" $ do
      -- 3rd derivative = 0 (foo is quadratic)
      TD.nthDerivativeTagged 3 foo 2.0 `shouldBe` 0.0

    it "should compute 4th-order derivative of foo at x=2" $ do
      -- 4th derivative = 0
      TD.nthDerivativeTagged 4 foo 2.0 `shouldBe` 0.0

  describe "Staging (expected failures)" $ do
    it "should stage foo into a jaxpr" $ do
      let expected =
            Jaxpr
              { getParams = ["v_1"]
              , getEquations =
                  [ Equation "v_2" Add [VarAtom "v_1", LitAtom 3.0]
                  , Equation "v_3" Mul [VarAtom "v_1", VarAtom "v_2"]
                  ]
              , getReturn = VarAtom "v_3"
              }
      runStage stageFoo `shouldBe` expected

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
