{-# LANGUAGE RankNTypes #-}

module TaglessSpec (spec) where

import Test.Hspec
import MiniJax.Common
import MiniJax.Tagless
import MiniJax.Tagless.Eval
import MiniJax.Tagless.JVP.Dynamic
import qualified MiniJax.Tagless.JVP.TaggedDynamic as TD
import qualified MiniJax.Tagless.JVP.Static as TS

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
      let makeDual primalVal tangentVal = Dual primalVal tangentVal
          derivative func x = tangent (runJVP (func (makeDual x 1.0)))
          testFunc x = do
            let g _ = return x
            let shouldBeZero = derivative g 0.0
            z <- lit shouldBeZero
            mul x z
          result = derivative testFunc 0.0
      -- Expect the wrong answer (1.0) due to perturbation confusion bug
      -- Correct answer would be 0.0, but Dynamic gives 1.0
      result `shouldBe` 1.0

    it "Static should pass perturbation confusion" $ do
      -- Static uses type-level tags to avoid perturbation confusion
      let derivative :: (forall s. TS.SDual s -> TS.Static s (TS.SDual s)) -> Float -> Float
          derivative func x = TS.runStaticTangent $ do
            input <- TS.staticDual x 1.0
            func input
          testFunc :: TS.SDual s -> TS.Static s (TS.SDual s)
          testFunc x = do
            let g :: TS.SDual s' -> TS.Static s' (TS.SDual s')
                g _ = TS.liftStatic x
            let shouldBeZero = derivative g 0.0
            z <- lit shouldBeZero
            mul x z
          result = derivative testFunc 0.0
      result `shouldBe` 0.0

    it "TaggedDynamic should pass perturbation confusion" $ do
      -- TaggedDynamic uses runtime tags to avoid perturbation confusion
      -- The derivative function creates a new TaggedDynamic context for each differentiation
      let derivative :: (TD.TaggedDual -> TD.TaggedDynamic TD.TaggedDual) -> Float -> Float
          derivative func x = TD.tangent (TD.runTaggedDynamic $ do
            input <- TD.taggedDual x 1.0  -- Create input with current interpreter's tag
            funcResult <- func input
            TD.liftTagged funcResult)  -- Lift result to treat values from other contexts as constants
          testFunc :: TD.TaggedDual -> TD.TaggedDynamic TD.TaggedDual
          testFunc x = do
            -- g is constant in its argument, so its derivative should be 0
            let g :: TD.TaggedDual -> TD.TaggedDynamic TD.TaggedDual
                g _ = return x  -- Returns x from outer context
            let shouldBeZero = derivative g 0.0
            z <- lit shouldBeZero  -- This creates a TaggedDual with tangent=0 (constant)
            mul x z
          result = derivative testFunc 0.0
      result `shouldBe` 0.0
