module TaglessSpec (spec) where

import Test.Hspec
import MiniJax

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