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