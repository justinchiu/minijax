module EvalSpec (spec) where

import Test.Hspec
import MiniJax

spec :: Spec
spec = do
  describe "Basic evaluation" $ do
    it "should evaluate add" $ do
      add 2.0 3.0 `shouldBe` 5.0

    it "should evaluate mul" $ do
      mul 2.0 3.0 `shouldBe` 6.0

    it "should evaluate foo(2.0) = 10.0" $ do
      let foo x = mul x (add x 3.0)
      foo 2.0 `shouldBe` 10.0