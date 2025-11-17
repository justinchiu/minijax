module BasicTypesSpec (spec) where

import Test.Hspec
import MiniJax

spec :: Spec
spec = do
  describe "Op type" $ do
    it "should support equality" $ do
      Add == Add `shouldBe` True
      Add == Mul `shouldBe` False
      Mul == Mul `shouldBe` True

    it "should be showable" $ do
      show Add `shouldBe` "Add"
      show Mul `shouldBe` "Mul"

  describe "Atom type" $ do
    it "should create variable atoms" $ do
      let x = VarAtom "x"
      x `shouldBe` VarAtom "x"
      x `shouldNotBe` VarAtom "y"

    it "should create literal atoms" $ do
      let three = LitAtom 3.0
      three `shouldBe` LitAtom 3.0
      three `shouldNotBe` LitAtom 4.0

    it "should be showable" $ do
      show (VarAtom "x") `shouldBe` "VarAtom \"x\""
      show (LitAtom 3.0) `shouldBe` "LitAtom 3.0"