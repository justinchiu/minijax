module InterpreterSpec (spec) where

import Test.Hspec
import MiniJax

spec :: Spec
spec = do
  describe "Interpreter-based evaluation" $ do
    it "should evaluate with EvalInterpreter" $ do
      -- The add and mul functions should now dispatch to current interpreter
      -- With the default EvalInterpreter, should work same as before
      interpretOp evalInterpreter Add [2.0, 3.0] `shouldBe` 5.0
      interpretOp evalInterpreter Mul [2.0, 3.0] `shouldBe` 6.0

    it "should handle the foo example" $ do
      -- Using the interpreter-dispatched add/mul
      let foo x = mul x (add x 3.0)
      foo 2.0 `shouldBe` 10.0